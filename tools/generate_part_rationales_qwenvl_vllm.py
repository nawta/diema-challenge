""" (vLLM variant): Qwen3-VL rationale generation at scale.

See: / tools/generate_part_rationales_qwenvl.py (transformers-
based single-clip baseline, kept for debugging/regression).

vLLM continuous batching drops the single-clip wall-clock from ~60s to an
amortised ~10-20s/clip on the 30B-A3B MoE, thanks to shared weights + packed
KV cache across concurrent videos. Useful sweet spot: ``--batch-size 16`` on
a 97GB GPU (one Blackwell 6000).

Key design differences vs the transformers script:
  - Model is loaded via ``vllm.LLM`` once. We tear it down only when the
    whole job exits.
  - Inputs are posed as ``chat(conversations, sampling_params)`` messages.
    The ``file://<path>`` scheme in the video content tells vLLM to read the
    mp4 directly via its own preprocessor, so no qwen_vl_utils dependency.
  - All other pipeline guards are identical: ``--split train`` hard-coded,
    append-only audit log, rationale cache key = sha256(render_key || prompt
    || model), idempotent skips of existing files.

Usage::

    python tools/generate_part_rationales_qwenvl_vllm.py \\
        --split train --batch-size 16 --max-clips 100

    # Full sweep (estimated ~25-50 hours on single H100/Blackwell GPU):
    python tools/generate_part_rationales_qwenvl_vllm.py \\
        --split train --batch-size 16

Implementation note on continuous batching: vLLM's scheduler will itself
subdivide a list of N conversations into runnable chunks based on available
KV-cache. We still hand it an outer --batch-size chunk so we can flush
rationale JSON + audit log periodically (protects against SIGKILL mid-run).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep names aligned with the non-vLLM script so tooling downstream uses a
# single source of truth.
from tools.generate_part_rationales_qwenvl import (  # noqa: E402
    MODEL_ID,
    PROMPT_CARD_PATH,
    SCHEMA_KEYS,
    _now_iso,
    _parse_json_or_repair,
    load_prompt_card,
    rationale_cache_key,
)


def _build_conversation(video_path: Path, prompt_card: dict) -> list[dict]:
    """Build a conversation in vLLM's OpenAI-compatible chat format.

    vLLM 0.19's chat handler expects ``type="video_url"`` (not ``"video"``)
    with the payload under ``video_url.url`` — mirroring the OpenAI
    multimodal chat schema. ``file://`` URIs are accepted and resolved by
    vLLM's multimodal loader. See
    vllm.entrypoints.chat_utils._parse_chat_message_content_part.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt_card["system"]}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path.resolve()}"},
                },
                {"type": "text", "text": prompt_card["user"]},
            ],
        },
    ]


_PART_KEYS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")


def _post_process_output(raw_text: str) -> tuple[dict, str | None]:
    """Strip <think>...</think>, parse JSON, auto-fill missing keys.

    Returns (rationale_dict, error_msg_or_None).

     failure-mode guard: if *all six* body-part strings are empty
    after parsing (token-cap ate the JSON body), treat the rationale as a
    hard failure instead of silently writing an all-placeholder row. This
    bumps the audit log's error count but prevents wasted slots in the
    rationale cache.
    """
    text = raw_text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    try:
        rationale = _parse_json_or_repair(text)
    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"
    missing = [k for k in SCHEMA_KEYS if k not in rationale]
    if missing:
        for k in missing:
            rationale[k] = "" if k != "salient_intervals" else []
        rationale["uncertainty_notes"] = (
            (rationale.get("uncertainty_notes") or "").rstrip()
            + f" [auto-filled missing keys: {missing}]"
        ).strip()
    # All-parts-empty detector.
    if all(not str(rationale.get(p, "")).strip() for p in _PART_KEYS):
        return {}, "all-parts-empty (token budget likely consumed by <think>)"
    return rationale, None


def _load_vllm_model(
    gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int,
):
    """Import-and-construct vllm.LLM. Kept inside a function so the heavy
    vllm import is paid once per process.
    """
    from vllm import LLM

    click.echo(f"Loading {MODEL_ID} via vLLM ... (first load takes ~2-4 min)")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        # Needed for the video processor; quiet down some warnings.
        trust_remote_code=True,
        # Let vLLM decide tensor parallel (we have 1 GPU).
        tensor_parallel_size=1,
        # vLLM 0.19 sandboxes local-file URIs by default; whitelist our
        # render output directory so ``file://output/rationale_cache/renders/…``
        # loads are accepted.
        allowed_local_media_path=str(Path("output/rationale_cache/renders").resolve()),
    )
    click.echo(f"  loaded in {time.time() - t0:.1f}s")
    return llm


@click.command()
@click.option("--split", required=True, type=click.Choice(["train"]),
              help="Only 'train' is accepted — test split is never sent to the VLM.")
@click.option("--render-dir", default="output/rationale_cache/renders")
@click.option("--output-dir", default="output/rationale_cache/rationales")
@click.option("--audit-log", default="output/rationale_cache/audit_log.jsonl")
@click.option("--max-clips", default=0, type=int,
              help="If >0, process only the first N queued entries (smoke test).")
@click.option("--batch-size", default=16, type=int,
              help="Number of conversations fed to llm.chat() per flush cycle. "
              "vLLM itself will continuously batch within this chunk.")
@click.option("--max-new-tokens", default=6144, type=int,
              help="Thinking-mode budget. audit on 218 rationales "
                   "showed 6/218 (2.8%) at 4096 emit JSON shell with empty "
                   "part strings (<think>…</think> ate the whole budget). "
                   "6144 drops that failure mode to near-zero.")
@click.option("--max-model-len", default=16384, type=int,
              help="vLLM KV-cache sequence length. Includes prompt + vision "
              "token budget + max_new_tokens. Qwen3-VL uses many vision tokens "
              "(~1000 per 64-frame 350x400 clip), so keep ≥ 8k.")
@click.option("--max-num-seqs", default=8, type=int,
              help="Max concurrent sequences in the vLLM scheduler. Reduce if "
              "you hit OOM; raise for more throughput.")
@click.option("--gpu-mem-util", default=0.90, type=float,
              help="Fraction of GPU memory vLLM is allowed to use.")
def main(
    split: str,
    render_dir: str,
    output_dir: str,
    audit_log: str,
    max_clips: int,
    batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_mem_util: float,
) -> None:
    # Hard split guard.
    if split != "train":
        raise click.ClickException("split must be 'train' — test is never sent to the VLM.")

    # Point HF cache at data (same as other tools).
    os.environ.setdefault("HF_HOME", "data/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "data/hf_cache")
    # vLLM uses transformers under the hood for the processor; decord remains
    # the video backend.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    prompt_card = load_prompt_card()
    prompt_version = str(prompt_card["schema_version"])

    render_dir_p = Path(render_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    audit_log_p = Path(audit_log)
    audit_log_p.parent.mkdir(parents=True, exist_ok=True)

    index_path = render_dir_p / "render_index.jsonl"
    if not index_path.exists():
        raise click.ClickException(
            f"render index not found: {index_path} "
            "(run tools/render_motion_video.py first)"
        )

    # Queue
    todo: list[dict] = []
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stem = row["stem"]
            render_key = row["cache_key"]
            rat_key = rationale_cache_key(render_key, prompt_version, MODEL_ID)
            if (output_dir_p / f"{rat_key}.json").exists():
                continue
            video_path = render_dir_p / f"{render_key}.mp4"
            if not video_path.exists():
                click.echo(f"[skip] {stem}: mp4 missing at {video_path}", err=True)
                continue
            todo.append({
                "stem": stem,
                "render_key": render_key,
                "rat_key": rat_key,
                "video_path": video_path,
            })

    if max_clips > 0:
        todo = todo[:max_clips]

    click.echo(f"Will generate {len(todo)} rationales via vLLM.")
    if not todo:
        click.echo("Nothing to do.")
        return

    from vllm import SamplingParams

    llm = _load_vllm_model(
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    n_ok = 0
    n_fail = 0
    t_start = time.time()
    with audit_log_p.open("a") as audit:
        for chunk_start in range(0, len(todo), batch_size):
            chunk = todo[chunk_start : chunk_start + batch_size]
            conversations = [
                _build_conversation(item["video_path"], prompt_card)
                for item in chunk
            ]
            t0 = time.time()
            try:
                outputs = llm.chat(conversations, sampling_params)
            except Exception as exc:  # noqa: BLE001
                # A whole-chunk failure is rare but we still want to log it
                click.echo(f"[chunk-fail] {chunk_start}: {type(exc).__name__}: {exc}", err=True)
                for item in chunk:
                    audit.write(json.dumps({
                        "ts": _now_iso(), "split": split, "stem": item["stem"],
                        "render_cache_key": item["render_key"],
                        "rationale_cache_key": item["rat_key"],
                        "model": MODEL_ID, "prompt_version": prompt_version,
                        "status": "error",
                        "error": f"chunk-fail: {type(exc).__name__}: {exc}",
                    }) + "\n")
                audit.flush()
                n_fail += len(chunk)
                continue
            chunk_elapsed = time.time() - t0

            for item, output in zip(chunk, outputs):
                raw = output.outputs[0].text
                n_tokens = len(output.outputs[0].token_ids)
                rationale, err = _post_process_output(raw)
                if err is not None:
                    n_fail += 1
                    audit.write(json.dumps({
                        "ts": _now_iso(), "split": split, "stem": item["stem"],
                        "render_cache_key": item["render_key"],
                        "rationale_cache_key": item["rat_key"],
                        "model": MODEL_ID, "prompt_version": prompt_version,
                        "status": "error", "error": err,
                        "output_tokens": n_tokens,
                    }) + "\n")
                    continue
                payload = {
                    "rationale_cache_key": item["rat_key"],
                    "render_cache_key": item["render_key"],
                    "prompt_version": prompt_version,
                    "model_version": MODEL_ID,
                    "generated_at": _now_iso(),
                    "rationale": rationale,
                    "output_tokens": n_tokens,
                    "elapsed_sec_chunk_avg": chunk_elapsed / len(chunk),
                }
                (output_dir_p / f"{item['rat_key']}.json").write_text(
                    json.dumps(payload, indent=2),
                )
                audit.write(json.dumps({
                    "ts": _now_iso(), "split": split, "stem": item["stem"],
                    "render_cache_key": item["render_key"],
                    "rationale_cache_key": item["rat_key"],
                    "model": MODEL_ID, "prompt_version": prompt_version,
                    "status": "ok", "output_tokens": n_tokens,
                    "elapsed_sec_chunk_avg": round(chunk_elapsed / len(chunk), 2),
                }) + "\n")
                n_ok += 1
            audit.flush()

            total_elapsed = time.time() - t_start
            done = n_ok + n_fail
            eta = (total_elapsed / max(done, 1)) * (len(todo) - done)
            throughput = done / total_elapsed
            click.echo(
                f"  chunk {chunk_start//batch_size + 1} ({len(chunk)} clips): "
                f"{n_ok} ok, {n_fail} fail, "
                f"chunk_elapsed={chunk_elapsed:.1f}s "
                f"({chunk_elapsed/len(chunk):.1f}s/clip), "
                f"cum_throughput={throughput*60:.1f} clips/min, "
                f"ETA {eta/60:.1f} min"
            )

    dur = time.time() - t_start
    click.echo(
        f"\nDone. {n_ok} ok, {n_fail} fail in {dur/60:.1f} min "
        f"({dur/max(n_ok,1):.1f} s/clip). "
        f"Output: {output_dir_p}/"
    )


if __name__ == "__main__":
    main()
