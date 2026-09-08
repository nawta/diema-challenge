""" / exp061: Qwen3-VL rationale candidate-bank generator (v2.0).

See: / configs/prompts/motion_rationale_qwenvl_v2.0_{a,b,c}.yaml
Related:
- tools/generate_part_rationales_qwenvl_vllm.py (v1 single-candidate baseline)
- tools/rank_rationale_candidates.py (downstream selector)
- docs/cards/prompt_card_3AD-4_v1.0.md (v1 prompt card; v2 cards exported by
  ``tools/export_prompt_cards.py`` with ``--phase 3AK``)

Generates **3 candidates per clip** by running 3 prompt-ordering variants
(`a`: global-first / `b`: body-part-first / `c`: salient-interval-first)
through the same Qwen3-VL-30B-A3B-Thinking model. The downstream ranker
(`tools/rank_rationale_candidates.py`) scores each candidate on schema
validity, leakage absence, body-part completeness, and lexical specificity,
then writes the top-1 pick + alt to `output/rationale_cache_v2/selected/`.

Cache layout
------------
- input mp4: ``output/rationale_cache/renders/<render_key>.mp4`` (shared with v1).
- output JSON: ``output/rationale_cache_v2/rationales/<rat_key_v2>__{a,b,c}.json``
  where ``rat_key_v2 = sha256(render_key || schema_version || variant || model_id)``.
  The variant is mixed into the cache key so the 3 candidates do NOT collide
  even when callers expect a flat directory.
- audit log: ``output/rationale_cache_v2/audit_log.jsonl`` (append-only).

Usage::

    # Smoke (30 clips × 3 variants = 90 generations)
    python tools/generate_rationale_candidates_qwen.py \\
        --split train --max-clips 30 --batch-size 8 --max-num-seqs 4

    # Full corpus (~7644 clips × 3 = ~22.9k generations; ~60-70h on a single
    # 97GB Blackwell at ~10s/gen amortised)
    python tools/generate_rationale_candidates_qwen.py \\
        --split train --batch-size 16 --max-num-seqs 4

Implementation notes
--------------------
- We pay the vLLM model load (~2-4 min) ONCE per process, then iterate over
  the 3 variants by swapping system+user messages. The KV cache stays warm
  across variants because the video tokens dominate the prompt.
- Continuous batching is preserved by passing the full chunk-of-conversations
  to ``llm.chat()``. The outer ``--batch-size`` chunk only controls how often
  we flush the audit log to disk (protects against SIGKILL mid-run).
- Idempotent re-runs: any ``<rat_key_v2>__{variant}.json`` already on disk is
  skipped. This means a partial run can be resumed by re-invoking the script
  with the same arguments.
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

from tools.generate_part_rationales_qwenvl import (  # noqa: E402
    MODEL_ID,
    SCHEMA_KEYS,
    _now_iso,
    _parse_json_or_repair,
)
from tools.generate_part_rationales_qwenvl_vllm import (  # noqa: E402
    _PART_KEYS,
    _post_process_output,
)


PROMPT_VERSION_V2 = "2.0"

V2_PROMPT_DIR = Path("configs/prompts")
V2_PROMPT_BY_VARIANT = {
    "a": V2_PROMPT_DIR / "motion_rationale_qwenvl_v2.0_a.yaml",
    "b": V2_PROMPT_DIR / "motion_rationale_qwenvl_v2.0_b.yaml",
    "c": V2_PROMPT_DIR / "motion_rationale_qwenvl_v2.0_c.yaml",
}


def rationale_cache_key_v2(
    render_cache_key: str, prompt_version: str, variant: str, model_id: str,
) -> str:
    """Variant-aware cache key — collisions across variants are eliminated."""
    blob = f"{render_cache_key}|{prompt_version}|{variant}|{model_id}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_v2_prompt(variant: str) -> dict:
    path = V2_PROMPT_BY_VARIANT[variant]
    with path.open("r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    required = {"schema_version", "variant", "model", "system", "user"}
    missing = required - set(card)
    if missing:
        raise ValueError(f"prompt card {path} missing required keys: {missing}")
    if str(card["schema_version"]) != PROMPT_VERSION_V2:
        raise ValueError(
            f"prompt card {path} schema_version != {PROMPT_VERSION_V2}; got "
            f"{card['schema_version']!r}"
        )
    if card["variant"] != variant:
        raise ValueError(
            f"prompt card {path} variant field {card['variant']!r} != {variant!r}"
        )
    return card


def _build_conversation(video_path: Path, prompt_card: dict) -> list[dict]:
    """OpenAI-style chat for vLLM 0.19. ``video_url.url`` accepts ``file://``."""
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


def _load_vllm_model(
    gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int,
):
    from vllm import LLM
    click.echo(f"Loading {MODEL_ID} via vLLM ... (first load takes ~2-4 min)")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
        tensor_parallel_size=1,
        allowed_local_media_path=str(Path("output/rationale_cache/renders").resolve()),
    )
    click.echo(f"  loaded in {time.time() - t0:.1f}s")
    return llm


@click.command()
@click.option("--split", required=True, type=click.Choice(["train"]),
              help="Only 'train' is accepted — test split is never sent to the VLM.")
@click.option("--variants", default="a,b,c", show_default=True,
              help="Comma-separated subset of {a,b,c}. Default runs all 3.")
@click.option("--render-dir", default="output/rationale_cache/renders")
@click.option("--output-dir", default="output/rationale_cache_v2/rationales")
@click.option("--audit-log", default="output/rationale_cache_v2/audit_log.jsonl")
@click.option("--max-clips", default=0, type=int,
              help="If >0, process only the first N queued entries (smoke test). "
                   "Applies to the per-variant queue, so a value of 30 runs 30 "
                   "clips × #variants generations.")
@click.option("--batch-size", default=16, type=int,
              help="vLLM chat() chunk size (per variant). Each variant is run "
                   "as a separate pass — the model stays loaded between passes.")
@click.option("--max-new-tokens", default=6144, type=int,
              help="Same default as v1; thinking budget for emotion-free "
                   "kinematic JSON. 6144 was empirically below the 2.8% empty "
                   "rate observed at 4096 in ")
@click.option("--max-model-len", default=16384, type=int,
              help="vLLM KV-cache sequence length.")
@click.option("--max-num-seqs", default=8, type=int,
              help="Concurrent sequences in the vLLM scheduler.")
@click.option("--gpu-mem-util", default=0.85, type=float,
              help="Fraction of GPU memory vLLM may use. Default lower than v1 "
                   "(0.90) because we run 3 passes back-to-back and want "
                   "headroom for KV-cache reset between variants.")
@click.option("--temperature", default=0.5, type=float, show_default=True,
              help="Sampling temperature. plan calls for ONE-axis "
                   "variation: prompt-ordering (a/b/c). Temperature is held "
                   "constant across variants (default 0.5 — a middle ground "
                   "between v1's deterministic 0.0 and unconstrained 1.0).")
@click.option("--top-p", default=0.95, type=float, show_default=True)
def main(
    split: str,
    variants: str,
    render_dir: str,
    output_dir: str,
    audit_log: str,
    max_clips: int,
    batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_mem_util: float,
    temperature: float,
    top_p: float,
) -> None:
    if split != "train":
        raise click.ClickException("split must be 'train' — test is never sent to the VLM.")

    variant_list = [v.strip().lower() for v in variants.split(",") if v.strip()]
    for v in variant_list:
        if v not in V2_PROMPT_BY_VARIANT:
            raise click.ClickException(f"unknown variant {v!r}; expected a/b/c")

    os.environ.setdefault("HF_HOME", "data/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "data/hf_cache")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    prompt_cards = {v: load_v2_prompt(v) for v in variant_list}

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

    base_index: list[dict] = []
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stem = row["stem"]
            render_key = row["cache_key"]
            video_path = render_dir_p / f"{render_key}.mp4"
            if not video_path.exists():
                click.echo(f"[skip] {stem}: mp4 missing at {video_path}", err=True)
                continue
            base_index.append({"stem": stem, "render_key": render_key,
                               "video_path": video_path})

    if max_clips > 0:
        base_index = base_index[:max_clips]

    queues: dict[str, list[dict]] = {}
    for variant in variant_list:
        queue: list[dict] = []
        for entry in base_index:
            rat_key = rationale_cache_key_v2(
                entry["render_key"], PROMPT_VERSION_V2, variant, MODEL_ID,
            )
            out_path = output_dir_p / f"{rat_key}__{variant}.json"
            if out_path.exists():
                continue
            queue.append({**entry, "rat_key": rat_key, "out_path": out_path,
                          "variant": variant})
        queues[variant] = queue
        click.echo(f"variant {variant}: queued {len(queue)} / scanned "
                   f"{len(base_index)}")

    total_todo = sum(len(q) for q in queues.values())
    if total_todo == 0:
        click.echo("Nothing to do — all candidates already cached.")
        return
    click.echo(f"Will generate {total_todo} candidates total via vLLM.")

    from vllm import SamplingParams
    llm = _load_vllm_model(
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
    )

    n_ok_total = 0
    n_fail_total = 0
    t_start = time.time()
    with audit_log_p.open("a") as audit:
        for variant in variant_list:
            queue = queues[variant]
            prompt_card = prompt_cards[variant]
            click.echo(f"\n=== variant {variant} ({prompt_card['variant_label']}) "
                       f"— {len(queue)} clips ===")
            n_ok = 0
            n_fail = 0
            for chunk_start in range(0, len(queue), batch_size):
                chunk = queue[chunk_start : chunk_start + batch_size]
                conversations = [
                    _build_conversation(item["video_path"], prompt_card)
                    for item in chunk
                ]
                t0 = time.time()
                try:
                    outputs = llm.chat(conversations, sampling_params)
                except Exception as exc:  # noqa: BLE001
                    click.echo(f"[chunk-fail] variant={variant} {chunk_start}: "
                               f"{type(exc).__name__}: {exc}", err=True)
                    for item in chunk:
                        audit.write(json.dumps({
                            "ts": _now_iso(), "split": split,
                            "stem": item["stem"], "variant": variant,
                            "render_cache_key": item["render_key"],
                            "rationale_cache_key": item["rat_key"],
                            "model": MODEL_ID,
                            "prompt_version": PROMPT_VERSION_V2,
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
                            "ts": _now_iso(), "split": split,
                            "stem": item["stem"], "variant": variant,
                            "render_cache_key": item["render_key"],
                            "rationale_cache_key": item["rat_key"],
                            "model": MODEL_ID,
                            "prompt_version": PROMPT_VERSION_V2,
                            "status": "error", "error": err,
                            "output_tokens": n_tokens,
                        }) + "\n")
                        continue
                    payload = {
                        "rationale_cache_key": item["rat_key"],
                        "render_cache_key": item["render_key"],
                        "prompt_version": PROMPT_VERSION_V2,
                        "variant": variant,
                        "variant_label": prompt_card.get("variant_label", variant),
                        "model_version": MODEL_ID,
                        "generated_at": _now_iso(),
                        "rationale": rationale,
                        "output_tokens": n_tokens,
                        "elapsed_sec_chunk_avg": chunk_elapsed / max(1, len(chunk)),
                        "sampling": {
                            "temperature": temperature, "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                        },
                    }
                    item["out_path"].write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    n_ok += 1
                    audit.write(json.dumps({
                        "ts": _now_iso(), "split": split,
                        "stem": item["stem"], "variant": variant,
                        "render_cache_key": item["render_key"],
                        "rationale_cache_key": item["rat_key"],
                        "model": MODEL_ID,
                        "prompt_version": PROMPT_VERSION_V2,
                        "status": "ok",
                        "output_tokens": n_tokens,
                    }) + "\n")
                audit.flush()
                done_var = n_ok + n_fail
                rate = (n_ok / done_var * 100.0) if done_var else 0.0
                click.echo(
                    f"  [variant {variant}] done {done_var}/{len(queue)}: "
                    f"ok={n_ok} fail={n_fail} ({rate:.1f}% ok); "
                    f"chunk {chunk_elapsed:.1f}s ({chunk_elapsed/max(1,len(chunk)):.2f}s/clip)"
                )
            n_ok_total += n_ok
            n_fail_total += n_fail
            click.echo(f"variant {variant} complete: ok={n_ok} fail={n_fail}")

    elapsed_total = time.time() - t_start
    click.echo(
        f"\n=== exp061 v2 candidate generation complete ===\n"
        f"  ok={n_ok_total} fail={n_fail_total} "
        f"(total={n_ok_total + n_fail_total}); elapsed={elapsed_total/60:.1f} min"
    )


if __name__ == "__main__":
    main()
