""" (Tier 2): Qwen3-VL 12-class emotion soft-label generator.

See: / configs/prompts/emotion_softlabel_qwenvl_v1.0.yaml
Related:
- tools/generate_part_rationales_qwenvl_vllm.py (sister tool, shared vLLM loader)
- experiments/exp060_vlm_softlabel_kd/ (consumer of the resulting npy)
- docs/rule_and_ethics_checklist.md §3.3 (leakage carve-out)

Generates a (N_train, 12) numpy float32 array of probability distributions
over the 12 DIEM-A emotions, conditioned on the rendered stick-figure
motion video. Each row's 12 floats sum to ~1.0 (re-normalized after parse).

This is **train-only auxiliary supervision** — never imported by the
test inference path. See ethics checklist §3.3.

Output schema::

    {
        "soft_labels": (N_train, 12) float32 — sum-to-1 per row,
        "stems":       list[str] length N_train,
        "emotions":    [anger, contempt, …, surprise] (frozen order),
        "model":       Qwen/Qwen3-VL-30B-A3B-Thinking,
        "prompt_version": "1.0",
        "n_dropped":   int — clips that failed JSON parse,
    }

Usage::

    python tools/generate_emotion_softlabel_qwenvl_vllm.py \\
        --split train --batch-size 16 --max-num-seqs 4 \\
        --max-new-tokens 4096 --gpu-mem-util 0.85
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.generate_part_rationales_qwenvl import (  # noqa: E402
    MODEL_ID,
    _now_iso,
    _parse_json_or_repair,
    rationale_cache_key,
)
from tools.generate_part_rationales_qwenvl_vllm import (  # noqa: E402
    _build_conversation,
    _load_vllm_model,
)


PROMPT_CARD_PATH = Path("configs/prompts/emotion_softlabel_qwenvl_v1.0.yaml")

EMOTIONS: tuple[str, ...] = (
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def load_softlabel_prompt_card() -> dict:
    if not PROMPT_CARD_PATH.is_file():
        raise FileNotFoundError(
            f"prompt card not found: {PROMPT_CARD_PATH}"
        )
    with PROMPT_CARD_PATH.open("r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    required = {"schema_version", "model", "system", "user"}
    missing = required - set(card)
    if missing:
        raise ValueError(
            f"prompt card {PROMPT_CARD_PATH} missing keys {sorted(missing)}"
        )
    return card


def _post_process_softlabel(raw_text: str) -> tuple[np.ndarray | None, str | None]:
    """Strip <think>, parse JSON, validate + re-normalize.

    Returns (softlabel_vec_12 | None, error_msg_or_None).
    """
    text = raw_text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    try:
        obj = _parse_json_or_repair(text)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    vec = np.zeros(len(EMOTIONS), dtype=np.float32)
    for i, e in enumerate(EMOTIONS):
        v = obj.get(e)
        if v is None:
            return None, f"missing emotion key: {e}"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None, f"non-numeric prob for {e}: {v!r}"
        if not np.isfinite(v) or v < 0:
            return None, f"invalid prob for {e}: {v}"
        vec[i] = v

    s = float(vec.sum())
    if s <= 0 or not np.isfinite(s):
        return None, f"non-positive prob sum: {s}"
    # Re-normalize to sum-to-1 (VLM rounding usually keeps it within ±0.05).
    vec = vec / s
    return vec, None


@click.command()
@click.option(
    "--split", default="train", show_default=True,
    help="Hard-coded train-only; test split is never sent to the VLM.",
)
@click.option(
    "--render-dir", default="output/rationale_cache/renders", show_default=True,
)
@click.option(
    "--output", default="output/vlm_softlabel_train.npz", show_default=True,
    help="Stores (soft_labels, stems, emotions, meta) as a single npz.",
)
@click.option(
    "--audit-log", default="output/rationale_cache/softlabel_audit_log.jsonl",
    show_default=True,
)
@click.option("--max-clips", default=0, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--max-new-tokens", default=4096, type=int,
              help="No <think> JSON body; 4096 is comfortably enough.")
@click.option("--max-model-len", default=16384, type=int)
@click.option("--max-num-seqs", default=4, type=int)
@click.option("--gpu-mem-util", default=0.85, type=float)
def main(
    split: str, render_dir: str, output: str, audit_log: str,
    max_clips: int, batch_size: int, max_new_tokens: int,
    max_model_len: int, max_num_seqs: int, gpu_mem_util: float,
) -> None:
    if split != "train":
        raise click.ClickException("split must be 'train' — test never enters this loop.")

    os.environ.setdefault("HF_HOME", "data/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "data/hf_cache")
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    prompt_card = load_softlabel_prompt_card()
    prompt_version = str(prompt_card["schema_version"])
    click.echo(f"prompt_version={prompt_version}, model={MODEL_ID}")

    render_dir_p = Path(render_dir)
    output_p = Path(output)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    audit_log_p = Path(audit_log)
    audit_log_p.parent.mkdir(parents=True, exist_ok=True)

    index_path = render_dir_p / "render_index.jsonl"
    if not index_path.exists():
        raise click.ClickException(f"render index not found: {index_path}")

    todo: list[dict] = []
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
                click.echo(f"[skip] {stem}: mp4 missing", err=True)
                continue
            sl_key = rationale_cache_key(render_key, prompt_version, MODEL_ID + "/softlabel")
            todo.append({
                "stem": stem, "render_key": render_key,
                "sl_key": sl_key, "video_path": video_path,
            })

    if max_clips > 0:
        todo = todo[:max_clips]
    click.echo(f"Will generate {len(todo)} 12-class soft labels via vLLM.")
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
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens,
    )

    soft_labels = np.full((len(todo), len(EMOTIONS)), np.nan, dtype=np.float32)
    stems_out: list[str] = []
    n_ok = 0
    n_fail = 0
    t_start = time.time()
    with audit_log_p.open("a") as audit:
        for chunk_start in range(0, len(todo), batch_size):
            chunk = todo[chunk_start: chunk_start + batch_size]
            conversations = [
                _build_conversation(item["video_path"], prompt_card)
                for item in chunk
            ]
            t0 = time.time()
            try:
                outputs = llm.chat(conversations, sampling_params)
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[chunk-fail] {chunk_start}: {type(exc).__name__}: {exc}", err=True)
                for item in chunk:
                    audit.write(json.dumps({
                        "ts": _now_iso(), "split": split, "stem": item["stem"],
                        "render_cache_key": item["render_key"],
                        "softlabel_cache_key": item["sl_key"],
                        "model": MODEL_ID, "prompt_version": prompt_version,
                        "status": "error",
                        "error": f"chunk-fail: {type(exc).__name__}: {exc}",
                    }) + "\n")
                audit.flush()
                n_fail += len(chunk)
                continue
            chunk_elapsed = time.time() - t0

            for i, (item, output_obj) in enumerate(zip(chunk, outputs)):
                raw = output_obj.outputs[0].text
                n_tokens = len(output_obj.outputs[0].token_ids)
                vec, err = _post_process_softlabel(raw)
                row_idx = chunk_start + i
                stems_out.append(item["stem"])
                if err is not None:
                    n_fail += 1
                    audit.write(json.dumps({
                        "ts": _now_iso(), "split": split, "stem": item["stem"],
                        "render_cache_key": item["render_key"],
                        "softlabel_cache_key": item["sl_key"],
                        "model": MODEL_ID, "prompt_version": prompt_version,
                        "status": "error", "error": err,
                        "output_tokens": n_tokens,
                    }) + "\n")
                    continue
                soft_labels[row_idx] = vec
                n_ok += 1
                audit.write(json.dumps({
                    "ts": _now_iso(), "split": split, "stem": item["stem"],
                    "render_cache_key": item["render_key"],
                    "softlabel_cache_key": item["sl_key"],
                    "model": MODEL_ID, "prompt_version": prompt_version,
                    "status": "ok", "output_tokens": n_tokens,
                    "elapsed_sec_chunk_avg": round(chunk_elapsed / len(chunk), 2),
                }) + "\n")
            audit.flush()

            total_elapsed = time.time() - t_start
            done = n_ok + n_fail
            eta = (total_elapsed / max(done, 1)) * (len(todo) - done)
            click.echo(
                f"chunk {chunk_start // batch_size + 1} ({len(chunk)} clips): "
                f"{n_ok} ok, {n_fail} fail, "
                f"chunk_elapsed={chunk_elapsed:.1f}s "
                f"({chunk_elapsed/len(chunk):.1f}s/clip), "
                f"cum_throughput={done * 60 / max(total_elapsed, 1):.1f} clips/min, "
                f"ETA {eta / 60:.1f} min"
            )

    np.savez(
        output_p,
        soft_labels=soft_labels,
        stems=np.array(stems_out, dtype=object),
        emotions=np.array(list(EMOTIONS), dtype=object),
        meta=np.array(json.dumps({
            "prompt_version": prompt_version,
            "model": MODEL_ID,
            "schema_version": "1.0",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_total": len(todo),
            "n_ok": n_ok,
            "n_fail": n_fail,
        }), dtype=object),
    )
    click.echo(
        f"\nSaved: {output_p} "
        f"(N={len(stems_out)}, ok={n_ok}, fail={n_fail})"
    )


if __name__ == "__main__":
    main()
