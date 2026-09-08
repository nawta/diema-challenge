""" (local VLM variant): body-part rationale via Qwen3-VL-30B-A3B-Thinking.

See: / docs/rule_and_ethics_checklist.md §1.2 (local-VLM gate)
Related:
- tools/render_motion_video.py (produces the input mp4 files + render_index.jsonl)
- configs/prompts/motion_rationale_qwenvl_v1.0.yaml (system + user prompt card)
- tools/check_rationale_leakage.py (forbidden-token audit)

Loads Qwen3-VL-30B-A3B-Thinking once, then iterates over rendered mp4 files in
``output/rationale_cache/renders/`` and produces one JSON rationale per clip in
``output/rationale_cache/rationales/<rationale_cache_key>.json`` plus an
append-only audit log.

**Gate guards**:
  - ``--split train`` hard-coded. Refuses to run on the test split, period.
  - Requires an in-place ``render_index.jsonl`` mapping cache_key → stem.
  - Writes an audit log entry per API call with (timestamp, stem, cache_key,
    model, prompt_version, output_tokens).

**Cache key semantics**:
  rationale_cache_key = sha256(render_cache_key || prompt_version || model_version)
  → the same rendered video + same prompt + same model always produce the
  same rationale file name, enabling idempotent re-runs.

Usage::

    # Smoke on the renders already produced
    python tools/generate_part_rationales_qwenvl.py \\
        --split train --limit 5

    # Full sweep once model is downloaded
    python tools/generate_part_rationales_qwenvl.py --split train
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


PROMPT_CARD_PATH = Path("configs/prompts/motion_rationale_qwenvl_v1.0.yaml")
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking"

SCHEMA_KEYS = (
    "global_motion", "head", "torso", "left_arm", "right_arm",
    "left_leg", "right_leg", "salient_intervals", "uncertainty_notes",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_prompt_card() -> dict:
    with PROMPT_CARD_PATH.open("r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    required = {"schema_version", "model", "system", "user"}
    missing = required - set(card)
    if missing:
        raise ValueError(
            f"prompt card {PROMPT_CARD_PATH} missing required keys: {missing}"
        )
    return card


def rationale_cache_key(render_cache_key: str, prompt_version: str, model_id: str) -> str:
    """Deterministic key for idempotent re-runs; see module docstring."""
    blob = f"{render_cache_key}|{prompt_version}|{model_id}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_json_or_repair(text: str) -> dict:
    """Extract a dict from the model output. Tolerate: trailing whitespace,
    surrounding code fences, leading/trailing junk around the JSON object,
    and un-terminated strings from token-budget truncation (best-effort).

    Parsing passes in order:
      1. Raw json.loads on the full stripped text
      2. Strip markdown ``` fences and retry
      3. Find outermost balanced ``{...}`` substring and parse that
      4. Give up and raise the most recent JSONDecodeError
    """
    stripped = text.strip()
    # Pass 1: raw
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Pass 2: markdown fences
    fenced = stripped
    if fenced.startswith("```"):
        parts = fenced.split("\n", 1)
        if len(parts) == 2:
            fenced = parts[1]
        if fenced.rstrip().endswith("```"):
            fenced = fenced.rstrip().rstrip("`").rstrip()
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Pass 3: balanced {...} scan to tolerate prefix/suffix junk
    candidate = stripped
    if "{" in candidate and "}" in candidate:
        start = candidate.index("{")
        # Walk forward tracking brace depth (respect strings with escapes)
        depth = 0
        in_str = False
        esc = False
        end = None
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
    # Last resort: re-raise from the raw parse
    return json.loads(stripped)  # will raise


def _build_messages(video_path: Path, prompt_card: dict) -> list[dict]:
    """Shape the chat messages per Qwen3-VL conventions."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt_card["system"]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path.resolve()}"},
                {"type": "text", "text": prompt_card["user"]},
            ],
        },
    ]


def load_model_and_processor(dtype: str = "auto"):
    """Import-and-load the Qwen3-VL model. Kept inside a function so the
    transformers import cost is paid once per process.

    Uses the **MoE class** (``Qwen3VLMoeForConditionalGeneration``) because
    ``Qwen3-VL-30B-A3B-Thinking`` is a mixture-of-experts checkpoint; the
    plain dense class silently drops expert weights and replaces them with
    freshly-initialised ``mlp.{up,down,gate}_proj.weight`` tensors, which
    produces garbage output.
    """
    import torch
    from transformers import AutoProcessor

    # Order matters: try the MoE class first, fall back to dense / auto.
    ModelCls = None
    for cls_name in (
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
    ):
        try:
            ModelCls = getattr(__import__("transformers", fromlist=[cls_name]), cls_name)
            click.echo(f"  using model class: {cls_name}")
            break
        except (ImportError, AttributeError):
            continue
    if ModelCls is None:
        from transformers import AutoModelForImageTextToText as ModelCls  # type: ignore
        click.echo("  using model class: AutoModelForImageTextToText (fallback)")

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    click.echo(f"Loading {MODEL_ID} ... (first load takes 1-2 min)")
    t0 = time.time()
    model = ModelCls.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    click.echo(f"  loaded in {time.time() - t0:.1f}s")
    return model, processor


def generate_rationale(
    model,
    processor,
    video_path: Path,
    prompt_card: dict,
    max_new_tokens: int = 2048,
) -> tuple[dict, int]:
    """Run the VLM end-to-end on one mp4. Returns (rationale_dict, n_output_tokens).

    Raises ``json.JSONDecodeError`` or ``ValueError`` if the model output
    can't be parsed into a JSON object.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    messages = _build_messages(video_path, prompt_card)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
    # Keep only the continuation (skip the prompt tokens)
    input_len = inputs["input_ids"].shape[1]
    gen_only = gen_ids[:, input_len:]
    output_tokens = int(gen_only.shape[1])
    decoded = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    # Qwen3-VL-Thinking emits <think>...</think> chain-of-thought before the answer.
    # Drop everything inside <think>...</think> and keep only what follows.
    if "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[1]
    rationale = _parse_json_or_repair(decoded)
    missing = [k for k in SCHEMA_KEYS if k not in rationale]
    if missing:
        # Fill missing keys with empty string so downstream consumers get a
        # stable shape. Record in uncertainty_notes.
        for k in missing:
            rationale[k] = "" if k != "salient_intervals" else []
        rationale["uncertainty_notes"] = (
            (rationale.get("uncertainty_notes") or "").rstrip()
            + f" [auto-filled missing keys: {missing}]"
        ).strip()
    return rationale, output_tokens


@click.command()
@click.option("--split", required=True, type=click.Choice(["train"]),
              help="Only 'train' is accepted — test split is never sent to the VLM.")
@click.option("--render-dir", default="output/rationale_cache/renders")
@click.option("--output-dir", default="output/rationale_cache/rationales")
@click.option("--audit-log", default="output/rationale_cache/audit_log.jsonl")
@click.option("--limit", default=0, type=int,
              help="If >0, process only the first N entries (smoke test).")
@click.option("--dtype", default="bf16",
              type=click.Choice(["auto", "bf16", "fp16"]))
@click.option("--max-new-tokens", default=2048, type=int,
              help="Max new tokens. Default 2048 because Qwen3-VL-Thinking "
              "can spend ~1k tokens inside <think>...</think> before the JSON "
              "answer; 1024 sometimes truncated.")
def main(
    split: str,
    render_dir: str,
    output_dir: str,
    audit_log: str,
    limit: int,
    dtype: str,
    max_new_tokens: int,
) -> None:
    # Hard split guard — redundant with click.Choice but documents intent.
    if split != "train":
        raise click.ClickException("split must be 'train' — test split is never sent to the VLM.")

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

    # Parse the index and skip rationales that already exist.
    todo: list[tuple[str, str]] = []
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stem = row["stem"]
            render_key = row["cache_key"]
            rat_key = rationale_cache_key(render_key, prompt_version, MODEL_ID)
            rat_path = output_dir_p / f"{rat_key}.json"
            if rat_path.exists():
                continue
            todo.append((stem, render_key))

    if limit > 0:
        todo = todo[:limit]

    click.echo(f"Will generate {len(todo)} rationales (split={split}).")
    if not todo:
        click.echo("Nothing to do (all rationales already present).")
        return

    # Point HF cache at data (same as download) so we don't need to duplicate.
    os.environ.setdefault("HF_HOME", "data/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "data/hf_cache")

    # qwen_vl_utils' default torchvision.io.read_video dispatcher is broken on
    # torchvision 0.26 (the function was removed). Force the decord backend,
    # which we've installed explicitly for this pipeline.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    model, processor = load_model_and_processor(dtype=dtype)

    start = time.time()
    n_ok = 0
    n_fail = 0
    with audit_log_p.open("a") as audit:
        for stem, render_key in todo:
            video_path = render_dir_p / f"{render_key}.mp4"
            if not video_path.exists():
                click.echo(f"[skip] {stem}: mp4 missing at {video_path}", err=True)
                continue
            rat_key = rationale_cache_key(render_key, prompt_version, MODEL_ID)
            try:
                t0 = time.time()
                rationale, n_tok = generate_rationale(
                    model, processor, video_path, prompt_card,
                    max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[fail] {stem}: {type(exc).__name__}: {exc}", err=True)
                n_fail += 1
                audit.write(json.dumps({
                    "ts": _now_iso(),
                    "split": split,
                    "stem": stem,
                    "render_cache_key": render_key,
                    "rationale_cache_key": rat_key,
                    "model": MODEL_ID,
                    "prompt_version": prompt_version,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }) + "\n")
                audit.flush()
                continue

            payload = {
                "rationale_cache_key": rat_key,
                "render_cache_key": render_key,
                "prompt_version": prompt_version,
                "model_version": MODEL_ID,
                "generated_at": _now_iso(),
                "rationale": rationale,
                "output_tokens": n_tok,
                "elapsed_sec": elapsed,
            }
            out_path = output_dir_p / f"{rat_key}.json"
            out_path.write_text(json.dumps(payload, indent=2))

            audit.write(json.dumps({
                "ts": _now_iso(),
                "split": split,
                "stem": stem,
                "render_cache_key": render_key,
                "rationale_cache_key": rat_key,
                "model": MODEL_ID,
                "prompt_version": prompt_version,
                "status": "ok",
                "output_tokens": n_tok,
                "elapsed_sec": round(elapsed, 2),
            }) + "\n")
            audit.flush()
            n_ok += 1
            if n_ok % 10 == 0:
                eta = (time.time() - start) / n_ok * (len(todo) - n_ok)
                click.echo(
                    f"  {n_ok}/{len(todo)} ok, {n_fail} fail, ETA {eta/60:.1f} min"
                )

    dur = time.time() - start
    click.echo(
        f"\nFinished: {n_ok} ok, {n_fail} fail in {dur/60:.1f} min. "
        f"Output: {output_dir_p}/, audit: {audit_log_p}"
    )


if __name__ == "__main__":
    main()
