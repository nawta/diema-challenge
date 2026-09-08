""" export paper-ready prompt card for a prompt YAML + runtime context.

See: / docs/rule_and_ethics_checklist.md §3.2 / docs/reproducibility_bundle.md
Related:
- configs/prompts/motion_rationale_qwenvl_v1.0.yaml (prompt source of truth)
- tools/generate_part_rationales_qwenvl_vllm.py (consumes the YAML)
- tools/render_motion_video.py (produces renders + sidecar .meta.json)

Compiles one Markdown card per prompt YAML so the paper appendix can cite an
exact, reproducible snapshot: prompt text, JSON schema, render settings,
model commit SHA, generation window, and runtime stats across the rationale
cache. Semver on the prompt YAML's ``schema_version`` field governs cache
invalidation:

- ``patch`` → typo / wording only, rationale cache stays valid
- ``minor`` → prompt structure changed, schema keys unchanged; cache stays valid
- ``major`` → output schema changed, rationale cache MUST be regenerated

Usage::

    # Default (train split, v1.0 prompt)
    python tools/export_prompt_cards.py

    # Explicit phase/version + non-default paths
    python tools/export_prompt_cards.py \\
        --prompt configs/prompts/motion_rationale_qwenvl_v1.0.yaml \\
        --phase 3AD-4 \\
        --rationale-dir output/rationale_cache/rationales \\
        --render-dir output/rationale_cache/renders \\
        --output docs/cards/prompt_card_3AD-4_v1.0.md
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")

# Keep aligned with tools/generate_part_rationales_qwenvl.py::SCHEMA_KEYS.
OUTPUT_SCHEMA_KEYS: tuple[str, ...] = (
    "global_motion", "head", "torso", "left_arm", "right_arm",
    "left_leg", "right_leg", "salient_intervals", "uncertainty_notes",
)


def parse_semver(s: str) -> tuple[int, int, int]:
    m = SEMVER_RE.match(str(s).strip())
    if not m:
        raise ValueError(
            f"schema_version must be MAJOR.MINOR[.PATCH], got {s!r}"
        )
    maj, minor, patch = m.group(1), m.group(2), m.group(3) or "0"
    return int(maj), int(minor), int(patch)


def load_prompt_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        card = yaml.safe_load(f)
    required = {"schema_version", "model", "system", "user"}
    missing = required - set(card)
    if missing:
        raise click.ClickException(
            f"{path}: missing required keys {sorted(missing)}"
        )
    parse_semver(card["schema_version"])
    return card


def load_one_render_meta(render_dir: Path) -> dict | None:
    """Return render_settings from the first .meta.json in render_dir, if any."""
    candidates = sorted(render_dir.glob("*.meta.json"))
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as f:
        return json.load(f).get("render_settings")


def hf_commit_sha(model_id: str) -> str | None:
    """Locate the HF snapshot commit SHA for ``model_id`` on disk.

    Looks under ``$HF_HOME`` then ``~/.cache/huggingface`` for
    ``models--<org>--<name>/refs/main``. Returns None if not cached
    locally (e.g., card exported on a fresh machine).
    """
    org_name = model_id.replace("/", "--")
    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home))
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface")
    for root in roots:
        ref = root / f"models--{org_name}" / "refs" / "main"
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    return None


def gather_rationale_stats(rationale_dir: Path, prompt_version: str) -> dict:
    """Aggregate runtime stats over rationale JSONs that match ``prompt_version``.

    Filters on rationale.prompt_version so a single-version card reports only
    relevant runs. Returns counts, tokens/elapsed statistics, and the
    [earliest, latest] generation window.
    """
    n_total = 0
    n_match = 0
    out_tokens: list[int] = []
    elapsed: list[float] = []
    generated_at: list[datetime] = []
    models_seen: set[str] = set()
    for p in sorted(rationale_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        n_total += 1
        if str(data.get("prompt_version")) != str(prompt_version):
            continue
        n_match += 1
        t = data.get("output_tokens")
        if isinstance(t, int):
            out_tokens.append(t)
        e = data.get("elapsed_sec_chunk_avg")
        if isinstance(e, (int, float)):
            elapsed.append(float(e))
        g = data.get("generated_at")
        if isinstance(g, str):
            try:
                generated_at.append(datetime.fromisoformat(g.replace("Z", "+00:00")))
            except ValueError:
                pass
        mv = data.get("model_version")
        if isinstance(mv, str):
            models_seen.add(mv)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {
            "n": len(xs),
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "p05": sorted(xs)[max(0, int(0.05 * len(xs)) - 1)],
            "p95": sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)) - 1)],
            "min": min(xs),
            "max": max(xs),
        }

    window = {}
    if generated_at:
        window = {
            "first": min(generated_at).isoformat(timespec="seconds"),
            "last": max(generated_at).isoformat(timespec="seconds"),
        }
    return {
        "n_total": n_total,
        "n_match": n_match,
        "output_tokens": _stats([float(x) for x in out_tokens]),
        "elapsed_sec_chunk_avg": _stats(elapsed),
        "models_seen": sorted(models_seen),
        "generation_window": window,
    }


def render_card_md(
    *,
    phase: str,
    prompt_yaml_path: Path,
    prompt_card: dict,
    render_settings: dict | None,
    model_commit: str | None,
    stats: dict,
    now_iso: str,
) -> str:
    ver = str(prompt_card["schema_version"])
    model_id = str(prompt_card["model"])
    major, minor, patch = parse_semver(ver)

    md: list[str] = [
        f"# Prompt card — {phase} v{ver}",
        "",
        "_Auto-generated by `tools/export_prompt_cards.py`. "
        "Do not hand-edit; bump the source YAML's `schema_version` instead._",
        "",
        "## 1. Identity",
        "",
        f"- **phase**: `{phase}`",
        f"- **prompt YAML**: `{prompt_yaml_path.as_posix()}`",
        f"- **prompt schema version**: `{ver}` (major={major}, minor={minor}, patch={patch})",
        f"- **model**: `{model_id}`",
        f"- **model HF commit SHA**: " + (f"`{model_commit}`" if model_commit else "_(not cached locally at export time)_"),
        f"- **card generated at**: `{now_iso}`",
        "",
        "## 2. Semver policy",
        "",
        "| bump | meaning | rationale cache impact |",
        "|---|---|---|",
        "| patch | typo / wording only | cache stays valid |",
        "| minor | prompt structure changed, output keys unchanged | cache stays valid |",
        "| major | output JSON schema changed | cache MUST be regenerated |",
        "",
        "A rationale JSON records `prompt_version` in its top-level metadata; "
        "after a major bump, drop any rationale whose version no longer matches.",
        "",
        "## 3. System prompt",
        "",
        "```text",
        str(prompt_card["system"]).rstrip(),
        "```",
        "",
        "## 4. User prompt",
        "",
        "```text",
        str(prompt_card["user"]).rstrip(),
        "```",
        "",
        "## 5. Output JSON schema",
        "",
        "Keys required on every rationale (enforced by `tools/generate_part_rationales_qwenvl.py::SCHEMA_KEYS`):",
        "",
    ]
    for k in OUTPUT_SCHEMA_KEYS:
        md.append(f"- `{k}`")
    md.extend([
        "",
        "All string-valued except `salient_intervals` which is an array of",
        "`{start_sec: float, end_sec: float, note: str}` objects (≤ 3 entries).",
        "",
        "## 6. Render settings (input video)",
        "",
    ])
    if render_settings:
        md.append("Frozen values read from a render `.meta.json` sidecar:")
        md.append("")
        md.append("| key | value |")
        md.append("|---|---|")
        for k, v in sorted(render_settings.items()):
            md.append(f"| `{k}` | `{v}` |")
    else:
        md.append("_(no `.meta.json` found under the render dir at export time)_")
    md.extend([
        "",
        "Leakage guard: output mp4 paths are sha256 hashes, no filename/metadata leak.",
        "",
        "## 7. Generation runtime stats",
        "",
        f"- rationales scanned: **{stats['n_total']}**",
        f"- rationales matching prompt version `{ver}`: **{stats['n_match']}**",
    ])
    if stats["models_seen"]:
        md.append(f"- model_version strings observed: {', '.join(f'`{m}`' for m in stats['models_seen'])}")
    if stats["generation_window"]:
        md.append(
            f"- generation window: `{stats['generation_window']['first']}` → "
            f"`{stats['generation_window']['last']}`"
        )
    md.append("")
    if stats["output_tokens"] or stats["elapsed_sec_chunk_avg"]:
        md.append("| metric | n | mean | median | p05 | p95 | min | max |")
        md.append("|---|---|---|---|---|---|---|---|")
        for key, label, fmt in [
            ("output_tokens", "output tokens (per rationale)", "{:.1f}"),
            ("elapsed_sec_chunk_avg", "elapsed sec / chunk (amortised)", "{:.2f}"),
        ]:
            s = stats.get(key) or {}
            if not s:
                continue
            md.append(
                f"| {label} | {s['n']} | "
                f"{fmt.format(s['mean'])} | {fmt.format(s['median'])} | "
                f"{fmt.format(s['p05'])} | {fmt.format(s['p95'])} | "
                f"{fmt.format(s['min'])} | {fmt.format(s['max'])} |"
            )
        md.append("")
    else:
        md.append("_(no matching rationale JSONs at export time — run `tools/generate_part_rationales_qwenvl_vllm.py` first)_")
        md.append("")

    md.extend([
        "## 8. Reproducibility",
        "",
        "To regenerate rationales against this exact prompt + model pair:",
        "",
        "```bash",
        "# 1. render 64-frame stick-figure videos (train split)",
        "python tools/render_motion_video.py \\",
        "    --npz data/diema_challenge/processed/motion_train_quat.npz \\",
        "    --batch train --fps 10 --view front_side_2view --max-frames 64",
        "",
        "# 2. vLLM batched rationale generation",
        "python tools/generate_part_rationales_qwenvl_vllm.py \\",
        "    --split train --batch-size 16 --max-num-seqs 4 --max-new-tokens 4096",
        "",
        "# 3. forbidden-token leakage audit (must report 0 blocked)",
        "python tools/check_rationale_leakage.py",
        "```",
        "",
        "Rationale cache key: `sha256(render_cache_key || prompt_version || model_id)` — "
        "idempotent across reruns. See `tools/generate_part_rationales_qwenvl.py::rationale_cache_key`.",
        "",
    ])
    return "\n".join(md)


def default_output_path(phase: str, version: str) -> Path:
    safe_phase = re.sub(r"[^A-Za-z0-9_\-]", "_", phase)
    return Path(f"docs/cards/prompt_card_{safe_phase}_v{version}.md")


def infer_phase_from_prompt_path(path: Path) -> str:
    """Fallback: use YAML stem (e.g., motion_rationale_qwenvl_v1.0 → motion_rationale_qwenvl)."""
    stem = path.stem
    return re.sub(r"_v\d+\.\d+(?:\.\d+)?$", "", stem)


@click.command()
@click.option(
    "--prompt",
    default="configs/prompts/motion_rationale_qwenvl_v1.0.yaml",
    show_default=True,
    help="Prompt YAML; must contain schema_version/model/system/user keys.",
)
@click.option(
    "--rationale-dir",
    default="output/rationale_cache/rationales",
    show_default=True,
    help="Dir of generated rationale JSONs (for runtime stats).",
)
@click.option(
    "--render-dir",
    default="output/rationale_cache/renders",
    show_default=True,
    help="Dir of render outputs incl. .meta.json sidecars (for render settings).",
)
@click.option(
    "--phase",
    default=None,
    help="Phase tag for the card filename (e.g., 3AD-4). Inferred from prompt stem if omitted.",
)
@click.option(
    "--output",
    default=None,
    help="Output markdown path. Default: docs/cards/prompt_card_<phase>_v<version>.md",
)
def main(prompt: str, rationale_dir: str, render_dir: str, phase: str | None, output: str | None) -> None:
    prompt_path = Path(prompt)
    if not prompt_path.is_file():
        raise click.ClickException(f"prompt YAML not found: {prompt_path}")
    card = load_prompt_yaml(prompt_path)
    ver = str(card["schema_version"])

    phase_tag = phase or infer_phase_from_prompt_path(prompt_path)
    out_path = Path(output) if output else default_output_path(phase_tag, ver)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_meta = load_one_render_meta(Path(render_dir))
    model_sha = hf_commit_sha(str(card["model"]))
    stats = gather_rationale_stats(Path(rationale_dir), ver)

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = render_card_md(
        phase=phase_tag,
        prompt_yaml_path=prompt_path,
        prompt_card=card,
        render_settings=render_meta,
        model_commit=model_sha,
        stats=stats,
        now_iso=now_iso,
    )
    out_path.write_text(md, encoding="utf-8")
    click.echo(f"Saved: {out_path}")
    click.echo(
        f"  phase={phase_tag} version={ver} "
        f"rationales_matched={stats['n_match']}/{stats['n_total']}"
    )


if __name__ == "__main__":
    main()
