"""Inject exp075 results into ``docs/analysis/exp075_headchain_only_verdict.md``.

Reads the JSON emitted by ``tools/exp075_aggregate.py`` and replaces the
TBD placeholders in §TL;DR table and §2 results table. Idempotent.

See: §5.3 / §11.5
Related: tools/exp075_aggregate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VERDICT = Path("docs/analysis/exp075_headchain_only_verdict.md")
JSON_PATH = Path("docs/analysis/exp075_aggregate.json")
CHANCE = 1.0 / 12.0


def _fmt(x: float | None, width: int = 5) -> str:
    return "TBD" if x is None else f"{x*100:.{width-3}f}"


def _format_row(variant: str, summary: dict) -> str:
    n_joints = summary["n_joints"]
    f1s = summary["f1s"]
    cells = [f1s[i] if i < len(f1s) else None for i in range(3)]
    mean = summary["mean"]
    std = summary["std"]
    cell_strs = [_fmt(c) + "%" for c in cells]
    mean_str = _fmt(mean) + "%" if mean is not None else "TBD"
    std_str = f"{std*100:.2f}" if std is not None else "TBD"
    label = {
        "a00_top4": "a00_top4",
        "a01_top10": "a01_top10",
        "a02_top10_arms": "a02_top10_arms",
        "a03_top10_legs": "a03_top10_legs",
    }[variant]
    return (
        f"| {label} | 0.435M | {cell_strs[0]} | {cell_strs[1]} | "
        f"{cell_strs[2]} | {mean_str} | {std_str} |"
    )


def main() -> int:
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found. Run tools/exp075_aggregate.py first.")
        return 2
    if not VERDICT.exists():
        print(f"ERROR: {VERDICT} not found.")
        return 2
    with JSON_PATH.open("r") as f:
        agg = json.load(f)

    summaries = agg["summaries"]
    criteria = agg["criteria"]

    # Replace TL;DR table (§TL;DR -- exact 4-row block under "| variant | params | ...")
    text = VERDICT.read_text(encoding="utf-8")

    rows_block = "\n".join(
        _format_row(v, summaries[v])
        for v in ("a00_top4", "a01_top10", "a02_top10_arms", "a03_top10_legs")
    )

    pattern = re.compile(
        r"(\| variant \| params \| fold 0 \| fold 1 \| fold 2 \| mean \| std \|\n"
        r"\|---\|---:\|---:\|---:\|---:\|---:\|---:\|\n)"
        r"(?:\| [^\n]*\n){4}",
        re.MULTILINE,
    )

    def _sub(match):
        return match.group(1) + rows_block + "\n"

    new_text, n_subs = pattern.subn(_sub, text)
    if n_subs == 0:
        print("WARNING: no TL;DR table replaced (regex did not match).")
    else:
        print(f"Replaced {n_subs} TL;DR table block(s).")

    # Replace Go-criteria checklist
    a_str = "✅ PASS" if criteria["A_stability"] else "❌ FAIL"
    b_str = "✅ PASS" if criteria["B_top4_above_chance"] else "❌ FAIL"
    c_str = "✅ PASS" if criteria["C_monotone"] else "⚠ FAIL/INSPECT"
    overall = "✅ GO (paper artifact ready)" if criteria["overall_go"] else "❌ NO-GO"

    pattern_go = re.compile(
        r"(- \*\*\(A\) Stability\*\*: all per-fold F1 > 10% \(no chance-level collapse\) — )TBD",
    )
    new_text = pattern_go.sub(rf"\1{a_str}", new_text)
    pattern_go = re.compile(
        r"(- \*\*\(B\) Top-4 ≫ chance\*\*: top-4 mean F1 ≥ 20% — )TBD",
    )
    new_text = pattern_go.sub(rf"\1{b_str}", new_text)
    pattern_go = re.compile(
        r"(- \*\*\(C\) Monotone\*\*: top-4 < top-10 ≤ top-10\+arms / top-10\+legs — )TBD",
    )
    new_text = pattern_go.sub(rf"\1{c_str}", new_text)

    VERDICT.write_text(new_text, encoding="utf-8")
    print(f"Wrote {VERDICT}")
    print(f"Overall verdict: {overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
