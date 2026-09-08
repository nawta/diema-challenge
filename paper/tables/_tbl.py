"""paper/tables/_tbl.py — shared LaTeX(booktabs)/CSV/Markdown table emitter.

Used by P9/P10/P11.  emit() writes {name}.tex, {name}.csv, {name}.md into
paper/tables/.  LaTeX uses booktabs (\\usepackage{booktabs} required) and
bolds the final row when bold_last=True.  Deterministic (sorted nothing,
fixed text) so P12 regen is byte-stable.
"""

from __future__ import annotations

import csv
from pathlib import Path

TBL_DIR = Path(__file__).resolve().parent


def _tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("±", r"$\pm$"),
                 ("→", r"$\rightarrow$"), ("↔", r"$\leftrightarrow$"),
                 ("≈", r"$\approx$"), ("≥", r"$\ge$"), ("≤", r"$\le$"),
                 ("★", r"$\star$"), ("·", r"$\cdot$"), ("−", "-"),
                 ("×", r"$\times$"), ("—", "---"), ("–", "--"),
                 ("§", r"\S{}"), ("…", r"\ldots{}"), ("’", "'"),
                 ("‘", "'"), ("“", "``"), ("”", "''"), ("ρ", r"$\rho$"),
                 ("Δ", r"$\Delta$"), ("σ", r"$\sigma$"),
                 ("λ", r"$\lambda$"), ("†", r"$^{\dagger}$"),
                 ("‡", r"$^{\ddagger}$")]:
        s = s.replace(a, b)
    return s


def emit(name: str, header: list[str], rows: list[list], *,
         colspec: str, bold_last: bool = False,
         notes: list[str] | None = None, caption: str = "") -> dict:
    notes = notes or []

    # E1.3 (a review) + 2026-05-20 post-fix: guard against header/row/
    # colspec width drift — catches the silent 'Extra alignment tab has been
    # changed to \\cr' pdflatex failure before it reaches the .tex.
    # Tokenise colspec properly (p{...} is ONE column; the previous naive
    # char-count regressed Table 2 because the 'c' in 'cm' was double-counted).
    import re as _re
    n_cols = len(_re.findall(r"p\{[^}]*\}|[lcr]", colspec))
    assert len(header) == n_cols, (
        f"emit({name}): header has {len(header)} cells but colspec='{colspec}' "
        f"declares {n_cols} columns")
    for i, r in enumerate(rows):
        assert len(r) == n_cols, (
            f"emit({name}): row {i} has {len(r)} cells but colspec='{colspec}' "
            f"declares {n_cols} columns")

    # ---- LaTeX (booktabs) ----
    tex = [f"% {name} — requires \\usepackage{{booktabs}}",
           "\\begin{table*}[t]", "\\centering",
           f"\\caption{{{_tex_escape(caption)}}}", f"\\label{{tab:{name}}}",
           "\\footnotesize", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(_tex_escape(h) for h in header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        cells = [_tex_escape(c) for c in r]
        if bold_last and i == len(rows) - 1:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        tex.append(" & ".join(cells) + r" \\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        tex.append("\\\\[2pt] \\parbox{\\linewidth}{\\raggedright\\scriptsize "
                   + " \\quad ".join(_tex_escape(n) for n in notes) + "}")
    tex += ["\\end{table*}", ""]
    (TBL_DIR / f"{name}.tex").write_text("\n".join(tex))

    # ---- CSV (no LaTeX) ----
    with open(TBL_DIR / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        for n in notes:
            w.writerow([f"# {n}"])

    # ---- Markdown preview ----
    md = [f"### {name}", ""]
    if caption:
        md += [f"_{caption}_", ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for i, r in enumerate(rows):
        cells = [str(c) for c in r]
        if bold_last and i == len(rows) - 1:
            cells = [f"**{c}**" for c in cells]
        md.append("| " + " | ".join(cells) + " |")
    md += [""] + [f"- _{n}_" for n in notes] + [""]
    (TBL_DIR / f"{name}.md").write_text("\n".join(md))

    return {"tex": str(TBL_DIR / f"{name}.tex"),
            "csv": str(TBL_DIR / f"{name}.csv"),
            "md": str(TBL_DIR / f"{name}.md")}
