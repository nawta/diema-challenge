"""TDD for the paper table emitter (_tbl.emit) + Table 1 E1.3 contract.

Table 1 contract was tightened on 2026-05-20 (E1.3 / a review):
the 9-way and 11-way-softmax intermediate rows were dropped, leaving
exactly 5 system rows; softmax-mean appears NOWHERE in the rendered
artifact (guide §5).

Run: conda run -n acii2026 pytest tests/test_paper_tables.py -q
"""

import csv
import sys
from pathlib import Path

TBL = Path(__file__).resolve().parent.parent / "paper" / "tables"
sys.path.insert(0, str(TBL))


def test_emit_contract(tmp_path, monkeypatch):
    import _tbl

    monkeypatch.setattr(_tbl, "TBL_DIR", tmp_path)
    out = _tbl.emit("t_unit", ["A", "B"], [["1", "x_1"], ["2", "y±2"]],
                    colspec="l c", bold_last=True, notes=["n1 50% & ×2"],
                    caption="cap_1")
    tex = Path(out["tex"]).read_text()
    assert "\\toprule" in tex and "\\midrule" in tex \
        and "\\bottomrule" in tex
    assert "\\textbf{2}" in tex and "\\textbf{y$\\pm$2}" in tex  # bold last
    assert "x\\_1" in tex and "50\\%" in tex and "$\\times$2" in tex
    assert "booktabs" in tex
    rows = list(csv.reader(open(out["csv"])))
    assert rows[0] == ["A", "B"] and rows[1] == ["1", "x_1"]
    assert "| A | B |" in Path(out["md"]).read_text()


def test_emit_marker_escape_and_single_run_in_note_block(tmp_path, monkeypatch):
    import _tbl

    monkeypatch.setattr(_tbl, "TBL_DIR", tmp_path)
    out = _tbl.emit("t_markers", ["ΔF1‡"], [["value†"]], colspec="c",
                    notes=["† dagger", "‡ double dagger"])
    tex = Path(out["tex"]).read_text()
    assert "$\\Delta$F1" in tex
    assert "$^{\\dagger}$" in tex and "$^{\\ddagger}$" in tex
    assert tex.count("\\parbox{\\linewidth}") == 1
    assert not any(marker in tex for marker in "†‡Δ★")


def test_emit_colspec_column_count_guard(tmp_path, monkeypatch):
    """emit() must reject header/row width != colspec column count."""
    import _tbl
    import pytest

    monkeypatch.setattr(_tbl, "TBL_DIR", tmp_path)
    with pytest.raises(AssertionError):                # colspec=3, header=2
        _tbl.emit("bad_hdr", ["A", "B"], [["1", "2"]], colspec="l c c")
    with pytest.raises(AssertionError):                # row width mismatch
        _tbl.emit("bad_row", ["A", "B"], [["1"]], colspec="l c")
    # p{Xcm} must count as ONE column, not as l/c/r chars inside 'cm'.
    # Table 2's actual colspec "l p{2.05cm} p{2.35cm} p{2.2cm} p{3.5cm}" = 5.
    _tbl.emit("p_ok", ["A", "B", "C", "D", "E"],
              [["1", "2", "3", "4", "5"]],
              colspec="l p{2.05cm} p{2.35cm} p{2.2cm} p{3.5cm}")
    with pytest.raises(AssertionError):                # 5-col colspec, 4-cell hdr
        _tbl.emit("p_bad", ["A", "B", "C", "D"], [["1", "2", "3", "4"]],
                  colspec="l p{2cm} p{2cm} p{2cm} p{2cm}")


def test_table1_artifacts_valid():
    """Table 1 E1.3 contract: 5 rows; canonical 33.86/34.03/36.80 present;
    bold final row; softmax/9-way phrasing absent (guide §5)."""
    tex = (TBL / "table1_main_results.tex")
    csvf = (TBL / "table1_main_results.csv")
    if not tex.exists():
        import pytest

        pytest.skip("run build_table1.py first")
    rows = list(csv.reader(open(csvf)))
    body = [r for r in rows[1:] if r and not r[0].startswith("#")]
    assert len(body) == 5, f"E1.3 contract: expect 5 rows, got {len(body)}"
    ci_col = rows[0].index("Macro-F1 95% CI")
    for r in body:
        assert r[ci_col].strip(), f"empty CI cell in row {r[0]}"
    t = tex.read_text()
    assert "\\textbf{11-way logit-mean (submitted)}" in t  # bold final row
    assert "33.86" in t and "34.03" in t       # canonical 7-way (per-fold/pooled)
    assert "36.80" in t                         # final per-fold mean
    # guide §5 negatives — softmax / 9-way row must NOT appear
    for forbidden in ("33.72", "35.78", "9-way", "softmax-mean", "softmax mean"):
        assert forbidden not in t, f"forbidden token '{forbidden}' in table1.tex"


def test_generated_table_marker_hygiene():
    """Every cell/header marker has one anchored note, and vice versa."""
    expected = {
        "table1_main_results.tex": ("$\\star$", "$^{\\ddagger}$"),
        "table3_explainability.tex": ("$^{\\dagger}$",),
    }
    for name, markers in expected.items():
        tex = (TBL / name).read_text()
        assert tex.count("\\parbox{\\linewidth}") == 1
        for marker in markers:
            assert marker in tex.split("\\bottomrule", 1)[0]
            assert tex.split("\\bottomrule", 1)[1].count(marker) == 1
        assert not any(raw in tex for raw in "†‡Δ★")

    for tex_path in TBL.glob("*.tex"):
        assert not any(raw in tex_path.read_text() for raw in "†‡Δ★")
