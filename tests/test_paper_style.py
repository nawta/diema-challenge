"""TDD for paper/figures/_style.py (P1 shared IEEE figure style).

Run: conda run -n acii2026 pytest tests/test_paper_style.py -q
"""

import importlib
import sys
from pathlib import Path

import matplotlib
import pytest

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


@pytest.fixture()
def style():
    """Import _style the SAME way figure scripts do (sibling import from
    inside paper/figures/), with rcParams reset to library defaults so we
    can prove setup_ieee() is not applied merely by importing."""
    if str(FIG_DIR) not in sys.path:
        sys.path.insert(0, str(FIG_DIR))
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    mod = importlib.import_module("_style")
    return importlib.reload(mod)


def test_constants(style):
    assert style.ONE_COL == 3.45
    assert style.TWO_COL == 7.16
    assert style.DPI_RASTER == 300


def test_palettes(style):
    assert len(style.PALETTE_QUAL) == 7
    for c in style.PALETTE_QUAL:
        assert c.startswith("#") and len(c) == 7
        int(c[1:], 16)  # valid hex
    assert style.PALETTE_SEQ == "viridis"
    assert style.PALETTE_DIV == "RdBu_r"


def test_setup_not_called_at_import(style):
    # default pdf.fonttype is 3 (Type-3). Import alone must not force 42.
    assert matplotlib.rcParams["pdf.fonttype"] != 42


def test_setup_ieee_embeds_fonts(style):
    style.setup_ieee()
    assert matplotlib.rcParams["pdf.fonttype"] == 42  # IEEE Xplore: embedded
    assert matplotlib.rcParams["ps.fonttype"] == 42
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["axes.linewidth"] == 0.5


def test_seed_all_reproducible(style):
    import numpy as np

    style.seed_all(42)
    a = np.random.rand(5)
    style.seed_all(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_size(style):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    style.set_size(fig, style.ONE_COL, 4.0)
    w, h = fig.get_size_inches()
    assert abs(w - 3.45) < 1e-9 and abs(h - 4.0) < 1e-9
    plt.close(fig)


def test_save_fig_outputs_pdf_and_png(style, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    # redirect output dir into tmp so the test never litters paper/figures/
    monkeypatch.setattr(style, "_FIG_DIR", tmp_path)
    style.setup_ieee()
    fig, ax = plt.subplots()
    ax.plot([0], [0], "o")
    pdf, png = style.save_fig(fig, "_unit_smoke")
    plt.close(fig)
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    assert pdf.suffix == ".pdf" and png.suffix == ".png"
