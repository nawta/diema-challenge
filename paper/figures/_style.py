"""paper/figures/_style.py — shared IEEE-conference matplotlib style (P1).

Design doc: paper/data_inventory.md (P0; canonical numbers + open questions).
Every figure script (P2-P8) starts with, run from inside paper/figures/:

    from _style import (setup_ieee, save_fig, set_size, seed_all,
                        ONE_COL, TWO_COL, PALETTE_QUAL, PALETTE_SEQ,
                        PALETTE_DIV)
    setup_ieee()
    seed_all(42)

Constraints honored:
  * setup_ieee() is NEVER called at import time (must be explicit).
  * pdf.fonttype=42 / ps.fonttype=42 -> TrueType fonts EMBEDDED (mandatory
    for IEEE Xplore: text stays selectable, not rasterized to shapes).
  * text.usetex=False -> no system LaTeX dependency (portable regen).
  * save_fig() strips the PDF CreationDate so P12 `regen_all.sh` can be
    byte-deterministic (with fixed seeds + sorted inputs).

Backend is forced to Agg at import (headless server; this is backend
selection, NOT styling, so it does not violate the "no setup at import"
rule).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; not a style mutation

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── IEEE conference template geometry (IEEEtran, US-letter 2-col) ──────────
ONE_COL = 3.45        # inch — single-column width
TWO_COL = 7.16        # inch — full (double-column) text width
DPI_RASTER = 300      # dpi for the PNG preview raster

_FIG_DIR = Path(__file__).resolve().parent

# ── colour-blind-safe qualitative palette ─────────────────────────────────
# seaborn.color_palette("colorblind") hex, itself adapted from
# Wong, B. (2011) "Points of view: Color blindness", Nature Methods 8:441.
# Deuteranopia/protanopia-safe. First 7 entries (= max distinct categories
# we use: 7 ensemble families / 3-tier bars / inductive-bias groups).
PALETTE_QUAL = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermillion
    "#CC78BC",  # purple
    "#CA9161",  # brown
    "#FBAFE4",  # pink
]
PALETTE_SEQ = "viridis"   # sequential heatmaps (magnitude; perceptually uniform)
PALETTE_DIV = "RdBu_r"    # diverging heatmaps (signed diffs; centre at 0)


def setup_ieee() -> None:
    """Apply IEEE-conference rcParams. MUST be called explicitly per script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,   # TrueType, embeddable (IEEE Xplore compliant)
        "ps.fonttype": 42,
        "text.usetex": False,  # portability: no LaTeX toolchain required
        "figure.dpi": DPI_RASTER,
        "savefig.dpi": DPI_RASTER,
    })


def set_size(fig, width_in: float, height_in: float) -> None:
    """Set the physical figure size in inches (use ONE_COL / TWO_COL)."""
    fig.set_size_inches(width_in, height_in)


def save_fig(fig, name: str, *, tight: bool = True):
    """Save ``fig`` to paper/figures/{name}.pdf (vector) AND .png (300 dpi).

    PDF: bbox_inches='tight', pad_inches=0.02, CreationDate stripped for
    deterministic regeneration. Returns (pdf_path, png_path).
    """
    pdf_path = _FIG_DIR / f"{name}.pdf"
    png_path = _FIG_DIR / f"{name}.png"
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    # metadata={'CreationDate': None} -> no timestamp -> byte-stable PDFs
    fig.savefig(pdf_path, format="pdf", metadata={"CreationDate": None}, **kw)
    fig.savefig(png_path, format="png", dpi=DPI_RASTER, **kw)
    return pdf_path, png_path


def seed_all(seed: int = 42) -> None:
    """Seed python / numpy / torch (if importable) for reproducible figures."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
