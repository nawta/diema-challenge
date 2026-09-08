""" pre-compute C3D stats for all DIEM-A clips into a single NPZ.

Related:
- diema/data/c3d_parser.py  (read_c3d, VICON_PRODUCTION_MARKERS)
- diema/features/c3d_stats.py (compute_c3d_stats, feature_names)
- experiments/exp056_c3d_stats_latefusion/ (consumer)

Scans ``data/diema_challenge/raw/c3d/{train,test}/`` for every .c3d file,
strips the subject prefix at parse time, computes the 51 per-clip features,
and writes the result as ``output/c3d_stats_cache.npz``:

    features (N, 51) float32
    filenames (N,)    str (stem only, e.g. "JP_06_joy_1_L" or "test0001")
    feature_names (F,) str (stable ordering, = diema.features.c3d_stats.feature_names())
    rate (N,)         float32 (sampling rate per file, always 120.0 expected)
    n_frames (N,)     int32
    dropout_rate (N,) float32

CPU-only; a full 9,936-clip sweep takes ~2-3 minutes on a single thread.

Usage::

    python tools/cache_c3d_stats.py
    # or override paths:
    python tools/cache_c3d_stats.py --c3d-root data/diema_challenge/raw/c3d \\
        --output output/c3d_stats_cache.npz
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.data.c3d_parser import read_c3d  # noqa: E402
from diema.features.c3d_stats import (  # noqa: E402
    compute_c3d_stats,
    feature_names,
    stats_to_vector,
)


def _process_one(path: str) -> tuple[str, np.ndarray, float, int, float] | None:
    """Single-clip worker: returns (stem, feature_vec, rate, n_frames, dropout_rate)."""
    p = Path(path)
    stem = p.stem
    try:
        d = read_c3d(p, strict=False)
    except Exception as exc:
        return (stem, None, 0.0, 0, 0.0) if False else None
    try:
        stats = compute_c3d_stats(
            d["markers"], d["marker_labels"],
            rate=d["rate"], residual=d["residual"],
        )
    except Exception:
        return None
    vec = stats_to_vector(stats)
    return (stem, vec, float(d["rate"]), int(d["n_frames"]),
            float(stats.get("marker_dropout_rate", 0.0)))


@click.command()
@click.option("--c3d-root", default="data/diema_challenge/raw/c3d",
              help="Directory containing train/ and test/ subfolders of .c3d files")
@click.option("--output", default="output/c3d_stats_cache.npz",
              help="Where to write the NPZ cache")
@click.option("--workers", default=0, type=int,
              help="Process-pool workers. 0 = serial (recommended, I/O-bound).")
@click.option("--max-files", default=0, type=int,
              help="If >0, process only the first N files (smoke test).")
def main(c3d_root: str, output: str, workers: int, max_files: int) -> None:
    root = Path(c3d_root)
    if not root.exists():
        raise click.ClickException(f"c3d root not found: {root}")

    train_files = sorted((root / "train").glob("*.c3d"))
    test_files = sorted((root / "test").glob("*.c3d"))
    all_files = train_files + test_files
    if max_files:
        all_files = all_files[:max_files]
    click.echo(f"Found {len(train_files)} train + {len(test_files)} test C3D files "
               f"→ processing {len(all_files)}.")

    stems: list[str] = []
    features: list[np.ndarray] = []
    rates: list[float] = []
    n_frames: list[int] = []
    dropout: list[float] = []

    def _collect(result):
        if result is None:
            return
        stem, vec, rate, n_f, drop = result
        stems.append(stem)
        features.append(vec)
        rates.append(rate)
        n_frames.append(n_f)
        dropout.append(drop)

    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, str(p)) for p in all_files]
            for i, fut in enumerate(as_completed(futures), 1):
                _collect(fut.result())
                if i % 500 == 0:
                    click.echo(f"  processed {i}/{len(all_files)}")
    else:
        for i, p in enumerate(all_files, 1):
            _collect(_process_one(str(p)))
            if i % 500 == 0:
                click.echo(f"  processed {i}/{len(all_files)}")

    click.echo(f"\nSuccessfully processed {len(features)} / {len(all_files)} files "
               f"({len(all_files) - len(features)} failures).")

    feat_arr = np.stack(features, axis=0)
    names = feature_names()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=feat_arr,
        filenames=np.asarray(stems),
        feature_names=np.asarray(names),
        rate=np.asarray(rates, dtype=np.float32),
        n_frames=np.asarray(n_frames, dtype=np.int32),
        dropout_rate=np.asarray(dropout, dtype=np.float32),
    )
    click.echo(f"Saved: {out}  (shape features={feat_arr.shape}, n_features={len(names)})")
    click.echo(f"  mean dropout rate across corpus: {np.mean(dropout):.4f}")


if __name__ == "__main__":
    main()
