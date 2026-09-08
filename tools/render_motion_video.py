""" render DIEM-A skeleton motion to MP4 (matplotlib 2D).

See: / docs/rule_and_ethics_checklist.md §1.2
Related:
- diema/data/parser.py  (filename → performer/intensity/emotion for LEAKAGE GUARD)
- diema/models/skeleton_graph.py::{DIEMA_JOINT_NAMES, DIEMA_INWARD_EDGES}
- tools/generate_part_rationales_qwenvl.py (downstream consumer)

For a given DIEM-A NPZ + filename, produce a motion-only MP4 video that
contains no label/filename/performer metadata. The output is:

- **skeleton25** render: 25 CTV joints + bone edges
- **front_side_2view** (default) or **front** single view
- constant camera, constant scale (hip-width normalised), white background,
  fixed line widths — everything Gemini / Qwen-VL needs to be able to see
  motion across clips without dataset-level confounds
- filename reduced to a cache key (sha256 of joint array + settings);
  original filename never appears in the mp4 path, metadata, or container

**Leakage guard** (docs/rule_and_ethics_checklist.md §2):
- output path is a 64-char hex hash, not the original stem
- ffmpeg metadata tags (`-metadata title=...`) are explicitly blanked
- no text overlay is drawn
- the accompanying `.meta.json` records the render settings (for
  reproducibility) but NOT the source filename — filename→hash lookup
  lives in a separate index file with HMAC'd identifiers

Backend: matplotlib 2D + ffmpeg. No OpenGL / EGL needed, runs on CPU,
headless OK. aitviewer-based 3D renderer is a future upgrade.

Usage::

    # Single clip (stem-based for test)
    python tools/render_motion_video.py \\
        --npz data/diema_challenge/processed/motion_train_quat.npz \\
        --filename JP_06_anger_1_H --fps 10 --view front_side_2view

    # Batch render whole train split
    python tools/render_motion_video.py \\
        --npz data/diema_challenge/processed/motion_train_quat.npz \\
        --batch train
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.models.skeleton_graph import (  # noqa: E402
    DIEMA_INWARD_EDGES,
    DIEMA_JOINT_NAMES,
)

# Pin the cache key / metadata schema; bump on any rendering-semantic change.
SCHEMA_VERSION = "1.0"


def _hip_width(joint_pos_frame: np.ndarray) -> float:
    """Estimate hip width (mm) to normalise scale.

    joint_pos_frame: (25, 3). Uses CTV nodes 17 (RightUpLeg) and 21
    (LeftUpLeg) per :data:`DIEMA_JOINT_NAMES`.
    """
    r_hip = joint_pos_frame[17]
    l_hip = joint_pos_frame[21]
    return float(np.linalg.norm(r_hip - l_hip)) + 1e-6


def normalise_joint_positions(joint_pos: np.ndarray) -> np.ndarray:
    """Return a copy of ``joint_pos`` (T, 25, 3) centred + hip-normalised.

    Centred on the virtual root (CTV node 0); scaled so the mean hip width
    across the clip equals 1. Result is in skeleton-scale units, not mm.
    """
    if joint_pos.shape[1:] != (25, 3):
        raise ValueError(f"joint_pos must be (T, 25, 3); got {joint_pos.shape}")
    out = joint_pos - joint_pos[:, 0:1, :]  # virtual_root-centred
    hw = np.mean([_hip_width(f) for f in out])
    out = out / hw
    return out


def render_cache_key(
    joint_pos: np.ndarray,
    render_settings: dict,
) -> str:
    """SHA-256 hex digest of (canonicalised joint tensor + frozen settings).

    Settings are JSON-encoded with sort_keys so any setting change flips
    the key. Matches the spec
    """
    canonical = np.round(np.ascontiguousarray(joint_pos.astype(np.float32)), 6)
    settings_blob = json.dumps(render_settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical.tobytes() + settings_blob).hexdigest()


def _render_frame_ax(ax, pts, edges, lims, title: str | None = None) -> None:
    """Plot one (25, 3) snapshot onto a 2D axis (X vs Z projection).

    `pts` is already in normalised skeleton units. We project onto the
    sagittal (Y axis) plane by dropping Y, so X = left/right, Z = up/down.
    """
    ax.clear()
    # bones first (background)
    for a, b in edges:
        xs = [pts[a, 0], pts[b, 0]]
        zs = [pts[a, 2], pts[b, 2]]
        ax.plot(xs, zs, "k-", linewidth=3.0, solid_capstyle="round", zorder=1)
    # joints as dots
    ax.scatter(pts[:, 0], pts[:, 2], s=25, c="black", zorder=2)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


def _side_view_points(pts: np.ndarray) -> np.ndarray:
    """Swap X and Y to produce a side view (Y projected instead of X)."""
    out = pts.copy()
    out[:, 0] = pts[:, 1]  # use Y (forward) as horizontal
    return out


def _clip_limits(
    joint_pos: np.ndarray, margin: float = 0.5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute (x_lim, z_lim) covering the full clip + margin.

    Input is expected in normalised units (hip_width=1). Adds ``margin``
    hip-widths of padding on every side.
    """
    x_min, x_max = float(joint_pos[..., 0].min()), float(joint_pos[..., 0].max())
    z_min, z_max = float(joint_pos[..., 2].min()), float(joint_pos[..., 2].max())
    return (
        (x_min - margin, x_max + margin),
        (z_min - margin, z_max + margin),
    )


def render_video(
    joint_pos: np.ndarray,
    out_path: Path,
    fps: int = 10,
    view: str = "front_side_2view",
    slowmo: float = 1.0,
) -> dict:
    """Render a (T, 25, 3) joint trajectory to MP4.

    Args:
        joint_pos: (T, 25, 3) world-coord joint trajectory.
        out_path: MP4 output path. Parent directory must exist.
        fps: output video fps (the joint trajectory is emitted frame-wise
            without time-warping unless ``slowmo < 1.0``).
        view: one of ``"front"`` or ``"front_side_2view"``.
        slowmo: playback-speed multiplier. 1.0 = native, 0.5 = half speed
            (videos are 2× longer). Implemented by duplicating each frame.

    Returns:
        Dict with ``"frames_rendered"``, ``"fps"``, ``"view"``, ``"slowmo"``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if view not in ("front", "front_side_2view"):
        raise ValueError(f"view must be 'front' or 'front_side_2view', got {view!r}")
    if slowmo <= 0 or slowmo > 1.0:
        raise ValueError("slowmo must be in (0, 1]")

    normed = normalise_joint_positions(joint_pos)
    lims_front = _clip_limits(normed, margin=0.5)
    if view == "front_side_2view":
        # side view uses swapped coordinates; compute limits separately
        side = np.stack([_side_view_points(f) for f in normed])
        lims_side = _clip_limits(side, margin=0.5)

    edges = list(DIEMA_INWARD_EDGES)

    # Frame duplication for slowmo
    frame_repeats = max(int(round(1.0 / slowmo)), 1)

    tmpdir = tempfile.mkdtemp(prefix="render_motion_")
    try:
        tmp_fmt = os.path.join(tmpdir, "frame_%05d.png")
        frame_idx = 0
        for t in range(normed.shape[0]):
            for _ in range(frame_repeats):
                if view == "front":
                    fig, ax = plt.subplots(figsize=(3.5, 4.0), dpi=100)
                    _render_frame_ax(ax, normed[t], edges, lims_front)
                else:  # front_side_2view
                    fig, axes = plt.subplots(
                        1, 2, figsize=(7.0, 4.0), dpi=100
                    )
                    _render_frame_ax(axes[0], normed[t], edges, lims_front)
                    _render_frame_ax(
                        axes[1], _side_view_points(normed[t]), edges, lims_side,
                    )
                fig.patch.set_facecolor("white")
                fig.tight_layout(pad=0.3)
                fig.savefig(tmp_fmt % frame_idx, facecolor="white")
                plt.close(fig)
                frame_idx += 1

        # ffmpeg merge. Use yuv420p + even dimensions for broad player compat.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", tmp_fmt,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-metadata", "title=",  # blank title (no leakage)
            "-metadata", "comment=",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        # clean up tmp frames
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    return {
        "frames_rendered": frame_idx,
        "fps": fps,
        "view": view,
        "slowmo": slowmo,
    }


def load_joint_positions_from_npz(
    npz_path: Path, filename_stem: str,
    preprocessed: dict | None = None,
) -> np.ndarray:
    """Extract (T, 25, 3) joint positions for one clip from a prepare-data NPZ.

    Re-runs FK via pybvh_ml to get `joint_pos` in world coordinates.

    Performance note: ``pybvh_ml.load_preprocessed`` loads the entire
    (multi-GB) NPZ. Callers in batch mode MUST pass ``preprocessed`` so the
    heavy read happens once; the single-clip CLI path pays the load cost
    per call.
    """
    if preprocessed is None:
        import pybvh_ml
        preprocessed = pybvh_ml.load_preprocessed(str(npz_path))
    filenames = preprocessed["filenames"]
    clips = preprocessed["clips"]
    try:
        idx = filenames.index(filename_stem)
    except ValueError as e:
        raise ValueError(
            f"stem {filename_stem!r} not found in {npz_path.name} "
            f"({len(filenames)} clips, first 3: {filenames[:3]})"
        ) from e

    clip = clips[idx]
    root_pos = clip["root_pos"]      # (F, 3)
    joint_data = clip["joint_data"]  # (F, J, 4) quaternions

    # Use the multi-stream helper to get joint_pos directly; it packs to (C, T, V)
    from diema.features.multi_stream import get_stream_features
    skeleton_info = preprocessed.get("skeleton_info", {})
    data_ctv, _ = get_stream_features(
        root_pos, joint_data,
        skeleton_info=skeleton_info,
        stream_type="joint_pos",
    )
    # data_ctv is (C=3, T, V=25). Transpose to (T, V, 3) for the renderer.
    if data_ctv.ndim != 3 or data_ctv.shape[0] != 3 or data_ctv.shape[2] != 25:
        raise ValueError(
            f"unexpected joint_pos shape {data_ctv.shape}; expected (3, T, 25)"
        )
    return np.asarray(data_ctv.transpose(1, 2, 0), dtype=np.float32)


@click.command()
@click.option("--npz", required=True,
              help="Path to a prepare-data NPZ (train or test).")
@click.option("--filename", default=None,
              help="Single clip stem to render (e.g. JP_06_anger_1_H).")
@click.option("--batch", default=None, type=click.Choice(["train", "test"]),
              help="Render every clip in the NPZ. Mutually exclusive with --filename.")
@click.option("--output-dir", default="output/rationale_cache/renders",
              help="Directory where <hash>.mp4 files are written.")
@click.option("--fps", default=10, type=int)
@click.option("--view", default="front_side_2view",
              type=click.Choice(["front", "front_side_2view"]))
@click.option("--slowmo", default=1.0, type=float)
@click.option("--limit", default=0, type=int,
              help="If >0, render only the first N clips (smoke test).")
@click.option("--max-frames", default=64, type=int,
              help="Temporally subsample each clip to at most this many frames "
              "before rendering. Default 64 matches the motion backbone's clip_length. "
              "Pass 0 for no subsample (full-length video).")
def main(
    npz: str,
    filename: str | None,
    batch: str | None,
    output_dir: str,
    fps: int,
    view: str,
    slowmo: float,
    limit: int,
    max_frames: int,
) -> None:
    env = EnvConfig()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = env.project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(npz)
    if not npz_path.exists():
        raise click.ClickException(f"NPZ not found: {npz_path}")

    if (filename is None) == (batch is None):
        raise click.ClickException("Specify exactly one of --filename or --batch")

    render_settings = {
        "schema_version": SCHEMA_VERSION,
        "fps": fps,
        "view": view,
        "slowmo": slowmo,
        "bg": "white",
        "scale": "hip_width",
        "line_width": 3,
        "max_frames": max_frames,
    }

    import pybvh_ml
    click.echo(f"Loading NPZ {npz_path.name} ... (one-time cost)")
    preprocessed = pybvh_ml.load_preprocessed(str(npz_path))
    stems = preprocessed["filenames"]
    click.echo(f"  loaded {len(stems)} clips")

    if filename is not None:
        target_stems = [filename]
    else:
        target_stems = stems
    if limit > 0:
        target_stems = target_stems[:limit]

    # Index written alongside renders: maps render_cache_key → filename stem.
    # Used by rationale generator for idempotency and for the
    # performer-identifier HMAC. Kept separate so leakage test can grep
    # the render dir alone without false positives.
    index_path = out_dir / "render_index.jsonl"

    n_ok = 0
    n_skip = 0
    with index_path.open("a") as idxf:
        for stem in target_stems:
            try:
                joint_pos = load_joint_positions_from_npz(
                    npz_path, stem, preprocessed=preprocessed,
                )
            except ValueError as exc:
                click.echo(f"[skip] {stem}: {exc}", err=True)
                n_skip += 1
                continue
            # Temporal subsample so long clips (up to ~7000 frames native) become
            # 64-frame videos suitable for VLM input and quick rendering.
            if max_frames > 0 and joint_pos.shape[0] > max_frames:
                idxs = np.linspace(
                    0, joint_pos.shape[0] - 1, num=max_frames, dtype=np.int64,
                )
                joint_pos = joint_pos[idxs]
            key = render_cache_key(joint_pos, render_settings)
            out_mp4 = out_dir / f"{key}.mp4"
            meta_path = out_dir / f"{key}.meta.json"
            if out_mp4.exists():
                n_skip += 1
                continue
            info = render_video(
                joint_pos, out_mp4, fps=fps, view=view, slowmo=slowmo,
            )
            meta = {
                "cache_key": key,
                "render_settings": render_settings,
                "frames_rendered": info["frames_rendered"],
                "n_clip_frames": int(joint_pos.shape[0]),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            idxf.write(json.dumps({"cache_key": key, "stem": stem}) + "\n")
            idxf.flush()
            n_ok += 1
            if n_ok % 25 == 0:
                click.echo(f"  rendered {n_ok}  (last stem: {stem}, key: {key[:12]}...)")

    click.echo(f"\nDone. rendered={n_ok}, skipped/existing={n_skip}, dir={out_dir}")


if __name__ == "__main__":
    main()
