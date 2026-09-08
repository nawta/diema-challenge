""" C3D parser wrapping ezc3d.

Related:
- diema/features/c3d_stats.py — consumes :func:`read_c3d` output
- data/diema_challenge/raw/c3d/{train,test}/*.c3d

Reads a DIEM-A Challenge C3D file and returns a compact dict:

    {
        "markers":       (F, M, 3) float32, 世界座標 xyz (no residual)
        "residual":      (F, M)    float32, per-frame marker dropout proxy
        "marker_labels": list[str], subject prefix stripped (e.g. "ARIEL", "LFHD")
        "subject":       str | None, stripped prefix (e.g. "JP_06") — exposed
                         for provenance but never propagated to features by default
        "rate":          float (Hz, 120 for DIEM-A)
        "n_markers":     int (57 for Vicon Production)
        "n_frames":      int
    }

**Leakage guard** (finding): The raw C3D POINT.LABELS
are prefixed with the subject id (e.g. ``JP_43:ARIEL`` for the anonymized test
clip ``test0001.c3d``). We strip the prefix here and surface it only via the
separate ``"subject"`` key so downstream feature code can enforce that
performer id never enters the feature vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_c3d(path: str | Path, strict: bool = True) -> dict:
    """Read a DIEM-A C3D file via ezc3d.

    Args:
        path: filesystem path to the .c3d file.
        strict: if True (default), validate that ``POINT.RATE`` is 120 Hz and
            the marker count is 57 (DIEM-A Vicon Production). Pass False for
            third-party / exotic files.

    Returns:
        dict with keys described in the module docstring.
    """
    import ezc3d

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"C3D not found: {p}")
    c = ezc3d.c3d(str(p))

    params = c["parameters"]
    rate = float(params["POINT"]["RATE"]["value"][0])
    n_used = int(params["POINT"]["USED"]["value"][0])
    labels_raw: list[str] = list(params["POINT"]["LABELS"]["value"])
    pts = c["data"]["points"]  # (4, M, F): xyz + residual

    if strict:
        if rate != 120.0:
            raise ValueError(f"unexpected sampling rate {rate} Hz (expected 120)")
        if n_used != 57:
            raise ValueError(f"unexpected marker count {n_used} (expected 57 Vicon Production)")

    # Strip subject prefix. DIEM-A labels are "<COUNTRY>_<NN>:<MARKER>".
    clean_labels: list[str] = []
    subjects_seen: set[str] = set()
    for lab in labels_raw:
        if ":" in lab:
            subj, marker = lab.split(":", 1)
            clean_labels.append(marker)
            subjects_seen.add(subj)
        else:
            clean_labels.append(lab)
    if len(subjects_seen) == 1:
        subject = next(iter(subjects_seen))
    elif len(subjects_seen) == 0:
        subject = None
    else:
        # Multi-subject recording — shouldn't happen on DIEM-A but don't crash.
        subject = "|".join(sorted(subjects_seen))

    # Reshape to (F, M, 3) + (F, M) for residual.
    if pts.shape[0] != 4:
        raise ValueError(f"expected POINT.points.shape[0]==4, got {pts.shape}")
    markers = np.ascontiguousarray(pts[:3].transpose(2, 1, 0).astype(np.float32))
    residual = np.ascontiguousarray(pts[3].transpose(1, 0).astype(np.float32))

    return {
        "markers": markers,
        "residual": residual,
        "marker_labels": clean_labels,
        "subject": subject,
        "rate": rate,
        "n_markers": int(markers.shape[1]),
        "n_frames": int(markers.shape[0]),
    }


# Vicon Production 57-marker layout (DIEM-A) — body-part groups.
# Source: inspection of data/diema_challenge/raw/c3d/train/JP_06_*.c3d
# (single authoritative layout across all 74 training performers).
VICON_PRODUCTION_PART_GROUPS: dict[str, list[str]] = {
    "head": ["ARIEL", "LFHD", "LBHD", "RFHD", "RBHD"],
    "torso_spine": ["C7", "T10", "LT10", "RT10", "CLAV", "STRN"],
    "l_arm": [
        "LFSH", "LTSH", "LBSH", "LUPA", "LELB", "LBEL",
        "LFRM", "LIWR", "LOWR", "LIHAND", "LOHAND",
    ],
    "r_arm": [
        "RFSH", "RTSH", "RBSH", "RUPA", "RELB", "RBEL",
        "RFRM", "RIWR", "ROWR", "RIHAND", "ROHAND",
    ],
    "pelvis": ["LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT"],
    "l_leg": ["LTHI", "LKNE", "LKNU", "LSHN", "LANK", "LHEL", "LMT5", "LMT1", "LTOE"],
    "r_leg": ["RTHI", "RKNE", "RKNU", "RSHN", "RANK", "RHEL", "RMT5", "RMT1", "RTOE"],
}

#: All Vicon Production marker names in the canonical CSV-insertion order
#: produced by inspection (see read_c3d docstring). Exported for tests.
VICON_PRODUCTION_MARKERS: list[str] = [
    m for group in VICON_PRODUCTION_PART_GROUPS.values() for m in group
]


def group_markers_by_part(
    marker_labels: list[str],
    groups: dict[str, list[str]] | None = None,
) -> dict[str, list[int]]:
    """Map each part name to the list of indices in ``marker_labels`` it spans.

    Unknown markers (not in any group) are silently skipped so the function
    tolerates alternative marker sets (e.g. a harmonized subset) without
    raising. Missing markers (in group but not in labels) likewise do not
    raise — they simply shrink the part's index list.

    Args:
        marker_labels: ordered list from :func:`read_c3d`.
        groups: mapping part → list of marker names. Defaults to
            :data:`VICON_PRODUCTION_PART_GROUPS`.

    Returns:
        ``{part: [indices into marker_labels]}`` with part names preserved.
    """
    groups = groups or VICON_PRODUCTION_PART_GROUPS
    name_to_idx = {name: i for i, name in enumerate(marker_labels)}
    out: dict[str, list[int]] = {}
    for part, names in groups.items():
        out[part] = [name_to_idx[n] for n in names if n in name_to_idx]
    return out
