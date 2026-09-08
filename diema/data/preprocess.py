"""BVH preprocessing pipeline: raw BVH files -> quaternion NPZ.

Stores data as quaternions — the canonical intermediate representation.
At training time, the Dataset converts to the target representation
(6D, Euler, etc.) specified in the experiment config.

See: diema_challenge_implementation_spec.md §6.1 — データ前処理
Related: diema/data/parser.py (filename parsing), tools/prepare_dataset.py (CLI)

Ported from: an internal baseline with additions:
  - Test-mode preprocessing (no labels)
  - Validation reporting
  - Augmentation at BVH level (optional)

NOTE: JP and TW BVH files have different Euler rotation orders (XZY vs ZYX).
pybvh_ml.preprocess_directory requires homogeneous rotation orders, so we
split by country prefix, preprocess separately, and merge NPZ outputs.
"""

import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pybvh_ml

from diema.data.parser import EMOTION_TO_IDX, is_valid_emotion, parse_emotion

logger = logging.getLogger(__name__)


def load_emo2idx(path: str | Path) -> dict[str, int]:
    """Load emotion-to-index mapping from a text file.

    Expected format: one 'emotion index' pair per line.
    If path is None, returns the default 12-class mapping.
    """
    if path is None:
        return EMOTION_TO_IDX.copy()
    emo2idx = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                emo2idx[parts[0]] = int(parts[1])
    return emo2idx


def _group_by_country(bvh_dir: Path) -> dict[str, list[Path]]:
    """Group BVH files by country prefix (e.g., JP_, TW_)."""
    groups = defaultdict(list)
    for f in sorted(bvh_dir.glob("*.bvh")):
        prefix = f.stem.split("_")[0]
        groups[prefix].append(f)
    return dict(groups)


def _merge_npz(part_paths: list[Path], output_path: Path) -> dict:
    """Merge multiple per-country NPZ files into a single NPZ.

    Renumbers clip indices and concatenates filenames/labels.
    Uses mean/std from the first part (quaternion stats are comparable
    across rotation orders since quaternions are order-independent).
    """
    all_data = {}
    all_filenames = []
    all_labels = []
    clip_offset = 0
    skel_info_json = None
    representation = None
    mean = None
    std = None

    for part_path in part_paths:
        data = np.load(part_path, allow_pickle=True)
        n = int(data["num_clips"])

        for i in range(n):
            new_idx = clip_offset + i
            all_data[f"clip_{new_idx}_root_pos"] = data[f"clip_{i}_root_pos"]
            all_data[f"clip_{new_idx}_joint_data"] = data[f"clip_{i}_joint_data"]
            if f"clip_{i}_joint_quats" in data:
                all_data[f"clip_{new_idx}_joint_quats"] = data[f"clip_{i}_joint_quats"]

        filenames = data["filenames"]
        all_filenames.extend(filenames.tolist())

        if "labels" in data:
            all_labels.extend(data["labels"].tolist())

        if mean is None:
            mean = data["mean"]
            std = data["std"]
            skel_info_json = str(data["skeleton_info_json"])
            representation = str(data["representation"])

        clip_offset += n

    save_dict = {
        "num_clips": np.array(clip_offset),
        "representation": np.array(representation),
        "filenames": np.array(all_filenames),
        "mean": mean,
        "std": std,
        "skeleton_info_json": np.array(skel_info_json),
    }
    save_dict.update(all_data)
    if all_labels:
        save_dict["labels"] = np.array(all_labels, dtype=np.int64)

    np.savez(output_path, **save_dict)

    return {
        "num_clips": clip_offset,
        "representation": representation,
        "filenames": all_filenames,
    }


def _preprocess_by_country(
    input_dir: Path,
    output_path: Path,
    label_fn=None,
) -> dict:
    """Preprocess BVH files split by country to avoid rotation order mismatch.

    JP (XZY) and TW (ZYX) have different Euler orders in BVH.
    Quaternion output is order-independent, so we process separately then merge.
    """
    groups = _group_by_country(input_dir)
    logger.info(f"Found {len(groups)} country groups: {list(groups.keys())}")

    part_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for country, files in sorted(groups.items()):
            country_dir = tmp / country
            country_dir.mkdir()
            for f in files:
                (country_dir / f.name).symlink_to(f)

            part_path = tmp / f"part_{country}.npz"
            logger.info(f"Preprocessing {country}: {len(files)} files")

            pybvh_ml.preprocess_directory(
                bvh_dir=country_dir,
                output_path=part_path,
                representation="quaternion",
                center_root=True,
                label_fn=label_fn,
            )
            part_paths.append(part_path)

        result = _merge_npz(part_paths, output_path)

    logger.info(f"Merged {result['num_clips']} clips to {output_path}")
    return result


def preprocess_train(
    input_dir: str | Path,
    output_path: str | Path,
    emo2idx: dict[str, int] | None = None,
) -> dict:
    """Preprocess training BVH files into quaternion NPZ.

    Extracts emotion labels from filenames. Handles mixed rotation
    orders (JP/TW) by preprocessing per-country then merging.

    Args:
        input_dir: directory containing .bvh files
        output_path: output .npz path
        emo2idx: emotion-to-index mapping. If None, uses default 12-class.

    Returns:
        Dict with preprocessing results (num_clips, representation, etc.)
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if emo2idx is None:
        emo2idx = EMOTION_TO_IDX.copy()

    bvh_files = sorted(input_dir.glob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {input_dir}")

    def label_fn(stem):
        emotion = parse_emotion(stem)
        return emo2idx[emotion]

    result = _preprocess_by_country(input_dir, output_path, label_fn=label_fn)
    logger.info(f"Preprocessed {result['num_clips']} train clips to {output_path}")
    return result


def preprocess_test(
    input_dir: str | Path,
    output_path: str | Path,
) -> dict:
    """Preprocess test BVH files into quaternion NPZ (no labels).

    All BVH files in the directory are included (no emotion filtering).
    Labels are set to -1 (unknown).

    Args:
        input_dir: directory containing test .bvh files
        output_path: output .npz path

    Returns:
        Dict with preprocessing results.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bvh_files = sorted(input_dir.glob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {input_dir}")

    def label_fn(stem):
        return -1

    result = _preprocess_by_country(input_dir, output_path, label_fn=label_fn)
    logger.info(f"Preprocessed {result['num_clips']} test clips to {output_path}")
    return result


def preprocess_with_augmentation(
    input_dir: str | Path,
    output_path: str | Path,
    emo2idx: dict[str, int] | None = None,
    augment_copies: int = 3,
    speed_range: tuple[float, float] = (0.8, 1.2),
    dropout_rate: float = 0.1,
) -> dict:
    """Preprocess training BVH with BVH-level augmentation.

    Generates augmented copies of included BVH files into a temp
    directory, then preprocesses everything at once.

    Args:
        input_dir: directory containing .bvh files
        output_path: output .npz path
        emo2idx: emotion-to-index mapping
        augment_copies: number of augmented copies per sample
        speed_range: (lo, hi) for speed perturbation
        dropout_rate: frame dropout rate

    Returns:
        Dict with preprocessing results.
    """
    import random
    import shutil
    import pybvh
    from pybvh.transforms import speed_perturbation, dropout_frames

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if emo2idx is None:
        emo2idx = EMOTION_TO_IDX.copy()

    valid_paths = [
        p for p in sorted(input_dir.glob("*.bvh"))
        if is_valid_emotion(p.stem)
    ]

    with tempfile.TemporaryDirectory() as aug_dir:
        aug_path = Path(aug_dir)

        for bvh_path in valid_paths:
            shutil.copy2(bvh_path, aug_path / bvh_path.name)

            for i in range(augment_copies):
                bvh = pybvh.read_bvh_file(bvh_path)

                if speed_range is not None:
                    factor = random.uniform(*speed_range)
                    bvh = speed_perturbation(bvh, factor)

                if dropout_rate > 0:
                    bvh = dropout_frames(bvh, dropout_rate)

                aug_stem = f"{bvh_path.stem}_aug{i:02d}"
                bvh.to_bvh_file(str(aug_path / f"{aug_stem}.bvh"))

        def label_fn(stem):
            emotion = parse_emotion(stem)
            return emo2idx[emotion]

        result = pybvh_ml.preprocess_directory(
            bvh_dir=aug_path,
            output_path=output_path,
            representation="quaternion",
            center_root=True,
            label_fn=label_fn,
        )

    logger.info(
        f"Preprocessed {result['num_clips']} augmented clips "
        f"({len(valid_paths)} originals x {augment_copies + 1}) to {output_path}"
    )
    return result


def validate_preprocessed(npz_path: str | Path) -> dict:
    """Validate a preprocessed NPZ file and return summary statistics.

    Args:
        npz_path: path to .npz file

    Returns:
        Dict with validation info: num_clips, sequence_lengths, label_distribution, etc.
    """
    preprocessed = pybvh_ml.load_preprocessed(npz_path)
    clips = preprocessed["clips"]
    labels = preprocessed.get("labels")
    filenames = preprocessed.get("filenames", [])

    seq_lengths = [clip["joint_data"].shape[0] for clip in clips]

    result = {
        "num_clips": len(clips),
        "seq_length_min": min(seq_lengths) if seq_lengths else 0,
        "seq_length_max": max(seq_lengths) if seq_lengths else 0,
        "seq_length_mean": sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
        "num_filenames": len(filenames),
    }

    if labels is not None:
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        result["label_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
        result["num_unique_labels"] = len(unique)

    if filenames:
        from diema.data.parser import parse_performer_id
        try:
            performers = {parse_performer_id(f) for f in filenames}
            result["num_performers"] = len(performers)
        except (IndexError, ValueError):
            pass  # test filenames (e.g., test0001) don't have performer info

    return result
