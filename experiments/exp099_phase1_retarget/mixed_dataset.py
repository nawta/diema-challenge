"""mixed_dataset.py — Stage 2 of exp099

`JointPosMixedDataset` returns (C=3, T, V=25) joint-position tensors
from a mix of original DIEM-A clips and retargeted clips.

Design notes
------------
Both branches share the same pipeline:
    quaternions + root_pos + offsets → augment (quat space) → FK → CTV pack

The only difference is which offsets feed FK:

- **Original**: per-clip source performer's raw BVH offsets ("honest" mode)
  OR the hardcoded JP_06 _DIEMA_OFFSETS_BVH ("flat" mode).
- **Retargeted**: target performer's raw BVH offsets (from .npz sidecar).

LPO fold rule: each retargeted clip's source performer must be in the
training set; this is enforced at __init__ time using the `train_performer_ids`
arg.

Mix ratio: the retargeted subset of size `int(n_orig * mix_ratio /
(1-mix_ratio))` is sampled ONCE in __init__ (not re-sampled per epoch).
When the LPO-filtered pool is smaller than that target it is upsampled
with replacement; otherwise a fixed subset is drawn. `__len__` = n_orig
+ n_retargeted. Per-epoch re-sampling is a proper TODO.

See: §5, §7
Related: diema/data/dataset.py (MotionDataset), diema/data/collate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.utils.data

import pybvh_ml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from diema.features.multi_stream import (  # noqa: E402
    _DIEMA_PARENTS_BVH, _load_reference_offsets,
    _pack_25_node, compute_joint_positions)
from diema.data.parser import parse_filename  # noqa: E402


CACHED_NPZ_DEFAULT = "data/diema_challenge/processed/motion_train_quat.npz"
RAW_OFFSETS_JSON = Path("output/retarget/phase1_raw_offsets.json")
PHASE1_MANIFEST = Path("output/retarget/phase1_manifest.json")


def _ctv_pack_jointpos(root_pos: np.ndarray,
                       joint_positions_24: np.ndarray) -> np.ndarray:
    """Pack (T, 24, 3) world joint positions + (T, 3) root to (3, T, 25).

    Matches production `diema.features.multi_stream._pack_25_node` +
    `get_stream_features("joint_pos")` convention:
    - Node 0 = root_pos (global translation)
    - Nodes 1-24 = world_pos - hips_pos (root-relative offsets)

    The root-relative subtraction at nodes 1-24 keeps body-shape variation
    O(body size) instead of O(global trajectory). Without it, std blows up
    by ~10× and BatchNorm in TokenEmbedding can't recover in 20 epochs
    (smoke run jointpos_honest fold_00: val_f1 stuck at 0.10 vs rotation_6d
    0.25).
    """
    T = root_pos.shape[0]
    assert joint_positions_24.shape == (T, 24, 3), (
        f"joint_positions_24 expected (T, 24, 3); got {joint_positions_24.shape}")
    # joint_positions_24[:, 0, :] is the hips (BVH idx 0), == root_pos by FK.
    root_relative = joint_positions_24 - joint_positions_24[:, 0:1, :]
    # _pack_25_node returns (C, T, 25)
    return _pack_25_node(root_pos.astype(np.float32),
                          root_relative.astype(np.float32))


def _build_augmentation_pipeline(
    *, augment_rotate: bool, augment_mirror: bool,
    augment_speed: bool, augment_noise_sigma: float,
    lr_joint_pairs: list[tuple[int, int]],
):
    """Build a pybvh_ml.AugmentationPipeline matching production a01."""
    steps = []
    if augment_rotate:
        # up_idx=2 (Z): the DIEM-A skeleton is Z-up in world space, so a
        # true vertical-axis (yaw) rotation must use axis 2. Production
        # exp004 passed up_idx=1 (Y), which tumbles the figure — see
        # docs/analysis up-axis investigation. This corrected smoke uses 2.
        steps.append((
            pybvh_ml.rotate_quaternions_vertical, 1.0,
            {"angle_deg": lambda rng: rng.uniform(-180, 180), "up_idx": 2},
        ))
    if augment_mirror:
        steps.append((
            pybvh_ml.mirror_quaternions, 0.5,
            {"lr_joint_pairs": lr_joint_pairs, "lateral_idx": 0},
        ))
    if augment_speed:
        steps.append((
            pybvh_ml.speed_perturbation_arrays, 1.0,
            {"factor": lambda rng: rng.uniform(0.8, 1.2)},
        ))
    if augment_noise_sigma > 0:
        steps.append((
            pybvh_ml.add_joint_noise_quaternions, 1.0,
            {"sigma_deg": augment_noise_sigma},
        ))
    return pybvh_ml.AugmentationPipeline(steps) if steps else None


class JointPosMixedDataset(torch.utils.data.Dataset):
    """Mixed-corpus joint_pos dataset for SkateFormer training.

    Parameters
    ----------
    original_split_entries : list of (filename, original_idx) tuples
        Train or val split entries for the original DIEM-A corpus (from
        diema.data.splits.generate_lpo_splits).
    train_performer_ids : set of str OR None
        Performer IDs in the training set (for the current fold). Used
        to filter the retargeted manifest so we don't leak source-performer
        motion across folds. Pass None to disable filtering (val/test sets
        get no retargeted samples).
    cached_npz_path : str
        Path to motion_train_quat.npz (default data/diema_challenge/...).
    phase1_manifest_path : Path
        Path to output/retarget/phase1_manifest.json.
    raw_offsets_path : Path
        Path to output/retarget/phase1_raw_offsets.json.
    original_fk_rule : "honest" | "flat"
        - "honest": each original clip uses its source performer's raw
          BVH offsets (load from raw_offsets cache).
        - "flat": all original clips use _DIEMA_OFFSETS_BVH (legacy).
    mix_ratio : float in [0, 1]
        Fraction of training samples drawn from the retargeted corpus.
        Mix ratio = retargeted / (retargeted + original). The retargeted
        count per epoch = int(n_orig * r / (1 - r)).
    clip_length : int
        Frames per sample.
    is_test : bool
        If True, use deterministic center sampling and disable augmentation.
        ALSO if True, mix_ratio is set to 0 (val/test never see retargeted).
    augmentation_pipeline : pybvh_ml.AugmentationPipeline or None
        Quaternion-space augmentations (applied to BOTH branches before FK).
    seed : int
        Base RNG seed.
    """

    def __init__(
        self,
        original_split_entries: list[tuple[str, int]],
        train_performer_ids: set[str] | None,
        cached_npz_path: str = CACHED_NPZ_DEFAULT,
        phase1_manifest_path: Path = PHASE1_MANIFEST,
        raw_offsets_path: Path = RAW_OFFSETS_JSON,
        original_fk_rule: Literal["honest", "flat"] = "honest",
        mix_ratio: float = 0.4,
        clip_length: int = 64,
        is_test: bool = False,
        augmentation_pipeline=None,
        seed: int = 42,
    ):
        super().__init__()
        if not 0.0 <= mix_ratio < 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1); got {mix_ratio}")
        if original_fk_rule not in ("honest", "flat"):
            raise ValueError(
                f"original_fk_rule must be 'honest' or 'flat'; got "
                f"{original_fk_rule}")
        self.clip_length = clip_length
        self.is_test = is_test
        self.augmentation_pipeline = augmentation_pipeline
        self.original_fk_rule = original_fk_rule
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

        # -- 1) Load cached npz once (preprocessed object holds clips lazily) --
        preprocessed = pybvh_ml.load_preprocessed(cached_npz_path)
        self._all_clips = preprocessed["clips"]   # list of dict
        self._all_labels = preprocessed.get("labels")
        self._all_filenames = preprocessed.get("filenames", [])

        # -- 2) Build original sample index ---------------------------------
        self._orig_indices = [int(idx) for _, idx in original_split_entries]

        # -- 3) Load raw offsets cache (for honest-FK + retargeted FK) ------
        offsets_cache = json.loads(raw_offsets_path.read_text())
        self._raw_offsets = {
            pid: np.asarray(arr, dtype=np.float32)
            for pid, arr in offsets_cache["offsets"].items()
        }
        self._flat_offsets = _load_reference_offsets().astype(np.float32)

        # -- 4) Build retargeted index (filtered to training performers) ----
        if mix_ratio > 0 and not is_test:
            manifest = json.loads(phase1_manifest_path.read_text())
            all_pairs = manifest["pairs"]
            if train_performer_ids is None:
                # No filtering — accept all
                eligible = all_pairs
            else:
                eligible = [
                    p for p in all_pairs
                    if p["performer"] in train_performer_ids
                ]
            n_pool_unique = len(eligible)
            self._n_retgt_pool = n_pool_unique
            if n_pool_unique == 0:
                raise ValueError(
                    "Retargeted pool is empty after the LPO source-performer "
                    "filter. Either mix_ratio should be 0, or train_performer_ids "
                    "does not intersect the manifest's source performers. "
                    f"(manifest pairs={len(all_pairs)}, "
                    f"train performers={len(train_performer_ids) if train_performer_ids else 'None'})")
            # Decide per-epoch retargeted count: target = mix_ratio of total.
            # n_retgt / (n_orig + n_retgt) = mix_ratio
            n_orig = len(self._orig_indices)
            target_n_retgt = int(round(
                n_orig * mix_ratio / max(1e-6, 1.0 - mix_ratio)))
            if target_n_retgt > len(eligible):
                # Pool too small (LPO filter shrinks it). Upsample with
                # replacement so we still hit the requested mix ratio.
                # Each duplicate gets different augmentation each step, so
                # this isn't degenerate.
                pick = self._rng.choice(
                    len(eligible), size=target_n_retgt, replace=True)
                eligible = [eligible[i] for i in pick]
                n_retgt_use = target_n_retgt
            else:
                # Subsample without replacement
                pick = self._rng.choice(
                    len(eligible), size=target_n_retgt, replace=False)
                eligible = [eligible[i] for i in pick]
                n_retgt_use = target_n_retgt
            self._retargeted_pairs = eligible
            self._n_retgt = n_retgt_use
            self._target_mix_ratio = mix_ratio
        else:
            self._retargeted_pairs = []
            self._n_retgt = 0
            self._target_mix_ratio = 0.0
            self._n_retgt_pool = 0

        # -- 5) Layout: first n_orig samples = originals; rest = retargeted
        #     This keeps DataLoader's shuffle / indexing simple.
        self._n_orig = len(self._orig_indices)
        self._total = self._n_orig + self._n_retgt

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reseed_rng(self, worker_id: int) -> None:
        self._rng = np.random.default_rng(self._base_seed + worker_id)

    def actual_mix_ratio(self) -> float:
        return self._n_retgt / max(1, self._total)

    def summary(self) -> dict:
        return {
            "n_original": self._n_orig,
            "n_retargeted_per_epoch": self._n_retgt,
            "total_per_epoch": self._total,
            "target_mix_ratio": self._target_mix_ratio,
            "actual_mix_ratio": self.actual_mix_ratio(),
            "original_fk_rule": self.original_fk_rule,
            "is_test": self.is_test,
            "n_retargeted_pool_unique": self._n_retgt_pool,
            "upsampled": self._n_retgt > self._n_retgt_pool,
        }

    # ------------------------------------------------------------------
    # Dataset contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        if idx < self._n_orig:
            return self._get_original(idx)
        return self._get_retargeted(idx - self._n_orig)

    # ------------------------------------------------------------------
    # Per-branch loaders
    # ------------------------------------------------------------------

    def _get_original(self, local_idx: int):
        orig_idx = self._orig_indices[local_idx]
        clip = self._all_clips[orig_idx]
        root_pos = clip["root_pos"].copy().astype(np.float32)   # (F, 3)
        joint_quats = clip["joint_data"].copy().astype(np.float32)  # (F, 24, 4)

        if not self.is_test and self.augmentation_pipeline is not None:
            joint_quats, root_pos = self.augmentation_pipeline(
                joint_quats, root_pos, rng=self._rng)

        if self.original_fk_rule == "honest":
            performer = self._performer_from_filename(self._all_filenames[orig_idx])
            offsets = self._raw_offsets.get(performer)
            if offsets is None:
                # Honest-FK requires every performer's raw offsets. A miss
                # would silently corrupt the honest-vs-flat comparison by
                # mixing flat-FK samples in. Fail loudly instead.
                raise KeyError(
                    f"honest-FK: no raw offsets for performer {performer!r} "
                    f"(clip {orig_idx}, file {self._all_filenames[orig_idx]!r}). "
                    f"Raw-offsets cache has {len(self._raw_offsets)} performers. "
                    f"Regenerate the cache or check parse_filename consistency.")
        else:
            offsets = self._flat_offsets

        return self._finalize(
            root_pos=root_pos,
            joint_quats=joint_quats,
            offsets=offsets,
            label=self._all_labels[orig_idx] if self._all_labels is not None else -1,
            filename=self._all_filenames[orig_idx] if self._all_filenames else "",
        )

    def _get_retargeted(self, local_idx: int):
        p = self._retargeted_pairs[local_idx]
        orig_idx = int(p["idx"])
        clip = self._all_clips[orig_idx]
        joint_quats = clip["joint_data"].copy().astype(np.float32)
        # Use the retarget's stored new_root_pos (ground-contact shifted),
        # NOT the original root_pos. Close the NpzFile handle explicitly.
        with np.load(p["output_npz"]) as sidecar:
            root_pos = np.asarray(sidecar["new_root_pos"], dtype=np.float32)
            target_offsets = np.asarray(
                sidecar["target_offsets"], dtype=np.float32)

        if not self.is_test and self.augmentation_pipeline is not None:
            joint_quats, root_pos = self.augmentation_pipeline(
                joint_quats, root_pos, rng=self._rng)

        return self._finalize(
            root_pos=root_pos,
            joint_quats=joint_quats,
            offsets=target_offsets,
            label=self._all_labels[orig_idx] if self._all_labels is not None else -1,
            filename=Path(p["output_npy"]).name,
        )

    def _finalize(
        self,
        *, root_pos: np.ndarray, joint_quats: np.ndarray,
        offsets: np.ndarray, label: int, filename: str,
    ):
        """FK + CTV pack + temporal sample + tensor conversion."""
        # Forward kinematics → (F, 24, 3)
        joint_positions = compute_joint_positions(
            root_pos=root_pos.astype(np.float32),
            joint_quats=joint_quats.astype(np.float32),
            offsets=offsets.astype(np.float32),
            parents=_DIEMA_PARENTS_BVH,
        )
        # CTV pack → (3, T, 25)
        data_ctv = _ctv_pack_jointpos(root_pos, joint_positions)

        # Temporal sample
        num_frames = data_ctv.shape[1]
        mode = "test" if self.is_test else "train"
        frame_idx = pybvh_ml.uniform_temporal_sample(
            num_frames, self.clip_length, mode=mode, rng=self._rng,
        )
        frame_idx = frame_idx % num_frames
        data_ctv = data_ctv[:, frame_idx, :]   # (3, clip_length, 25)

        return (
            torch.tensor(data_ctv, dtype=torch.float32),
            torch.tensor(int(label), dtype=torch.long),
            filename,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _performer_from_filename(self, fn: str) -> str:
        try:
            info = parse_filename(Path(fn).stem)
            return f"{info.nationality}_{info.performer_num}"
        except Exception:
            # Test files like "test0001" have no nationality/performer
            return ""
