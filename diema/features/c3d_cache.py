""" filename → C3D feature vector lookup.

Related:
- tools/cache_c3d_stats.py (produces the NPZ this class loads)
- diema/training/c3d_fusion_trainer.py (consumer)

Wraps an ``output/c3d_stats_cache.npz`` file and exposes a
``filename_stem → (51,) float32`` lookup, mirroring the contract of
``diema.features.text_cache.ScenarioTextCache``. Unknown filenames return
an all-zero vector so the late-fusion model can fall back to motion-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class C3DStatsCache:
    """Read-only lookup from filename stem to 51-D C3D feature vector.

    Args:
        cache_path: path to the NPZ produced by ``tools/cache_c3d_stats.py``.
        standardize: if True, z-score normalize each feature column across
            the whole corpus (robust per-feature stats saved in ``.stats``).
            Useful because raw values span many orders of magnitude
            (duration_sec vs jerk).

    Attributes:
        num_features: int, 51 on the canonical DIEM-A corpus.
        num_filenames: int, 9,936 on full train+test corpus.
        stats: dict with per-feature ``mean``/``std`` arrays (if standardize).
    """

    def __init__(self, cache_path: str | Path, standardize: bool = True):
        p = Path(cache_path)
        if not p.exists():
            raise FileNotFoundError(
                f"C3D cache not found: {p}. "
                "Build with `python tools/cache_c3d_stats.py` first."
            )
        data = np.load(p, allow_pickle=False)
        self.features: np.ndarray = data["features"].astype(np.float32)
        self.filenames: list[str] = [str(s) for s in data["filenames"].tolist()]
        self.feature_names: list[str] = [str(s) for s in data["feature_names"].tolist()]

        # ezc3d sometimes returns NaN for frames where a marker was dropped
        # (residual < 0). Those NaNs propagate through np.mean and can reach
        # ~10% of samples for the "global" (all-marker) stats. Replace with
        # zero on load so downstream scikit-learn / MLP consumers don't choke.
        # A column whose rate exceeds ``nan_warn_threshold`` gets a single
        # stderr warning so the user knows to audit the upstream cache.
        nan_mask = np.isnan(self.features)
        if nan_mask.any():
            import warnings
            col_rate = nan_mask.mean(axis=0)
            noisy = [
                self.feature_names[i] for i, r in enumerate(col_rate) if r > 0.05
            ]
            warnings.warn(
                f"C3DStatsCache: {int(nan_mask.any(axis=1).sum())}/"
                f"{len(self.filenames)} samples contain NaN in at least one "
                f"feature; replacing with 0. Columns with >5% NaN: {noisy}",
                stacklevel=2,
            )
            self.features = np.where(nan_mask, 0.0, self.features)
        inf_mask = np.isinf(self.features)
        if inf_mask.any():
            self.features = np.where(inf_mask, 0.0, self.features)

        if self.features.shape != (len(self.filenames), len(self.feature_names)):
            raise ValueError(
                f"cache shape mismatch: features {self.features.shape} vs "
                f"{(len(self.filenames), len(self.feature_names))}"
            )

        self._filename_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.filenames)
        }

        self.stats: dict[str, np.ndarray] = {}
        if standardize:
            # Robust scale: use per-column mean and std across the corpus.
            # Clamp tiny stds so constant columns become zero after standardize.
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            self.features = (self.features - mean) / std
            self.stats = {"mean": mean, "std": std}

    # -- public API -------------------------------------------------------

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_filenames(self) -> int:
        return len(self._filename_to_idx)

    def vector_for(self, filename: str) -> np.ndarray:
        """Return the (num_features,) feature vector for a filename.

        Accepts either the stem (``JP_06_joy_1_L``) or a full path
        (``.../JP_06_joy_1_L.bvh``); returns a zero vector for unknown
        filenames so the model can fall back to pure motion.
        """
        stem = Path(str(filename)).stem
        idx = self._filename_to_idx.get(stem)
        if idx is None:
            return np.zeros(self.num_features, dtype=np.float32)
        return self.features[idx].copy()

    def batch_tensor(self, filenames: list[str]) -> torch.Tensor:
        """Vectorized :meth:`vector_for`. Returns ``(N, num_features)`` float32."""
        vecs = np.stack([self.vector_for(fn) for fn in filenames], axis=0)
        return torch.from_numpy(vecs).float()

    def batch_mask(self, filenames: list[str]) -> torch.Tensor:
        """Return a bool tensor ``(N,)`` that is True when the C3D entry exists."""
        return torch.tensor(
            [Path(str(fn)).stem in self._filename_to_idx for fn in filenames],
            dtype=torch.bool,
        )
