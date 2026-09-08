""" scenario text cache + filename → (idx, embedding) lookup.

Related:
- diema/features/text_features.py — ScenarioTextEncoder / save_text_cache
- tools/build_scenario_cache.py — offline builder (supports --mask=emotion)
- diema/training/losses_motion_text.py — GlobalInfoNCELoss consumer

A thin wrapper that composes the two existing building blocks:
1. The on-disk ``{text: embedding}`` cache (unmasked or masked) built by
   ``tools/build_scenario_cache.py``.
2. A filename → scenario-text map derived from ``train_data.csv``.

Why a new module: trainers need *three* tensors per batch — the
L2-normalized embedding, a scenario index (int) for multi-positive masking,
and an optional mask flag for val/test samples with no text. Building those
inline in each run.py duplicates logic and risks fold leakage. This module
centralizes the lookup + integer assignment so every experiment reads the
same stable ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from diema.features.text_features import load_text_cache


class ScenarioTextCache:
    """Lookup ``filename -> (scenario_idx: int, embedding: Tensor)``.

    Scenario text embeddings are keyed by the raw scenario string (either
    unmasked or ``[MASK]``-masked depending on which cache file was loaded).
    The integer ``scenario_idx`` is assigned once on construction by sorting
    unique scenario strings lexicographically — this keeps the mapping stable
    across runs and across the masked / unmasked caches (both share the same
    set of scenario *strings* after masking, though the masked variant may
    merge some strings into duplicates).

    Args:
        cache_path: path to a ``.pt`` cache built by
            ``tools.build_scenario_cache.build_cache``.
        csv_path: path to ``train_data.csv`` for the filename → scenario map.
        filename_column: column name that holds the filename (defaults to
            ``original_name``, matching DIEM-A train CSV).
        scenario_column: column name that holds the scenario text.
        mask_emotion_on_the_fly: if True, re-apply ``tools.build_scenario_cache``
            masking to the CSV's raw scenario text before looking up the
            embedding. Needed when the CSV is still raw but the cache was
            built with ``--mask=emotion``. If the cache meta already has
            ``mask_mode=emotion``, this is auto-detected.
        forbidden_yaml: path to forbidden tokens YAML (only needed when
            re-masking on the fly).
    """

    _UNKNOWN_IDX = -1

    def __init__(
        self,
        cache_path: str | Path,
        csv_path: str | Path,
        filename_column: str = "original_name",
        scenario_column: str = "scenario",
        mask_emotion_on_the_fly: bool | None = None,
        forbidden_yaml: str | Path | None = None,
    ):
        import pandas as pd

        self.cache_path = Path(cache_path)
        self.csv_path = Path(csv_path)
        self.embeddings_raw, self.meta = load_text_cache(self.cache_path)

        if mask_emotion_on_the_fly is None:
            mask_emotion_on_the_fly = self.meta.get("mask_mode") == "emotion"

        self._mask_regex = None
        if mask_emotion_on_the_fly:
            from diema.features.emotion_mask import (
                compile_emotion_mask_regex,
                load_forbidden_tokens,
                mask_emotion_aliases,
            )
            yaml_path = Path(
                forbidden_yaml
                or self.meta.get("forbidden_yaml")
                or "configs/leakage/forbidden_tokens.yaml"
            )
            forbidden = load_forbidden_tokens(yaml_path)
            self._mask_regex = compile_emotion_mask_regex(forbidden["emotion_aliases"])
            self._mask_fn = mask_emotion_aliases

        if not self.csv_path.exists():
            raise FileNotFoundError(f"train CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        for col in (filename_column, scenario_column):
            if col not in df.columns:
                raise ValueError(f"'{col}' column missing from {self.csv_path}")
        df = df[[filename_column, scenario_column]].dropna().copy()
        df[scenario_column] = df[scenario_column].astype(str)
        if self._mask_regex is not None:
            df[scenario_column] = df[scenario_column].apply(
                lambda s: self._mask_fn(s, self._mask_regex)[0]
            )

        # Stable ordering for scenario_idx.
        unique_scenarios = sorted({s for s in df[scenario_column].tolist()
                                   if s in self.embeddings_raw})
        self._text_to_idx: dict[str, int] = {
            text: idx for idx, text in enumerate(unique_scenarios)
        }
        self._idx_to_text: list[str] = unique_scenarios

        # Stack embeddings in the same deterministic order as idx.
        embedding_dim = None
        stacked: list[torch.Tensor] = []
        for text in unique_scenarios:
            vec = self.embeddings_raw[text]
            if embedding_dim is None:
                embedding_dim = int(vec.numel())
            stacked.append(vec.to(torch.float32).reshape(-1))
        self._embedding_dim = embedding_dim or 0
        if stacked:
            self._embedding_matrix = torch.stack(stacked, dim=0)  # (U, D)
        else:
            self._embedding_matrix = torch.zeros(0, self._embedding_dim)

        # Build filename → scenario_idx map.
        self._filename_to_idx: dict[str, int] = {}
        dropped = 0
        for fname, text in zip(df[filename_column].tolist(), df[scenario_column].tolist()):
            idx = self._text_to_idx.get(text)
            if idx is None:
                dropped += 1
                continue
            # Strip any .bvh / .c3d / .fbx extension for a canonical key.
            stem = Path(str(fname)).stem
            self._filename_to_idx[stem] = idx
        self._dropped_rows = dropped
        if dropped > 0:
            import warnings
            warnings.warn(
                f"ScenarioTextCache: {dropped}/{len(df)} rows in {self.csv_path.name} "
                f"reference a scenario not present in cache {self.cache_path.name}. "
                "This usually means the cache was built from a different CSV or with "
                "different masking. Rebuild with tools/build_scenario_cache.py.",
                stacklevel=2,
            )

    # -- public API -------------------------------------------------------

    @property
    def num_unique(self) -> int:
        """Number of unique scenario strings in the cache."""
        return len(self._idx_to_text)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """(num_unique, D) stacked embeddings, aligned to scenario_idx."""
        return self._embedding_matrix

    @property
    def num_filenames(self) -> int:
        return len(self._filename_to_idx)

    @property
    def filename_to_idx(self) -> dict[str, int]:
        """Read-only view of the filename-stem → scenario_idx map.

        Consumers (e.g. ``TextAlignLightningModel``) should use this public
        accessor instead of touching ``_filename_to_idx`` directly, so the
        internal storage can evolve (e.g. to keep extensions if needed).
        """
        return dict(self._filename_to_idx)

    def idx_for(self, filename: str) -> int:
        """Return the scenario_idx for a filename.

        Returns ``-1`` if the filename is unknown (e.g. a test sample for which
        no scenario is available). Callers should treat -1 as a masked sample
        (exclude from the InfoNCE loss / set its positive_mask row to False).
        """
        stem = Path(str(filename)).stem
        return self._filename_to_idx.get(stem, self._UNKNOWN_IDX)

    def idx_for_batch(self, filenames: Iterable[str]) -> torch.Tensor:
        """Vectorized :meth:`idx_for`. Returns a ``(N,)`` int64 tensor."""
        return torch.tensor([self.idx_for(f) for f in filenames], dtype=torch.long)

    def embedding_for(self, filename: str, *, normalize: bool = True) -> torch.Tensor:
        """Return the scenario embedding for a filename, or a zero vector if unknown.

        Args:
            filename: training clip filename (extension optional).
            normalize: if True, return an L2-normalized copy (cosine space).

        Returns:
            (D,) float32 tensor.
        """
        idx = self.idx_for(filename)
        if idx == self._UNKNOWN_IDX:
            return torch.zeros(self._embedding_dim, dtype=torch.float32)
        vec = self._embedding_matrix[idx]
        if normalize:
            return vec / vec.norm().clamp_min(1e-12)
        return vec

    def positive_mask_for_batch(
        self,
        scenario_idx: torch.Tensor,
        emotion_labels: torch.Tensor,
        policy: str = "same_scenario_and_emotion",
    ) -> torch.Tensor:
        """Thin re-export of :func:`build_positive_mask` so callers can plumb
        the trainer without importing two modules.

        See :mod:`diema.training.losses_motion_text.build_positive_mask`.
        """
        from diema.training.losses_motion_text import build_positive_mask

        return build_positive_mask(
            scenario_idx=scenario_idx,
            emotion_labels=emotion_labels,
            policy=policy,
        )
