""" lookup filename_stem → (6, D) part embeddings + (6) mask.

See: (exp052)
Related:
- tools/build_rationale_text_cache.py — offline cache builder
- diema/features/text_cache.py — scenario (global) analogue
- diema/training/losses_motion_text.py::PartAlignmentLoss — consumer

The cache is produced by ``tools/build_rationale_text_cache.py`` and stores a
frozen tensor ``(N, 6, D)`` of sentence-transformer vectors for the six body
parts (head, torso, left_arm, right_arm, left_leg, right_leg). Rows for
clips with no generated rationale are simply absent — callers must handle
``idx_for == -1`` the same way the scenario cache does.

This module supplies the batch-time hot path: given a list of filename stems
(typically the dataloader's batch of sample IDs), it returns:

- ``z_parts_text``  : (B, 6, D) float32 stacked embeddings (zero-filled for unknown stems)
- ``part_mask``     : (B, 6) bool — False for unknown stems OR placeholder parts
- ``valid_row_mask``: (B,)   bool — False for clips with no rationale at all

``PartAlignmentLoss`` already respects ``part_mask`` (per-cell zero-out),
so a False row triggers zero gradient for that sample automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

__all__ = ["RationaleTextCache"]


class RationaleTextCache:
    """Read-only loader for the part-wise rationale embedding cache."""

    _UNKNOWN_IDX = -1

    def __init__(self, cache_path: str | Path):
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(f"rationale cache not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for required in ("embeddings", "part_mask", "stem_to_idx", "part_names"):
            if required not in payload:
                raise ValueError(
                    f"rationale cache {path} missing key {required!r}; rebuild with "
                    "tools/build_rationale_text_cache.py"
                )
        emb = payload["embeddings"]
        if emb.ndim != 3:
            raise ValueError(
                f"embeddings tensor must be 3-D (N, P, D); got shape {tuple(emb.shape)}"
            )
        self._embeddings: torch.Tensor = emb.to(torch.float32).contiguous()
        self._part_mask: torch.Tensor = payload["part_mask"].to(torch.bool).contiguous()
        if self._part_mask.shape != self._embeddings.shape[:2]:
            raise ValueError(
                f"part_mask shape {tuple(self._part_mask.shape)} does not match "
                f"embeddings[:2] {tuple(self._embeddings.shape[:2])}"
            )
        self._stem_to_idx: dict[str, int] = dict(payload["stem_to_idx"])
        self._part_names: list[str] = list(payload["part_names"])
        self._meta: dict = dict(payload.get("meta", {}))

    # -- shape / metadata --------------------------------------------------

    @property
    def num_rows(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def num_parts(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self._embeddings.shape[2])

    @property
    def part_names(self) -> list[str]:
        return list(self._part_names)

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    # -- lookup ------------------------------------------------------------

    def idx_for(self, filename: str) -> int:
        """Return row index for a filename stem, or -1 if absent from the cache."""
        return self._stem_to_idx.get(Path(str(filename)).stem, self._UNKNOWN_IDX)

    def idx_for_batch(self, filenames: Iterable[str]) -> torch.Tensor:
        return torch.tensor([self.idx_for(f) for f in filenames], dtype=torch.long)

    def embedding_for_batch(
        self, filenames: Iterable[str], *, normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_parts_text, part_mask, valid_row_mask) for a batch.

        Args:
            filenames: iterable of filename stems (extensions optional).
            normalize: L2-normalize embeddings along the D axis before return.

        Shapes (P = num_parts, D = embedding_dim, B = len(filenames)):
          - ``z_parts_text``  : (B, P, D) float32. Zero-filled rows for unknowns.
          - ``part_mask``     : (B, P)    bool.   Per-part validity.
          - ``valid_row_mask``: (B,)      bool.   Whether the whole sample has any rationale.
        """
        idx = self.idx_for_batch(filenames)
        B = idx.numel()
        P = self.num_parts
        D = self.embedding_dim
        z = torch.zeros(B, P, D, dtype=torch.float32)
        m = torch.zeros(B, P, dtype=torch.bool)
        valid = idx >= 0
        known_positions = valid.nonzero(as_tuple=True)[0]
        known_rows = idx[valid]
        if known_rows.numel() > 0:
            z[known_positions] = self._embeddings[known_rows]
            m[known_positions] = self._part_mask[known_rows]
        if normalize:
            z = torch.nn.functional.normalize(z, p=2, dim=2)
        return z, m, valid

    # -- diagnostics -------------------------------------------------------

    def coverage_stats(self) -> dict[str, float]:
        """Fraction of (row, part) cells with True mask; useful for sanity checks."""
        n_cells = self._part_mask.numel()
        if n_cells == 0:
            return {"cells": 0, "true_fraction": 0.0, "per_part": []}
        per_part = self._part_mask.float().mean(dim=0).tolist()
        return {
            "cells": int(n_cells),
            "true_fraction": float(self._part_mask.float().mean()),
            "per_part": per_part,
        }
