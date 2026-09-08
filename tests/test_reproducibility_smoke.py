"""Smoke tests for reproducibility.

Verifies that seed-based operations produce identical results.

See: diema_challenge_implementation_spec.md §6.5 — seed固定
"""

import pytest

from diema.data.splits import generate_lpo_splits
from utils.seed import seed_everything


class TestReproducibility:
    """Ensure deterministic behavior across runs."""

    def test_splits_are_deterministic(self, sample_filenames):
        """Same filenames + same num_folds = same splits."""
        splits_a = generate_lpo_splits(sample_filenames, num_folds=3)
        splits_b = generate_lpo_splits(sample_filenames, num_folds=3)

        for a, b in zip(splits_a, splits_b):
            assert a["train"] == b["train"]
            assert a["val"] == b["val"]

    def test_seed_everything_deterministic(self):
        """seed_everything should make numpy/random/torch deterministic."""
        import numpy as np
        import torch
        import random

        seed_everything(42)
        a1 = np.random.rand(5)
        b1 = random.random()
        c1 = torch.randn(3)

        seed_everything(42)
        a2 = np.random.rand(5)
        b2 = random.random()
        c2 = torch.randn(3)

        assert np.allclose(a1, a2)
        assert b1 == b2
        assert torch.allclose(c1, c2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different outputs."""
        import numpy as np

        seed_everything(42)
        a1 = np.random.rand(10)

        seed_everything(123)
        a2 = np.random.rand(10)

        assert not np.allclose(a1, a2)
