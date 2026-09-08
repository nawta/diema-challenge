"""Tests for style_summary feature extractor."""

import torch
import pytest

from diema.features.style_summary import compute_style_summary, STYLE_DIM


class TestStyleSummary:
    def test_output_shape(self):
        x = torch.randn(4, 6, 64, 25)
        s = compute_style_summary(x)
        assert s.shape == (4, STYLE_DIM)

    def test_no_gradient_flow(self):
        x = torch.randn(2, 6, 64, 25, requires_grad=True)
        s = compute_style_summary(x)
        assert not s.requires_grad

    def test_deterministic(self):
        x = torch.randn(3, 6, 32, 25)
        s1 = compute_style_summary(x)
        s2 = compute_style_summary(x)
        assert torch.equal(s1, s2)

    def test_batch_size_one(self):
        x = torch.randn(1, 6, 64, 25)
        s = compute_style_summary(x)
        assert s.shape == (1, STYLE_DIM)
        assert torch.isfinite(s).all()

    def test_short_sequence_returns_zeros(self):
        """T=1 means no frame differences; should return zeros."""
        x = torch.randn(2, 6, 1, 25)
        s = compute_style_summary(x)
        assert s.shape == (2, STYLE_DIM)
        assert (s == 0).all()

    def test_style_dim_is_21(self):
        assert STYLE_DIM == 21

    def test_different_inputs_give_different_summaries(self):
        torch.manual_seed(0)
        x1 = torch.randn(2, 6, 64, 25)
        x2 = torch.randn(2, 6, 64, 25) * 5  # different scale
        s1 = compute_style_summary(x1)
        s2 = compute_style_summary(x2)
        assert not torch.allclose(s1, s2, atol=1e-3)
