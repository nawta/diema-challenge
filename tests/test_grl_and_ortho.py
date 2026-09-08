"""Tests for primitives: gradient_reversal and orthogonality_loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from diema.training.grl import gradient_reversal
from diema.training.losses import orthogonality_loss


class TestGradientReversal:
    def test_forward_is_identity(self):
        x = torch.randn(4, 5, requires_grad=True)
        y = gradient_reversal(x, lambda_=0.7)
        assert torch.equal(y, x)

    def test_backward_flips_and_scales(self):
        x = torch.ones(3, requires_grad=True)
        y = gradient_reversal(x, lambda_=2.5)
        # downstream "loss" = y.sum() so grad d_loss/d_y = ones
        y.sum().backward()
        # GRL should produce d_loss/d_x = -lambda_ * 1 = -2.5
        expected = -2.5 * torch.ones(3)
        assert torch.allclose(x.grad, expected)

    def test_lambda_zero_zeros_grad(self):
        x = torch.ones(3, requires_grad=True)
        y = gradient_reversal(x, lambda_=0.0)
        y.sum().backward()
        # -0 * grad_output = zeros
        assert torch.allclose(x.grad, torch.zeros(3))

    def test_lambda_negative_keeps_sign(self):
        # Negative lambda is unusual but allowed: grad becomes +|lambda| * grad
        x = torch.ones(3, requires_grad=True)
        y = gradient_reversal(x, lambda_=-1.5)
        y.sum().backward()
        expected = 1.5 * torch.ones(3)
        assert torch.allclose(x.grad, expected)

    def test_grl_in_full_module(self):
        """End-to-end: GRL chained through a linear layer should send a
        sign-flipped signal back to the upstream params."""
        backbone = nn.Linear(8, 4)
        adv_head = nn.Linear(4, 3)

        x = torch.randn(6, 8)
        y_targets = torch.tensor([0, 1, 2, 0, 1, 2])

        feat = backbone(x)
        feat_grl = gradient_reversal(feat, lambda_=0.1)
        logits = adv_head(feat_grl)
        loss = nn.functional.cross_entropy(logits, y_targets)
        loss.backward()

        # Adv head should have a "normal" gradient (not flipped).
        assert adv_head.weight.grad is not None
        assert adv_head.weight.grad.abs().sum() > 0
        # Backbone should have grads — the sign is the flipped version of
        # what plain backprop would produce. We just verify it's non-zero.
        assert backbone.weight.grad is not None
        assert backbone.weight.grad.abs().sum() > 0

    def test_accepts_non_contiguous_input(self):
        """Regression: non-contiguous tensors (from transpose / permute)
        must flow through GRL without a RuntimeError. Earlier versions of
        this file used ``return x.view_as(x)`` which crashes on strided
        views; after the fix the forward returns ``x`` directly.
        """
        x = torch.randn(4, 8, 3, requires_grad=True)
        # Transpose makes the tensor non-contiguous in memory.
        x_t = x.transpose(0, 2)
        assert not x_t.is_contiguous()

        y = gradient_reversal(x_t, lambda_=0.5)
        # Loss on the non-contiguous view — backward must propagate.
        y.sum().backward()

        assert x.grad is not None
        # d_loss/d_x is the same sign pattern as -0.5 * ones_like(x_t),
        # reshaped back to x.shape through the transpose.
        assert x.grad.abs().sum() > 0
        # Sign check: summing over all elements gives -0.5 * numel.
        assert x.grad.sum().item() == pytest.approx(-0.5 * x.numel())


class TestOrthogonalityLoss:
    """Verifies `orthogonality_loss` computes ||(zc_norm.T @ zs_norm) / N||_F^2.

    Dividing by N *before* squaring (rather than dividing the sum of squares
    by N) makes the loss genuinely batch-size invariant — see the docstring
    of :func:`diema.training.losses.orthogonality_loss`.
    """

    def test_feature_wise_uncorrelated_is_zero(self):
        """When the cross-correlation matrix across the batch is zero,
        the loss is exactly zero. This is the "feature-wise uncorrelated"
        case: across the batch, every column of zc is orthogonal to every
        column of zs after row-normalization.
        """
        # Construct zc/zs where each row is a unit vector and whose
        # column-wise cross-correlation matrix vanishes across the batch.
        # Rows of zc always point to +e0 (unit vector), rows of zs
        # alternate between +e1 and -e1 (also unit vectors). Then
        # cross[0,1] = mean(+1, -1, +1, -1) = 0, and all other entries
        # are trivially zero.
        zc = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        zs = torch.tensor([
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        loss = orthogonality_loss(zc, zs)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_identical_pair_nonzero_and_bounded(self):
        """When every row of zc == zs is the same unit vector (e.g., all
        rows equal ``[1, 0]``), the sample-averaged cross-correlation is
        the outer product ``e0 ⊗ e0.T = [[1, 0], [0, 0]]`` and the squared
        Frobenius norm is 1.0. This is the "feature-wise fully correlated"
        case and represents the maximum for unit-norm rows in this 2D
        layout.
        """
        z = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        loss_same = orthogonality_loss(z, z)
        assert loss_same.item() == pytest.approx(1.0, abs=1e-6)

    def test_identical_outscores_uncorrelated(self):
        """Sanity: the "identical rows" construction should produce a
        strictly larger penalty than a feature-wise-uncorrelated pair at
        the same batch size.
        """
        zc = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        zs_uncorr = torch.tensor([
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        loss_same = orthogonality_loss(zc, zc)
        loss_uncorr = orthogonality_loss(zc, zs_uncorr)
        assert loss_same.item() > loss_uncorr.item()
        assert loss_uncorr.item() == pytest.approx(0.0, abs=1e-7)

    def test_scale_invariant(self):
        torch.manual_seed(0)
        zc = torch.randn(8, 6)
        zs = torch.randn(8, 6)
        loss_ref = orthogonality_loss(zc, zs)
        loss_scaled = orthogonality_loss(zc * 100.0, zs * 0.001)
        assert torch.allclose(loss_ref, loss_scaled, atol=1e-5)

    def test_batch_size_invariant(self):
        """Concatenating identical copies of a batch must leave the loss
        unchanged. This is the key property the /N-before-squaring fix
        gives us over the old (sum of squares / N) form.
        """
        torch.manual_seed(42)
        zc = torch.randn(6, 5)
        zs = torch.randn(6, 5)
        loss_small = orthogonality_loss(zc, zs)
        # Duplicate the batch — distribution unchanged, N doubled.
        zc2 = torch.cat([zc, zc], dim=0)
        zs2 = torch.cat([zs, zs], dim=0)
        loss_big = orthogonality_loss(zc2, zs2)
        assert torch.allclose(loss_small, loss_big, atol=1e-6)

    def test_nonneg(self):
        torch.manual_seed(42)
        for _ in range(10):
            zc = torch.randn(5, 7)
            zs = torch.randn(5, 4)
            assert orthogonality_loss(zc, zs).item() >= 0.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            orthogonality_loss(torch.randn(3, 4, 5), torch.randn(3, 4, 5))
        with pytest.raises(ValueError):
            orthogonality_loss(torch.randn(3, 4), torch.randn(5, 4))

    def test_grad_flows(self):
        zc = torch.randn(4, 6, requires_grad=True)
        zs = torch.randn(4, 6, requires_grad=True)
        loss = orthogonality_loss(zc, zs)
        loss.backward()
        assert zc.grad is not None
        assert zs.grad is not None
        assert zc.grad.abs().sum() > 0
        assert zs.grad.abs().sum() > 0
