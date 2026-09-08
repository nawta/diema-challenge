"""Gradient Reversal Layer (GRL) for adversarial training.

Provides a custom autograd function that acts as identity in the forward
pass and multiplies the upstream gradient by ``-lambda`` in the backward
pass. This is the canonical building block from Ganin & Lempitsky 2015
("Unsupervised Domain Adaptation by Backpropagation") used by domain-
adversarial models, and is reused here for content/style
factorization.

In the GRL is applied **only to z_content** before feeding
it into a small performer-classifier branch. The intent is to encourage
``z_content`` to *not* contain performer-identity information, while
``z_style`` is left free to capture style cues — the opposite of the
"erase all subject identity" approach that tried (and which the
n002 Phase F-3 study showed is not the right move for DIEM-A).

Usage::

    from diema.training.grl import gradient_reversal

    # Forward pass: identity. Backward pass: grad <- -lambda * grad.
    z_content_adv = gradient_reversal(z_content, lambda_=0.05)
    perf_logits = self.perf_classifier(z_content_adv)
    L_adv = F.cross_entropy(perf_logits, performer_labels)

Related: diema/training/losses.py (orthogonality_loss)
"""

from __future__ import annotations

import torch


class _GradientReversalFn(torch.autograd.Function):
    """Identity in forward, sign-flipped scaled gradient in backward.

    Forward returns ``x`` directly instead of ``x.view_as(x)`` so that
    non-contiguous inputs (e.g. produced by ``transpose`` or ``permute``)
    do not raise ``RuntimeError``. The gradient still flows back to the
    original ``x`` because autograd tracks the op regardless of the alias.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply the Gradient Reversal Layer to ``x``.

    Forward pass returns ``x`` unchanged. Backward pass multiplies the
    incoming gradient by ``-lambda_``.

    Args:
        x: input tensor (any shape, requires_grad if you want gradient flow)
        lambda_: scalar reversal strength. ``0.0`` short-circuits to plain
            identity (no gradient flip), positive values reverse-and-scale,
            negative values are unusual but allowed.

    Returns:
        A tensor that aliases ``x`` for the forward computation but whose
        gradient with respect to ``x`` is ``-lambda_ * grad_output``.
    """
    return _GradientReversalFn.apply(x, lambda_)


__all__ = ["gradient_reversal"]
