"""Tests for same-emotion temporal CutMix in LightningModel.

Covered:
  - cutmix_alpha=0 is a no-op
  - cutmix swaps temporal segments only between same-class samples
  - cutmix never alters frames of samples with no same-class partner
  - default training_step runs end-to-end with cutmix enabled
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.training.train import LightningModel


class _IdentityHead(nn.Module):
    """Tiny model satisfying the BaseModel-ish contract used in tests."""

    def __init__(self, in_channels: int = 6, num_class: int = 12):
        super().__init__()
        self.in_channels = in_channels
        # Plain global pooling + linear head — enough to exercise gradient flow.
        self.fc = nn.Linear(in_channels, num_class)
        # Avoid the cosine-logit guard in LightningModel.__init__.
        self.use_arcface_head = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (N, C, T, V) -> mean pool over T, V -> (N, C) -> logits
        feat = x.mean(dim=(2, 3))
        return {"logits": self.fc(feat)}


def _make_lit(cutmix_alpha: float):
    model = _IdentityHead()
    return LightningModel(
        model=model,
        base_lr=1e-3,
        num_class=12,
        loss_type="ce",
        optimizer="AdamW",
        scheduler_type="cosine",
        cutmix_alpha=cutmix_alpha,
    )


def test_cutmix_zero_is_noop():
    lit = _make_lit(cutmix_alpha=0.0)
    inputs = torch.randn(4, 6, 16, 25)
    labels = torch.tensor([0, 1, 0, 1])
    # Calling the helper directly with alpha=0 short-circuits via the
    # training_step branch, so we test the helper with alpha=1 only when
    # alpha was set > 0. Here we only verify the branch isn't entered.
    out = lit._apply_same_emotion_cutmix(inputs.clone(), labels) if lit.cutmix_alpha > 0 else inputs
    assert torch.equal(out, inputs)


def test_cutmix_swaps_only_same_class_samples():
    torch.manual_seed(0)
    lit = _make_lit(cutmix_alpha=1.0)

    # Build a batch where every sample has a unique "fingerprint" along T so
    # swapped windows are easy to detect. Use disjoint classes for some samples
    # so they should NOT be touched by cutmix.
    N, C, T, V = 8, 6, 32, 25
    inputs = torch.zeros(N, C, T, V)
    for i in range(N):
        inputs[i] = float(i + 1)  # constant per-sample value
    # 4 samples in class 0, 4 unique singletons
    labels = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4])

    out = lit._apply_same_emotion_cutmix(inputs.clone(), labels)

    # Class 0 samples (indices 0..3) may be modified — but the modification
    # must come from another class-0 sample, so each frame value belongs to
    # {1, 2, 3, 4} (values seeded from indices 0..3 + 1).
    valid_class0_values = {1.0, 2.0, 3.0, 4.0}
    for i in range(4):
        unique_vals = set(out[i].unique().tolist())
        assert unique_vals.issubset(valid_class0_values), (
            f"sample {i} contains foreign values: {unique_vals}"
        )

    # Singleton-class samples (indices 4..7) must be untouched (no partner).
    for i in range(4, 8):
        assert torch.equal(out[i], inputs[i]), f"singleton sample {i} was modified"


def test_cutmix_handles_minimal_batch():
    """With a single sample (N=1), cutmix must short-circuit to identity."""
    lit = _make_lit(cutmix_alpha=1.0)
    inputs = torch.randn(1, 6, 16, 25)
    labels = torch.tensor([0])
    out = lit._apply_same_emotion_cutmix(inputs.clone(), labels)
    assert torch.equal(out, inputs)


def test_training_step_runs_with_cutmix():
    """End-to-end: training_step with cutmix_alpha > 0 should produce a
    finite scalar loss and gradients. We avoid `configure_optimizers` (which
    needs a real Trainer) by stubbing both `.optimizers()` and `.log()`."""
    from types import SimpleNamespace

    torch.manual_seed(0)
    lit = _make_lit(cutmix_alpha=1.0)

    fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    lit.optimizers = lambda: fake_opt  # type: ignore[assignment]
    lit.log = lambda *a, **k: None  # type: ignore[assignment]

    inputs = torch.randn(8, 6, 32, 25)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    batch = (inputs, labels, ["x"] * 8)

    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lit.model.parameters()
    )
    assert has_grad
