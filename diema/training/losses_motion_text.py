"""Motion-text correspondence losses for

See: (multi-positive InfoNCE + part alignment)
Related:
- diema/training/losses.py — single-positive `info_nce_loss` (kept for back-compat)
- diema/features/text_features.py — ScenarioTextEncoder / text cache
- diema/models/skeleton_graph.py — DIEMA_BODY_PARTS (source of truth for N_BODY_PARTS)

This module adds the multi-positive contrastive formulation required by
 exp051 (same-scenario-and-emotion positives) and the part-wise
alignment loss used by exp052. The module is self-contained so the
single-positive InfoNCE in `losses.py` can continue to be used elsewhere
without an import cycle.

Key design points:
- `GlobalInfoNCELoss` takes a `positive_mask (N, N)` so the caller decides the
  pairing policy (diagonal / same_scenario / same_emotion / same_scenario_and_emotion).
  `None` falls back to diagonal (single-positive CLIP-style).
- `PartAlignmentLoss` honors a `part_mask (N, P)` — `part_mask=False` zeros both
  the forward loss and the backward gradient for that part. Fully-masked rows
  return 0 without producing NaN.
- All math sits in fp32; the caller decides whether to `.float()` / `.half()`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "global_multi_positive_info_nce",
    "part_alignment_loss",
    "GlobalInfoNCELoss",
    "PartAlignmentLoss",
]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def global_multi_positive_info_nce(
    z_motion: torch.Tensor,
    t_text: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
    symmetric: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Multi-positive supervised InfoNCE (Khosla+ 2020, SupCon style).

    Args:
        z_motion: (N, D) motion-side projection.
        t_text:   (N, D) text-side embedding (same D).
        positive_mask: (N, N) bool/float. ``positive_mask[i, j] = True`` means
            text j is a positive for motion i. Must always include the diagonal.
            If ``None``, falls back to identity (single-positive CLIP InfoNCE).
        temperature: softmax temperature (> 0).
        symmetric: if True, average the motion→text and text→motion directions
            (CLIP convention).
        eps: clamp-min for L2 normalization / log-sum-exp stability.

    Returns:
        Scalar loss tensor. Formulation: for each row i,
            L_i = - log (sum_{j in P(i)} exp(sim_ij/τ)) / (sum_k exp(sim_ik/τ))
        averaged over rows. Rows with no positives (should be rare since the
        diagonal is always a positive) contribute 0.

    Shape contract:
        z_motion / t_text must be 2-D with matching batch and feature sizes.
        positive_mask, if given, must be (N, N) with the same device as z_motion.
    """
    if z_motion.ndim != 2 or t_text.ndim != 2:
        raise ValueError(
            f"expected 2-D tensors, got z_motion={tuple(z_motion.shape)} "
            f"t_text={tuple(t_text.shape)}"
        )
    if z_motion.shape != t_text.shape:
        raise ValueError(
            f"z_motion {tuple(z_motion.shape)} and t_text {tuple(t_text.shape)} "
            "must share the same shape — project one side through Linear(D_other, D) "
            "before calling this."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    N = z_motion.size(0)
    device = z_motion.device

    if positive_mask is None:
        positive_mask = torch.eye(N, dtype=torch.bool, device=device)
    else:
        if positive_mask.shape != (N, N):
            raise ValueError(
                f"positive_mask must be (N, N) = ({N}, {N}), got "
                f"{tuple(positive_mask.shape)}"
            )
        positive_mask = positive_mask.to(device=device)

    # Always allocate a fresh float tensor so we never mutate a caller's mask.
    # `.float()` is a no-op on already-float32 tensors, so an explicit clone
    # is required.
    pos_f = positive_mask.to(dtype=torch.float32, copy=True)
    diag_idx = torch.arange(N, device=device)
    # Diagonal must always be a positive. If the caller supplied a mask that
    # does not set all diagonal entries, warn once so callers can fix their
    # code — silently patching it masks upstream bugs.
    if not bool(positive_mask.to(torch.bool)[diag_idx, diag_idx].all()):
        import warnings
        warnings.warn(
            "positive_mask is missing diagonal True entries; forcing them to 1. "
            "Callers should construct masks via build_positive_mask() or ensure "
            "their comparison includes self-pairs.",
            stacklevel=2,
        )
    pos_f[diag_idx, diag_idx] = 1.0

    zm = _l2_normalize(z_motion, eps=eps)
    tt = _l2_normalize(t_text, eps=eps)
    logits_m2t = (zm @ tt.transpose(0, 1)) / temperature  # (N, N)

    # Numerically stable log_softmax: subtract per-row max first.
    logits_m2t_shift = logits_m2t - logits_m2t.max(dim=1, keepdim=True).values
    exp_logits = logits_m2t_shift.exp()
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
    log_prob = logits_m2t_shift - torch.log(denom)

    num_pos = pos_f.sum(dim=1).clamp_min(1.0)  # avoid divide-by-zero
    mean_log_prob_pos = (pos_f * log_prob).sum(dim=1) / num_pos
    loss_m2t = -mean_log_prob_pos.mean()
    if not symmetric:
        return loss_m2t

    # text→motion: transpose both the sim matrix and the positive mask.
    logits_t2m = (tt @ zm.transpose(0, 1)) / temperature
    logits_t2m_shift = logits_t2m - logits_t2m.max(dim=1, keepdim=True).values
    exp_t = logits_t2m_shift.exp()
    denom_t = exp_t.sum(dim=1, keepdim=True).clamp_min(eps)
    log_prob_t = logits_t2m_shift - torch.log(denom_t)
    pos_t = pos_f.transpose(0, 1)
    num_pos_t = pos_t.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos_t = (pos_t * log_prob_t).sum(dim=1) / num_pos_t
    loss_t2m = -mean_log_prob_pos_t.mean()
    return 0.5 * (loss_m2t + loss_t2m)


def part_alignment_loss(
    z_parts: torch.Tensor,
    t_parts: torch.Tensor,
    part_mask: torch.Tensor,
    mode: str = "cosine",
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Part-wise alignment loss between motion-part and text-part projections.

    Args:
        z_parts: (N, P, D) motion-part projection.
        t_parts: (N, P, D) text-side part embedding (from Gemini rationale).
        part_mask: (N, P) bool/float. ``part_mask[n, p] = False`` excludes part p
            for sample n from both the loss and the backward gradient.
        mode: ``"cosine"`` (default) = `1 - cos(z, t)` per (n, p);
              ``"infonce"`` = batch-wise InfoNCE per part p (uses diagonal pairing
              across the batch).
        temperature: only used for ``mode="infonce"``.
        eps: clamp-min for normalization.

    Returns:
        Scalar loss. Averages over *valid* (n, p) cells; returns 0 if all-masked.
    """
    if z_parts.shape != t_parts.shape:
        raise ValueError(
            f"z_parts {tuple(z_parts.shape)} and t_parts {tuple(t_parts.shape)} "
            "must share the same shape"
        )
    if z_parts.ndim != 3:
        raise ValueError(f"z_parts must be 3-D (N, P, D), got {tuple(z_parts.shape)}")
    if part_mask.shape != z_parts.shape[:2]:
        raise ValueError(
            f"part_mask {tuple(part_mask.shape)} must equal z_parts.shape[:2] "
            f"= {tuple(z_parts.shape[:2])}"
        )
    mask_f = part_mask.float()
    mask_sum = mask_f.sum()
    if mask_sum.item() == 0:
        # Return a zero that still participates in the graph so the caller
        # does not special-case this branch. Using `z_parts.sum() * 0` keeps
        # the grad-graph connected if grads are accidentally enabled.
        return (z_parts.sum() * 0.0).squeeze()

    if mode == "cosine":
        zn = _l2_normalize(z_parts, eps=eps)
        tn = _l2_normalize(t_parts, eps=eps)
        per_cell = 1.0 - (zn * tn).sum(dim=-1)  # (N, P)
        weighted = per_cell * mask_f
        return weighted.sum() / mask_sum

    if mode == "infonce":
        # Per-part batch-wise InfoNCE, then average over parts.
        N, P, _D = z_parts.shape
        zn = _l2_normalize(z_parts, eps=eps)
        tn = _l2_normalize(t_parts, eps=eps)
        part_losses = []
        part_weights = []
        for p in range(P):
            valid = mask_f[:, p]  # (N,)
            n_valid = int(valid.sum().item())
            if n_valid < 2:
                # Need ≥ 2 valid rows to form a meaningful batch.
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            z_v = zn[idx, p]
            t_v = tn[idx, p]
            logits = (z_v @ t_v.transpose(0, 1)) / temperature
            targets = torch.arange(n_valid, device=z_v.device, dtype=torch.long)
            loss_p = 0.5 * (F.cross_entropy(logits, targets)
                            + F.cross_entropy(logits.transpose(0, 1), targets))
            part_losses.append(loss_p)
            part_weights.append(float(n_valid))
        if not part_losses:
            return (z_parts.sum() * 0.0).squeeze()
        weights = torch.tensor(part_weights, device=z_parts.device, dtype=torch.float32)
        stacked = torch.stack(part_losses)
        return (stacked * weights).sum() / weights.sum()

    raise ValueError(f"mode must be 'cosine' or 'infonce', got {mode!r}")


class GlobalInfoNCELoss(nn.Module):
    """nn.Module wrapper around :func:`global_multi_positive_info_nce` for configs."""

    def __init__(self, temperature: float = 0.1, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        z_motion: torch.Tensor,
        t_text: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return global_multi_positive_info_nce(
            z_motion,
            t_text,
            positive_mask=positive_mask,
            temperature=self.temperature,
            symmetric=self.symmetric,
        )


class PartAlignmentLoss(nn.Module):
    """nn.Module wrapper around :func:`part_alignment_loss`."""

    def __init__(self, mode: str = "cosine", temperature: float = 0.1):
        super().__init__()
        if mode not in ("cosine", "infonce"):
            raise ValueError(f"mode must be 'cosine' or 'infonce', got {mode!r}")
        self.mode = mode
        self.temperature = temperature

    def forward(
        self,
        z_parts: torch.Tensor,
        t_parts: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        return part_alignment_loss(
            z_parts,
            t_parts,
            part_mask=part_mask,
            mode=self.mode,
            temperature=self.temperature,
        )


def build_positive_mask(
    scenario_idx: torch.Tensor | None = None,
    emotion_labels: torch.Tensor | None = None,
    policy: str = "same_scenario_and_emotion",
    exclude_unknown: bool = True,
) -> torch.Tensor:
    """Construct a (N, N) positive mask from batch metadata.

    Args:
        scenario_idx: (N,) int tensor — unique-scenario-id per sample.
            ``-1`` means "unknown" (e.g. test-set rows without a scenario).
        emotion_labels: (N,) int tensor — emotion class label per sample.
            ``-1`` means "unknown" (e.g. test-set rows without a label).
        policy:
            - ``"same_scenario_and_emotion"``: both tags must match.
            - ``"same_scenario"``: only scenario must match.
            - ``"same_emotion"``: only emotion must match.
            - ``"diagonal"``: identity (single-positive CLIP InfoNCE).
        exclude_unknown: if True (default), rows and columns whose relevant
            metadata is ``-1`` are never treated as positives for each other,
            except along the diagonal (where we always keep the self-pair
            because the loss uses self as a trivial anchor). This guards
            against the "two unknown samples look equal therefore positive"
            bug that a naive ``a == b`` comparison introduces.

    Returns:
        Bool tensor (N, N) with diagonal forced True.

     recommends same_scenario_and_emotion to guard against
    scenarios acted under a different emotion polluting the positive set.
    """
    if policy == "diagonal":
        if scenario_idx is None and emotion_labels is None:
            raise ValueError("need at least one of scenario_idx/emotion_labels to infer N")
        ref = scenario_idx if scenario_idx is not None else emotion_labels
        return torch.eye(ref.size(0), dtype=torch.bool, device=ref.device)

    if policy == "same_scenario":
        if scenario_idx is None:
            raise ValueError("same_scenario requires scenario_idx")
        mask = scenario_idx[:, None] == scenario_idx[None, :]
        if exclude_unknown:
            valid = scenario_idx >= 0
            mask = mask & valid[:, None] & valid[None, :]
        return _force_diagonal(mask)

    if policy == "same_emotion":
        if emotion_labels is None:
            raise ValueError("same_emotion requires emotion_labels")
        mask = emotion_labels[:, None] == emotion_labels[None, :]
        if exclude_unknown:
            valid = emotion_labels >= 0
            mask = mask & valid[:, None] & valid[None, :]
        return _force_diagonal(mask)

    if policy == "same_scenario_and_emotion":
        if scenario_idx is None or emotion_labels is None:
            raise ValueError(
                "same_scenario_and_emotion requires both scenario_idx and emotion_labels"
            )
        mask = (scenario_idx[:, None] == scenario_idx[None, :]) & (
            emotion_labels[:, None] == emotion_labels[None, :]
        )
        if exclude_unknown:
            valid = (scenario_idx >= 0) & (emotion_labels >= 0)
            mask = mask & valid[:, None] & valid[None, :]
        return _force_diagonal(mask)

    raise ValueError(f"unknown positive-pairing policy {policy!r}")


def _force_diagonal(mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``mask`` with the diagonal forced True."""
    out = mask.clone()
    n = out.size(0)
    idx = torch.arange(n, device=out.device)
    out[idx, idx] = True
    return out
