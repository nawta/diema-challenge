"""Loss functions for emotion classification.

Provides CrossEntropy, Focal Loss, Class-Balanced CE, ArcFace, and
ArcFace+CE joint loss for skeleton-based emotion classification.

See: diema_challenge_implementation_spec.md §6.5,,
Related: diema/training/train.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reduces the contribution of easy-to-classify examples, focusing
    training on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default: 2.0)
        alpha: per-class weights or None (default: None)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss using effective number of samples.

    CB(p, y) = (1 - beta^n_y) / (1 - beta) * CE(p, y)

    where n_y is the number of samples for class y and beta is close to 1.

    Args:
        samples_per_class: list of sample counts per class
        beta: effective number hyperparameter (default: 0.9999)
        loss_type: 'ce' or 'focal'
        gamma: focal loss gamma (only used if loss_type='focal')
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        loss_type: str = "ce",
        gamma: float = 2.0,
    ):
        super().__init__()
        effective_num = [1.0 - beta**n for n in samples_per_class]
        weights = [1.0 / en if en > 0 else 0.0 for en in effective_num]
        total = sum(weights)
        weights = [w / total * len(weights) for w in weights]

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            return FocalLoss(gamma=self.gamma, alpha=self.weights)(logits, targets)
        return F.cross_entropy(logits, targets, weight=self.weights)


class ArcFaceLoss(nn.Module):
    """ArcFace loss with additive angular margin.

    **Requires a cosine classifier head.** The input must be
    ``cos(theta) = normalize(features) @ normalize(weight).T`` in ``[-1, 1]``,
    as produced by :class:`diema.models.heads.ArcMarginHead`. Feeding plain
    ``nn.Linear`` logits to this loss is **silently wrong** — the math that
    computes ``sin(theta) = sqrt(1 - cos^2)`` will clamp anything outside
    ``[-1, 1]`` and produce a meaningless gradient. To enable ArcFace on
    STGCN++ / CTR-GCN / SkateFormer, pass ``use_arcface_head=True`` to the
    model constructor (see also :class:`ArcFaceCELoss`).

    Args:
        s: scale factor (default 32 — Bilzard 14th place ASL Signs setting)
        m: additive angular margin (default 0.2)
        easy_margin: if True, skip the hard cut-off used by Deng et al. (2019)
        label_smoothing: applied to the cross-entropy on top of the
            margined cosine logits
    """

    def __init__(
        self,
        s: float = 32.0,
        m: float = 0.2,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.label_smoothing = label_smoothing
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            cosine: (N, C) cosine similarities (already in [-1, 1])
            targets: (N,) class indices
        """
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


# Re-export ArcMarginHead from the canonical location in models/heads.py so
# that existing callers of ``from diema.training.losses import ArcMarginHead``
# keep working. New code should import directly from ``diema.models.heads``.
from diema.models.heads import ArcMarginHead  # noqa: E402


class ArcFaceCELoss(nn.Module):
    """Joint ArcFace + CrossEntropy loss on cosine logits (Bilzard 14th).

    Computes ``alpha * arcface(cos, y) + (1 - alpha) * ce(cos * s, y)``. Both
    branches consume the SAME tensor of cosine similarities produced by
    :class:`diema.models.heads.ArcMarginHead`:

      - The ArcFace branch applies the additive angular margin (which
        assumes its inputs live in [-1, 1]) and then scales by ``s`` before
        the softmax.
      - The CE branch now also multiplies the cosine similarities by ``s``
        before computing cross-entropy, so the two branches are computed on
        comparable scales and the CE signal is non-degenerate (without the
        scale, cosine logits sit in [-1, 1] and CE would be almost flat).

    The model **must** use ``ArcMarginHead`` (or another head that produces
    unit-normalised cosine logits); feeding plain ``nn.Linear`` logits to
    this loss is undefined behaviour. See :func:`get_loss_fn` for the
    validation hint.

    Bilzard's writeup states that ArcFace alone is unstable and **must** be
    combined with CE — alpha=0.5 is the published setting and label
    smoothing is reported to give no extra gain on top of ArcFace, so the
    default is 0.0.

    Args:
        s: ArcFace scale (shared between the ArcFace and CE branches)
        m: ArcFace additive angular margin
        alpha: weight on the ArcFace component (default 0.5)
        label_smoothing: applied to the CE component only
    """

    def __init__(
        self,
        s: float = 32.0,
        m: float = 0.2,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.s = s
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.arcface = ArcFaceLoss(s=s, m=m, label_smoothing=0.0)

    def forward(self, cosine_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Both branches use the same cosine logits but the CE branch scales
        # them so the softmax sees comparable magnitudes.
        ce = F.cross_entropy(
            cosine_logits * self.s, targets, label_smoothing=self.label_smoothing,
        )
        af = self.arcface(cosine_logits, targets)
        return self.alpha * af + (1.0 - self.alpha) * ce


#: Loss types that require the model to produce cosine logits via
#: :class:`diema.models.heads.ArcMarginHead`. If you select one of these
#: losses you must also pass ``use_arcface_head=True`` to the model config
#: (or construct the model with ``use_arcface_head=True``). Plain
#: ``nn.Linear`` logits fed to these losses give silently wrong gradients.
COSINE_LOGIT_LOSS_TYPES = frozenset({"arcface", "arcface_ce"})


def info_nce_loss(
    z: torch.Tensor,
    t: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss between two aligned embedding sets.

    Used by train-time text supervision: z is the
    motion-side projection, ``t`` is the frozen sentence-transformer
    embedding of the scenario description, and we want to pull aligned
    pairs together while pushing unaligned pairs apart.

    Both sides are L2-normalized along the feature axis before computing
    cosine similarities, and the loss is divided by ``temperature`` to
    sharpen the softmax. With ``symmetric=True`` the returned value is
    ``0.5 * (CE(row softmax, eye) + CE(col softmax, eye))`` — the
    "symmetric InfoNCE" form from CLIP — which does not preferentially
    optimize one side.

    Args:
        z: (N, D) first embedding batch (e.g., motion features).
        t: (N, D) second embedding batch (e.g., text features).
            Both sides **must** share the same feature dim ``D`` —
            project one side through a ``Linear(D_other, D)`` layer
            before calling this if they differ. The enforcement raises
            a ``ValueError`` so mismatched shapes fail loudly.
        temperature: softmax temperature. Default 0.1. CLIP uses 0.07
            (learnable), sentence-transformers often uses 0.05. This
            is a sensitive hyperparameter — treat the default as a
            reasonable starting point, not a universal optimum.
        symmetric: if True, average the row-wise and column-wise losses.
        eps: numerical floor for L2 normalization.

    Returns:
        Scalar non-negative loss tensor.
    """
    if z.ndim != 2 or t.ndim != 2:
        raise ValueError(
            f"info_nce_loss expects (N, D) tensors, got "
            f"{tuple(z.shape)} and {tuple(t.shape)}"
        )
    if z.size(0) != t.size(0):
        raise ValueError(
            f"batch sizes mismatch: {z.size(0)} vs {t.size(0)}"
        )
    if z.size(1) != t.size(1):
        raise ValueError(
            "info_nce_loss requires matching feature dims on both sides. "
            f"Got z.shape[1]={z.size(1)}, t.shape[1]={t.size(1)} — project "
            "one side with a Linear layer before calling this."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    N = z.size(0)
    z_norm = z / z.norm(dim=1, keepdim=True).clamp_min(eps)
    t_norm = t / t.norm(dim=1, keepdim=True).clamp_min(eps)
    logits = (z_norm @ t_norm.transpose(0, 1)) / temperature  # (N, N)
    targets = torch.arange(N, device=z.device, dtype=torch.long)
    loss_z = F.cross_entropy(logits, targets)
    if not symmetric:
        return loss_z
    loss_t = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_z + loss_t)


def orthogonality_loss(
    z_content: torch.Tensor,
    z_style: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Frobenius-norm penalty on the (batch-averaged) cross-correlation
    matrix of two feature blocks.

    Encourages ``z_content`` and ``z_style`` to be linearly decorrelated
    in the Barlow-Twins / factor-analysis sense: we first L2-row-normalise
    both embeddings (making the loss scale-invariant), then compute the
    feature-wise cross-correlation matrix **averaged over the batch**::

        cross[k, l] = (1/N) * sum_i zc_norm[i, k] * zs_norm[i, l]

    and return its squared Frobenius norm ``||cross||_F^2``. Dividing
    **before** squaring (rather than dividing the sum of squares by ``N``)
    makes the loss genuinely batch-size invariant: doubling ``N`` while
    keeping the per-sample distribution fixed leaves the loss unchanged.

    Used by content/style dual-head models alongside a weak GRL
    branch on ``z_content`` only — the orthogonality term keeps the two
    embedding blocks from collapsing onto each other at the feature level.

    Note: this is a **feature-wise** decorrelation (columns of ``zc`` are
    decorrelated from columns of ``zs`` across the batch). It does *not*
    force each sample's ``z_content[i]`` to be perpendicular to its own
    ``z_style[i]``; if you want per-sample orthogonality instead, compute
    ``((normalize(zc) * normalize(zs)).sum(dim=1)).pow(2).mean()``.

    Args:
        z_content: (N, D) content embedding
        z_style:   (N, D') style embedding
        eps: numerical floor for L2 normalisation

    Returns:
        Scalar non-negative loss tensor.
    """
    if z_content.ndim != 2 or z_style.ndim != 2:
        raise ValueError(
            f"orthogonality_loss expects (N, D) tensors, got "
            f"{tuple(z_content.shape)} and {tuple(z_style.shape)}"
        )
    if z_content.size(0) != z_style.size(0):
        raise ValueError(
            f"batch sizes mismatch: {z_content.size(0)} vs {z_style.size(0)}"
        )
    zc = z_content / (z_content.norm(dim=1, keepdim=True).clamp_min(eps))
    zs = z_style / (z_style.norm(dim=1, keepdim=True).clamp_min(eps))
    cross = (zc.transpose(0, 1) @ zs) / float(z_content.size(0))  # (D, D')
    return cross.pow(2).sum()


class SupConLoss(nn.Module):
    """Performer-aware Supervised Contrastive Loss.

    Extends standard SupCon (Khosla et al., NeurIPS 2020) with
    cross-performer weighting: same-emotion pairs from *different*
    performers receive higher weight (``w_strong``) to encourage
    performer-invariant content alignment, while same-performer pairs
    get lower weight (``w_weak``).

    When ``performer_ids`` is ``None``, falls back to standard SupCon
    (all positives weighted equally).

    Related: diema/training/supcon_trainer.py

    Args:
        temperature: contrastive temperature (default 0.1)
        w_strong: weight for cross-performer positives (default 2.0)
        w_weak: weight for same-performer positives (default 0.5)
        base_temperature: denominator temperature for loss scaling
            (default 0.07, following the SupCon paper)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        w_strong: float = 2.0,
        w_weak: float = 0.5,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.w_strong = w_strong
        self.w_weak = w_weak
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        performer_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the SupCon loss.

        Args:
            features: L2-normalised embeddings ``(N, D)`` on the unit
                hypersphere.
            labels: emotion class labels ``(N,)`` (long).
            performer_ids: optional performer indices ``(N,)`` (long).
                When provided, positives are re-weighted by performer
                membership; when ``None``, all positives get weight 1.

        Returns:
            Scalar loss.
        """
        if features.ndim != 2:
            raise ValueError(
                f"SupConLoss expects (N, D), got {tuple(features.shape)}"
            )
        N = features.size(0)
        if N < 2:
            return features.new_tensor(0.0)

        # Cosine similarity / temperature  (N, N)
        sim = features @ features.T / self.temperature

        # Mask: same-emotion positives (exclude self-pair diagonal)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        pos_mask = label_eq & self_mask  # (N, N)

        # Weight matrix: default 1.0 for all positives
        weight = pos_mask.float()
        if performer_ids is not None:
            perf_eq = performer_ids.unsqueeze(0) == performer_ids.unsqueeze(1)
            # Cross-performer positives get w_strong, same-performer get w_weak
            cross_perf = pos_mask & ~perf_eq
            same_perf = pos_mask & perf_eq
            weight = (
                cross_perf.float() * self.w_strong
                + same_perf.float() * self.w_weak
            )

        # Log-sum-exp trick for numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Denominator: sum over all non-self pairs (positives + negatives)
        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted mean of log-probs over positive pairs
        weight_sum = weight.sum(dim=1).clamp_min(1e-12)  # (N,)
        mean_log_prob = (weight * log_prob).sum(dim=1) / weight_sum

        # Samples with no positives contribute zero loss
        has_pos = pos_mask.any(dim=1)
        # Scaling follows Khosla et al. and the official SupContrast
        # reference: -(temperature / base_temperature).
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = (loss * has_pos.float()).sum() / has_pos.float().sum().clamp_min(1.0)
        return loss


# ---------------------------------------------------------------------------
# Center Loss (v2 — SupCon replacement for small data)
# ---------------------------------------------------------------------------


class CenterLoss(nn.Module):
    """Per-class center loss (Wen et al., ECCV 2016).

    Pulls each sample's feature toward its class center. Much more
    stable than pairwise SupCon on small datasets (8K) because it
    does not depend on in-batch positive pair quality.

    The class centers are **learnable parameters** updated with their
    own learning rate (typically ``center_lr=0.5``, much higher than
    backbone LR). This is standard practice from the original paper.

    See: v2
    Related: diema/training/supcon_trainer.py

    Args:
        num_classes: number of emotion classes.
        feat_dim: backbone feature dimensionality.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean L2 distance from features to their class centers.

        Args:
            features: ``(N, D)`` feature vectors (NOT necessarily normalized).
            labels: ``(N,)`` integer class labels.

        Returns:
            Scalar loss.
        """
        # Gather the center for each sample's class
        batch_centers = self.centers[labels]  # (N, D)
        loss = (features - batch_centers).pow(2).sum(dim=1).mean()
        return loss


# ---------------------------------------------------------------------------
# Exponential Lambda Warmup (shared infrastructure)
# ---------------------------------------------------------------------------


def exp_lambda_warmup(
    epoch: int,
    lambda_target: float,
    tau: float = 20.0,
) -> float:
    """Exponential warmup: ``lambda_target * (1 - exp(-epoch / tau))``.

    At epoch 0: ~0. At epoch tau: ~63% of target. At epoch 3*tau: ~95%.
    This prevents auxiliary losses from overwhelming the primary CE
    during critical early backbone formation.

    Args:
        epoch: current training epoch.
        lambda_target: final lambda value.
        tau: time constant (default 20 epochs).

    Returns:
        Effective lambda for this epoch.
    """
    import math
    return lambda_target * (1.0 - math.exp(-epoch / max(tau, 1e-6)))


# ---------------------------------------------------------------------------
# MoE regularization
# ---------------------------------------------------------------------------


def moe_load_balancing_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    """Switch Transformer load balancing loss: ``K * Σ_k f_k * p_k``.

    Encourages uniform expert utilization across the batch.

    Args:
        routing_weights: softmax routing probs ``(N, K)``.

    Returns:
        Scalar loss.
    """
    N, K = routing_weights.shape
    # f_k = fraction of tokens routed to expert k (hard assignment)
    assignments = routing_weights.argmax(dim=1)  # (N,)
    f = torch.zeros(K, device=routing_weights.device, dtype=routing_weights.dtype)
    counts = torch.bincount(assignments, minlength=K).float()
    f = counts / N
    # p_k = mean routing probability for expert k
    p = routing_weights.mean(dim=0)  # (K,)
    return K * (f * p).sum()


def moe_entropy_regularization(routing_weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy of routing distribution (to maximize).

    High entropy = uniform routing = good.

    Args:
        routing_weights: softmax routing probs ``(N, K)``.

    Returns:
        Scalar: mean per-sample entropy (positive). Negate to maximize.
    """
    eps = 1e-12
    entropy = -(routing_weights * (routing_weights + eps).log()).sum(dim=1)
    return entropy.mean()


def moe_expert_diversity_loss(expert_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Penalize cosine similarity between expert outputs.

    Encourages experts to produce distinct logit distributions.

    Args:
        expert_outputs: list of ``K`` tensors ``(N, C)``.

    Returns:
        Scalar: mean pairwise cosine similarity (lower is better).
    """
    K = len(expert_outputs)
    if K < 2:
        return expert_outputs[0].new_tensor(0.0)
    # Stack and L2-normalize: (K, N, C)
    stacked = torch.stack(expert_outputs, dim=0)
    normed = F.normalize(stacked, dim=2)
    # Use |cos| to penalize both correlation and anti-correlation,
    # pushing experts toward orthogonal (decorrelated) outputs.
    total = normed.new_tensor(0.0)
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            cos_sim = (normed[i] * normed[j]).sum(dim=1).abs().mean()
            total = total + cos_sim
            count += 1
    return total / count


def get_loss_fn(loss_type: str = "ce", **kwargs) -> nn.Module:
    """Factory function for loss functions.

    Args:
        loss_type: 'ce', 'focal', 'class_balanced', 'arcface', or 'arcface_ce'
        **kwargs: additional arguments for the loss

    Returns:
        Loss module instance.

    Note:
        Callers selecting ``arcface`` / ``arcface_ce`` are responsible for
        ensuring the model produces cosine logits (see
        :data:`COSINE_LOGIT_LOSS_TYPES`). LightningModel enforces this at
        construction time when both the loss type and a ``model`` reference
        are available.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "class_balanced":
        return ClassBalancedLoss(**kwargs)
    elif loss_type == "arcface":
        return ArcFaceLoss(**kwargs)
    elif loss_type == "arcface_ce":
        return ArcFaceCELoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            "Available: ce, focal, class_balanced, arcface, arcface_ce"
        )
