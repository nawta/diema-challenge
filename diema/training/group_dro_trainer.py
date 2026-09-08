"""GroupDRO (Distributionally Robust Optimization) training module.

Implements the online worst-group re-weighting from:
    Sagawa et al., "Distributionally Robust Neural Networks", ICLR 2020.
    https://arxiv.org/abs/1911.08731

Plan reference: ~/.claude/plans/replicated-gathering-raven.md (Track B)
TODO reference: Phase ACC Track B — B0: GroupDRO trainer

Design rationale
----------------
Prior auxiliary-loss approaches (SupCon, MoE, DANN) all added representation-
level objectives that competed with the primary CE loss and caused backbone
collapse. GroupDRO adds NO new parameters — it only reweights the existing CE
loss toward the worst-performing performer group per mini-batch. This is the
key property that makes it safe: the loss surface remains CE everywhere.

The performer-group assignment is derived from the filename already present in
every train batch (``(inputs, labels, filenames)`` — see
:func:`diema.data.collate.motion_collate_fn`). No dataset or collator changes
are required.

See also
--------
- diema/training/train.py:LightningModel            — base class
- diema/training/center_loss_trainer.py             — subclass template
- diema/training/dual_head_trainer.py               — performer-id parser import
- diema/data/collate.py                             — batch structure reference
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.training.train import LightningModel
# Import the performer-pair parser from dual_head_trainer — IMPORT, don't copy.
from diema.training.dual_head_trainer import parse_performer_pair_from_filename


# ---------------------------------------------------------------------------
# Pure functions (module-level, no side effects, fully testable without PL)
# ---------------------------------------------------------------------------


def country_group_id(performer_id: str) -> int:
    """Map a DIEM-A performer ID or filename to a country-level group index.

    DIEM-A has two recording countries: Japan (JP, index 0) and Taiwan
    (TW, index 1). The mapping is coarse-grained by design — using 2 groups
    rather than ~18 performers keeps each mini-batch group non-empty even at
    batch_size=128.

    Args:
        performer_id: a string whose leading two characters encode country,
            e.g. ``"JP_06"`` (from parse_performer_pair_from_filename),
            ``"JP_06_anger_01.bvh"``, or any filename that starts with
            ``"JP"`` / ``"TW"``.

    Returns:
        0 for Japan, 1 for Taiwan.

    Raises:
        ValueError: if the two-letter prefix is neither ``"JP"`` nor ``"TW"``.
    """
    stem = str(performer_id)
    # Accept filenames and stems alike — just look at the first two characters.
    prefix = stem[:2].upper()
    if prefix == "JP":
        return 0
    if prefix == "TW":
        return 1
    raise ValueError(
        f"Unknown country prefix '{prefix}' in performer_id={performer_id!r}. "
        "Expected 'JP' (group 0) or 'TW' (group 1)."
    )


def update_group_weights(
    q: torch.Tensor,
    group_losses_present: dict[int, float | torch.Tensor],
    present_group_ids: list[int],
    eta: float,
) -> torch.Tensor:
    """Online multiplicative GroupDRO weight update (Sagawa et al. 2020, Eq. 7).

    Updates group weights for groups present in the current mini-batch via::

        q_g ← q_g * exp(eta * L_g)          for g in present_group_ids
        q   ← q / sum(q)                     renormalise onto the simplex

    Groups absent from the batch keep their unnormalised weight so that the
    simplex projection implicitly down-weights them. After renormalisation the
    full ``q`` vector sums to 1 and is non-negative (since all entries are
    strictly positive before projection and the update is multiplicative).

    Args:
        q: current unnormalised group weights ``(num_groups,)`` — will NOT be
            modified in place; a new tensor is returned.
        group_losses_present: mapping from ``group_id`` (int) to the MEAN CE
            loss for that group in this mini-batch (scalar float or 0-d tensor).
        present_group_ids: list of group ids whose losses are in the mapping.
        eta: step-size / learning rate for the multiplicative update (η > 0).
            Sagawa et al. use η = 0.01 as the default.

    Returns:
        New weight vector ``(num_groups,)`` on the probability simplex
        (non-negative, sums to 1), same dtype/device as ``q``.
    """
    q_new = q.clone()
    for g in present_group_ids:
        L_g = group_losses_present[g]
        if isinstance(L_g, torch.Tensor):
            L_g = L_g.detach().item()
        # Multiplicative exponentiated-gradient update: q_g <- q_g * exp(eta*L_g).
        # Clamp the exponent for numerical safety — the online update can
        # overflow if eta*L_g is large. CE is usually modest so this is a guard,
        # not a regular path. (Codex review 2026-06-12.)
        exp_arg = min(eta * L_g, 50.0)
        q_new[g] = q[g] * torch.exp(
            torch.tensor(exp_arg, dtype=q.dtype, device=q.device)
        )
    # Renormalise onto the simplex.
    q_sum = q_new.sum().clamp_min(1e-12)
    return q_new / q_sum


def group_dro_loss(
    per_sample_ce: torch.Tensor,
    group_ids: list[int],
    q: torch.Tensor,
    num_groups: int,
    eta: float,
    update: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the GroupDRO mini-batch loss and (optionally) update group weights.

    Implements the online GroupDRO objective from Sagawa et al. (2020):

        min_{theta}  max_{q in Delta}  sum_g  q_g * (1/n_g) sum_{i in g} CE(f(x_i), y_i)

    In the online setting, instead of solving the inner max exactly, we do one
    gradient step on the dual (the q update)::

        L_g     = mean_{i in g} per_sample_ce[i]     for groups present in batch
        q_g     ← q_g * exp(eta * L_g)                multiplicative ascent
        q       ← q / sum(q)                          project back to simplex
        loss    = sum_{g in present} q_g_new * L_g    weighted objective

    Absent groups contribute zero to both numerator and denominator of the
    update but retain their (unnormalized) weight for future batches.

    Args:
        per_sample_ce: ``(N,)`` float tensor of per-sample cross-entropy losses,
            computed with ``reduction='none'``. Must have ``requires_grad=True``
            (or be part of a computation graph) so that ``.backward()`` works on
            the returned scalar.
        group_ids: ``(N,)``-length list of integer group assignments
            ``g_i ∈ {0, ..., num_groups-1}``. Samples with group -1 or any out-
            of-range index are silently excluded.
        q: current group weight vector ``(num_groups,)`` — NOT updated in place;
            if ``update=True`` a new tensor is returned.
        num_groups: total number of groups ``G``.
        eta: DRO learning rate for the q update.
        update: if True (default), also return the updated q; if False, return
            the current q unchanged (useful for validation / debugging).

    Returns:
        Tuple ``(loss, q_new)`` where:
        - ``loss``: differentiable scalar, the GroupDRO weighted objective.
        - ``q_new``: updated (and renormalised) weight vector of shape
          ``(num_groups,)``. If ``update=False``, this is identical to the
          input ``q``.
    """
    N = per_sample_ce.shape[0]
    device = per_sample_ce.device
    dtype = per_sample_ce.dtype

    # Compute per-group mean CE for groups present in this batch.
    group_loss_sums: dict[int, torch.Tensor] = {}
    group_counts: dict[int, int] = {}
    for i, g in enumerate(group_ids):
        if g < 0 or g >= num_groups:
            continue  # skip unknown groups
        if g not in group_loss_sums:
            group_loss_sums[g] = per_sample_ce[i]
            group_counts[g] = 1
        else:
            group_loss_sums[g] = group_loss_sums[g] + per_sample_ce[i]
            group_counts[g] += 1

    present_groups = sorted(group_loss_sums.keys())

    if not present_groups:
        # Degenerate batch (all unknown groups) — fall back to plain mean CE.
        return per_sample_ce.mean(), q.clone()

    group_mean_losses: dict[int, torch.Tensor] = {
        g: group_loss_sums[g] / group_counts[g] for g in present_groups
    }

    # Update q using the detached group losses (dual variable, not the primal).
    group_losses_for_update: dict[int, float | torch.Tensor] = {
        g: v.detach() for g, v in group_mean_losses.items()
    }

    if update:
        q_new = update_group_weights(q, group_losses_for_update, present_groups, eta)
    else:
        q_new = q.clone()

    # Primal GroupDRO objective (Sagawa et al. 2020) using the post-update dual
    # weights q_new (detached below). We sum over groups PRESENT in this batch
    # and renormalise their q to sum to 1 within the batch.
    # NOTE (Codex review 2026-06-12): present-group renormalisation is a
    # deliberate, documented deviation from strict Sagawa (which keeps the global
    # q and lets absent groups simply drop their mass). For the COUNTRY grouping
    # (2 groups) both JP and TW appear in nearly every batch_size=128 batch, so
    # sum_present(q) ≈ 1 and the two formulations are numerically near-identical
    # here. Revisit if escalating to per-performer grouping (many groups absent
    # per batch, where the renormalisation would change the gradient scale).
    present_q_sum = sum(q_new[g].item() for g in present_groups)
    if present_q_sum < 1e-12:
        # Pathological — fall back to mean CE.
        return per_sample_ce.mean(), q_new

    loss = per_sample_ce.new_tensor(0.0)
    for g in present_groups:
        # Normalise q over present groups so they sum to 1 within the batch.
        # This is needed to keep the loss scalar O(1) regardless of which
        # subset of groups appears in the batch.
        weight_g = q_new[g].item() / present_q_sum
        loss = loss + weight_g * group_mean_losses[g]

    return loss, q_new


# ---------------------------------------------------------------------------
# Lightning subclass
# ---------------------------------------------------------------------------


class GroupDROLightningModel(LightningModel):
    """Lightning subclass for GroupDRO performer-invariant training.

    Trains the backbone with a GroupDRO-reweighted cross-entropy loss that
    up-weights batches from the worst-performing performer group. This directly
    attacks the diagnosed 10.8× performer confound without adding any new
    parameters or auxiliary objectives, avoiding the backbone-collapse failure
    modes of SupCon / MoE / DANN.

    The ``q`` buffer (group weights) is a non-trainable buffer updated in-place
    during each training step. It is saved with the model checkpoint so
    inference-time grouping is not required.

    Warmup behaviour
    ----------------
    For the first ``warmup_epochs_dro`` epochs, plain mean CE is used (q
    update disabled). This lets the backbone form before the reweighting can
    distort the optimisation landscape. After warmup the online GroupDRO
    objective is used.

    CutMix note
    -----------
    CutMix mixes temporal windows between samples — if those samples belong to
    different groups, their group assignment is ambiguous. We disable CutMix
    during the DRO phase (epoch >= warmup_epochs_dro) and allow it during
    warmup. Concretely: if ``cutmix_alpha > 0`` the cutmix hook is called
    only while ``self.current_epoch < warmup_epochs_dro``. This is documented
    so users know why they might see CutMix enabled during warmup but not after.

    Mixup is unconditionally forbidden (mixing convexly between two samples
    from different groups gives a fractional group, which the integer-indexed
    group DRO cannot represent).

    See: Phase ACC Track B — B0
    Plan: ~/.claude/plans/replicated-gathering-raven.md Track B (GroupDRO)
    Related:
        diema/training/train.py::LightningModel
        diema/training/center_loss_trainer.py::CenterLossLightningModel
        diema/training/dual_head_trainer.py::parse_performer_pair_from_filename

    Args:
        model: any :class:`diema.models.base.BaseModel` instance.
        base_lr: backbone learning rate.
        num_class: number of emotion classes (for metrics).
        group_mode: grouping granularity — currently only ``"country"``
            (JP=0, TW=1) is supported.
        num_groups: number of groups ``G``; must match ``group_mode``
            (2 for ``"country"``).
        eta_dro: DRO dual learning rate ``η`` for the q update (default 0.01).
        warmup_epochs_dro: number of epochs to use plain mean CE before
            switching to the GroupDRO reweighting (default 10).
        loss_type: primary loss function name (forwarded to parent).
        loss_kwargs: kwargs for the loss function (e.g. ``label_smoothing``).
        optimizer / scheduler_type / weight_decay / warmup_epochs:
            forwarded verbatim to :class:`LightningModel`.
        cutmix_alpha: passed to parent; applied only during warmup.
        gaussian_noise_sigma: forwarded to parent.
    """

    # _HPARAM_IGNORE lists constructor args that are not serialisable by
    # save_hyperparameters. The parent class checks this list via getattr.
    _HPARAM_IGNORE = ["model"]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        group_mode: str = "country",
        num_groups: int = 2,
        eta_dro: float = 0.01,
        warmup_epochs_dro: int = 10,
        loss_type: str = "ce",
        loss_kwargs: dict | None = None,
        optimizer: str = "AdamW",
        scheduler_type: str = "cosine_warmup",
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        cutmix_alpha: float = 0.0,
        gaussian_noise_sigma: float = 0.0,
        mixup_alpha: float = 0.0,
        **extra_kwargs,
    ):
        if mixup_alpha > 0.0:
            raise ValueError(
                "GroupDROLightningModel forbids mixup_alpha > 0. Mixing samples "
                "convexly across group boundaries creates fractional group "
                "assignments that the integer-indexed GroupDRO loss cannot "
                "represent. Pass mixup_alpha=0.0."
            )
        if group_mode not in ("country",):
            raise ValueError(
                f"Unsupported group_mode={group_mode!r}. Currently only "
                "'country' (JP=0, TW=1) is supported."
            )

        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            loss_type=loss_type,
            loss_kwargs=loss_kwargs,
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            cutmix_alpha=cutmix_alpha,
            gaussian_noise_sigma=gaussian_noise_sigma,
            mixup_alpha=0.0,  # always 0 — validated above
        )

        self.group_mode = group_mode
        self.num_groups = int(num_groups)
        self.eta_dro = float(eta_dro)
        self.warmup_epochs_dro = int(warmup_epochs_dro)

        # Register q as a non-trainable buffer so it is checkpointed with the
        # model but excluded from optimizer parameter groups.
        self.register_buffer(
            "q",
            torch.ones(self.num_groups, dtype=torch.float32) / self.num_groups,
        )

    def _filenames_to_group_ids(self, filenames: list[str]) -> list[int]:
        """Map a batch of filenames to integer group ids.

        Returns -1 for any filename whose country prefix cannot be determined.
        """
        ids: list[int] = []
        for fn in filenames:
            try:
                ids.append(country_group_id(fn))
            except ValueError:
                ids.append(-1)
        return ids

    def _select_loss(
        self,
        per_sample_ce: torch.Tensor,
        group_ids: list[int],
        current_epoch: int,
    ) -> tuple[torch.Tensor, bool]:
        """Select between plain mean CE (warmup) and GroupDRO loss.

        Factored as a pure(-ish) helper so tests can call it directly without
        going through the full Lightning training infrastructure.

        Args:
            per_sample_ce: ``(N,)`` per-sample CE loss with ``requires_grad``.
            group_ids: integer group assignment per sample.
            current_epoch: ``self.current_epoch`` (passed explicitly for
                testability via monkeypatching).

        Returns:
            Tuple ``(loss, dro_active)`` where ``dro_active`` indicates whether
            the GroupDRO reweighting was applied.
        """
        if current_epoch < self.warmup_epochs_dro:
            # Warmup: plain mean CE; q is NOT updated.
            return per_sample_ce.mean(), False

        # DRO phase: compute GroupDRO objective and update q buffer.
        loss, q_new = group_dro_loss(
            per_sample_ce=per_sample_ce,
            group_ids=group_ids,
            q=self.q,
            num_groups=self.num_groups,
            eta=self.eta_dro,
            update=True,
        )
        # In-place update of the registered buffer.
        self.q.copy_(q_new)
        return loss, True

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch

        # Gaussian weight noise (stash so on_after_backward can restore).
        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            # CutMix: only during warmup to avoid group-assignment ambiguity.
            # After warmup_epochs_dro the DRO phase starts and CutMix is off.
            if self.cutmix_alpha > 0.0 and self.current_epoch < self.warmup_epochs_dro:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            out = self(inputs)

            # Per-sample CE with label_smoothing inherited from loss_fn.
            # We reconstruct per-sample reduction='none' from the loss_fn's
            # configuration. For nn.CrossEntropyLoss we can use F.cross_entropy
            # directly with the same label_smoothing extracted from loss_fn.
            # This avoids duplicating the label_smoothing config.
            label_smoothing = getattr(self.loss_fn, "label_smoothing", 0.0)
            per_sample_ce = F.cross_entropy(
                out["logits"], labels,
                reduction="none",
                label_smoothing=label_smoothing,
            )

            # Group ids from filenames.
            group_ids = self._filenames_to_group_ids(filenames)

            loss, dro_active = self._select_loss(
                per_sample_ce=per_sample_ce,
                group_ids=group_ids,
                current_epoch=self.current_epoch,
            )

            self.log("train_loss", loss, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
            self.log("dro_active", float(dro_active), prog_bar=False)

            # Log per-group q weights (cheap: just two scalars for country mode).
            if dro_active:
                for g in range(self.num_groups):
                    self.log(f"q_group_{g}", self.q[g].item(), prog_bar=False)

        except BaseException:
            self._restore_noise_weights()
            raise

        return loss

    # validation_step / test_step / predict_step / configure_optimizers are
    # inherited unchanged from LightningModel — OOFPredictionCallback and all
    # downstream tooling work as-is.
