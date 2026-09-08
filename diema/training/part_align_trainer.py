""" training integration for part-wise rationale alignment.

:class:`PartAlignLightningModel` is a thin subclass of
:class:`diema.training.train.LightningModel` that adds the part-wise
:func:`diema.training.losses_motion_text.part_alignment_loss` on top of
the standard CE. No validation / test behavior changes.

Loss::

    total = CE(logits, y) + λ_part * PartAlignmentLoss(z_parts, t_parts,
                                                       part_mask)

Where ``z_parts`` comes from :class:`PartAlignConv1DTr_Model` (shape
``(N, P, D)``), ``t_parts`` is looked up via :class:`RationaleTextCache`,
and ``part_mask`` is False for any (sample, part) cell whose rationale is
a placeholder (``"no distinctive motion"``) or whose sample has no cached
rationale.

See: (exp052)
Related:
- diema/models/conv1d_transformer/partalign_conv1d_tr_model.py
- diema/features/rationale_text_cache.py
- diema/training/losses_motion_text.py::PartAlignmentLoss
- diema/training/text_align_trainer.py (global analogue)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from diema.training.losses_motion_text import part_alignment_loss
from diema.training.train import LightningModel

_SUPPORTED_MODES = frozenset({"cosine", "infonce"})


class PartAlignLightningModel(LightningModel):
    """Lightning subclass for joint CE + part-wise rationale alignment.

    The rationale embedding tensor ``part_embedding_tensor`` is ``(N_cached,
    P, D)`` and is registered as a buffer so it rides with the model's
    device. ``filename_to_part_idx`` maps ``stem → row`` in that tensor.
    Samples whose stem is absent from the map receive ``part_mask=False``
    for every part and contribute zero gradient to the alignment loss.

    Args:
        filename_to_part_idx: map filename.stem → row in ``part_embedding_tensor``.
        part_embedding_tensor: ``(N_cached, P, D)`` float32, typically
            produced by :meth:`diema.features.rationale_text_cache.RationaleTextCache`.
        part_mask_tensor: ``(N_cached, P)`` bool, False for placeholder parts.
        part_loss_weight: λ for the part alignment loss (exp052 a00 default 0.05).
        part_loss_mode: ``"cosine"`` (default) or ``"infonce"``.
        part_loss_temperature: only used when mode=="infonce" (default 0.1).
        part_loss_warmup_epochs: CE-only warmup before part loss turns on.
        lightning_kwargs: forwarded to :class:`LightningModel` parent.
    """

    _HPARAM_IGNORE = [
        "filename_to_part_idx",
        "part_embedding_tensor",
        "part_mask_tensor",
    ]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        filename_to_part_idx: dict[str, int],
        part_embedding_tensor: torch.Tensor,
        part_mask_tensor: torch.Tensor,
        part_loss_weight: float = 0.05,
        part_loss_mode: str = "cosine",
        part_loss_temperature: float = 0.1,
        part_loss_warmup_epochs: int = 0,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        if part_loss_mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"part_loss_mode must be in {sorted(_SUPPORTED_MODES)}, "
                f"got {part_loss_mode!r}"
            )
        if part_embedding_tensor.ndim != 3:
            raise ValueError(
                f"part_embedding_tensor must be 3-D (N, P, D); got "
                f"{tuple(part_embedding_tensor.shape)}"
            )
        if part_mask_tensor.shape != part_embedding_tensor.shape[:2]:
            raise ValueError(
                "part_mask_tensor shape "
                f"{tuple(part_mask_tensor.shape)} does not match "
                f"part_embedding_tensor[:2] "
                f"{tuple(part_embedding_tensor.shape[:2])}"
            )

        try:
            self.save_hyperparameters(ignore=self._HPARAM_IGNORE)
        except Exception:
            pass

        self.filename_to_part_idx = dict(filename_to_part_idx)
        self.part_loss_weight = float(part_loss_weight)
        self.part_loss_mode = part_loss_mode
        self.part_loss_temperature = float(part_loss_temperature)
        self.part_loss_warmup_epochs = int(part_loss_warmup_epochs)

        self.register_buffer(
            "part_embedding_tensor",
            part_embedding_tensor.to(torch.float32).clone(),
            persistent=True,
        )
        self.register_buffer(
            "part_mask_tensor",
            part_mask_tensor.to(torch.bool).clone(),
            persistent=True,
        )

    # -- lookup helpers ---------------------------------------------------

    def _batch_part_idx(
        self, filenames: list[str], device: torch.device,
    ) -> torch.Tensor:
        ids = [
            self.filename_to_part_idx.get(Path(str(fn)).stem, -1)
            for fn in filenames
        ]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _lookup_targets(
        self, part_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(t_parts (B, P, D), part_mask (B, P))`` for a batch.

        Unknown rows (idx < 0) get zero-filled targets and all-False mask
        — they contribute zero gradient through :func:`part_alignment_loss`.
        """
        B = part_idx.shape[0]
        P = int(self.part_embedding_tensor.shape[1])
        D = int(self.part_embedding_tensor.shape[2])
        device = self.part_embedding_tensor.device
        t = torch.zeros(B, P, D, dtype=torch.float32, device=device)
        m = torch.zeros(B, P, dtype=torch.bool, device=device)
        valid = part_idx >= 0
        if valid.any():
            positions = valid.nonzero(as_tuple=True)[0]
            rows = part_idx[valid]
            t[positions] = self.part_embedding_tensor.index_select(0, rows)
            m[positions] = self.part_mask_tensor.index_select(0, rows)
        return t, m

    # -- training loop ----------------------------------------------------

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch

        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            if self.cutmix_alpha > 0.0:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            used_mixup = False
            if self.mixup_alpha > 0.0:
                used_mixup = True
                lam = float(
                    torch.distributions.Beta(
                        self.mixup_alpha, self.mixup_alpha
                    ).sample()
                )
                perm = torch.randperm(inputs.size(0), device=inputs.device)
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                labels_b = labels[perm]
                out = self(inputs)
                L_emo = lam * self.loss_fn(out["logits"], labels) + (
                    1.0 - lam
                ) * self.loss_fn(out["logits"], labels_b)
            else:
                out = self(inputs)
                L_emo = self.loss_fn(out["logits"], labels)

            total = L_emo
            self.log("train_loss_emo", L_emo, prog_bar=True)

            if (
                self.part_loss_weight > 0
                and "z_parts" in out
                and self.current_epoch >= self.part_loss_warmup_epochs
                and not used_mixup
            ):
                part_idx = self._batch_part_idx(filenames, inputs.device)
                z_parts = out["z_parts"]
                t_parts, part_mask = self._lookup_targets(part_idx)
                # Ensure targets are L2-normalized before the cosine/infonce loss.
                t_parts = torch.nn.functional.normalize(t_parts, dim=-1)
                L_part = part_alignment_loss(
                    z_parts,
                    t_parts,
                    part_mask=part_mask,
                    mode=self.part_loss_mode,
                    temperature=self.part_loss_temperature,
                )
                total = total + self.part_loss_weight * L_part
                self.log("train_loss_part", L_part)
                self.log(
                    "part_valid_frac",
                    (part_idx >= 0).float().mean().item(),
                )
                self.log(
                    "part_mask_frac",
                    part_mask.float().mean().item(),
                )

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
