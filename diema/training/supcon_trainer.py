""" training integration for SupCon + emotion classification.

:class:`SupConLightningModel` is a thin subclass of
:class:`diema.training.train.LightningModel` that adds a
performer-aware Supervised Contrastive loss to the standard CE.

Loss::

    total = CE(emo_logits, y_emotion)
          + supcon_lambda * SupConLoss(z_sc, y_emotion, y_performer)

The performer label reuses the same mapping infrastructure as
:mod:`diema.training.dual_head_trainer`.

Related: diema/models/conv1d_transformer/supcon_conv1d_tr_model.py,
         diema/training/losses.py::SupConLoss,
         diema/training/dual_head_trainer.py (performer mapping)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.training.train import LightningModel
from diema.training.losses import SupConLoss
from diema.training.dual_head_trainer import _perf_idx_from_filename


class SupConLightningModel(LightningModel):
    """Lightning subclass for joint CE + SupCon training.

    Overrides ``training_step`` to compute the combined loss. Validation
    and test steps fall back to the parent (only uses ``out["logits"]``).

    Args:
        filename_to_perf_idx: map from sample filename to performer index.
        num_performer: number of distinct performers.
        pair_to_perf_idx: optional fallback map.
        supcon_lambda: weight on the SupCon loss (default 0.05).
        supcon_warmup_epochs: CE-only warmup before enabling SupCon
            (default 5).
        supcon_temperature: contrastive temperature (default 0.1).
        supcon_w_strong: weight for cross-performer positives.
        supcon_w_weak: weight for same-performer positives.
        All other kwargs are forwarded to :class:`LightningModel`.
    """

    # Exclude large / non-serializable dicts from Lightning's
    # save_hyperparameters (tuple keys cause OmegaConf KeyValidationError).
    _HPARAM_IGNORE = ["filename_to_perf_idx", "pair_to_perf_idx"]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        filename_to_perf_idx: dict[str, int],
        num_performer: int,
        pair_to_perf_idx: dict[tuple[str, int], int] | None = None,
        supcon_lambda: float = 0.05,
        supcon_warmup_epochs: int = 5,
        supcon_temperature: float = 0.1,
        supcon_w_strong: float = 2.0,
        supcon_w_weak: float = 0.5,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        self.filename_to_perf_idx = dict(filename_to_perf_idx)
        self.pair_to_perf_idx = dict(pair_to_perf_idx) if pair_to_perf_idx else {}
        self.num_performer = int(num_performer)
        self.supcon_lambda = float(supcon_lambda)
        self.supcon_warmup_epochs = int(supcon_warmup_epochs)
        self.supcon_loss_fn = SupConLoss(
            temperature=supcon_temperature,
            w_strong=supcon_w_strong,
            w_weak=supcon_w_weak,
        )

    def _get_performer_ids(
        self,
        filenames: list[str],
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return performer index tensor ``(N,)`` or ``None`` if mapping
        is empty / all unknown."""
        if not self.filename_to_perf_idx and not self.pair_to_perf_idx:
            return None
        pair_map = self.pair_to_perf_idx if self.pair_to_perf_idx else None
        ids = []
        known = 0
        for fn in filenames:
            idx = _perf_idx_from_filename(fn, self.filename_to_perf_idx, pair_map)
            if idx is not None and 0 <= idx < max(self.num_performer, 1):
                ids.append(idx)
                known += 1
            else:
                ids.append(0)
        if known == 0:
            return None
        return torch.tensor(ids, dtype=torch.long, device=device)

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch

        # Gaussian weight noise from parent
        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            # augmentations from parent
            if self.cutmix_alpha > 0.0:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            # Flag: mixup corrupts embeddings, so SupCon must be
            # disabled when mixup is active (mixed embeddings + unmixed
            # labels breaks positive/negative pair semantics).
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

            # SupCon loss (after warmup, skip when mixup was applied)
            if (
                self.supcon_lambda > 0
                and "z_sc" in out
                and self.current_epoch >= self.supcon_warmup_epochs
                and not used_mixup
            ):
                performer_ids = self._get_performer_ids(filenames, inputs.device)
                L_sc = self.supcon_loss_fn(out["z_sc"], labels, performer_ids)
                total = total + self.supcon_lambda * L_sc
                self.log("train_loss_supcon", L_sc)

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
