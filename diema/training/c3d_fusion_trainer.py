""" (exp056): training integration for C3D late-fusion.

Related:
- diema/models/conv1d_transformer/c3d_latefusion_model.py (model)
- diema/features/c3d_cache.py (feature lookup)
- experiments/exp056_c3d_stats_latefusion/ (consumer)

Thin subclass of :class:`diema.training.train.LightningModel` that plumbs a
batch-level C3D feature tensor into the model's extended ``forward`` signature.
Everything else (loss, optimizer, augmentation, Gaussian noise, mixup, etc.)
is inherited unchanged — the fusion is purely at the model input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.training.train import LightningModel


class C3DFusionLightningModel(LightningModel):
    """CE-only Lightning wrapper that feeds ``c3d_vec`` into the model.

    Attributes:
        c3d_cache: :class:`diema.features.c3d_cache.C3DStatsCache` lookup.
        missing_c3d_as_zero: if True (default) samples whose filename is
            not in the cache get an all-zero C3D vector (motion-only fallback).
    """

    _HPARAM_IGNORE = ["c3d_cache"]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        c3d_cache,
        missing_c3d_as_zero: bool = True,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model, base_lr=base_lr, num_class=num_class,
            **lightning_kwargs,
        )
        self.c3d_cache = c3d_cache
        self.missing_c3d_as_zero = bool(missing_c3d_as_zero)

        # Skip C3D embeddings from hparams.yaml — both cache and backbone may be large.
        try:
            self.save_hyperparameters(ignore=self._HPARAM_IGNORE)
        except Exception:
            pass

    # -- forward helpers --------------------------------------------------

    def _c3d_tensor(self, filenames: list[str], device: torch.device) -> torch.Tensor:
        """Build the (N, D_c3d) C3D feature tensor for the current batch."""
        return self.c3d_cache.batch_tensor(filenames).to(device)

    def _forward_with_c3d(
        self,
        inputs: torch.Tensor,
        filenames: list[str],
    ) -> dict[str, torch.Tensor]:
        c3d = self._c3d_tensor(filenames, inputs.device)
        return self.model(inputs, c3d_vec=c3d)

    # -- training / val / test --------------------------------------------

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch

        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            if self.cutmix_alpha > 0.0:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            if self.mixup_alpha > 0.0:
                lam = float(
                    torch.distributions.Beta(
                        self.mixup_alpha, self.mixup_alpha
                    ).sample()
                )
                perm = torch.randperm(inputs.size(0), device=inputs.device)
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                labels_b = labels[perm]
                out = self._forward_with_c3d(inputs, filenames)
                loss = lam * self.loss_fn(out["logits"], labels) + (
                    1.0 - lam
                ) * self.loss_fn(out["logits"], labels_b)
            else:
                out = self._forward_with_c3d(inputs, filenames)
                loss = self.loss_fn(out["logits"], labels)

            self.log("train_loss", loss, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, filenames = batch
        out = self._forward_with_c3d(inputs, filenames)
        loss = self.loss_fn(out["logits"], labels)
        predicted = torch.argmax(out["logits"], dim=1)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))
        self.val_acc(predicted, labels)
        self.val_f1(predicted, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, batch_size=len(labels))
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True, batch_size=len(labels))

    def test_step(self, batch, batch_idx):
        inputs, labels, filenames = batch
        out = self._forward_with_c3d(inputs, filenames)
        predicted = torch.argmax(out["logits"], dim=1)
        self.test_acc(predicted, labels)
        self.test_f1(predicted, labels)
        self.log("test_acc", self.test_acc, on_epoch=True, batch_size=len(labels))
        self.log("test_f1", self.test_f1, on_epoch=True, batch_size=len(labels))
