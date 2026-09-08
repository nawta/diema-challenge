""" v2: CE + CenterLoss training (SupCon replacement).

CenterLoss is much more stable than pairwise SupCon on small datasets
(8K) because it does not depend on in-batch positive pair quality.
Each sample is pulled toward its class center, with exponential
lambda warmup to prevent interference with early backbone formation.

Loss::

    total = CE(emo_logits, y_emotion)
          + lambda_eff * CenterLoss(features, y_emotion)

where ``lambda_eff = lambda_center * (1 - exp(-epoch / tau))``.

See: v2
Related: diema/training/losses.py::CenterLoss,
         diema/training/losses.py::exp_lambda_warmup
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched

from diema.training.train import LightningModel
from diema.training.losses import CenterLoss, exp_lambda_warmup


class CenterLossLightningModel(LightningModel):
    """Lightning subclass for CE + CenterLoss training.

    The CenterLoss centers are learnable parameters with their own LR
    (``center_lr``, default 0.5). The lambda is warmed up exponentially
    from 0 to ``lambda_center`` with time constant ``lambda_tau``.

    Args:
        num_class: number of emotion classes.
        feat_dim: backbone feature dimensionality.
        lambda_center: target center loss weight (default 0.001).
        lambda_tau: exponential warmup time constant (default 20).
        center_lr: learning rate for class centers (default 0.5).
        All other kwargs forwarded to :class:`LightningModel`.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        feat_dim: int,
        lambda_center: float = 0.001,
        lambda_tau: float = 20.0,
        center_lr: float = 0.5,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        self.center_loss_fn = CenterLoss(
            num_classes=num_class, feat_dim=feat_dim,
        )
        self.lambda_center = float(lambda_center)
        self.lambda_tau = float(lambda_tau)
        self.center_lr = float(center_lr)

    def configure_optimizers(self):
        """Override to give center params a separate LR.

        The parent's configure_optimizers() calls self.parameters()
        which includes center_loss_fn. We exclude center params from
        the main param group and add them with center_lr instead.
        """
        # Collect center param ids to exclude from backbone group
        center_param_ids = {id(p) for p in self.center_loss_fn.parameters()}
        backbone_params = [
            p for p in self.parameters()
            if id(p) not in center_param_ids
        ]

        # Build optimizer with backbone params only
        opt_cls = {
            "SGD": lambda ps: torch.optim.SGD(
                ps, lr=self.base_lr, momentum=0.9, nesterov=True,
                weight_decay=self.weight_decay,
            ),
            "Adam": lambda ps: torch.optim.Adam(
                ps, lr=self.base_lr, weight_decay=self.weight_decay,
            ),
            "AdamW": lambda ps: torch.optim.AdamW(
                ps, lr=self.base_lr, weight_decay=self.weight_decay,
            ),
        }
        optimizer = opt_cls[self.optimizer_name](backbone_params)

        # Add center params with their own LR (no weight decay)
        optimizer.add_param_group({
            "params": list(self.center_loss_fn.parameters()),
            "lr": self.center_lr,
            "weight_decay": 0.0,
        })

        # Build scheduler (reuse parent's logic)
        import torch.optim.lr_scheduler as sched
        if self.scheduler_type == "cosine":
            scheduler = sched.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs,
            )
        elif self.scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            warmup = LinearLR(
                optimizer, start_factor=1e-4, total_iters=self.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs - self.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:  # step
            scheduler = sched.StepLR(
                optimizer,
                step_size=self.scheduler_params[0],
                gamma=self.scheduler_params[1],
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        inputs, labels, _filenames = batch

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
                out = self(inputs)
                L_emo = lam * self.loss_fn(out["logits"], labels) + (
                    1.0 - lam
                ) * self.loss_fn(out["logits"], labels_b)
            else:
                out = self(inputs)
                L_emo = self.loss_fn(out["logits"], labels)

            total = L_emo
            self.log("train_loss_emo", L_emo, prog_bar=True)

            # CenterLoss with exponential warmup
            if "features" in out and self.lambda_center > 0:
                lambda_eff = exp_lambda_warmup(
                    self.current_epoch, self.lambda_center, self.lambda_tau,
                )
                L_center = self.center_loss_fn(out["features"], labels)
                total = total + lambda_eff * L_center
                self.log("train_loss_center", L_center)
                self.log("lambda_center_eff", lambda_eff)

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
