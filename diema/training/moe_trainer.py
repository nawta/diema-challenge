""" training integration for Style-Conditioned MoE.

:class:`MoELightningModel` extends :class:`LightningModel` with MoE
regularization losses: load balancing, entropy, and expert diversity.

Loss::

    total = CE(logits, y_emo)
          + lambda_lb  * load_balancing(routing_weights)
          + lambda_ent * (-entropy(routing_weights))
          + lambda_div * expert_diversity(expert_logits)

Related: diema/models/conv1d_transformer/moe_head_model.py,
         diema/training/losses.py (MoE regularization functions)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.training.train import LightningModel
from diema.training.losses import (
    moe_load_balancing_loss,
    moe_entropy_regularization,
    moe_expert_diversity_loss,
)


class MoELightningModel(LightningModel):
    """Lightning subclass for MoE head training.

    Args:
        lambda_lb: load balancing loss weight (default 0.01).
        lambda_ent: entropy regularization weight (default 0.01).
            Negated inside training_step to maximize entropy.
        lambda_div: expert diversity loss weight (default 0.005).
        All other kwargs forwarded to :class:`LightningModel`.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        lambda_lb: float = 0.01,
        lambda_ent: float = 0.01,
        lambda_div: float = 0.005,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        self.lambda_lb = float(lambda_lb)
        self.lambda_ent = float(lambda_ent)
        self.lambda_div = float(lambda_div)

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

            if "routing_weights" in out:
                rw = out["routing_weights"]

                if self.lambda_lb > 0:
                    L_lb = moe_load_balancing_loss(rw)
                    total = total + self.lambda_lb * L_lb
                    self.log("train_loss_lb", L_lb)

                if self.lambda_ent > 0:
                    L_ent = moe_entropy_regularization(rw)
                    total = total - self.lambda_ent * L_ent  # maximize entropy
                    self.log("train_routing_entropy", L_ent)

            if "expert_logits" in out and self.lambda_div > 0:
                L_div = moe_expert_diversity_loss(out["expert_logits"])
                total = total + self.lambda_div * L_div
                self.log("train_loss_div", L_div)

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
