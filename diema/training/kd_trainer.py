""" (Tier 2): VLM logit knowledge-distillation trainer.

See: (exp060) / docs/rule_and_ethics_checklist.md §3.3
Related:
- diema/training/train.py::LightningModel (parent)
- tools/generate_emotion_softlabel_qwenvl_vllm.py (produces the soft-label .npz)

Loss::

    total = CE(logits, y) + λ_kd · T² · KL(softmax(logits/T) || softmax(teacher/T))

Where ``teacher`` is a 12-class probability distribution from Qwen3-VL,
loaded from ``output/vlm_softlabel_train.npz``. T is the temperature
(default 2.0). Following Hinton 2015, the KL term is multiplied by T²
so the gradient magnitude does not vanish as T grows.

Train-only constraint (ethics §3.3): the soft-label tensor is registered
as a buffer that lives in train code only — there is no test-side import
path, and ``tests/test_softlabel_train_only.py`` enforces this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.training.train import LightningModel


def softlabel_kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Hinton-style KL divergence on softened student vs hard teacher distribution.

    Args:
        student_logits: (N, C) raw classifier logits.
        teacher_probs:  (N, C) probability distribution (rows sum to 1, all > 0).
        temperature: T > 0. Scales logits before softmax. Loss is multiplied by T².
        eps: clamp_min for log stability.

    Returns:
        Scalar KL loss averaged over the batch.
    """
    if student_logits.shape != teacher_probs.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs teacher {tuple(teacher_probs.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    log_p = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = teacher_probs.clamp_min(eps)
    # KL(teacher || student) = Σ teacher · (log teacher - log student)
    log_q_teacher = torch.log(p_teacher / p_teacher.sum(dim=-1, keepdim=True))
    kl_per_row = (p_teacher * (log_q_teacher - log_p)).sum(dim=-1)
    return (temperature ** 2) * kl_per_row.mean()


class SoftLabelKdLightningModel(LightningModel):
    """CE + KD on (frozen) Qwen3-VL 12-class soft labels."""

    _HPARAM_IGNORE = ["filename_to_softlabel_idx", "softlabel_tensor"]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        filename_to_softlabel_idx: dict[str, int],
        softlabel_tensor: torch.Tensor,
        kd_loss_weight: float = 0.5,
        kd_temperature: float = 2.0,
        kd_warmup_epochs: int = 0,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        if softlabel_tensor.ndim != 2 or softlabel_tensor.shape[1] != num_class:
            raise ValueError(
                f"softlabel_tensor must be (N, {num_class}); got {tuple(softlabel_tensor.shape)}"
            )
        if kd_temperature <= 0:
            raise ValueError(f"kd_temperature must be > 0, got {kd_temperature}")

        try:
            self.save_hyperparameters(ignore=self._HPARAM_IGNORE)
        except Exception:
            pass

        self.filename_to_softlabel_idx = dict(filename_to_softlabel_idx)
        self.kd_loss_weight = float(kd_loss_weight)
        self.kd_temperature = float(kd_temperature)
        self.kd_warmup_epochs = int(kd_warmup_epochs)

        self.register_buffer(
            "softlabel_tensor",
            softlabel_tensor.to(torch.float32).clone(),
            persistent=True,
        )

    def _batch_softlabel(
        self, filenames: list[str], device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(teacher (B, C), valid (B,))`` for a batch.

        ``valid[i] = False`` means the stem has no soft label (drop from KD).
        Unknown rows return uniform 1/C as a placeholder (zeroed by valid mask).
        """
        ids = [
            self.filename_to_softlabel_idx.get(Path(str(fn)).stem, -1)
            for fn in filenames
        ]
        idx = torch.tensor(ids, dtype=torch.long, device=device)
        valid = idx >= 0
        B = idx.numel()
        C = int(self.softlabel_tensor.shape[1])
        teacher = torch.full(
            (B, C), 1.0 / C, dtype=torch.float32, device=device,
        )
        if valid.any():
            positions = valid.nonzero(as_tuple=True)[0]
            rows = idx[valid]
            teacher[positions] = self.softlabel_tensor.index_select(0, rows)
        return teacher, valid

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
                self.kd_loss_weight > 0
                and self.current_epoch >= self.kd_warmup_epochs
                and not used_mixup
            ):
                teacher, valid = self._batch_softlabel(filenames, inputs.device)
                if valid.any():
                    sub_logits = out["logits"][valid]
                    sub_teacher = teacher[valid]
                    L_kd = softlabel_kd_loss(
                        sub_logits, sub_teacher,
                        temperature=self.kd_temperature,
                    )
                    total = total + self.kd_loss_weight * L_kd
                    self.log("train_loss_kd", L_kd)
                    self.log("kd_valid_frac", valid.float().mean().item())

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
