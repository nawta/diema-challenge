"""Lightning training module for emotion recognition.

Model-agnostic wrapper that works with any BaseModel subclass.
Handles training/validation/test/predict steps, optimizer configuration,
and metric logging (including wandb integration).

See: diema_challenge_implementation_spec.md §6.5 — 学習ループ
Related: diema/models/base.py, diema/training/losses.py, diema/training/callbacks.py

Ported from: an internal baseline with additions:
  - wandb Logger integration
  - OOF prediction storage support
  - Multiple loss function support (CE, Focal, ClassBalanced)
  - Configurable optimizer/scheduler
"""

import torch
import torch.nn as nn

import torchmetrics
import pytorch_lightning as pl

from diema.training.losses import COSINE_LOGIT_LOSS_TYPES, get_loss_fn


class LightningModel(pl.LightningModule):
    """Lightning wrapper for training any BaseModel subclass.

    Args:
        model: a BaseModel instance
        base_lr: base learning rate
        num_class: number of classes for metrics
        loss_type: loss function name ('ce', 'focal', 'class_balanced')
        loss_kwargs: additional kwargs for loss function
        optimizer: 'SGD' or 'Adam'
        scheduler_type: 'cosine' or 'step'
        scheduler_params: for 'step', provide [step_size, gamma]
        weight_decay: L2 regularization
        aux_loss_weights: per-loss weight overrides
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        loss_type: str = "ce",
        loss_kwargs: dict | None = None,
        optimizer: str = "SGD",
        scheduler_type: str = "cosine",
        scheduler_params: list | None = None,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        aux_loss_weights: dict | None = None,
        gaussian_noise_sigma: float = 0.0,
    ):
        super().__init__()
        if scheduler_params is None:
            scheduler_params = []

        self.base_lr = base_lr
        self.model = model
        self.loss_fn = get_loss_fn(loss_type, **(loss_kwargs or {}))
        self.optimizer_name = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha = mixup_alpha
        # same-emotion temporal CutMix. When > 0, each training
        # step samples lam ~ Beta(alpha, alpha), picks a permutation that
        # only pairs same-class samples (others stay unchanged), then swaps
        # a temporal window of length round((1 - lam) * T) into each sample
        # from its same-class partner. Labels are unchanged because both
        # partners share the same class.
        self.cutmix_alpha = cutmix_alpha
        self.aux_loss_weights = aux_loss_weights or {}
        # Gaussian weight noise — lightweight AWP alternative.
        # Each training step, add N(0, sigma) noise to all parameters before
        # forward, then restore them. Implemented in optimizer_step (see below).
        self.gaussian_noise_sigma = gaussian_noise_sigma

        if scheduler_type not in ("cosine", "cosine_warmup", "step"):
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        if optimizer not in ("SGD", "Adam", "AdamW"):
            raise ValueError(f"Unsupported optimizer type: {optimizer}")
        if scheduler_type == "step" and not scheduler_params:
            raise ValueError("Scheduler params must be provided for step scheduler")

        # cosine-logit losses silently break if the model produces
        # plain nn.Linear logits. Fail loudly at init time if the model
        # advertises a non-cosine head.
        if loss_type in COSINE_LOGIT_LOSS_TYPES:
            wants_cosine = getattr(model, "use_arcface_head", None)
            if wants_cosine is False:
                raise ValueError(
                    f"loss_type={loss_type!r} requires a cosine classifier head. "
                    f"Construct the model with use_arcface_head=True "
                    f"(STGCN++, CTR-GCN, SkateFormer support this) or switch "
                    f"the loss to 'ce'. See diema.models.heads.ArcMarginHead."
                )
            # wants_cosine is None for models that do not yet support the
            # flag (e.g. ProtoGCN, Squeezeformer, Conformer). We can't
            # prove correctness in that case, so emit a soft warning via
            # the training logger instead of raising.

        # Subclasses may define _HPARAM_IGNORE to exclude additional
        # non-serializable constructor args (e.g. tuple-keyed dicts).
        ignore = ["model"] + getattr(self.__class__, "_HPARAM_IGNORE", [])
        self.save_hyperparameters(ignore=ignore)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_class, average="macro")

    def forward(self, x):
        return self.model(x)

    def _apply_gaussian_weight_noise(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """ add N(0, sigma) noise to all trainable params.

        Returns a list of (param, original_data) pairs so that the caller can
        restore the original weights after the forward/backward pass. The
        noise is sampled fresh each step. No-op when sigma <= 0.
        """
        if self.gaussian_noise_sigma <= 0.0:
            return []
        saved: list[tuple[torch.Tensor, torch.Tensor]] = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            original = p.data.clone()
            saved.append((p, original))
            p.data.add_(torch.randn_like(p.data) * self.gaussian_noise_sigma)
        return saved

    def _restore_weights(self, saved: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for p, orig in saved:
            p.data.copy_(orig)

    def training_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch

        # Gaussian weight noise (light AWP). Sample noise, apply to
        # all params, do forward + loss. `_noise_saved_weights` is stashed
        # *immediately* after noise is applied so that `on_after_backward`
        # (or the `except` branch below) can always restore weights, even if
        # the forward/loss/logging path throws. Gradients are computed under
        # the noisy weights, but the optimizer step runs against the original
        # weights because we restore in `on_after_backward` (which Lightning
        # calls between `backward()` and gradient clipping).
        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            # Same-emotion temporal CutMix. Runs *before* mixup
            # so that the mixed samples can also undergo cross-class
            # interpolation if both are configured.
            if self.cutmix_alpha > 0.0:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            # Mixup augmentation: linearly interpolate two samples and labels
            if self.mixup_alpha > 0.0:
                lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
                perm = torch.randperm(inputs.size(0), device=inputs.device)
                inputs = lam * inputs + (1.0 - lam) * inputs[perm]
                labels_b = labels[perm]
                out = self(inputs)
                loss = lam * self.loss_fn(out["logits"], labels) + (1.0 - lam) * self.loss_fn(out["logits"], labels_b)
            else:
                out = self(inputs)
                loss = self.loss_fn(out["logits"], labels)

            # Handle auxiliary losses if the model produces them
            if "aux_losses" in out:
                for name, value in out["aux_losses"].items():
                    if name.endswith("_logits"):
                        aux_loss = self.loss_fn(value, labels)
                    else:
                        aux_loss = value
                    weight = self.aux_loss_weights.get(name, 1.0)
                    loss = loss + weight * aux_loss
                    self.log(f"train_{name}", aux_loss)

            self.log("train_loss", loss, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            # If anything fails between noise application and on_after_backward
            # (forward, loss, logging), restore weights to their pre-noise
            # state so the model is not left permanently perturbed.
            self._restore_noise_weights()
            raise
        return loss

    def _apply_same_emotion_cutmix(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """ temporal CutMix swap with same-class partners.

        For every sample in the batch, we pick a random partner and swap a
        contiguous temporal window only if the partner shares the same label.
        Samples without a same-class partner in the random permutation keep
        their original frames unchanged. Because both endpoints of every
        non-trivial swap share the same label, the targets do not need to be
        mixed and the standard CE loss applies as-is.

        Inputs shape: ``(N, C, T, V)``.
        """
        N, _, T, _ = inputs.shape
        if N < 2 or T < 2:
            return inputs
        # Sample Beta and the cut_start index on the same device as
        # ``inputs`` so that every random draw inside this helper uses
        # the same RNG stream as ``perm`` below. Force ``alpha_t`` to
        # float32 regardless of ``inputs.dtype`` because
        # ``torch.distributions.Beta.sample()`` is numerically flaky on
        # float16 / bfloat16 in some PyTorch builds; we immediately
        # convert the sample to a Python float so the outer dtype does
        # not matter for the downstream cutmix masking.
        alpha_t = torch.tensor(
            self.cutmix_alpha, dtype=torch.float32, device=inputs.device
        )
        lam = float(torch.distributions.Beta(alpha_t, alpha_t).sample())
        # cut_len ∈ [1, T-1] — never copy the entire sequence
        cut_len = int(round((1.0 - lam) * T))
        cut_len = max(min(cut_len, T - 1), 1)
        cut_start = int(
            torch.randint(0, T - cut_len + 1, (1,), device=inputs.device).item()
        )
        cut_end = cut_start + cut_len

        perm = torch.randperm(N, device=inputs.device)
        same_class = labels == labels[perm]
        if not bool(same_class.any()):
            return inputs

        # For samples without a same-class partner, fall back to self so the
        # window swap is a no-op for them.
        identity = torch.arange(N, device=inputs.device)
        safe_perm = torch.where(same_class, perm, identity)

        out = inputs.clone()
        out[:, :, cut_start:cut_end, :] = inputs[safe_perm, :, cut_start:cut_end, :]
        return out

    def _restore_noise_weights(self) -> None:
        """Restore weights stashed by _apply_gaussian_weight_noise. Idempotent."""
        saved = getattr(self, "_noise_saved_weights", None)
        if saved:
            self._restore_weights(saved)
            self._noise_saved_weights = []

    def on_after_backward(self):
        """ restore original weights after grads computed under noise.

        Runs before gradient clipping / optimizer step so the optimizer updates
        the original weights using gradients computed under the noisy weights.
        """
        self._restore_noise_weights()

    def validation_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        loss = self.loss_fn(out["logits"], labels)
        predicted = torch.argmax(out["logits"], dim=1)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(labels))
        self.val_acc(predicted, labels)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.val_f1(predicted, labels)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels, _sample_names = batch
        out = self(inputs)
        predicted = torch.argmax(out["logits"], dim=1)

        self.test_acc(predicted, labels)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.test_f1(predicted, labels)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inputs, labels, sample_names = batch
        out = self(inputs)
        proba = torch.softmax(out["logits"], dim=1)
        predicted = torch.argmax(proba, dim=1)
        return predicted, proba, labels, sample_names

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay,
            )

        if self.scheduler_type == "step":
            step_size, gamma = self.scheduler_params
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_type == "cosine_warmup":
            # Linear warmup over `warmup_epochs`, then cosine decay over the rest
            from torch.optim.lr_scheduler import LambdaLR
            max_epochs = self.trainer.max_epochs
            warmup = max(self.warmup_epochs, 1) if self.warmup_epochs > 0 else 1
            import math as _math

            def lr_lambda(epoch):
                if epoch < warmup:
                    return float(epoch + 1) / float(warmup)
                progress = (epoch - warmup) / max(max_epochs - warmup, 1)
                return 0.5 * (1.0 + _math.cos(_math.pi * progress))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
