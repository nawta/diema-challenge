""" training integration for scenario-text alignment.

:class:`TextAlignLightningModel` is a thin subclass of
:class:`diema.training.train.LightningModel` that adds a multi-positive
:func:`diema.training.losses_motion_text.global_multi_positive_info_nce` loss
on top of the standard CE. No validation / test behavior changes.

Loss::

    total = CE(logits, y) + λ_global * GlobalInfoNCE(z_text, t_scenario,
                                                     positive_mask)

Where ``positive_mask`` is built per-batch from
``(scenario_idx, emotion_labels)`` under the ``same_scenario_and_emotion``
policy by default (TODO 3AE-5 recommendation to avoid cross-emotion leakage).

See: (exp051)
Related:
- diema/models/conv1d_transformer/textalign_conv1d_tr_model.py (backbone wrapper)
- diema/features/text_cache.py (ScenarioTextCache filename→idx+embedding lookup)
- diema/training/losses_motion_text.py (GlobalInfoNCELoss)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diema.training.train import LightningModel
from diema.training.losses_motion_text import (
    build_positive_mask,
    global_multi_positive_info_nce,
)

_SUPPORTED_POLICIES = frozenset({
    "same_scenario_and_emotion",
    "same_scenario",
    "same_emotion",
    "diagonal",
})


class TextAlignLightningModel(LightningModel):
    """Lightning subclass for joint CE + multi-positive InfoNCE on scenario text.

    Override :meth:`training_step` to compute the combined loss. Validation
    and test steps fall back to the parent (only uses ``out["logits"]``).

    Args:
        filename_to_scenario_idx: map filename.stem → scenario_idx.
        scenario_embedding_matrix: ``(num_unique, text_dim)`` L2-normalized
            tensor aligned with scenario_idx. Stored as a buffer so it moves
            with the LightningModule's device.
        text_loss_weight: λ for the global InfoNCE loss (default 0.1, exp051 a00).
        text_loss_warmup_epochs: CE-only warmup before the text loss turns on
            (default 0, matching TMR start-from-step-0 convention).
        text_loss_temperature: softmax temperature for the InfoNCE (default 0.1).
        positive_pairing: one of ``{"same_scenario_and_emotion" (default),
            "same_scenario", "same_emotion", "diagonal"}``.
        text_loss_symmetric: CLIP-style m↔t averaging (default True).
        lightning_kwargs: forwarded to :class:`LightningModel` parent.

    Design:
        The scenario text embeddings are kept on CPU in the cache (
        design) and copied once into a Lightning buffer so they are placed on
        the model's device at the first training step. ``filename_to_scenario_idx``
        is stored as a plain dict (not hyperparameter) since it is big and
        unserializable via OmegaConf.
    """

    _HPARAM_IGNORE = ["filename_to_scenario_idx", "scenario_embedding_matrix"]

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        filename_to_scenario_idx: dict[str, int],
        scenario_embedding_matrix: torch.Tensor,
        text_loss_weight: float = 0.1,
        text_loss_warmup_epochs: int = 0,
        text_loss_temperature: float = 0.1,
        positive_pairing: str = "same_scenario_and_emotion",
        text_loss_symmetric: bool = True,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        if positive_pairing not in _SUPPORTED_POLICIES:
            raise ValueError(
                f"positive_pairing must be in {sorted(_SUPPORTED_POLICIES)}, "
                f"got {positive_pairing!r}"
            )

        # Ignore the large / non-serializable members so they don't end up in
        # hparams.yaml (the embedding matrix is ~2623×384 float32 and the
        # filename dict is ~8000 entries — neither belongs in the run summary).
        # Lightning's save_hyperparameters merges with ancestors' state, so this
        # call layers on top of the parent's bookkeeping.
        try:
            self.save_hyperparameters(ignore=self._HPARAM_IGNORE)
        except Exception:
            # Parent may have already snapshot hparams; silently tolerate.
            pass

        self.filename_to_scenario_idx = dict(filename_to_scenario_idx)
        self.text_loss_weight = float(text_loss_weight)
        self.text_loss_warmup_epochs = int(text_loss_warmup_epochs)
        self.text_loss_temperature = float(text_loss_temperature)
        self.positive_pairing = positive_pairing
        self.text_loss_symmetric = bool(text_loss_symmetric)

        # Register the embedding matrix as a (non-parameter) buffer so it
        # travels with the model's device and is present in the checkpoint's
        # state_dict. `tools/strip_text_branch.py` can drop it
        # before submission without affecting inference.
        if scenario_embedding_matrix.ndim != 2:
            raise ValueError(
                f"scenario_embedding_matrix must be 2-D, got "
                f"{tuple(scenario_embedding_matrix.shape)}"
            )
        self.register_buffer(
            "scenario_embedding_matrix",
            scenario_embedding_matrix.to(torch.float32).clone(),
            persistent=True,
        )

    # -- internal helpers --------------------------------------------------

    def _batch_scenario_idx(
        self,
        filenames: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        from pathlib import Path
        ids = []
        for fn in filenames:
            stem = Path(str(fn)).stem
            ids.append(self.filename_to_scenario_idx.get(stem, -1))
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _compute_text_loss(
        self,
        z_text: torch.Tensor,
        labels: torch.Tensor,
        scenario_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run GlobalInfoNCE on the subset of samples with a known scenario_idx.

        Samples with ``scenario_idx == -1`` (e.g. test clips or scenarios not
        covered by the cache) are dropped from the contrastive batch — they do
        not receive or contribute gradient through this loss.
        """
        valid = scenario_idx >= 0
        if valid.sum().item() < 2:
            # Need ≥ 2 samples to form a meaningful positive mask.
            return z_text.new_zeros(()).squeeze()

        z_v = z_text[valid]
        idx_v = scenario_idx[valid]
        labels_v = labels[valid]
        t_v = self.scenario_embedding_matrix.index_select(0, idx_v).to(z_text.dtype)

        positive_mask = build_positive_mask(
            scenario_idx=idx_v,
            emotion_labels=labels_v,
            policy=self.positive_pairing,
        ).to(z_text.device)

        return global_multi_positive_info_nce(
            z_v,
            t_v,
            positive_mask=positive_mask,
            temperature=self.text_loss_temperature,
            symmetric=self.text_loss_symmetric,
        )

    # -- training loop -----------------------------------------------------

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch

        # Gaussian weight noise follows the parent LightningModel pattern:
        # apply here, let ``on_after_backward`` restore on the normal path, and
        # only restore in the ``except`` branch below for exception paths. Do not
        # convert this to try/finally — that would double-restore once here and
        # once in the hook, leaving the model permanently un-noised but also
        # hiding the hook invariant. See diema/training/train.py::on_after_backward.
        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            # same-emotion CutMix, parent method.
            if self.cutmix_alpha > 0.0:
                inputs = self._apply_same_emotion_cutmix(inputs, labels)

            used_mixup = False
            if self.mixup_alpha > 0.0:
                # Mixup mixes across emotion classes and corrupts the
                # scenario-positive mask semantics, so skip text loss when
                # mixup is active (parity with SupConLightningModel).
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
                self.text_loss_weight > 0
                and "z_text" in out
                and self.current_epoch >= self.text_loss_warmup_epochs
                and not used_mixup
            ):
                scenario_idx = self._batch_scenario_idx(filenames, inputs.device)
                L_text = self._compute_text_loss(
                    out["z_text"], labels, scenario_idx
                )
                total = total + self.text_loss_weight * L_text
                self.log("train_loss_text", L_text)
                # `.item()` already returns a Python float; no outer cast needed.
                self.log("text_valid_frac", (scenario_idx >= 0).float().mean().item())

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
