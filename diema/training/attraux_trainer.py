""" / exp065 — LightningModel with 8 scenario-attribute auxiliary heads.

See: / docs/results_summary.md §10.9
Related:
- diema/training/train.py (LightningModel — base)
- diema/features/scenario_attrs.py (8-attr schema)
- output/scenario_attrs_rule.json (per-stem attr index dump)

Design
------
The base ``region_aware_conv1d_transformer`` already returns
``{"logits": (N, num_class), "features": (N, feat_dim)}``. We add 8 linear
heads, one per scenario attribute, on top of ``features``::

    feat (N, 6*dim) ──┬─▶ existing fc -> emotion logits  (N, 12)
                       └─▶ 8 × Linear(feat_dim, 3)       (N, 3) per attr

Loss::

    L_emotion = CE(logits, label)
    L_attr    = mean over 8 attrs of CE(attr_logits[k], attr_target[k])
    L_total   = L_emotion + λ_attr * L_attr

Attribute targets per clip come from
``output/scenario_attrs_rule.json`` (built by ``probe_scenario_attrs_classifiability.py``);
the LightningModel loads this once at construction and looks up by sample
``stem`` (each batch carries the list-of-filenames as the third element).

Test-time inference does NOT require the attribute heads; they are
auxiliary supervision only. Submission paths read only ``logits``.

Why a separate LightningModel
-----------------------------
The base ``LightningModel`` has an ``aux_losses`` hook but it labels every
``_logits`` aux output with the MAIN ``labels`` tensor. Our 8 attribute heads
each have their own per-clip target, so we override training/validation/test
steps to do the right multi-target lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from diema.features.scenario_attrs import ATTR_ORDER, ATTR_VALUES
from diema.training.losses import get_loss_fn
from diema.training.train import LightningModel


# Map (attr_name, value_str) → small int (0..2). Used to convert the JSON dump
# (which stores string values) into long tensors at LightningModule init.
_ATTR_VAL_TO_IDX: dict[str, dict[str, int]] = {
    a: {v: i for i, v in enumerate(ATTR_VALUES[a])} for a in ATTR_ORDER
}
_NUM_ATTRS = len(ATTR_ORDER)


def load_stem_to_attr_indices(json_path: Path) -> dict[str, torch.Tensor]:
    """Load the JSON dump and return ``{stem: LongTensor of length 8}``.

    Index order follows ``ATTR_ORDER``. Values are 0-indexed within the
    ``ATTR_VALUES[attr]`` vocabulary.
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw: dict[str, dict[str, str]] = json.load(f)
    out: dict[str, torch.Tensor] = {}
    for stem, attrs in raw.items():
        idx_list = []
        for a in ATTR_ORDER:
            v = attrs.get(a)
            mapping = _ATTR_VAL_TO_IDX[a]
            if v not in mapping:
                # Default to the middle / "neutral" / "mid" value for the attr.
                # We pick whichever index 1 is, which by convention is the
                # middle value in our 3-class schemata.
                idx_list.append(1)
            else:
                idx_list.append(mapping[v])
        out[stem] = torch.tensor(idx_list, dtype=torch.long)
    return out


class RegionAwareWithAttrHeads(nn.Module):
    """Wraps a region_aware_conv1d_transformer and adds 8 linear attr heads.

    The wrapped module's ``forward`` must return a dict with keys
    ``"logits"`` (N, num_class) and ``"features"`` (N, feat_dim). The wrapper
    appends ``"attr_logits"`` (N, 8, 3) — first dim is the canonical
    ``ATTR_ORDER`` order, last dim is the 3 attribute values.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model
        feat_dim = int(getattr(base_model, "feature_dim", 0))
        if feat_dim <= 0:
            raise ValueError(
                f"base_model {type(base_model).__name__} must expose feature_dim "
                f"property (got {feat_dim!r})"
            )
        # Each attribute is 3-class.
        self.attr_heads = nn.ModuleList(
            [nn.Linear(feat_dim, 3) for _ in range(_NUM_ATTRS)]
        )

    @property
    def feature_dim(self) -> int:
        return int(self.base.feature_dim)

    @property
    def output_dim(self) -> int:
        return int(self.base.output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(x)
        feats = out["features"]                                  # (N, D)
        attr_logits = torch.stack(
            [head(feats) for head in self.attr_heads], dim=1
        )                                                         # (N, 8, 3)
        return {**out, "attr_logits": attr_logits}


class AttrAuxLightningModel(LightningModel):
    """LightningModel that adds CE losses on 8 attribute heads.

    Wraps the base LightningModel: emotion CE is unchanged. The auxiliary
    loss term is::

        L_attr = (1/8) * Σ_k CE(attr_logits[:, k, :], attr_target[:, k])

    We multiply by ``lambda_attr`` and add to the main emotion loss.

    Attribute targets are looked up per-batch via the sample stems supplied
    by the dataloader collate_fn. Stems missing from the JSON dump fall back
    to the per-attribute middle value (index 1) — this preserves training
    correctness without crashing on edge cases.
    """

    _HPARAM_IGNORE = ["stem_to_attr"]

    def __init__(
        self,
        model: nn.Module,
        stem_to_attr_path: str,
        lambda_attr: float,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.lambda_attr = float(lambda_attr)
        path = Path(stem_to_attr_path)
        if not path.is_file():
            raise FileNotFoundError(f"stem_to_attr_path not found: {path}")
        self.stem_to_attr: dict[str, torch.Tensor] = load_stem_to_attr_indices(path)
        # Stash a fallback (1, 1, …, 1) to keep the dtype/shape stable when a
        # stem misses the JSON dump (shouldn't happen, but defensive).
        self._fallback = torch.ones(_NUM_ATTRS, dtype=torch.long)
        # Per-attribute val-acc / val-f1 in addition to the main metrics.
        self.val_attr_acc = nn.ModuleList([
            torchmetrics.Accuracy(task="multiclass", num_classes=3)
            for _ in range(_NUM_ATTRS)
        ])

    # ------------- batch helpers -------------

    def _attr_targets_for_batch(
        self, sample_names: Iterable[str], device: torch.device,
    ) -> torch.Tensor:
        """Map sample stems → (N, 8) long tensor on ``device``."""
        rows = []
        for s in sample_names:
            stem = Path(str(s)).stem if not isinstance(s, str) or "/" in str(s) else str(s)
            v = self.stem_to_attr.get(stem, self._fallback)
            rows.append(v)
        return torch.stack(rows, dim=0).to(device=device)

    def _attr_loss(
        self, attr_logits: torch.Tensor, attr_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Mean per-attribute CE; returns (mean, per_attr_loss_list).

        ``attr_logits``: (N, 8, 3). ``attr_targets``: (N, 8).
        """
        per_attr_loss: list[torch.Tensor] = []
        for k in range(_NUM_ATTRS):
            per_attr_loss.append(
                F.cross_entropy(attr_logits[:, k, :], attr_targets[:, k])
            )
        mean = torch.stack(per_attr_loss).mean()
        return mean, per_attr_loss

    # ------------- training / validation steps -------------

    def training_step(self, batch, batch_idx):
        # Skip the noise-injection / mixup / cutmix paths from the base for
        # simplicity here; aux head training does not benefit from them in
        # the plan and avoids complicating the loss calc. (We can
        # re-enable later if smoke shows F1 lift but we want regularization.)
        inputs, labels, sample_names = batch
        out = self(inputs)
        ce_loss = self.loss_fn(out["logits"], labels)
        attr_targets = self._attr_targets_for_batch(sample_names, inputs.device)
        attr_loss_mean, per_attr = self._attr_loss(out["attr_logits"], attr_targets)
        total = ce_loss + self.lambda_attr * attr_loss_mean

        self.log("train_loss", total, prog_bar=True, batch_size=inputs.size(0))
        self.log("train_ce", ce_loss, prog_bar=False, batch_size=inputs.size(0))
        self.log("train_attr_loss", attr_loss_mean, prog_bar=True, batch_size=inputs.size(0))
        for k, name in enumerate(ATTR_ORDER):
            self.log(f"train_attr_{name}", per_attr[k], batch_size=inputs.size(0))
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=False)
        return total

    def validation_step(self, batch, batch_idx):
        inputs, labels, sample_names = batch
        out = self(inputs)
        ce_loss = self.loss_fn(out["logits"], labels)
        attr_targets = self._attr_targets_for_batch(sample_names, inputs.device)
        attr_loss_mean, _ = self._attr_loss(out["attr_logits"], attr_targets)
        total = ce_loss + self.lambda_attr * attr_loss_mean

        preds = out["logits"].argmax(dim=1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        for k in range(_NUM_ATTRS):
            attr_preds = out["attr_logits"][:, k, :].argmax(dim=1)
            self.val_attr_acc[k].update(attr_preds, attr_targets[:, k])

        self.log("val_loss", total, prog_bar=True, batch_size=inputs.size(0))
        self.log("val_ce", ce_loss, batch_size=inputs.size(0))
        self.log("val_attr_loss", attr_loss_mean, batch_size=inputs.size(0))
        return total

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        for k, name in enumerate(ATTR_ORDER):
            self.log(f"val_attr_acc_{name}", self.val_attr_acc[k].compute())
        self.val_acc.reset(); self.val_f1.reset()
        for k in range(_NUM_ATTRS):
            self.val_attr_acc[k].reset()

    def test_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        out = self(inputs)
        preds = out["logits"].argmax(dim=1)
        self.test_acc.update(preds, labels)
        self.test_f1.update(preds, labels)
        return None

    def on_test_epoch_end(self) -> None:
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.test_acc.reset(); self.test_f1.reset()
