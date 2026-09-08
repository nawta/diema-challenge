""" training integration for content/style dual-head models.

:class:`DualHeadLightningModel` is a thin subclass of
:class:`diema.training.train.LightningModel` that knows how to route
performer labels to the style / adversarial heads of a
:class:`diema.models.dual_head.DualHead_Model` backbone.

Loss:

    total = CE(emo_logits, y_emotion)
          + loss_weight_style  * CE(style_logits, y_performer)
          + loss_weight_adv    * CE(adv_logits,   y_performer)     # GRL applied inside model
          + loss_weight_ortho  * orthogonality_loss(z_content, z_style)

The performer label is derived from the input filename via the
pre-computed ``filename_to_perf_idx`` map (built from
``train_data.csv``). Samples whose filename is not in the map fall back
to performer index 0 and are EXCLUDED from the style/adv loss via a
mask — this keeps val / test loops happy when (country, actor) IDs are
unknown.

Related: diema/models/dual_head/dual_head_model.py,
         diema/training/grl.py, diema/training/losses.py::orthogonality_loss
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diema.training.train import LightningModel
from diema.training.losses import orthogonality_loss


_FILENAME_PREFIX_RE = re.compile(r"^([A-Z]{2})_(\d+)")


def build_filename_to_performer_index(
    train_csv_path: str | Path,
) -> tuple[dict[str, int], int]:
    """Parse ``train_data.csv`` into a ``filename → performer index`` map.

    The performer identity is the ``(country, actor_id)`` pair — e.g.
    ``(JP, 6)`` and ``(TW, 6)`` are different performers. The map is
    keyed by the filename stem (``original_name`` column in the CSV),
    which matches the filenames embedded in the NPZ file.

    Robustness notes:

    * ``actor_id`` is forced to ``int`` and ``country`` to ``str`` before
      sorting so that a stray string-typed column cannot produce
      lexicographic ordering (``"10" < "2"``) or mixed-type sort errors.
    * On DIEM-A the NPZ filenames exactly equal ``original_name``
      (verified: 7992 / 7992 match). If that contract ever changes
      (e.g., augmentation suffix ``_augNN``), callers should use the
      companion :func:`parse_performer_pair_from_filename` /
      :func:`build_pair_to_performer_index` helpers to map by
      ``(country, actor_id)`` directly instead of by filename.

    Args:
        train_csv_path: path to ``train_data.csv``.

    Returns:
        Tuple ``(filename_to_idx, num_performers)`` where ``num_performers``
        equals the number of unique ``(country, actor_id)`` pairs.
    """
    import pandas as pd

    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"train CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"original_name", "country", "actor_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_data.csv missing columns: {missing}")

    # Force types to guarantee stable sort order. Without this a
    # string-typed actor_id column would sort lexicographically
    # ("10" < "2") and shuffle the index map across runs.
    countries = df["country"].astype(str)
    actor_ids = pd.to_numeric(df["actor_id"], errors="raise").astype(int)
    original_names = df["original_name"].astype(str)

    pairs = sorted({(c, int(a)) for c, a in zip(countries, actor_ids)})
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

    filename_to_idx: dict[str, int] = {}
    for name, c, a in zip(original_names, countries, actor_ids):
        filename_to_idx[name] = pair_to_idx[(c, int(a))]
    return filename_to_idx, len(pairs)


_PAIR_RE = re.compile(r"^([A-Z]{2})_(\d+)")


def parse_performer_pair_from_filename(filename: str) -> tuple[str, int] | None:
    """Return ``(country, actor_id)`` parsed from a DIEM-A filename.

    DIEM-A filenames follow the pattern
    ``{country}_{actor_id:02d}_{emotion}_{scenario}_{intensity}[...]``
    where the country is a 2-letter ISO-like prefix (``JP`` / ``TW``)
    and the actor id is a zero-padded integer. Anything after the
    actor id is ignored, so the helper is robust to optional aug /
    suffix variants the preprocessing pipeline might introduce later.
    Returns ``None`` if the filename does not match.
    """
    stem = Path(filename).stem
    m = _PAIR_RE.match(stem)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def build_pair_to_performer_index(
    train_csv_path: str | Path,
) -> tuple[dict[tuple[str, int], int], int]:
    """Return ``({(country, actor_id): idx}, num_performers)``.

    Same backing data as :func:`build_filename_to_performer_index` but
    keyed directly by the parsed pair, making it robust to any filename
    decoration downstream. Useful if you want to look up the performer
    index from a sample without depending on CSV join keys.
    """
    filename_to_idx, num = build_filename_to_performer_index(train_csv_path)
    pair_to_idx: dict[tuple[str, int], int] = {}
    for name, idx in filename_to_idx.items():
        pair = parse_performer_pair_from_filename(name)
        if pair is not None:
            pair_to_idx[pair] = idx
    return pair_to_idx, num


def _perf_idx_from_filename(
    filename: str,
    filename_to_idx: dict[str, int],
    pair_to_idx: dict[tuple[str, int], int] | None = None,
) -> int | None:
    """Return the performer index for a filename, or None if unknown.

    Lookup order:
      1. Literal filename against ``filename_to_idx``.
      2. ``Path(filename).stem`` (strips a single extension).
      3. ``(country, actor_id)`` parsed from the filename against
         ``pair_to_idx`` — this is robust to augmentation suffixes,
         timestamp decorations, and other downstream filename changes.

    Args:
        filename: NPZ sample filename as embedded in the preprocessed
            data.
        filename_to_idx: map from literal filename to performer index.
        pair_to_idx: optional fallback map from ``(country, actor_id)``
            to performer index. When provided, unknown filenames will
            be resolved via prefix parsing.

    Returns:
        The performer index or ``None`` if none of the lookups match.
    """
    if filename in filename_to_idx:
        return filename_to_idx[filename]
    stem = Path(filename).stem
    if stem in filename_to_idx:
        return filename_to_idx[stem]
    if pair_to_idx is not None:
        pair = parse_performer_pair_from_filename(filename)
        if pair is not None and pair in pair_to_idx:
            return pair_to_idx[pair]
    return None


class DualHeadLightningModel(LightningModel):
    """Lightning subclass that trains a :class:`DualHead_Model` with
    the joint loss (emotion + style + adversarial + ortho).

    This reuses the parent class's optimizer / scheduler / metrics /
    Gaussian-noise / mixup / cutmix plumbing, and *overrides* the loss
    computation in ``training_step`` to take advantage of the dual-head
    forward outputs. The validation / test / predict steps fall back to
    the parent implementation, which only looks at ``out["logits"]``.

    Args:
        filename_to_perf_idx: map from sample filename (string) to
            performer index (int). Unknown filenames are excluded from
            the style / adv losses via a sample mask.
        num_performer: number of distinct performer classes (used only
            to sanity-check indices). Pass ``0`` to disable the
            performer-side losses entirely.
        loss_weight_style: multiplier on CE(style_logits, y_performer).
            Default 1.0.
        loss_weight_adv: multiplier on CE(adv_logits, y_performer) —
            NOTE that the actual adversarial strength is the product of
            this weight AND the GRL lambda set inside the DualHead_Model
            constructor. Default 1.0.
        loss_weight_ortho: multiplier on ``orthogonality_loss(z_content,
            z_style)``. Default 0.01.
        All other kwargs are forwarded to :class:`LightningModel`.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        num_class: int,
        filename_to_perf_idx: dict[str, int],
        num_performer: int,
        pair_to_perf_idx: dict[tuple[str, int], int] | None = None,
        loss_weight_style: float = 1.0,
        loss_weight_adv: float = 1.0,
        loss_weight_ortho: float = 0.01,
        **lightning_kwargs,
    ):
        super().__init__(
            model=model,
            base_lr=base_lr,
            num_class=num_class,
            **lightning_kwargs,
        )
        if num_performer < 0:
            raise ValueError(f"num_performer must be >= 0, got {num_performer}")
        # mixup/cutmix are silently ignored by our override;
        # warn loudly so configs don't get dropped on the floor.
        if getattr(self, "mixup_alpha", 0.0) > 0.0:
            raise ValueError(
                "DualHeadLightningModel overrides training_step and does "
                "not honor mixup_alpha — pass mixup_alpha=0.0 or use the "
                "plain LightningModel."
            )
        if getattr(self, "cutmix_alpha", 0.0) > 0.0:
            raise ValueError(
                "DualHeadLightningModel overrides training_step and does "
                "not honor cutmix_alpha — pass cutmix_alpha=0.0 or use "
                "the plain LightningModel."
            )
        self.filename_to_perf_idx = dict(filename_to_perf_idx)
        self.pair_to_perf_idx = dict(pair_to_perf_idx) if pair_to_perf_idx else {}
        self.num_performer = int(num_performer)
        self.loss_weight_style = float(loss_weight_style)
        self.loss_weight_adv = float(loss_weight_adv)
        self.loss_weight_ortho = float(loss_weight_ortho)
        # Separate loss for performer classification (no label smoothing
        # from the emotion loss config) — a plain CE is safer for the
        # adversarial path.
        self.perf_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _perf_labels_and_mask(
        self,
        filenames: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (perf_labels, valid_mask) for a batch of filenames.

        ``perf_labels[i]`` is 0 if the filename was unknown (so the CE
        does not index-error); ``valid_mask[i]`` is 1.0 for known
        filenames and 0.0 for unknown. Callers should multiply the
        per-sample loss by ``valid_mask`` before reducing.

        Implementation note: labels and mask are built on CPU as plain
        Python lists and promoted to a single CUDA tensor each via one
        ``torch.tensor(..., device=device)`` call. This avoids N tiny
        CUDA scalar writes per training step.
        """
        N = len(filenames)
        labels_cpu: list[int] = [0] * N
        mask_cpu: list[float] = [0.0] * N
        pair_map = self.pair_to_perf_idx if self.pair_to_perf_idx else None
        for i, fn in enumerate(filenames):
            idx = _perf_idx_from_filename(fn, self.filename_to_perf_idx, pair_map)
            if idx is None or idx < 0 or idx >= max(self.num_performer, 1):
                continue
            labels_cpu[i] = idx
            mask_cpu[i] = 1.0
        labels = torch.tensor(labels_cpu, dtype=torch.long, device=device)
        mask = torch.tensor(mask_cpu, dtype=torch.float32, device=device)
        return labels, mask

    def training_step(self, batch, batch_idx):
        inputs, labels, filenames = batch
        # Gaussian-noise / mixup / cutmix hooks come from the parent
        # class — since we override training_step entirely, we
        # reimplement the bare minimum (emotion-only CE + dual-head
        # aux losses). This avoids inheriting the nested try/except
        # and keeps the dual-head path easy to reason about.

        # Gaussian noise, if enabled, applied around the
        # forward call so the dual-head branches also train under
        # perturbed weights.
        self._noise_saved_weights = self._apply_gaussian_weight_noise()
        try:
            out = self(inputs)
            if "logits" not in out:
                raise RuntimeError(
                    "DualHeadLightningModel expects the model to return a "
                    "dict with a 'logits' key (plus optional z_content / "
                    "z_style / style_logits / adv_logits)."
                )
            emo_logits = out["logits"]
            L_emo = self.loss_fn(emo_logits, labels)
            total = L_emo
            self.log("train_loss_emo", L_emo, prog_bar=True)

            if self.num_performer > 0 and (
                "style_logits" in out or "adv_logits" in out
            ):
                perf_labels, mask = self._perf_labels_and_mask(
                    filenames, inputs.device
                )
                # Keep the "valid sample count" as a tensor so we do not
                # force a CUDA->CPU sync every step; clamp_min(1) keeps
                # the divisor non-zero when every filename is unknown
                # (the numerator is also zero in that case, so the loss
                # contribution is exactly zero either way).
                n_valid = mask.sum().clamp_min(1.0)
                if "style_logits" in out and self.loss_weight_style > 0:
                    per_sample = self.perf_loss_fn(
                        out["style_logits"], perf_labels
                    )
                    L_style = (per_sample * mask).sum() / n_valid
                    total = total + self.loss_weight_style * L_style
                    self.log("train_loss_style", L_style)
                if "adv_logits" in out and self.loss_weight_adv > 0:
                    per_sample = self.perf_loss_fn(
                        out["adv_logits"], perf_labels
                    )
                    L_adv = (per_sample * mask).sum() / n_valid
                    total = total + self.loss_weight_adv * L_adv
                    self.log("train_loss_adv", L_adv)

            if (
                self.loss_weight_ortho > 0
                and "z_content" in out
                and "z_style" in out
            ):
                L_ortho = orthogonality_loss(out["z_content"], out["z_style"])
                total = total + self.loss_weight_ortho * L_ortho
                self.log("train_loss_ortho", L_ortho)

            self.log("train_loss", total, prog_bar=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=False)
        except BaseException:
            self._restore_noise_weights()
            raise
        return total
