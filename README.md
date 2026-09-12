# diema-challenge

Emotion recognition from motion capture — the code behind our **Best Performance
Award** entry to the **DIEM-A Challenge at MMAC @ ACII 2026**.

The task is 12-class emotion classification from full-body motion-capture
recordings. This repository contains the model implementations, training and
feature-extraction code, ensembling and calibration tools, the explainability
(motion-to-text rationale) pipeline, and the scripts that produce the paper's
figures and tables.

> **The datasets are not included and cannot be redistributed.** See
> [Data availability](#data-availability) before you try to run anything.

---

## Data availability

This repository ships **code only**. The **DIEM-A** and **AIDE** datasets, and
**every artifact derived from them** — motion-capture recordings, extracted
skeletons and features, generated rationales, rendered motion videos, labels,
cross-validation splits, and out-of-fold predictions — are **deliberately
excluded** and **cannot be redistributed**.

The DIEM-A dataset is governed by a User License Agreement whose terms prohibit
redistribution of the data or of any derivative of it. Access to the data is
granted by the dataset providers only. To request access, please contact the
**DIEM-A / MMAC @ ACII 2026 challenge organizers**; researchers who wish to
reproduce our pipeline on their own copy of the data are also welcome to contact
the authors.

Because the data is absent, the training, feature-extraction, and figure scripts
here will not run end-to-end out of the box. They are provided so that the method
is fully inspectable and reproducible **once you have obtained the data through
the proper channel**. Point the code at your local copy with:

```bash
export DIEMA_DATA_ROOT=/path/to/your/diema_challenge
```

## Method overview

The submitted system is a logit-mean ensemble that combines:

- **Skeleton-action models** trained from scratch on the challenge data —
  ST-GCN++ (baseline), a Region-Aware Conv-Transformer (our best single model),
  CTR-GCN, SkateFormer, ProtoGCN, and Conformer / Squeezeformer sequence models.
- **External-pretraining transfer** — frozen features from motion / video
  backbones pretrained on large public corpora (MotionBERT, MAMP, PoseC3D),
  fused with the from-scratch models.
- **Explainability (ACII bonus task)** — a motion-to-text pipeline that produces
  per-part rationales and Laban-Movement-Analysis-inspired attributes, evaluated
  with saliency, counterfactual motion edits, and prototype explanation cards.

See [`NOTICE`](NOTICE) for attribution of the ported / adapted model
architectures.

## Results (development)

Macro-F1 and Accuracy are reported as **10-fold leave-performers-out (LPO)
per-fold mean ± SD**, the official challenge convention, on the development
(out-of-fold) split. Fusion is a mean of raw logits.

| System | Trainable params (M) | Macro-F1 (mean ± SD) | Accuracy (mean ± SD) |
| --- | --- | --- | --- |
| ST-GCN++ (reproduced baseline) | 1.41 | 25.73 ± 4.03 | 27.54 ± 3.92 |
| Best single (Region-Aware) | 1.03 | 30.05 ± 3.48 | 30.87 ± 3.43 |
| 7-way ensemble | 12.98 | 33.86 ± 2.92 | 34.68 ± 3.00 |
| **11-way logit-mean (submitted)** | 12.98 | **36.80 ± 4.00** | 37.40 ± 4.06 |

Numbers reproduce `paper/tables/table1_main_results.csv`. These are
development-set (out-of-fold) figures; see the paper for the full protocol and
confidence intervals.

## Results (hidden test set)

On the challenge's hidden test set (18 held-out performers, 1,944 clips; labels
kept by the organisers), the submitted 11-way ensemble scored **37.23% Macro-F1**
and **37.50% accuracy** on the organisers' final leaderboard
(<https://sites.google.com/view/mmac-acii-2026/program-results>) and received
the challenge's Best Performance Award. This is a single held-out score with no
fold-level spread; it sits inside the 36.80 ± 4.00% range of the development
result above.

## Repository structure

```
diema/            # core library: data, features, models, training, evaluation, explain
  ├─ data/        # mocap parsing, splits, datamodule
  ├─ features/    # skeleton conversions, kinematic / C3D features
  ├─ models/      # STGCN++, CTR-GCN, SkateFormer, ProtoGCN, Conformer, Squeezeformer, ...
  ├─ training/    # Lightning training loops, losses, dual-head trainer
  ├─ evaluation/  # metrics, calibration
  └─ explain/     # saliency, counterfactuals, motion-to-text rationale
tools/            # CLI utilities: build splits, caches, ensembling, feature extraction, rationale generation
experiments/      # per-experiment run scripts and configs (exp001 … exp099)
configs/          # shared Hydra / prompt configs
tests/            # pytest suite
paper/            # figure and table generation scripts (data-dependent)
utils/            # environment/config, IO, metrics, seeding
```

## Installation

Python ≥ 3.10.

```bash
# with conda (the environment used for development is named "acii2026")
conda create -n acii2026 python=3.10 && conda activate acii2026

pip install -e ".[dev]"
# optional mocap toolchain (C3D parsing, aitviewer rendering, rationale generation):
pip install -e ".[mocap]"
```

> **Note.** The BVH preprocessing path depends on `pybvh` / `pybvh-ml`, which are
> the authors' internal mocap tooling and are not published on PyPI. They are
> only needed to convert raw motion capture into the model inputs; the model,
> training, and evaluation code do not require them. Contact the authors if you
> need this preprocessing step.

Copy `.env.example` to `.env` and fill in any values you need (e.g. an API key
for the optional motion-to-text rationale step). `.env` is git-ignored.

## Usage

Most tools are Click CLIs; run with `--help` for options. A typical flow, once
`DIEMA_DATA_ROOT` points at a local copy of the data:

```bash
python -m tools.build_splits            # build the LPO cross-validation splits
python experiments/exp034_regionaware_convtr/run.py   # train the best single model
python -m tools.calibrate_ensemble      # fuse member logits
```

## Testing

```bash
pytest -q
```

Some tests require the dataset and will be skipped or fail without it; the rest
exercise the parsers, model shapes, losses, and utility code with synthetic
inputs.

## Citation

If you use this code, please cite the accompanying paper (DIEM-A Challenge,
MMAC @ ACII 2026). A preprint is on arXiv at <https://arxiv.org/abs/2609.02510>,
and the talk page with slides is at <https://nawta.github.io/mmac2026/>. A
BibTeX entry will be added here once the proceedings are published.

## License

Code in this repository is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)). Ported and adapted third-party model architectures retain
their upstream licenses; see [`NOTICE`](NOTICE) for details. The datasets and all
data-derived artifacts are **not** covered by this license and are **not**
included (see [Data availability](#data-availability)).
