""" build emotion-class / performer motion prototypes in feature space.

See: (exp058)
Related:
- tools/explain_part_masking.py (shares _build_val_loader + checkpoint helpers)
- diema/models/conv1d_transformer/conv1d_transformer_model.py (feature_dim=dim)
- diema/data/parser.py (EMOTION_TO_IDX, filename → performer)

For a trained experiment, this tool:

1. Rebuilds the model from ``config.yaml`` and loads best.ckpt.
2. Runs a single forward pass over the fold's TRAIN split, collecting the
   pre-classifier features (model output "features" key — all registered
   backbones expose this by contract).
3. Computes two prototype sets:
   - **Class prototype** (per emotion, method 'mean'): centroid of the
     training features for that class.
   - **Class medoid** (method 'medoid'): the actual sample whose features
     are closest to the class centroid — useful for qualitative paper
     examples ("the most typical anger motion").
4. Computes the **performer prototype** (per performer actor_id) as the
   centroid of all that performer's training clips. ``sample − performer``
   gives the emotion-only direction; that residual is the part of the
   feature that cannot be explained by the performer's resting style.
5. Writes ``output/explain/prototypes_<exp>.pt`` with class_proto,
   performer_proto, the medoid indices / filenames, and the
   (sample, class) deviation table so localised-feedback and
    reports can consume it without re-inference.

Usage::

    python tools/build_motion_prototypes.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --fold 0 \\
        --device cuda
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: E402
from utils.env import EnvConfig  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402
from diema.data.parser import EMOTION_TO_IDX, parse_filename  # noqa: E402
from diema.data.splits import generate_lpo_splits  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402


NUM_CLASS = 12


def _build_train_loader(cfg_dict: dict, fold: int, num_folds: int,
                        batch_size: int = 128) -> torch.utils.data.DataLoader:
    """Construct a deterministic TRAIN-fold DataLoader (no augmentation)."""
    from diema.data.dataset import MotionDataset
    from torch.utils.data import DataLoader

    env = EnvConfig()
    data_path = cfg_dict.get("data_path", "")
    path = Path(data_path) if data_path else env.processed_dir / "motion_train_quat.npz"
    if not path.is_absolute():
        path = env.project_root / path
    if not path.exists():
        raise FileNotFoundError(f"train NPZ not found: {path}")

    preprocessed = pybvh_ml.load_preprocessed(str(path))
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)
    split_dict = splits[fold]
    train_indices = [idx for _, idx in split_dict["train"]]

    ds = MotionDataset(
        data_path=str(path),
        indices=train_indices,
        clip_length=cfg_dict.get("clip_length", 64),
        target_repr=cfg_dict.get("target_repr", "6d"),
        is_test=True,  # disable augmentation even on the TRAIN split
        augmentation_pipeline=None,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)


@torch.no_grad()
def _collect_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return ``(features (N, D), labels (N,), filenames)``.

    The features come from the backbone's ``"features"`` output key — the
    pre-classifier tensor all registered BaseModel subclasses expose. We
    concatenate on CPU to keep GPU memory bounded.
    """
    model = model.to(device).eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    fnames: list[str] = []
    for batch in loader:
        x, y, fn = batch
        x = x.to(device)
        out = model(x)
        if "features" not in out:
            raise RuntimeError(
                "model output is missing 'features'; all BaseModel subclasses "
                "are required to expose this by the model I/O contract"
            )
        feats.append(out["features"].cpu())
        labels.append(y.cpu())
        fnames.extend(fn)
    return torch.cat(feats, 0), torch.cat(labels, 0), fnames


def _build_class_prototypes(features: torch.Tensor, labels: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """(class_mean (C, D), class_medoid_idx (C,)).

    ``class_medoid_idx[c]`` points into the ``features`` rows for class ``c``.
    We use Euclidean distance to the class centroid to pick the medoid;
    cosine would give the same ranking on L2-normalised features but the
    backbones expose un-normalised features so Euclidean is the honest pick.
    """
    D = features.size(1)
    mean = torch.zeros(NUM_CLASS, D)
    medoid = torch.full((NUM_CLASS,), -1, dtype=torch.long)
    for c in range(NUM_CLASS):
        mask = labels == c
        if not mask.any():
            continue
        class_feat = features[mask]
        mu = class_feat.mean(dim=0)
        mean[c] = mu
        # Index within the full features tensor of the medoid.
        dist = (class_feat - mu).norm(dim=1)
        relative_idx = int(dist.argmin().item())
        absolute_idx = int(mask.nonzero(as_tuple=True)[0][relative_idx].item())
        medoid[c] = absolute_idx
    return mean, medoid


def _build_performer_prototypes(
    features: torch.Tensor,
    filenames: list[str],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return ``(perf_mean (P, D), perf_id_to_idx)`` keyed by ``f"{country}_{actor_id:02d}"``.

    Parses each filename with :func:`diema.data.parser.parse_filename`. Rows
    whose filename fails parsing are silently skipped (the feature is dropped
    from every prototype). Empty skip set is common because training data
    follows the strict ``JP_06_...`` convention.
    """
    D = features.size(1)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, fn in enumerate(filenames):
        try:
            info = parse_filename(fn)
        except Exception:
            continue
        key = f"{info.nationality}_{int(info.performer_num):02d}"
        buckets[key].append(i)
    perf_ids = sorted(buckets.keys())
    perf_mean = torch.zeros(len(perf_ids), D)
    for p, key in enumerate(perf_ids):
        idx = torch.tensor(buckets[key], dtype=torch.long)
        perf_mean[p] = features[idx].mean(dim=0)
    return perf_mean, {k: i for i, k in enumerate(perf_ids)}


def _compute_deviation_table(
    features: torch.Tensor,
    labels: torch.Tensor,
    filenames: list[str],
    class_mean: torch.Tensor,
    perf_mean: torch.Tensor,
    perf_id_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """Per-sample distances to class and performer prototypes.

    Returns a dict with:
    - ``class_distance``: (N,) distance of each sample to its own class prototype
    - ``class_distances_all``: (N, C) distance to every class prototype (for
      paper prototype-nearest-neighbour figures)
    - ``performer_distance``: (N,) distance to own performer prototype
    - ``residual_norm``: (N,) ||(sample − perf_mean) − (class_mean − perf_mean)||
      — the sample's departure from "their own performer emoting this class"
    """
    N = features.size(0)
    class_distances_all = torch.cdist(features, class_mean)  # (N, C)
    class_distance = class_distances_all.gather(1, labels.view(-1, 1)).squeeze(1)

    own_perf_idx = torch.full((N,), -1, dtype=torch.long)
    for i, fn in enumerate(filenames):
        try:
            info = parse_filename(fn)
        except Exception:
            continue
        key = f"{info.nationality}_{int(info.performer_num):02d}"
        own_perf_idx[i] = perf_id_to_idx.get(key, -1)

    perf_for_sample = torch.where(
        own_perf_idx >= 0,
        own_perf_idx,
        torch.zeros_like(own_perf_idx),
    )
    performer_proto = perf_mean[perf_for_sample]
    performer_distance = (features - performer_proto).norm(dim=1)
    # Residual == how far from the "performer-style corrected class centroid"
    class_for_sample = class_mean[labels]
    expected = performer_proto + (class_for_sample - performer_proto)  # == class_for_sample
    residual = features - expected
    residual[own_perf_idx < 0] = 0  # unparsed rows get a zero residual
    residual_norm = residual.norm(dim=1)

    return {
        "class_distance": class_distance.numpy(),
        "class_distances_all": class_distances_all.numpy(),
        "performer_distance": performer_distance.numpy(),
        "residual_norm": residual_norm.numpy(),
        "own_perf_idx": own_perf_idx.numpy(),
    }


@click.command()
@click.option("--experiment", required=True)
@click.option("--fold", default=0, type=int)
@click.option("--num-folds", default=10, type=int)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--output-path", default=None,
              help="Path to .pt output; default output/explain/prototypes_<exp>_fold<NN>.pt")
def main(experiment: str, fold: int, num_folds: int, device: str,
         batch_size: int, output_path: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    exp_dir = env.artifacts_dir / experiment
    fold_dir = exp_dir / f"fold_{fold:02d}"
    ckpt = fold_dir / "best.ckpt"
    cfg_path = fold_dir / "config.yaml"
    if not (ckpt.exists() and cfg_path.exists()):
        raise click.ClickException(
            f"missing best.ckpt or config.yaml in {fold_dir}"
        )
    cfg_dict = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
    builder_ns = _cfg_to_builder_ns(cfg_dict)
    model = build_model(builder_ns)
    _load_checkpoint_into_model(ckpt, model)

    loader = _build_train_loader(cfg_dict, fold=fold, num_folds=num_folds,
                                 batch_size=batch_size)
    click.echo(f"Collecting features from fold_{fold:02d} TRAIN split...")
    features, labels, filenames = _collect_features(model, loader, dev)
    click.echo(f"  features: {tuple(features.shape)}  labels: {tuple(labels.shape)}")

    class_mean, class_medoid = _build_class_prototypes(features, labels)
    perf_mean, perf_id_to_idx = _build_performer_prototypes(features, filenames)
    click.echo(f"  class prototypes: {tuple(class_mean.shape)}")
    click.echo(f"  performer prototypes: {tuple(perf_mean.shape)} ({len(perf_id_to_idx)} performers)")

    devs = _compute_deviation_table(
        features, labels, filenames, class_mean, perf_mean, perf_id_to_idx,
    )

    out = {
        "experiment": experiment,
        "fold": fold,
        "class_mean": class_mean,
        "class_medoid_idx": class_medoid,
        "medoid_filenames": [filenames[int(i)] if int(i) >= 0 else ""
                             for i in class_medoid],
        "performer_mean": perf_mean,
        "performer_id_to_idx": perf_id_to_idx,
        "deviation": devs,
        "feature_dim": int(features.size(1)),
        "num_samples": int(features.size(0)),
    }
    output = (
        Path(output_path) if output_path
        else env.project_root / "output/explain" /
             f"prototypes_{experiment}_fold{fold:02d}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    click.echo(f"Saved: {output}")

    # Print a quick sanity summary.
    click.echo("\nMedoid per emotion (closest real train clip to centroid):")
    from diema.data.parser import IDX_TO_EMOTION
    for c in range(NUM_CLASS):
        if out["class_medoid_idx"][c] >= 0:
            click.echo(
                f"  {IDX_TO_EMOTION[c]:>10}: {out['medoid_filenames'][c]}"
            )


if __name__ == "__main__":
    main()
