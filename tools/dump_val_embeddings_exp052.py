"""/8 companion: dump val-fold z_parts + labels + stems for exp052.

See: / experiments/exp052_gap_generated_rationale/run.py
Related:
- tools/dump_val_embeddings_exp051.py (global scenario sibling)
- tools/eval_motion_text_retrieval.py (consumes z_motion for retrieval metrics)
- tools/export_part_importance.py  (consumes z_parts_raw + part_mask)

Emits a richer .pt than exp051 so both downstream consumers share a single
provenance file::

    {
        "z_motion":    (N, D) float32   — mean of L2-normalized z_parts,
                                          re-normalized (retrieval probe uses this)
        "z_parts_raw": (N, P, D) float32 — pre-normalize projection per part
        "part_mask":   (N, P)    bool    — rationale-cache mask per (sample, part);
                                           False for unknown-stem rows (all-False)
        "labels":      (N,)      int64
        "stems":       list[str] length N
        "target_kind": "rationale"
        "exp_id":      <ckpt parent's parent dir>
    }

Usage (after exp052 a00 fold_0 trains and the rationale cache is built)::

    python tools/dump_val_embeddings_exp052.py \\
        --checkpoint output/artifacts/exp052_gap_generated_rationale_a00/fold_00/best.ckpt \\
        --rationale-cache output/rationale_text_cache.pt \\
        --fold 0 \\
        --output /tmp/exp052_a00_fold00_val.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--rationale-cache", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Cache built by tools/build_rationale_text_cache.py; feeds part_mask.",
)
@click.option("--fold", default=0, type=int, show_default=True)
@click.option("--num-folds", default=10, type=int, show_default=True)
@click.option(
    "--data-path", default="data/diema_challenge/processed/motion_train_quat.npz",
    show_default=True,
)
@click.option("--batch-size", default=64, type=int, show_default=True)
@click.option("--num-workers", default=4, type=int, show_default=True)
@click.option(
    "--output", "output_path", required=True, type=click.Path(dir_okay=False),
)
@click.option("--device", default=None)
def main(
    checkpoint: str, rationale_cache: str,
    fold: int, num_folds: int,
    data_path: str, batch_size: int, num_workers: int,
    output_path: str, device: str | None,
) -> None:
    import pybvh_ml

    from diema.data.collate import MotionDataModule
    from diema.data.splits import generate_lpo_splits
    from diema.features.rationale_text_cache import RationaleTextCache

    ckpt_path = Path(checkpoint)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or {}
    if "model" not in hparams:
        raise click.ClickException(
            "hparams['model'] not found — exp052 checkpoint should embed it "
            "(see PartAlignLightningModel / LightningModel save_hyperparameters)."
        )
    model = hparams["model"]
    state = ckpt["state_dict"]
    inner = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(inner, strict=True)
    model.eval()

    if device:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    click.echo(f"Model on device: {dev}")

    rat_cache = RationaleTextCache(rationale_cache)
    click.echo(
        f"Rationale cache: N={rat_cache.num_rows} rows, "
        f"P={rat_cache.num_parts}, D={rat_cache.embedding_dim}"
    )

    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)
    split_dict = splits[fold]
    click.echo(
        f"Fold {fold}: train={len(split_dict['train'])}, "
        f"val={len(split_dict['val'])}"
    )

    datamodule = MotionDataModule(
        data_path=str(data_path),
        split_dict=split_dict,
        clip_length=64,
        batch_size=batch_size,
        num_workers=num_workers,
        target_repr="6d",
        augmentation_pipeline=None,
        debug=False,
    )
    datamodule.setup(stage="fit")

    zm_all: list[torch.Tensor] = []
    zr_all: list[torch.Tensor] = []
    label_all: list[int] = []
    stem_all: list[str] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            x, y, stems = batch
            x = x.to(dev, non_blocking=True)
            out = model(x)
            if "z_parts" not in out or "z_parts_raw" not in out:
                raise click.ClickException(
                    "checkpoint does not produce 'z_parts' / 'z_parts_raw' "
                    "(wrong exp family — expected PartAlignConv1DTr_Model)."
                )
            # Motion → rationale retrieval: mean-pool L2-normalized per-part
            # vectors, then re-normalize. Matches the rationale target
            # aggregation in tools/eval_motion_text_retrieval.py.
            z_mean = torch.nn.functional.normalize(out["z_parts"].mean(dim=1), dim=1)
            zm_all.append(z_mean.cpu())
            zr_all.append(out["z_parts_raw"].cpu())
            label_all.extend(int(v) for v in y.tolist())
            stem_all.extend(str(s) for s in stems)

    z_motion = torch.cat(zm_all, dim=0)
    z_parts_raw = torch.cat(zr_all, dim=0)
    labels = torch.tensor(label_all, dtype=torch.long)

    # Per-sample part_mask from the rationale cache (False when stem absent).
    _, part_mask, _valid = rat_cache.embedding_for_batch(stem_all)

    torch.save({
        "z_motion": z_motion,
        "z_parts_raw": z_parts_raw,
        "part_mask": part_mask,
        "labels": labels,
        "stems": stem_all,
        "target_kind": "rationale",
        "exp_id": ckpt_path.parent.parent.name,
    }, out_path)
    click.echo(
        f"Saved: {out_path} (N={z_motion.shape[0]}, "
        f"D={z_motion.shape[1]}, parts={z_parts_raw.shape[1]})"
    )


if __name__ == "__main__":
    main()
