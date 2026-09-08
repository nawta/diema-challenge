""" companion: dump val-fold z_text + labels + stems for exp051.

See: (exp054 retrieval probe) / experiments/exp051_tmr_scenario_global/run.py
Related:
- tools/eval_motion_text_retrieval.py (consumer of the .pt produced here)
- diema/models/conv1d_transformer/textalign_conv1d_tr_model.py

Reads an exp051 Lightning checkpoint (``hparams["model"]`` holds the pickled
:class:`TextAlignConv1DTr_Model`), reconstructs the val fold's dataloader,
runs the model in eval mode to collect the motion→text projection for each
clip, and writes a .pt in the schema expected by
:mod:`tools.eval_motion_text_retrieval`:

    {
        "z_motion":    (N, D) float32   — L2-normalized z_text
        "labels":      (N,)   int64     — emotion class ids
        "stems":       list[str] length N
        "target_kind": "scenario"
        "exp_id":      <ckpt parent dir>
    }

The tool is exp051-specific because the backbone instantiation and val
dataloader assembly are 1:1 inherited from experiments/exp051_tmr_scenario_global/run.py.
 exp052 will ship its own dumper with the analogous shape
(z_motion = z_parts.mean(dim=1)).

Usage::

    python tools/dump_val_embeddings_exp051.py \\
        --checkpoint output/artifacts/exp051_tmr_scenario_global_a00/fold_00/best.ckpt \\
        --fold 0 \\
        --output /tmp/exp051_a00_fold00_val.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
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
@click.option("--device", default=None, help="cuda / cuda:0 / cpu. Default: auto.")
def main(
    checkpoint: str, fold: int, num_folds: int,
    data_path: str, batch_size: int, num_workers: int,
    output_path: str, device: str | None,
) -> None:
    import pybvh_ml

    from diema.data.collate import MotionDataModule
    from diema.data.splits import generate_lpo_splits

    ckpt_path = Path(checkpoint)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or {}
    if "model" not in hparams:
        raise click.ClickException(
            "hparams['model'] not found — checkpoint does not embed the model "
            "object. exp051 a00 fold_0 should include it; re-check ckpt source."
        )
    model = hparams["model"]
    # State-dict keys live under "model." due to Lightning wrapping.
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

    # Build val-fold dataloader (mirrors experiments/exp051_.../run.py).
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
    # MotionDataModule.setup("fit") creates both train and val datasets; the
    # "validate" stage is not recognized (only "fit" and "test"/"predict").
    datamodule.setup(stage="fit")

    z_all: list[torch.Tensor] = []
    label_all: list[int] = []
    stem_all: list[str] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            x, y, stems = batch
            x = x.to(dev, non_blocking=True)
            out = model(x)
            if "z_text" not in out:
                raise click.ClickException(
                    "model output does not include 'z_text' — wrong exp family"
                )
            # Already L2-normalized inside the model, but re-normalize to be safe
            # in case a future variant drops the F.normalize in forward.
            z = torch.nn.functional.normalize(out["z_text"], dim=1)
            z_all.append(z.cpu())
            label_all.extend(int(v) for v in y.tolist())
            stem_all.extend(str(s) for s in stems)

    z_motion = torch.cat(z_all, dim=0)
    labels = torch.tensor(label_all, dtype=torch.long)
    torch.save({
        "z_motion": z_motion,
        "labels": labels,
        "stems": stem_all,
        "target_kind": "scenario",
        "exp_id": ckpt_path.parent.parent.name,
    }, out_path)
    click.echo(
        f"Saved: {out_path} "
        f"(N={z_motion.shape[0]}, D={z_motion.shape[1]})"
    )


if __name__ == "__main__":
    main()
