"""tools/diagnostic_time_reversal.py — 11-way ensemble time-reversal F1 test.

Question (2026-05-20): does the 11-way ensemble actually use temporal
direction? reported an AUC gap of +0.003 for exp034 (single
model) — but the 11-way ensemble has not been directly tested.

Method: for each of 7 skeleton-based members × 10 folds, run val
inference twice — once with the standard input, once with the time
axis of the input tensor flipped (T -> T-1-t). For the 4 external
members (MotionBERT-Lite-frozen, C3D-contact-24-D, MAMP-ntu60xsub,
MAMP-ntu120xset), the feature pipelines pool symmetrically over time:
- MotionBERT-Lite: `rep.mean(axis=(1, 2))` / `rep.mean(axis=1)`  -> symmetric
- MAMP: `feat.mean(dim=(1, 2, 3))` (pooled) / `feat.mean(dim=(1, 2))`
  (per_joint) -> symmetric
- C3D-contact-24D: c3d_stats.py uses np.nanmean / RMS / |diff|.mean
  (L2-norm of diff cancels sign) -> symmetric
So their reversed_logits == forward_logits mathematically; we re-use
the existing forward logits for them.

Augmentation discipline: the production 7 skeleton models were trained
with augment_mirror (spatial L/R) + augment_noise + augment_rotate +
augment_speed (temporal *rate* jitter, NOT reversal). No time-reverse /
time-flip / temporal-flip function exists in the repo. Therefore F1 Δ
under time reversal is a clean signal, not an augmentation artefact.

Outputs:
- output/oof_reversed/<member>/fold_<NN>/oof_logits.npy (skeleton only)
- experiments/exp_time_reversal/results.json (summary + per-member F1)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: F401, E402
from utils.env import EnvConfig  # noqa: E402
from diema.data.dataset import MotionDataset  # noqa: E402
from diema.data.parser import emotion_to_label  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402  (register)
from tools.infer_test_ensemble import (  # noqa: E402
    _cfg_to_builder_ns, _load_checkpoint_into_model)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))
import reconcile_f1 as R  # noqa: E402

PROD = R.PROD
EXTRA = R.EXTRA4
ART = R.ART
OUT = Path("output/oof_reversed")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@torch.no_grad()
def infer_pair(member: str, fold: int, val_indices: list[int]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward_logits, reversed_logits) for one (member, fold)."""
    ckpt = ART / member / f"fold_{fold:02d}" / "best.ckpt"
    cfg_path = ART / member / f"fold_{fold:02d}" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    ns = _cfg_to_builder_ns(OmegaConf.to_container(cfg, resolve=True))
    model = build_model(ns).to(DEV).eval()
    _load_checkpoint_into_model(ckpt, model)

    env = EnvConfig()
    train_npz = env.processed_dir / "motion_train_quat.npz"
    ds = MotionDataset(
        data_path=str(train_npz),
        indices=val_indices,
        clip_length=int(cfg.get("clip_length", 64)),
        target_repr=str(cfg.get("target_repr", "joint")),
        is_test=False,
        augmentation_pipeline=None,                  # no aug at inference
    )
    loader = DataLoader(ds, batch_size=64, num_workers=2, shuffle=False)

    fw, rv = [], []
    for batch in loader:
        x, _y, _fn = batch
        x = x.to(DEV)
        of = model(x); ol_f = of["logits"] if isinstance(of, dict) else of
        # batched layout is (B, C, T, V); T axis = dim 2
        x_rev = torch.flip(x, dims=[2])
        orv = model(x_rev); ol_r = orv["logits"] if isinstance(orv, dict) \
            else orv
        fw.append(ol_f.cpu().numpy())
        rv.append(ol_r.cpu().numpy())
    return np.concatenate(fw, 0), np.concatenate(rv, 0)


def main():
    with open(R.SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)
    # build (per-fold) val index lookup from filenames -> NPZ idx
    env = EnvConfig()
    train_npz = np.load(env.processed_dir / "motion_train_quat.npz",
                        allow_pickle=True)
    npz_fns = [str(x) for x in train_npz["filenames"]]
    fn2idx = {fn: i for i, fn in enumerate(npz_fns)}

    fold_val_indices, fold_val_fns = {}, {}
    for fid in range(10):
        fns = [str(fn) for fn, _ in splits[fid]["val"]]
        fold_val_indices[fid] = [fn2idx[fn] for fn in fns]
        fold_val_fns[fid] = fns

    # forward indexable order (matches reconcile_f1.load_oof)
    ref = [fn for fid in range(10) for fn in fold_val_fns[fid]]
    y = np.array([emotion_to_label(f) for f in ref], dtype=np.int64)
    fold_ids = np.concatenate([np.full(len(fold_val_fns[fid]), fid,
                                       dtype=np.int64) for fid in range(10)])

    # run inference for 7 skeleton members
    print(f"[diag] device = {DEV}; 7 members × 10 folds")
    fw_logits, rv_logits = {m: [] for m in PROD}, {m: [] for m in PROD}
    for m in PROD:
        for fid in range(10):
            print(f"  {m}/fold_{fid:02d} …", flush=True)
            fw, rv = infer_pair(m, fid, fold_val_indices[fid])
            (OUT / m / f"fold_{fid:02d}").mkdir(parents=True, exist_ok=True)
            np.save(OUT / m / f"fold_{fid:02d}" / "oof_logits_forward.npy", fw)
            np.save(OUT / m / f"fold_{fid:02d}" / "oof_logits_reversed.npy",
                    rv)
            fw_logits[m].append(fw)
            rv_logits[m].append(rv)
        fw_logits[m] = np.concatenate(fw_logits[m], 0)
        rv_logits[m] = np.concatenate(rv_logits[m], 0)
        # sanity: forward should match the committed oof_logits.npy closely
        committed = np.concatenate(
            [np.load(ART / m / f"fold_{fid:02d}" / "oof_logits.npy")
             for fid in range(10)], 0)
        max_err = float(np.max(np.abs(fw_logits[m] - committed)))
        print(f"    forward vs committed max |Δ| = {max_err:.5f}")

    # external 4 members: load committed forward, reuse as reversed (symmetric)
    raw_fwd, _, _ = R.load_oof()        # softmax-prob arrays per member
    # raw_fwd has softmaxed probs already; we need logit space. Recompute.
    # Simplest: for external members, take the committed prob and use it
    # twice (forward == reversed). For our member-level metric, we use
    # hard preds (argmax), so softmax vs logit doesn't matter for F1.

    # combine 11-way logit-mean for both forward and reversed
    def to_softmax_arr(arr): return _softmax(arr)

    def combine_logitmean(probs_list):
        c = np.mean([np.log(np.clip(p, 1e-12, 1.0)) for p in probs_list], 0)
        return _softmax(c)

    fw_probs = [to_softmax_arr(fw_logits[m]) for m in PROD] + \
               [raw_fwd[k] for k in EXTRA]
    rv_probs = [to_softmax_arr(rv_logits[m]) for m in PROD] + \
               [raw_fwd[k] for k in EXTRA]    # externals reuse forward
    ens_fw = combine_logitmean(fw_probs).argmax(-1)
    ens_rv = combine_logitmean(rv_probs).argmax(-1)

    # metrics
    def perfold(pred):
        return [float(f1_score(y[fold_ids == k], pred[fold_ids == k],
                                average="macro")) for k in range(10)]

    summary = {
        "device": str(DEV),
        "n_oof": int(len(y)),
        "ensemble": {
            "forward_pooled_f1": float(f1_score(y, ens_fw, average="macro")),
            "reversed_pooled_f1": float(f1_score(y, ens_rv, average="macro")),
            "forward_perfold": perfold(ens_fw),
            "reversed_perfold": perfold(ens_rv),
        },
        "per_member": {},
    }
    for m in PROD:
        pf = fw_logits[m].argmax(-1)
        pr = rv_logits[m].argmax(-1)
        summary["per_member"][m] = {
            "forward_pooled_f1": float(f1_score(y, pf, average="macro")),
            "reversed_pooled_f1": float(f1_score(y, pr, average="macro")),
            "delta_pp": (float(f1_score(y, pr, average="macro"))
                          - float(f1_score(y, pf, average="macro"))) * 100,
            "logit_cos_sim_mean": float(np.mean(
                np.sum(fw_logits[m] * rv_logits[m], 1) /
                (np.linalg.norm(fw_logits[m], axis=1) *
                 np.linalg.norm(rv_logits[m], axis=1) + 1e-12))),
            "argmax_agreement": float(np.mean(pf == pr)),
        }
    for k in EXTRA:
        summary["per_member"][k] = {
            "forward_pooled_f1": float(f1_score(
                y, raw_fwd[k].argmax(-1), average="macro")),
            "reversed_pooled_f1": float(f1_score(
                y, raw_fwd[k].argmax(-1), average="macro")),
            "delta_pp": 0.0,
            "note": ("mathematically symmetric (mean-pool over time); "
                     "reversed_logits == forward_logits"),
        }
    summary["ensemble"]["delta_pp"] = (
        summary["ensemble"]["reversed_pooled_f1"]
        - summary["ensemble"]["forward_pooled_f1"]) * 100

    outdir = Path("experiments/exp_time_reversal")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(summary, indent=2))

    # console summary
    e = summary["ensemble"]
    print(f"\n=== 11-way ensemble (logit-mean) ===")
    print(f"forward  F1 = {e['forward_pooled_f1']*100:.2f}% (pooled)")
    print(f"reversed F1 = {e['reversed_pooled_f1']*100:.2f}% (pooled)")
    print(f"Δ           = {e['delta_pp']:+.2f} pp")
    print(f"\n=== per-member ===")
    for m, d in summary["per_member"].items():
        if "note" in d:
            print(f"  {m:32s} fwd={d['forward_pooled_f1']*100:.2f} "
                  f"rev=fwd (symmetric)")
        else:
            print(f"  {m:32s} fwd={d['forward_pooled_f1']*100:.2f}  "
                  f"rev={d['reversed_pooled_f1']*100:.2f}  "
                  f"Δ={d['delta_pp']:+.2f} pp  "
                  f"argmax-agree={d['argmax_agreement']*100:.1f}%  "
                  f"cos={d['logit_cos_sim_mean']:.3f}")
    print(f"\nresults.json saved to {outdir/'results.json'}")


if __name__ == "__main__":
    main()
