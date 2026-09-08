"""exp086 PoseC3D frozen-probe — 3-fold smoke + (if PASS) 10-fold + ensemble.

 gates:
  - Smoke (3-fold LPO): mean Macro-F1 ≥ 24.0%
  - OOF: vs best current ensemble baseline Δ ≥ +0.20pp
  - Per-class (single add): ≤4 classes regress >0.30pp
  - Per-class (fusion final): ≤2 classes regress >0.50pp

Baselines compared: 7-way (33.72%), 9-way (exp083, 34.31%),
11-way (exp085 9+both MAMP, 35.78% — current best). PoseC3D as a
heatmap-3D-CNN is the most architecturally orthogonal source so far.

PoseC3D features are pooled-only (512-D, ResNet3dSlowOnly global avg pool);
no per_joint variant (it's a CNN, not a token model).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

MEMBERS = [
    "exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
    "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00",
]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fold_idx(splits, filenames, fid):
    f = splits[fid]
    pos = {str(fn): i for i, fn in enumerate(filenames)}
    tr = np.array([pos[str(it[0])] for it in f["train"] if str(it[0]) in pos], dtype=np.int64)
    va = np.array([pos[str(it[0])] for it in f["val"] if str(it[0]) in pos], dtype=np.int64)
    return tr, va


def align(logits, fns, ref):
    m = {str(fn): i for i, fn in enumerate(fns)}
    out = np.zeros((len(ref), logits.shape[1]), dtype=np.float32)
    for i, fn in enumerate(ref):
        out[i] = logits[m[str(fn)]]
    return softmax(out)


@click.command()
@click.option("--features-npz", required=True)
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--mb-logits", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy")
@click.option("--mb-fns", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy")
@click.option("--c3d-logits", default="output/predictions/c3d_contact_only/oof_logits.npy")
@click.option("--c3d-fns", default="output/predictions/c3d_contact_only/oof_filenames.npy")
@click.option("--mamp1", default="output/predictions/mamp_ntu60xsub/oof_logits.npy")
@click.option("--mamp1-fns", default="output/predictions/mamp_ntu60xsub/oof_filenames.npy")
@click.option("--mamp2", default="output/predictions/mamp_ntu120xset/oof_logits.npy")
@click.option("--mamp2-fns", default="output/predictions/mamp_ntu120xset/oof_filenames.npy")
@click.option("--tag", default="ntu60xsub")
@click.option("--output-dir", default="experiments/exp086_posec3d_frozen_probe")
@click.option("--force-fusion/--gate-only", default=False,
              help="run 10-fold + fusion even if smoke ABORTs (exp083/072 "
                   "orthogonality integration search; standalone verdict "
                   "stays NO-GO)")
def main(features_npz, splits_pkl, artifacts_root, mb_logits, mb_fns, c3d_logits,
         c3d_fns, mamp1, mamp1_fns, mamp2, mamp2_fns, tag, output_dir, force_fusion):
    t0 = time.time()
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([emotion_to_label(str(fn)) for fn in filenames], dtype=np.int64)
    feats = np.asarray(npz["features_pooled"], dtype=np.float32)
    print(f"[exp086:{tag}] features {feats.shape}, "
          f"mean-var/dim {feats.var(axis=0).mean():.4f}, "
          f"dead {(feats.std(axis=0)<1e-4).sum()}/{feats.shape[1]}")
    if np.isnan(feats).any() or np.isinf(feats).any():
        print("ABORT: NaN/Inf"); sys.exit(2)

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    # 3-fold smoke sweep (pooled only) × C
    smoke = []
    for c_reg in (0.1, 1.0, 10.0):
        f1s = []
        for fid in (0, 1, 2):
            tr, va = fold_idx(splits, filenames, fid)
            sc = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1,
                                     solver="lbfgs", random_state=42)
            clf.fit(sc.transform(feats[tr]), labels[tr])
            yp = clf.predict(sc.transform(feats[va]))
            f1s.append(f1_score(labels[va], yp, average="macro"))
        smoke.append({"C": c_reg, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))})
        print(f"[smoke] C={c_reg:<5} 3-fold {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    best = max(smoke, key=lambda s: s["mean"])
    smoke_decision = "PASS" if best["mean"] >= 0.24 else "ABORT"
    print(f"[smoke] BEST C={best['C']} {best['mean']*100:.2f}% gate 24.0% → {smoke_decision}")

    out = {"tag": tag, "smoke": smoke, "smoke_best": best,
           "smoke_decision": smoke_decision, "gate": 0.24,
           "standalone_verdict": "GO" if smoke_decision == "PASS" else "NO-GO"}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if smoke_decision == "ABORT" and force_fusion:
        print("[exp086] smoke ABORT but --force-fusion: running integration "
              "search (exp083/exp072 precedent; standalone stays NO-GO)")
    if smoke_decision == "PASS" or force_fusion:
        # 10-fold OOF at best C
        c_reg = best["C"]
        NUM = 12
        oof = np.zeros((len(feats), NUM), dtype=np.float32)
        ff = []
        for fid in range(10):
            tr, va = fold_idx(splits, filenames, fid)
            sc = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(C=c_reg, max_iter=2000, n_jobs=-1,
                                     solver="lbfgs", random_state=42)
            clf.fit(sc.transform(feats[tr]), labels[tr])
            pr = clf.predict_proba(sc.transform(feats[va]))
            for j, cls in enumerate(clf.classes_):
                oof[va, int(cls)] = np.log(np.clip(pr[:, j], 1e-12, 1.0))
            ff.append(f1_score(labels[va], clf.predict(sc.transform(feats[va])), average="macro"))
        pred_dir = Path(f"output/predictions/posec3d_{tag}")
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / "oof_logits.npy", oof)
        np.save(pred_dir / "oof_filenames.npy", filenames)
        print(f"[10fold] mean {np.mean(ff)*100:.2f}% ± {np.std(ff)*100:.2f}%")

        # ensemble: build OOF-concat order
        oof_fns = np.array([str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]])
        yt = np.array([emotion_to_label(fn) for fn in oof_fns], dtype=np.int64)
        root = Path(artifacts_root)
        mp = []
        for nm in MEMBERS:
            fl = [np.load(root / f"{nm}/fold_{fid:02d}/oof_logits.npy") for fid in range(10)]
            mp.append(softmax(np.concatenate(fl, axis=0)))
        seven = np.mean(mp, axis=0)
        s_f1 = f1_score(yt, seven.argmax(-1), average="macro")
        mb = align(np.load(mb_logits), np.load(mb_fns, allow_pickle=True), oof_fns)
        c3 = align(np.load(c3d_logits), np.load(c3d_fns, allow_pickle=True), oof_fns)
        m1 = align(np.load(mamp1), np.load(mamp1_fns, allow_pickle=True), oof_fns)
        m2 = align(np.load(mamp2), np.load(mamp2_fns, allow_pickle=True), oof_fns)
        pc = align(oof, filenames, oof_fns)
        nine = (seven * 7 + mb + c3) / 9
        n_f1 = f1_score(yt, nine.argmax(-1), average="macro")
        eleven = (seven * 7 + mb + c3 + m1 + m2) / 11
        e_f1 = f1_score(yt, eleven.argmax(-1), average="macro")
        pc_e = f1_score(yt, eleven.argmax(-1), average=None, labels=list(range(NUM)))

        combos = {
            "10way_9+posec3d": (seven * 7 + mb + c3 + pc) / 10,
            "12way_11+posec3d": (seven * 7 + mb + c3 + m1 + m2 + pc) / 12,
        }
        print(f"\n[7way] {s_f1*100:.2f}% | [9way] {n_f1*100:.2f}% | [11way] {e_f1*100:.2f}%")
        ens = {"seven": float(s_f1), "nine": float(n_f1), "eleven": float(e_f1),
               "tenfold_mean": float(np.mean(ff)), "tenfold_std": float(np.std(ff))}
        for name, probs in combos.items():
            yp = probs.argmax(-1)
            f1 = float(f1_score(yt, yp, average="macro"))
            base_f1, base_pc = (n_f1, f1_score(yt, nine.argmax(-1), average=None, labels=list(range(NUM)))) \
                if "10way" in name else (e_f1, pc_e)
            d = (f1 - base_f1) * 100
            pcn = f1_score(yt, yp, average=None, labels=list(range(NUM)))
            r30 = [IDX_TO_EMOTION[c] for c in range(NUM) if (pcn[c]-base_pc[c])*100 < -0.30]
            r50 = [IDX_TO_EMOTION[c] for c in range(NUM) if (pcn[c]-base_pc[c])*100 < -0.50]
            base = "9way" if "10way" in name else "11way"
            print(f"[{name}] {f1*100:.2f}% Δ vs {base} {d:+.2f}pp | reg>0.30 {len(r30)} {r30} | reg>0.50 {len(r50)}")
            ens[name] = {"macro_f1": f1, "delta_pp": d, "vs": base,
                         "reg_0.30": r30, "reg_0.50": r50}
        out["ensemble"] = ens

    out["elapsed_sec"] = time.time() - t0
    with open(Path(output_dir) / f"results_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[exp086:{tag}] saved results_{tag}.json ({out['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
