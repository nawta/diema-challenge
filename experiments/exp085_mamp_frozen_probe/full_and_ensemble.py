"""exp085 — MAMP full 10-fold OOF + 9→10/11-way ensemble eval.

Generates per-fold OOF logits for one or more MAMP feature variants, then
evaluates whether adding MAMP to the exp083 9-way (7 prod + MB + C3D)
lifts Macro-F1 under the gates.

 gates:
  - OOF: vs exp083 9-way (OOF 34.31%) Δ ≥ +0.20pp
  - Per-class (single add): ≤4 classes regress >0.30pp
  - Per-class (fusion final): ≤2 classes regress >0.50pp

Saves OOF logits to output/predictions/mamp_<tag>/{oof_logits,oof_filenames}.npy
(same schema as exp081/072, so exp088 fusion can reuse them).

Reuses the 9-way construction from experiments/exp083_9way_ensemble/run_9way.py.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label

MEMBERS = [
    "exp002_a04_smooth",
    "exp003_ctrgcn_a00",
    "exp004_skateformer_a01",
    "exp006_protogcn_a00",
    "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00",
    "exp034_regionaware_convtr_a00",
]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def get_fold_indices(splits, filenames, fold_idx):
    fold = splits[fold_idx]
    name_to_pos = {str(fn): i for i, fn in enumerate(filenames)}
    tr = np.array([name_to_pos[str(it[0])] for it in fold["train"] if str(it[0]) in name_to_pos], dtype=np.int64)
    va = np.array([name_to_pos[str(it[0])] for it in fold["val"] if str(it[0]) in name_to_pos], dtype=np.int64)
    return tr, va


def gen_oof(features_npz, feat_source, c_reg, splits, max_iter=2000):
    npz = np.load(features_npz, allow_pickle=True)
    filenames = np.asarray(npz["filenames"])
    labels = np.array([emotion_to_label(str(fn)) for fn in filenames], dtype=np.int64)
    if feat_source == "pooled":
        feats = np.asarray(npz["features_pooled"], dtype=np.float32)
    else:
        pj = np.asarray(npz["features_per_joint"], dtype=np.float32)
        feats = pj.reshape(pj.shape[0], -1)
    NUM_CLASS = 12
    oof_logits = np.zeros((len(feats), NUM_CLASS), dtype=np.float32)
    per_fold = []
    for fid in range(10):
        tr, va = get_fold_indices(splits, filenames, fid)
        scaler = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(C=c_reg, max_iter=max_iter, n_jobs=-1, solver="lbfgs", random_state=42)
        clf.fit(scaler.transform(feats[tr]), labels[tr])
        proba = clf.predict_proba(scaler.transform(feats[va]))
        for j, cls in enumerate(clf.classes_):
            oof_logits[va, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        y_pred = clf.predict(scaler.transform(feats[va]))
        per_fold.append(float(f1_score(labels[va], y_pred, average="macro")))
    return oof_logits, filenames, np.mean(per_fold), np.std(per_fold)


def align(logits, fns, ref_fns):
    fn_to_idx = {str(fn): i for i, fn in enumerate(fns)}
    out = np.zeros((len(ref_fns), logits.shape[1]), dtype=np.float32)
    for i, fn in enumerate(ref_fns):
        out[i] = logits[fn_to_idx[str(fn)]]
    return softmax(out)


@click.command()
@click.option("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
@click.option("--artifacts-root", default="output/artifacts")
@click.option("--mb-logits", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy")
@click.option("--mb-fns", default="output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy")
@click.option("--c3d-logits", default="output/predictions/c3d_contact_only/oof_logits.npy")
@click.option("--c3d-fns", default="output/predictions/c3d_contact_only/oof_filenames.npy")
@click.option("--variants", default="ntu60xsub:data/diema_challenge/processed/mamp_ntu60xsub_train.npz:per_joint_flat:0.1,ntu120xset:data/diema_challenge/processed/mamp_ntu120xset_train.npz:pooled:0.1",
              help="comma-sep tag:npz:feat_source:C list")
@click.option("--results-json", default="experiments/exp085_mamp_frozen_probe/full_and_ensemble_results.json")
def main(splits_pkl, artifacts_root, mb_logits, mb_fns, c3d_logits, c3d_fns, variants, results_json):
    t0 = time.time()
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)

    oof_fns_concat = []
    for fid in range(10):
        for fn, _ in splits[fid]["val"]:
            oof_fns_concat.append(str(fn))
    oof_fns_concat = np.array(oof_fns_concat)
    y_true = np.array([emotion_to_label(fn) for fn in oof_fns_concat], dtype=np.int64)
    N, NUM_CLASS = len(oof_fns_concat), 12
    root = Path(artifacts_root)

    # 7-way
    member_probs = []
    for name in MEMBERS:
        fl = [np.load(root / f"{name}/fold_{fid:02d}/oof_logits.npy") for fid in range(10)]
        member_probs.append(softmax(np.concatenate(fl, axis=0)))
    seven = np.mean(member_probs, axis=0)
    seven_pred = seven.argmax(-1)
    seven_f1 = float(f1_score(y_true, seven_pred, average="macro"))
    print(f"[7way] Macro-F1 {seven_f1*100:.2f}%")

    mb = align(np.load(mb_logits), np.load(mb_fns, allow_pickle=True), oof_fns_concat)
    c3d = align(np.load(c3d_logits), np.load(c3d_fns, allow_pickle=True), oof_fns_concat)
    nine = (seven * 7 + mb + c3d) / 9
    nine_pred = nine.argmax(-1)
    nine_f1 = float(f1_score(y_true, nine_pred, average="macro"))
    print(f"[9way exp083 baseline] Macro-F1 {nine_f1*100:.2f}%")

    pc_nine = f1_score(y_true, nine_pred, average=None, labels=list(range(NUM_CLASS)))

    var_specs = []
    for spec in variants.split(","):
        tag, npz, src, c = spec.split(":")
        var_specs.append((tag, npz, src, float(c)))

    mamp_probs = {}
    out = {"seven_way_macro_f1": seven_f1, "nine_way_macro_f1": nine_f1, "variants": {}}
    for tag, npz, src, c in var_specs:
        oof_logits, fns, mean_f1, std_f1 = gen_oof(npz, src, c, splits)
        pred_dir = Path(f"output/predictions/mamp_{tag}")
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / "oof_logits.npy", oof_logits)
        np.save(pred_dir / "oof_filenames.npy", fns)
        probs = align(oof_logits, fns, oof_fns_concat)
        mamp_probs[tag] = probs
        alone_f1 = float(f1_score(y_true, probs.argmax(-1), average="macro"))
        print(f"\n[{tag}] 10-fold mean {mean_f1*100:.2f}% ± {std_f1*100:.2f}%, "
              f"OOF-concat alone {alone_f1*100:.2f}%, src={src} C={c}")

        ten = (seven * 7 + mb + c3d + probs) / 10
        ten_pred = ten.argmax(-1)
        ten_f1 = float(f1_score(y_true, ten_pred, average="macro"))
        d_vs9 = (ten_f1 - nine_f1) * 100
        pc_ten = f1_score(y_true, ten_pred, average=None, labels=list(range(NUM_CLASS)))
        reg30 = [IDX_TO_EMOTION[c2] for c2 in range(NUM_CLASS) if (pc_ten[c2]-pc_nine[c2])*100 < -0.30]
        reg50 = [IDX_TO_EMOTION[c2] for c2 in range(NUM_CLASS) if (pc_ten[c2]-pc_nine[c2])*100 < -0.50]
        print(f"[10way 9+{tag}] Macro-F1 {ten_f1*100:.2f}% Δ vs 9way {d_vs9:+.2f}pp "
              f"| reg>0.30pp: {len(reg30)} {reg30} | reg>0.50pp: {len(reg50)}")
        out["variants"][tag] = {
            "feat_source": src, "C": c,
            "tenfold_mean_macro_f1": mean_f1, "tenfold_std": std_f1,
            "alone_oof_macro_f1": alone_f1,
            "ten_way_macro_f1": ten_f1, "delta_vs_9way_pp": d_vs9,
            "reg_0.30pp": reg30, "reg_0.50pp": reg50,
            "gate_oof_pass": bool(d_vs9 >= 0.20),
            "gate_perclass_single_pass": bool(len(reg30) <= 4),
        }

    # 11-way: 9 + both MAMP variants (if 2 variants)
    if len(mamp_probs) == 2:
        tags = list(mamp_probs.keys())
        eleven = (seven * 7 + mb + c3d + mamp_probs[tags[0]] + mamp_probs[tags[1]]) / 11
        el_pred = eleven.argmax(-1)
        el_f1 = float(f1_score(y_true, el_pred, average="macro"))
        d11 = (el_f1 - nine_f1) * 100
        pc_el = f1_score(y_true, el_pred, average=None, labels=list(range(NUM_CLASS)))
        reg30 = [IDX_TO_EMOTION[c2] for c2 in range(NUM_CLASS) if (pc_el[c2]-pc_nine[c2])*100 < -0.30]
        reg50 = [IDX_TO_EMOTION[c2] for c2 in range(NUM_CLASS) if (pc_el[c2]-pc_nine[c2])*100 < -0.50]
        print(f"\n[11way 9+{tags[0]}+{tags[1]}] Macro-F1 {el_f1*100:.2f}% "
              f"Δ vs 9way {d11:+.2f}pp | reg>0.30pp {len(reg30)} {reg30} | reg>0.50pp {len(reg50)}")
        out["eleven_way"] = {
            "macro_f1": el_f1, "delta_vs_9way_pp": d11,
            "reg_0.30pp": reg30, "reg_0.50pp": reg50,
            "gate_oof_pass": bool(d11 >= 0.20),
            "gate_perclass_fusion_pass": bool(len(reg50) <= 2),
        }

    out["elapsed_sec"] = time.time() - t0
    Path(results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[exp085] saved {results_json} ({out['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
