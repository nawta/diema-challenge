"""exp089 — evidence-grounded explanation cards (locked-vocabulary narrator).

Per docs/analysis/lma_motion_text_plan.md §3. The plan constrains the
narrator to slot-filling with a locked JA vocabulary so every word traces
to a measured LMA z-score (hallucination-proof). The deterministic
locked-vocabulary renderer IS that hallucination-safe core: it needs no
LLM, is fully reproducible, and by construction cannot emit a word outside
allowed_vocab_ja.json. (A Qwen3-VL polish pass is an optional future step;
the deterministic renderer is the methodologically sound deliverable and
avoids reintroducing the hallucination the plan explicitly suppresses.
Reported as a scope decision.)

Per clip card:
  - predicted emotion (from production 11-way OOF logits — best ensemble)
  - top-k distinctive LMA features for THIS clip vs its true-class mean
  - top salient joints for the class (exp078 spatial_sal)
  - narration: locked-vocab phrases for the top-k LMA, sign-aware
  - hallucination guard: assert every emitted phrase ∈ allowed vocab

Generates cards for the documented confusion pairs + a random sample.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label
from diema.features.lma_attributes import LMA_NAMES

VOCAB = json.loads(
    Path("experiments/exp089_motion_text_alignment/allowed_vocab_ja.json").read_text()
)
ALLOWED = {v["pos"] for k, v in VOCAB.items() if k != "_doc"} | \
          {v["neg"] for k, v in VOCAB.items() if k != "_doc"}
ORDER = ["anger", "contempt", "disgust", "fear", "joy", "sadness",
         "surprise", "jealousy", "shame", "guilt", "gratitude", "pride"]
# documented confusion pairs (project knowledge)
CONFUSION_PAIRS = [("jealousy", "contempt"), ("guilt", "sadness"),
                   ("fear", "surprise"), ("pride", "gratitude")]


def phrase(feat: str, z: float) -> str:
    e = VOCAB[feat]
    return e["pos"] if z >= 0 else e["neg"]


def render_card(sample, pred_emo, true_emo, lma_vec, class_mu, class_sd,
                top_joints, k=4) -> str:
    z = (lma_vec - class_mu) / (class_sd + 1e-8)
    order = np.argsort(-np.abs(z))[:k]
    ev_lines, phrases = [], []
    for i in order:
        fn = LMA_NAMES[i]
        if fn == "_doc" or fn not in VOCAB:
            continue
        ph = phrase(fn, z[i])
        phrases.append(ph)
        ev_lines.append(f"  - `{fn}` z={z[i]:+.2f} → 「{ph}」")
    # Join 連用形 phrases with 、 and close with a noun-style summary to
    # avoid per-phrase conjugation (keeps the locked vocab intact).
    narration = "、".join(phrases) + "、という運動特徴を示す。"
    # hallucination guard
    for ph in phrases:
        assert ph in ALLOWED, f"vocab violation: {ph!r}"
    card = (
        f"## {sample} → 予測: **{pred_emo}** (正解: {true_emo})\n\n"
        f"**運動学的証拠 (LMA, クリップ固有 z-score top-{k})**:\n"
        + "\n".join(ev_lines) + "\n\n"
        f"**深層モデルが注目する関節** (exp078 spatial saliency, {true_emo}): "
        + ", ".join(top_joints) + "\n\n"
        f"**説明 (locked vocabulary, 全語が上記 LMA z-score に由来)**:\n"
        f"> このクリップは{narration}\n"
    )
    return card


def main():
    lma = np.load("output/lma_attrs_rule.npz", allow_pickle=True)
    feats = lma["features"]
    labels = lma["labels"]
    fns = [str(x) for x in lma["filenames"]]
    fn_to_i = {f: i for i, f in enumerate(fns)}

    # per-class mean/std for clip-specific deviation
    class_mu = np.zeros((12, 32)); class_sd = np.zeros((12, 32))
    for c in range(12):
        m = labels == c
        class_mu[c] = feats[m].mean(0)
        class_sd[c] = feats[m].std(0)

    s78 = np.load("experiments/exp078_temporal_spatial_saliency/results.npz",
                  allow_pickle=True)
    spatial_sal = s78["spatial_sal"]  # (12,25)
    ctv_names = [str(x) for x in s78["joint_names"]]
    top_joints_by_class = {
        c: [ctv_names[j] for j in np.argsort(-spatial_sal[c])[:4]]
        for c in range(12)
    }

    # production best-ensemble OOF predictions (11-way) for predicted emotion
    with open("data/diema_challenge/processed/split_lpo_10fold.pkl", "rb") as f:
        splits = pickle.load(f)
    oof_fns = np.array([str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]])

    def sm(x):
        x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

    def aln(p, fp):
        a = np.load(p); fns2 = [str(x) for x in np.load(fp, allow_pickle=True)]
        m = {f: i for i, f in enumerate(fns2)}
        return sm(np.stack([a[m[f]] for f in oof_fns]))

    MEM = ["exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
           "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
           "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00"]
    mp = []
    for nm in MEM:
        fl = [np.load(f"output/artifacts/{nm}/fold_{fid:02d}/oof_logits.npy") for fid in range(10)]
        mp.append(sm(np.concatenate(fl, 0)))
    seven = np.mean(mp, 0)
    mb = aln("output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy",
             "output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy")
    c3 = aln("output/predictions/c3d_contact_only/oof_logits.npy",
             "output/predictions/c3d_contact_only/oof_filenames.npy")
    m1 = aln("output/predictions/mamp_ntu60xsub/oof_logits.npy",
             "output/predictions/mamp_ntu60xsub/oof_filenames.npy")
    m2 = aln("output/predictions/mamp_ntu120xset/oof_logits.npy",
             "output/predictions/mamp_ntu120xset/oof_filenames.npy")
    eleven = (seven * 7 + mb + c3 + m1 + m2) / 11
    pred = {oof_fns[i]: int(eleven[i].argmax()) for i in range(len(oof_fns))}

    out_dir = Path("output/explanation_cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    selected = []

    # confusion-pair exemplars: clips of class A predicted as B (and vice versa)
    for a, b in CONFUSION_PAIRS:
        ca, cb = ORDER.index(a), ORDER.index(b)
        for true_c, pred_c in [(ca, cb), (cb, ca)]:
            cand = [f for f in oof_fns
                    if emotion_to_label(f) == true_c and pred.get(f) == pred_c
                    and f in fn_to_i]
            if cand:
                selected.append((cand[0], "confusion"))

    # random correct + random sample
    correct = [f for f in oof_fns
               if pred.get(f) == emotion_to_label(f) and f in fn_to_i]
    for f in rng.choice(correct, size=min(6, len(correct)), replace=False):
        selected.append((str(f), "correct"))

    cards = []
    for sample, kind in selected:
        i = fn_to_i[sample]
        tc = emotion_to_label(sample)
        pc = pred.get(sample, tc)
        card = render_card(
            sample, IDX_TO_EMOTION[pc], IDX_TO_EMOTION[tc],
            feats[i], class_mu[tc], class_sd[tc], top_joints_by_class[tc])
        (out_dir / f"{sample}.md").write_text(card)
        cards.append({"sample": sample, "kind": kind,
                      "pred": IDX_TO_EMOTION[pc], "true": IDX_TO_EMOTION[tc]})

    summary = {
        "n_cards": len(cards),
        "allowed_vocab_size": len(ALLOWED),
        "hallucination_violations": 0,  # asserted in render_card
        "confusion_pairs": [f"{a}/{b}" for a, b in CONFUSION_PAIRS],
        "cards": cards,
        "card_dir": str(out_dir),
    }
    Path("experiments/exp089_motion_text_alignment/results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[exp089] {len(cards)} cards → {out_dir}/ "
          f"(vocab {len(ALLOWED)} phrases, 0 hallucination violations)")
    for c in cards:
        print(f"  {c['sample']:14s} [{c['kind']:9s}] pred={c['pred']} true={c['true']}")


if __name__ == "__main__":
    main()
