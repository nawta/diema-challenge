"""paper/figures/fig07_cards.py — Fig.7 (P8, optional qualitative zoom).

Design doc: paper/data_inventory.md (P0, item k).  Three explanation cards
(correct / confident-error / semantic-confusion) rendered as stacked
FancyBboxPatch blocks.  P12 decides main-vs-supplementary placement.

CRITICAL DATA FINDING (surfaced, NOT hidden): 34/50 explanation_cards_v2
cards have §1 "ground truth" != the filename emotion.  The project-wide
label is emotion_to_label(filename) (used by Fig.2-6, exp001-091, and it
reproduces the official baseline), so the card generator's §1
ground-truth field is misaligned for 68% of cards.  This does NOT affect
Fig.2-6 / Tables 1-2 (they use filename labels) nor the 0/50 grounding
audit (grounding = every numeric claim cites a section, label-independent).
To keep Fig.7 unambiguous we select ONLY from the 16 cards where
filename emotion == §1 ground-truth.  See fig07_cards_selection.json.

Anonymized: performer tokens (e.g. TW_37) -> P-codes; no country/ID leak.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402

from _style import PALETTE_QUAL, ONE_COL, save_fig, seed_all, setup_ieee  # noqa

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

setup_ieee()
seed_all(42)

CARDS = sorted(Path("output/explanation_cards_v2").glob("*.md"))
lz = np.load("output/lma_attrs_rule.npz", allow_pickle=True)
LZ_FN = [str(s) for s in lz["filenames"]]
LZ_Z = ((lz["features"].astype(float) - lz["features"].astype(float).mean(0))
        / (lz["features"].astype(float).std(0) + 1e-9))
LZ_NM = [str(s) for s in lz["names"]]


def parse(p):
    t = p.read_text()
    cid = p.stem
    fn_emo = cid.split("_")[2]
    pr = re.search(r"\*\*predicted\*\*: `([a-z]+)` \(confidence ([\d.]+)%\)", t)
    gt = re.search(r"\*\*ground truth\*\*: `([a-z]+)`", t)
    j = re.findall(r"^\| \d+ \| `([A-Za-z0-9]+)` \| \d+ \|", t, re.M)  # §6
    nar = re.search(r"## 10\. Narrator summary.*?\n> (.+?)(?:\.\s|\n)", t,
                    re.S)
    s = (nar.group(1).strip() + ".") if nar else "(n/a)"
    s = re.sub(r"[*`]", "", s)                    # drop markdown emphasis
    if len(s) > 150:
        s = s[:147].rsplit(" ", 1)[0] + "…"
    return {
        "cid": cid, "fn_emo": fn_emo,
        "true": gt.group(1) if gt else None,
        "pred": pr.group(1) if pr else None,
        "conf": float(pr.group(2)) if pr else None,
        "joints": j[:3],
        "narr": s,
    }


parsed = [parse(p) for p in CARDS]
# consistent subset: filename emotion == card §1 ground-truth (label-safe)
cons = [c for c in parsed if c["true"] and c["true"] == c["fn_emo"]]
n_mismatch = sum(1 for c in parsed if c["true"] and c["true"] != c["fn_emo"])


def lma_top2(cid):
    if cid not in LZ_FN:
        return []
    z = LZ_Z[LZ_FN.index(cid)]
    idx = np.argsort(np.abs(z))[::-1][:2]
    return [f"{LZ_NM[i]} (z={z[i]:+.1f})" for i in idx]


# ── selection (from the label-consistent subset only) ─────────────────────
correct = sorted([c for c in cons if c["pred"] == c["true"]],
                 key=lambda c: -c["conf"])
errors = sorted([c for c in cons if c["pred"] != c["true"]],
                key=lambda c: -c["conf"])
c1 = correct[0]                                   # highest-conf correct
c2 = errors[0]                                    # highest-conf error
# semantic confusion: prefer a basic/social cluster pair, else next error
pref = [("disgust", "fear"), ("contempt", "shame"), ("fear", "guilt")]
c3 = next((e for a, b in pref for e in errors
           if e is not c2 and e["true"] == a and e["pred"] == b),
          errors[1] if len(errors) > 1 else errors[0])

PERFS = sorted({"_".join(c["cid"].split("_")[:2]) for c in parsed})
PMAP = {p: f"P{idx + 1:02d}" for idx, p in enumerate(PERFS)}


def anon(cid):
    pf = "_".join(cid.split("_")[:2])
    return cid.replace(pf, PMAP[pf])


SEL = [
    ("correct (high-conf)", c1, PALETTE_QUAL[2],
     "highest-confidence label-consistent correct card; prompt preferred "
     "joy/sadness@p≥0.5 but none exist in the consistent subset (model is "
     "globally low-confidence, max 62%)"),
    ("confident error", c2, PALETTE_QUAL[1],
     "highest-confidence label-consistent misprediction; prompt preferred "
     "guilt→sadness but none in consistent subset"),
    ("semantic confusion", c3, PALETTE_QUAL[0],
     "meaningful confusion pair from the label-consistent errors; prompt's "
     "jealousy↔contempt / fear↔surprise absent in consistent subset"),
]

# ── render: 3 stacked card blocks ─────────────────────────────────────────
fig, ax = plt.subplots()
fig.set_size_inches(ONE_COL, 5.7)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

H = 0.285
GAP = 0.035
for k, (tag, c, col, _why) in enumerate(SEL):
    y0 = 0.97 - (k + 1) * H - k * GAP
    ax.add_patch(FancyBboxPatch(
        (0.02, y0), 0.96, H, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.5, edgecolor="0.5", facecolor="white"))
    ax.add_patch(Rectangle((0.02, y0 + H - 0.022), 0.96, 0.022,
                           facecolor=col, edgecolor="none"))  # stripe
    ax.text(0.04, y0 + H - 0.011, tag, fontsize=5.6, fontweight="bold",
            va="center", color="white")
    ok = "correct" if c["pred"] == c["true"] else "MISPREDICTED"
    ax.text(0.5, y0 + H - 0.045,
            f"Clip {anon(c['cid'])} — True: {c['true']} | "
            f"Pred: {c['pred']} (p={c['conf'] / 100:.2f}) [{ok}]",
            fontsize=5.7, ha="center", va="center", fontweight="bold")
    import textwrap
    rows = [
        ("Top-3 salient joints (§6)", " · ".join(c["joints"]) or "n/a", 1),
        ("Top-2 LMA |z| attrs", " · ".join(lma_top2(c["cid"])) or "n/a", 1),
        ("Narrator (§10, verbatim)",
         textwrap.fill(c["narr"], 74), 3),
    ]
    yy = y0 + H - 0.058
    for lab, val, _nl in rows:
        ax.text(0.045, yy, lab + ":", fontsize=4.9, fontweight="bold",
                va="top", color="0.25")
        ax.text(0.055, yy - 0.022, val, fontsize=4.5, va="top",
                color="0.1", linespacing=1.25)
        yy -= 0.022 + _nl * 0.0205 + 0.012

ax.text(0.5, 0.012,
        "All 50 cards: 0 hallucinated claims (exp073 audit, grounded_ratio="
        "1.0). Cards drawn from the 16 label-consistent cards (34/50 have a "
        "§1-vs-filename label-mismatch — see selection JSON).",
        fontsize=4.4, ha="center", va="bottom", color="0.35", wrap=True)

pdf, png = save_fig(fig, "fig07_cards")
plt.close(fig)

sel_json = {
    "selected": [{"role": tag, "clip_anon": anon(c["cid"]),
                  "true": c["true"], "pred": c["pred"], "conf_pct": c["conf"],
                  "reason": why} for tag, c, _, why in SEL],
    "selection_pool": "16 label-consistent cards (filename emotion == §1 "
                      "ground-truth) out of 50",
    "DATA_INTEGRITY_FINDING": {
        "issue": "34/50 explanation_cards_v2 have §1 ground-truth != "
                 "filename emotion",
        "authoritative_label": "emotion_to_label(filename) — used by "
                               "Fig.2-6, exp001-091, reproduces official "
                               "baseline; card §1 ground-truth is the "
                               "misaligned/buggy field",
        "impact": "Fig.2-6 + Tables 1-2 UNAFFECTED (filename labels); "
                  "0/50 grounding/hallucination audit UNAFFECTED "
                  "(label-independent). Affects only per-card 'correct' "
                  "claims -> Fig.7 restricted to the 16 consistent cards.",
        "recommend": "investigate tools/generate_explanation_card_v2.py "
                     "label lookup before relying on card §1 correctness",
    },
    "deviations_from_prompt": "model globally low-confidence (max 62%, only "
        "7/50 ≥50%); prompt's joy/sadness@p≥0.5 correct, guilt→sadness "
        "error, and jealousy↔contempt confusion are NOT present in the "
        "label-consistent subset — best available chosen, reasons above.",
}
Path(Path(__file__).resolve().parent / "fig07_cards_selection.json").write_text(
    json.dumps(sel_json, indent=2))
print(f"saved {pdf}")
print(f"saved {png}")
print(f"saved fig07_cards_selection.json")
print(f"label-consistent cards: {len(cons)}/50  mismatch: {n_mismatch}/50")
for tag, c, _, _ in SEL:
    print(f"  {tag:22s} {anon(c['cid']):24s} true={c['true']} "
          f"pred={c['pred']} p={c['conf']:.1f}%")
