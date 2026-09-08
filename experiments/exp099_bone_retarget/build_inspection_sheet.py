"""build_inspection_sheet.py — Stage 4b of exp099.

Sample 50 (source, retargeted) pairs uniformly across 12 emotions and
render side-by-side mp4 for nawta's manual quality scoring (1-5 + free
text notes). Emit `inspection_form.md` template.

After nawta fills the form, `aggregate_inspection.py` (Stage 4c) reads
it and updates `PHASE0_VERDICT.md` with the final combined verdict.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bvh_offset_probe import find_representative_bvh, extract_24joint_offsets  # noqa: E402
from diema.features.multi_stream import compute_joint_positions  # noqa: E402
from tools.render_motion_video import render_video  # noqa: E402

CACHED_NPZ = Path("data/diema_challenge/processed/motion_train_quat.npz")
MANIFEST = Path("output/retarget/smoke_manifest.json")
OUT_SHEET = Path("output/retarget/inspection_sheet")
OUT_FORM = Path("output/retarget/inspection_form.md")

N_INSPECT = 50
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    OUT_SHEET.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    pairs = manifest["pairs"]
    print(f"[stage4b] loaded {len(pairs)} smoke pairs")

    # Sample uniformly across 12 emotions
    by_emo = defaultdict(list)
    for i, p in enumerate(pairs):
        by_emo[p["emotion"]].append(i)
    emotions = sorted(by_emo.keys())
    per_emo = N_INSPECT // len(emotions)       # 4
    remainder = N_INSPECT - per_emo * len(emotions)  # 50 - 4*12 = 2
    inspect_idx = []
    for ei, emo in enumerate(emotions):
        bucket = by_emo[emo]
        n = per_emo + (1 if ei < remainder else 0)
        n = min(n, len(bucket))
        sel = rng.choice(len(bucket), size=n, replace=False)
        inspect_idx.extend([bucket[i] for i in sel])
    rng.shuffle(inspect_idx)
    inspect_idx = inspect_idx[:N_INSPECT]
    print(f"[stage4b] selected {len(inspect_idx)} pairs (target {N_INSPECT}, "
          f"~{per_emo}-{per_emo+1} per emotion)")

    # Cache performer offsets needed
    needed = set()
    for k in inspect_idx:
        needed.add(pairs[k]["performer"])
        needed.add(pairs[k]["target_performer"])
    offsets_cache = {pid: extract_24joint_offsets(
                        find_representative_bvh(pid))[0].astype(np.float32)
                     for pid in needed}
    print(f"[stage4b] cached offsets for {len(offsets_cache)} performers")

    print(f"[stage4b] loading cached npz {CACHED_NPZ}")
    npz = np.load(CACHED_NPZ, allow_pickle=True)

    # Render loop
    print(f"[stage4b] rendering {N_INSPECT} side-by-side pairs...")
    rendered = []
    for k, mi in enumerate(inspect_idx):
        p = pairs[mi]
        slot = k + 1                  # pair_id 1..50
        slot_dir = OUT_SHEET / f"pair_{slot:02d}"
        slot_dir.mkdir(exist_ok=True)
        i = p["idx"]
        joint_quats = npz[f"clip_{i}_joint_data"].astype(np.float32)
        root_pos = npz[f"clip_{i}_root_pos"].astype(np.float32)
        src_off = offsets_cache[p["performer"]]
        # Source FK
        src_pos_24 = compute_joint_positions(
            root_pos=root_pos, joint_quats=joint_quats, offsets=src_off)
        src_pos_25 = np.concatenate([root_pos[:, None, :], src_pos_24],
                                      axis=1).astype(np.float32)
        # Retargeted (load saved npy)
        tgt_pos_25 = np.load(p["output_npy"]).astype(np.float32)
        # Cap to ≤120 frames for compact rendering
        if src_pos_25.shape[0] > 120:
            src_pos_25 = src_pos_25[:120]
            tgt_pos_25 = tgt_pos_25[:120]

        src_mp4 = slot_dir / "source.mp4"
        tgt_mp4 = slot_dir / "retargeted.mp4"
        sxs_mp4 = slot_dir / "side_by_side.mp4"
        render_video(joint_pos=src_pos_25, out_path=src_mp4,
                      fps=10, view="front_side_2view", slowmo=1.0)
        render_video(joint_pos=tgt_pos_25, out_path=tgt_mp4,
                      fps=10, view="front_side_2view", slowmo=1.0)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_mp4), "-i", str(tgt_mp4),
            "-filter_complex", "hstack=inputs=2", str(sxs_mp4),
        ], check=True)
        rendered.append({
            "slot": slot,
            "pair_id": mi,
            "emotion": p["emotion"],
            "source_performer": p["performer"],
            "target_performer": p["target_performer"],
            "source_country": p["country"],
            "target_country": p["target_country"],
            "bone_cosine_distance": p["bone_cosine_distance"],
            "side_by_side_mp4": str(sxs_mp4),
        })
        if (k + 1) % 10 == 0:
            print(f"  ... {k+1}/{N_INSPECT}")
    print(f"[stage4b] {len(rendered)} pairs rendered")

    # Emit inspection_form.md template (50 rows, 1-5 scoring + notes)
    form = [
        "# Inspection form — exp099 Stage 4b",
        "",
        "**Task**: view each `side_by_side.mp4` (source LEFT, "
        "retargeted RIGHT) and rate the **motion quality** of the "
        "retargeted clip on a 1-5 scale:",
        "",
        "- **5** — natural, plausible motion; no visible artefacts",
        "- **4** — mostly natural; minor scaling oddities",
        "- **3** — recognizable motion but noticeable artefacts "
        "(foot sliding, etc.)",
        "- **2** — severe artefacts (self-penetration, broken pose)",
        "- **1** — broken / unusable",
        "",
        "**Scoring threshold for GO**: mean ≥ 3.5 across 50 "
        "(framework §2). Mean ≥ 4.0 unlocks full mix ratio 0.4; "
        "3.5-4.0 unlocks 0.25 mix; < 3.5 = NO-GO.",
        "",
        "Fill the `score` column with a number 1-5. Add free-text "
        "notes in the `notes` column (optional but encouraged).",
        "",
        "| slot | mp4 | emotion | src → tgt | bone_d | score | notes |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for r in sorted(rendered, key=lambda r: r["slot"]):
        relmp4 = Path(r["side_by_side_mp4"]).relative_to(
            OUT_FORM.parent)
        form.append(
            f"| {r['slot']:02d} | `{relmp4}` | {r['emotion']} | "
            f"{r['source_performer']} ({r['source_country']}) → "
            f"{r['target_performer']} ({r['target_country']}) | "
            f"{r['bone_cosine_distance']:.4f} |   |   |")
    form.append("")
    form.append("---")
    form.append("**After filling**: run "
                "`conda run -n acii2026 python "
                "experiments/exp099_bone_retarget/aggregate_inspection.py`")
    form.append("to update `output/retarget/PHASE0_VERDICT.md` "
                "with the final combined GO/NO-GO.")
    OUT_FORM.write_text("\n".join(form))
    print(f"[stage4b] → {OUT_FORM}")

    # Also save the metadata json for aggregator
    (OUT_SHEET / "metadata.json").write_text(
        json.dumps({"n_inspect": len(rendered), "seed": SEED,
                     "pairs": rendered}, indent=2))
    print(f"[stage4b] DONE")


if __name__ == "__main__":
    main()
