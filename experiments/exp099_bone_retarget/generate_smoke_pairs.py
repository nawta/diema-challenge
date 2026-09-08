"""generate_smoke_pairs.py — Stage 3 of exp099.

Generate 500 cross-country retargeted clips for quality
assurance. Pairing rule per prompt:

- 250 JP source → TW target (most-dissimilar TW performer by bone
  cosine distance), 250 TW source → JP target.
- Emotion-stratified sampling across all 12 emotions (≈ 42 per
  emotion, seed=42 reproducible).
- For each pair: load cached quaternion + root_pos, swap target offsets
  via `retarget_clip`, save (T, 25, 3) joint positions in CTV layout
  (virtual_root prepended) + manifest entry.
- After 500 pairs, render 5 spot-check + 3 worst-distance side-by-side
  videos for human inspection (Stage 3 deliverable).

LPO fold information: the manifest records `source_performer` so
dataloader can assign retargeted clips to the source performer's fold
(leakage avoidance per decision framework §7).

Related:
- retarget_core.retarget_clip  (Stage 2 kernel)
- bvh_offset_probe.extract_24joint_offsets  (per-performer raw offsets)
- tools/render_motion_video.render_video    (side-by-side mp4)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bvh_offset_probe import find_representative_bvh, extract_24joint_offsets  # noqa
from retarget_core import retarget_clip  # noqa
from diema.data.parser import parse_filename  # noqa
from tools.render_motion_video import render_video  # noqa

PROPORTIONS_JSON = Path("output/retarget/bone_proportions.json")
CACHED_NPZ = Path("data/diema_challenge/processed/motion_train_quat.npz")
OUT_SMOKE = Path("output/retarget/smoke")
OUT_RENDERS = Path("output/retarget/smoke_renders")
MANIFEST = Path("output/retarget/smoke_manifest.json")

N_PAIRS = 500            # 250 JP→TW + 250 TW→JP
SEED = 42


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return 1.0 - float((a @ b) / (na * nb + 1e-12))


def load_performer_offsets(performer_ids: list[str]) -> dict[str, np.ndarray]:
    """Cache raw BVH offsets for each performer (one rep BVH each)."""
    out = {}
    for pid in performer_ids:
        path = find_representative_bvh(pid)
        offsets, _ = extract_24joint_offsets(path)
        out[pid] = offsets.astype(np.float32)
    return out


def main():
    rng = np.random.default_rng(SEED)
    OUT_SMOKE.mkdir(parents=True, exist_ok=True)
    OUT_RENDERS.mkdir(parents=True, exist_ok=True)

    # -- 1) Load performer metadata ----------------------------------------
    bone_meta = json.loads(PROPORTIONS_JSON.read_text())
    performers = bone_meta["performers"]
    performer_ids = sorted(performers.keys())
    by_country: dict[str, list[str]] = defaultdict(list)
    for pid, info in performers.items():
        by_country[info["country"]].append(pid)
    print(f"[stage3] {len(performer_ids)} performers "
          f"({len(by_country['JP'])} JP + {len(by_country['TW'])} TW)")

    # -- 2) Cache offsets for all performers ----------------------------
    print(f"[stage3] caching raw offsets for {len(performer_ids)} performers...")
    perf_offsets = load_performer_offsets(performer_ids)

    # -- 3) Load cached npz and build clip metadata ---------------------
    print(f"[stage3] loading {CACHED_NPZ}")
    npz = np.load(CACHED_NPZ, allow_pickle=True)
    filenames = [str(x) for x in npz["filenames"]]
    n_clips = int(npz["num_clips"])
    assert n_clips == len(filenames) == 7992, \
        f"expected 7992 cached clips, got {n_clips}/{len(filenames)}"

    # Parse each filename to (performer, country, emotion)
    clip_meta = []
    for i, fn in enumerate(filenames):
        try:
            info = parse_filename(Path(fn).stem)
            clip_meta.append({
                "idx": i,
                "filename": fn,
                "performer": f"{info.nationality}_{info.performer_num}",
                "country": info.nationality,
                "emotion": info.emotion,
            })
        except Exception as e:
            print(f"  [warn] skip {fn}: {e}")
    print(f"[stage3] parsed {len(clip_meta)} clip metadata entries")

    # -- 4) Build emotion-balanced sample of 500 source clips -----------
    # Goal: per emotion bucket, sample ~42 clips, with ~50/50 JP/TW
    # source split.
    by_emo: dict[str, list[dict]] = defaultdict(list)
    for c in clip_meta:
        by_emo[c["emotion"]].append(c)
    emotions = sorted(by_emo.keys())
    pairs_per_emo = N_PAIRS // len(emotions)        # 41
    remainder = N_PAIRS - pairs_per_emo * len(emotions)  # 500 - 41*12 = 8
    sources = []
    for ei, emo in enumerate(emotions):
        bucket = by_emo[emo]
        # try to balance JP/TW within the bucket
        jp_clips = [c for c in bucket if c["country"] == "JP"]
        tw_clips = [c for c in bucket if c["country"] == "TW"]
        n_emo = pairs_per_emo + (1 if ei < remainder else 0)
        n_jp = n_emo // 2
        n_tw = n_emo - n_jp
        jp_idx = rng.choice(len(jp_clips), size=min(n_jp, len(jp_clips)),
                             replace=False)
        tw_idx = rng.choice(len(tw_clips), size=min(n_tw, len(tw_clips)),
                             replace=False)
        sources.extend([jp_clips[i] for i in jp_idx])
        sources.extend([tw_clips[i] for i in tw_idx])
    rng.shuffle(sources)
    print(f"[stage3] sampled {len(sources)} source clips "
          f"(target: {N_PAIRS}); ~{pairs_per_emo}-{pairs_per_emo+1} per emotion")

    # -- 5) Cross-country pairing with max-cosine-distance --------------
    pairings = []
    proportions = {pid: np.array(info["proportions"])
                    for pid, info in performers.items()}
    for src in sources:
        target_country = "TW" if src["country"] == "JP" else "JP"
        target_candidates = by_country[target_country]
        # max cosine distance from source's *own* bone proportions
        src_props = proportions[src["performer"]]
        dists = [(pid, cosine_distance(src_props, proportions[pid]))
                  for pid in target_candidates]
        dists.sort(key=lambda x: -x[1])  # descending
        # top-decile: top 10 % of candidates
        top_decile = max(3, len(dists) // 10)
        tgt_pid, tgt_dist = dists[rng.integers(0, top_decile)]
        pairings.append({
            **src,
            "target_performer": tgt_pid,
            "target_country": target_country,
            "bone_cosine_distance": tgt_dist,
        })
    print(f"[stage3] paired {len(pairings)} pairs (JP→TW: "
          f"{sum(1 for p in pairings if p['country']=='JP')}, TW→JP: "
          f"{sum(1 for p in pairings if p['country']=='TW')})")

    # -- 6) Retarget each pair ------------------------------------------
    print(f"[stage3] retargeting {len(pairings)} clips...")
    for k, p in enumerate(pairings):
        i = p["idx"]
        joint_quats = npz[f"clip_{i}_joint_data"].astype(np.float32)
        root_pos = npz[f"clip_{i}_root_pos"].astype(np.float32)
        src_off = perf_offsets[p["performer"]]
        tgt_off = perf_offsets[p["target_performer"]]
        out = retarget_clip(
            joint_quats=joint_quats, root_pos=root_pos,
            source_offsets=src_off, target_offsets=tgt_off,
            preserve_ground_contact=True,
        )
        # CTV layout: prepend root_pos (= virtual_root) so shape becomes (T, 25, 3)
        ctv_pos = np.concatenate(
            [out["new_root_pos"][:, None, :], out["joint_positions"]],
            axis=1).astype(np.float32)
        # save
        src_stem = Path(p["filename"]).stem
        out_name = (f"{src_stem}__from_{p['country']}_to_"
                     f"{p['target_country']}__to_{p['target_performer']}.npy")
        np.save(OUT_SMOKE / out_name, ctv_pos)
        p["output_npy"] = str(OUT_SMOKE / out_name)
        p["retarget_meta"] = out["meta"]
        # also save full retarget output for LMA (joint_quats + new_root_pos
        # + new_offsets) in a sidecar npz for Stage 4a
        sidecar = OUT_SMOKE / f"{out_name[:-4]}.npz"
        np.savez_compressed(
            sidecar,
            joint_quats=joint_quats,
            new_root_pos=out["new_root_pos"],
            source_offsets=src_off,
            target_offsets=tgt_off,
        )
        p["output_npz"] = str(sidecar)
        if (k + 1) % 100 == 0:
            print(f"  ... {k+1}/{len(pairings)}")
    print(f"[stage3] retarget done; {len(pairings)} (.npy + .npz) saved")

    # -- 7) Save manifest -----------------------------------------------
    manifest = {
        "n_pairs": len(pairings),
        "seed": SEED,
        "pairing_rule": ("cross-country; per source pick a top-decile "
                          "max-cosine-distance target from the opposite "
                          "country"),
        "lpo_fold_rule": ("retargeted clip belongs to the SOURCE "
                           "performer's fold (leakage avoidance)"),
        "ctv_layout": ("(T, 25, 3) with virtual_root at node 0 "
                        "(== new_root_pos), BVH joints at nodes 1-24"),
        "pairs": pairings,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=float))
    print(f"[stage3] manifest → {MANIFEST}")

    # -- 8) Render 5 spot-check + 3 worst-distance side-by-side ---------
    print(f"[stage3] rendering 5 spot-check + 3 worst-distance pairs")
    # 5 spot checks: random sample
    spot_idx = list(rng.choice(len(pairings), size=5, replace=False))
    # 3 worst: top-3 by bone_cosine_distance
    worst_idx = sorted(range(len(pairings)),
                        key=lambda k: -pairings[k]["bone_cosine_distance"])[:3]
    render_set = [(idx, "spot") for idx in spot_idx] + \
                  [(idx, "worst") for idx in worst_idx]

    def _render_side_by_side(pair, tag, slot):
        """Render source + retargeted side-by-side for one pair."""
        i = pair["idx"]
        src_jq = npz[f"clip_{i}_joint_data"].astype(np.float32)
        src_rp = npz[f"clip_{i}_root_pos"].astype(np.float32)
        # source FK in BVH-24 then CTV-pack (virtual_root prepended)
        from diema.features.multi_stream import compute_joint_positions
        src_off = perf_offsets[pair["performer"]]
        src_pos_24 = compute_joint_positions(
            src_rp, src_jq, offsets=src_off)
        src_pos_25 = np.concatenate([src_rp[:, None, :], src_pos_24], axis=1).astype(np.float32)
        tgt_pos_25 = np.load(pair["output_npy"]).astype(np.float32)
        # Subsample to ≤120 frames for short render (keep first 120 if longer)
        if src_pos_25.shape[0] > 120:
            src_pos_25 = src_pos_25[:120]
            tgt_pos_25 = tgt_pos_25[:120]

        src_mp4 = OUT_RENDERS / f"{tag}_{slot:02d}_source.mp4"
        tgt_mp4 = OUT_RENDERS / f"{tag}_{slot:02d}_retargeted.mp4"
        sxs_mp4 = OUT_RENDERS / f"{tag}_{slot:02d}_side_by_side.mp4"
        render_video(joint_pos=src_pos_25, out_path=src_mp4,
                      fps=10, view="front_side_2view", slowmo=1.0)
        render_video(joint_pos=tgt_pos_25, out_path=tgt_mp4,
                      fps=10, view="front_side_2view", slowmo=1.0)
        # ffmpeg hstack
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_mp4), "-i", str(tgt_mp4),
            "-filter_complex", "hstack=inputs=2", str(sxs_mp4),
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        return sxs_mp4

    for slot, (idx, tag) in enumerate(render_set):
        p = pairings[idx]
        sxs = _render_side_by_side(p, tag, slot)
        print(f"  [{tag} {slot}] source={p['performer']} ({p['country']}) "
              f"→ target={p['target_performer']} ({p['target_country']}); "
              f"d={p['bone_cosine_distance']:.4f}; emo={p['emotion']}")
        print(f"    → {sxs}")

    print("[stage3] DONE")


if __name__ == "__main__":
    main()
