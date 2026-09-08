"""generate_full_corpus.py — Stage 1 of exp099

Generate 4000 cross-country retargeted clips for mixed-corpus
training. Same pairing rule as Stage 3 (top-decile max-cosine-
distance, emotion-stratified, seed=42), scaled 8x.

Outputs:
- `output/retarget/phase1_corpus/<src_stem>__from_X_to_Y__to_<tgt_pid>.npy`
  shape (T, 25, 3) CTV-layout joint positions (virtual_root at node 0).
- `.npz` sidecar with joint_quats / new_root_pos / source_offsets / target_offsets.
- `output/retarget/phase1_manifest.json` — per-pair metadata incl.
  `source_performer` (for LPO fold assignment) and `retarget_meta`.
- `output/retarget/phase1_raw_offsets.json` — cache of raw 24-joint
  rest-pose offsets per performer (24, 3 float lists). Needed by Stage 2
  for the **honest-FK** path on original (non-retargeted) samples.

Resumability: existing output npy files are skipped on re-run.

Related:
- experiments/exp099_bone_retarget/generate_smoke_pairs.py (Stage 3)
- experiments/exp099_bone_retarget/retarget_core.py (FK kernel)
- experiments/exp099_bone_retarget/bvh_offset_probe.py (raw-BVH offset reader)
- experiments/exp099_phase1_retarget
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp099_bone_retarget"))

from bvh_offset_probe import find_representative_bvh, extract_24joint_offsets  # noqa: E402
from retarget_core import retarget_clip  # noqa: E402
from diema.data.parser import parse_filename  # noqa: E402


PROPORTIONS_JSON = Path("output/retarget/bone_proportions.json")
CACHED_NPZ = Path("data/diema_challenge/processed/motion_train_quat.npz")
OUT_CORPUS = Path("output/retarget/phase1_corpus")
MANIFEST = Path("output/retarget/phase1_manifest.json")
RAW_OFFSETS_JSON = Path("output/retarget/phase1_raw_offsets.json")

N_PAIRS = 4000           # 2000 JP→TW + 2000 TW→JP
SEED = 42


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return 1.0 - float((a @ b) / (na * nb + 1e-12))


def load_performer_offsets(performer_ids: list[str]) -> dict[str, np.ndarray]:
    """Extract raw BVH offsets (24, 3) for each performer from a rep BVH."""
    out: dict[str, np.ndarray] = {}
    for pid in performer_ids:
        path = find_representative_bvh(pid)
        offsets, _ = extract_24joint_offsets(path)
        out[pid] = offsets.astype(np.float32)
    return out


def save_raw_offsets_cache(perf_offsets: dict[str, np.ndarray]) -> None:
    """Persist (24, 3) raw offsets per performer as JSON.

    Stage 2's honest-FK path reads this to inject per-clip offsets into
    forward kinematics for original (non-retargeted) samples.
    """
    payload = {
        "_meta": {
            "n_performers": len(perf_offsets),
            "shape": "24 x 3 (BVH-24 joint order, units = BVH file units)",
            "purpose": ("Per-performer raw rest-pose offsets, extracted from "
                         "one representative BVH per performer. Used by Stage 2 "
                         "MixedDataset honest-FK path."),
        },
        "offsets": {
            pid: arr.astype(float).tolist()
            for pid, arr in perf_offsets.items()
        },
    }
    RAW_OFFSETS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_OFFSETS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[stage1] raw offsets cache → {RAW_OFFSETS_JSON} "
          f"({len(perf_offsets)} performers)")


def main():
    rng = np.random.default_rng(SEED)
    OUT_CORPUS.mkdir(parents=True, exist_ok=True)

    # -- 1) Load performer metadata ----------------------------------------
    bone_meta = json.loads(PROPORTIONS_JSON.read_text())
    performers = bone_meta["performers"]
    performer_ids = sorted(performers.keys())
    by_country: dict[str, list[str]] = defaultdict(list)
    for pid, info in performers.items():
        by_country[info["country"]].append(pid)
    print(f"[stage1] {len(performer_ids)} performers "
          f"({len(by_country['JP'])} JP + {len(by_country['TW'])} TW)")

    # -- 2) Cache offsets for all performers ----------------------------
    print(f"[stage1] caching raw offsets for {len(performer_ids)} performers...")
    perf_offsets = load_performer_offsets(performer_ids)
    save_raw_offsets_cache(perf_offsets)

    # -- 3) Load cached npz and build clip metadata ---------------------
    print(f"[stage1] loading {CACHED_NPZ}")
    npz = np.load(CACHED_NPZ, allow_pickle=True)
    filenames = [str(x) for x in npz["filenames"]]
    n_clips = int(npz["num_clips"])
    assert n_clips == len(filenames) == 7992, \
        f"expected 7992 cached clips, got {n_clips}/{len(filenames)}"

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
    print(f"[stage1] parsed {len(clip_meta)} clip metadata entries")

    # -- 4) Emotion-stratified sampling of N_PAIRS source clips ----------
    by_emo: dict[str, list[dict]] = defaultdict(list)
    for c in clip_meta:
        by_emo[c["emotion"]].append(c)
    emotions = sorted(by_emo.keys())
    pairs_per_emo = N_PAIRS // len(emotions)           # 333
    remainder = N_PAIRS - pairs_per_emo * len(emotions)  # 4000 - 333*12 = 4

    sources = []
    for ei, emo in enumerate(emotions):
        bucket = by_emo[emo]
        jp_clips = [c for c in bucket if c["country"] == "JP"]
        tw_clips = [c for c in bucket if c["country"] == "TW"]
        n_emo = pairs_per_emo + (1 if ei < remainder else 0)
        n_jp = n_emo // 2
        n_tw = n_emo - n_jp
        # Sample with replacement only if the bucket is too small (rare).
        replace_jp = len(jp_clips) < n_jp
        replace_tw = len(tw_clips) < n_tw
        jp_idx = rng.choice(len(jp_clips), size=n_jp, replace=replace_jp)
        tw_idx = rng.choice(len(tw_clips), size=n_tw, replace=replace_tw)
        sources.extend([jp_clips[i] for i in jp_idx])
        sources.extend([tw_clips[i] for i in tw_idx])
        if replace_jp or replace_tw:
            print(f"  [info] emotion={emo}: replace JP={replace_jp} "
                  f"(have {len(jp_clips)} need {n_jp}), "
                  f"replace TW={replace_tw} (have {len(tw_clips)} need {n_tw})")
    rng.shuffle(sources)
    print(f"[stage1] sampled {len(sources)} source clips "
          f"(target: {N_PAIRS}); pairs_per_emo≈{pairs_per_emo}")

    # -- 5) Cross-country pairing with top-decile max-cosine-distance ---
    pairings = []
    proportions = {pid: np.array(info["proportions"])
                   for pid, info in performers.items()}
    for src in sources:
        target_country = "TW" if src["country"] == "JP" else "JP"
        target_candidates = by_country[target_country]
        src_props = proportions[src["performer"]]
        dists = [(pid, cosine_distance(src_props, proportions[pid]))
                 for pid in target_candidates]
        dists.sort(key=lambda x: -x[1])  # descending
        top_decile = max(3, len(dists) // 10)
        tgt_pid, tgt_dist = dists[rng.integers(0, top_decile)]
        pairings.append({
            **src,
            "target_performer": tgt_pid,
            "target_country": target_country,
            "bone_cosine_distance": tgt_dist,
        })
    n_jp_to_tw = sum(1 for p in pairings if p['country'] == 'JP')
    n_tw_to_jp = sum(1 for p in pairings if p['country'] == 'TW')
    print(f"[stage1] paired {len(pairings)} pairs "
          f"(JP→TW: {n_jp_to_tw}, TW→JP: {n_tw_to_jp})")

    # -- 6) Retarget each pair (resumable) ------------------------------
    print(f"[stage1] retargeting {len(pairings)} clips "
          f"(skipping pre-existing outputs)...")
    n_skipped = 0
    for k, p in enumerate(pairings):
        src_stem = Path(p["filename"]).stem
        out_name = (f"{src_stem}__from_{p['country']}_to_"
                    f"{p['target_country']}__to_{p['target_performer']}.npy")
        out_npy = OUT_CORPUS / out_name
        sidecar = OUT_CORPUS / f"{out_name[:-4]}.npz"
        p["output_npy"] = str(out_npy)
        p["output_npz"] = str(sidecar)

        if out_npy.exists() and sidecar.exists():
            # Already done; reload meta for manifest consistency
            try:
                meta_pkl = OUT_CORPUS / f"{out_name[:-4]}.meta.json"
                if meta_pkl.exists():
                    p["retarget_meta"] = json.loads(meta_pkl.read_text())
                else:
                    p["retarget_meta"] = {"reused_existing": True}
            except Exception:
                p["retarget_meta"] = {"reused_existing": True}
            n_skipped += 1
            if (k + 1) % 500 == 0:
                print(f"  ... {k+1}/{len(pairings)} (skipped existing: {n_skipped})")
            continue

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
        ctv_pos = np.concatenate(
            [out["new_root_pos"][:, None, :], out["joint_positions"]],
            axis=1).astype(np.float32)
        np.save(out_npy, ctv_pos)
        np.savez_compressed(
            sidecar,
            joint_quats=joint_quats,
            new_root_pos=out["new_root_pos"],
            source_offsets=src_off,
            target_offsets=tgt_off,
        )
        # also persist meta for fast resume
        (OUT_CORPUS / f"{out_name[:-4]}.meta.json").write_text(
            json.dumps(out["meta"]))
        p["retarget_meta"] = out["meta"]

        if (k + 1) % 500 == 0:
            print(f"  ... {k+1}/{len(pairings)} "
                  f"(skipped existing: {n_skipped})")
    print(f"[stage1] retarget done; "
          f"{len(pairings) - n_skipped} new + {n_skipped} skipped existing")

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
        "raw_offsets_cache": str(RAW_OFFSETS_JSON),
        "pairs": pairings,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=float))
    print(f"[stage1] manifest → {MANIFEST}")

    print("[stage1] DONE")


if __name__ == "__main__":
    main()
