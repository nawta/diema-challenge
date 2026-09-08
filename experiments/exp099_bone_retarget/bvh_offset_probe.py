"""bvh_offset_probe.py — Step 3 #5 smoke + Stage 1a KILL GATE.

Question: are per-performer raw BVH offsets actually different from
JP_06's hardcoded reference (`_DIEMA_OFFSETS_BVH` in
diema/features/multi_stream.py L40), or did the EDA-blessed
"all DIEM-A samples share the same skeleton structure" assumption
already flatten the shape signal into rotations?

If raw BVH offsets are uniform across performers (max relative
deviation < 5 %), the cached `motion_train_quat.npz` pipeline already
collapses bone-shape into rotation pattern, and a bone-proportion
retarget is redundant by construction — should ABORT.

If raw BVH offsets differ ≥ 5 %, the cached data has dropped shape
information that retargeting can synthesize back; proceeds.

Design note (related: diema/models/skeleton_graph.py L25-51):
Bvh.joint_count = 24 (BVH-24 space; CTV adds a virtual_root at idx 0).
We extract the 24 named-joint offsets via Bvh.nodes filtering.

Usage:
    python experiments/exp099_bone_retarget/bvh_offset_probe.py \
        --performers JP_06,TW_01,JP_07
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pybvh

RAW_BVH_TRAIN = Path("data/diema_challenge/raw/bvh/train")
KILL_GATE_THRESHOLD = 0.05   # 5 % max relative deviation


def find_representative_bvh(performer_id: str) -> Path:
    """Return one BVH file for the given performer (deterministic pick:
    first emotion=anger, scenario=1, intensity=H if available; otherwise
    earliest alphabetical)."""
    matches = sorted(RAW_BVH_TRAIN.glob(f"{performer_id}_*.bvh"))
    if not matches:
        raise FileNotFoundError(
            f"no BVH file found for performer {performer_id} under "
            f"{RAW_BVH_TRAIN}")
    # prefer anger_1_H
    pref = [m for m in matches if "anger_1_H" in m.name]
    return pref[0] if pref else matches[0]


def extract_24joint_offsets(bvh_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load BVH, return (24, 3) offset array + ordered joint names."""
    b = pybvh.read_bvh_file(str(bvh_path))
    name_to_offset = {n.name: np.asarray(n.offset, dtype=np.float64)
                       for n in b.nodes
                       if hasattr(n, "name") and hasattr(n, "offset")}
    offsets = np.stack([name_to_offset[name] for name in b.joint_names])
    assert offsets.shape == (24, 3), \
        f"expected (24, 3) offsets, got {offsets.shape}"
    return offsets, list(b.joint_names)


def compute_height(offsets: np.ndarray, joint_names: list[str]) -> float:
    """Approximate skeleton standing height = |Hips→Head z-stack| +
    |Hips→Toe z-stack|. Cumulates spine→head chain and hip→toe chain;
    uses absolute coordinate-sum of bone offsets (BVH local frame)."""
    spine_chain = ["Spine", "Spine1", "Spine2", "Spine3",
                    "Neck", "Neck1", "Head"]
    leg_chain = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
    spine_h = sum(np.linalg.norm(offsets[joint_names.index(j)])
                   for j in spine_chain if j in joint_names)
    leg_h = sum(np.linalg.norm(offsets[joint_names.index(j)])
                 for j in leg_chain if j in joint_names)
    return float(spine_h + leg_h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--performers", type=str, default="JP_06,TW_01,JP_07",
                    help="comma-separated performer IDs")
    p.add_argument("--reference", type=str, default="JP_06",
                    help="reference performer for deviation calculation")
    args = p.parse_args()

    performers = args.performers.split(",")
    if args.reference not in performers:
        performers = [args.reference] + performers
    print(f"[probe] performers: {performers}  reference: {args.reference}")
    print(f"[probe] raw BVH dir: {RAW_BVH_TRAIN}")
    print(f"[probe] kill-gate threshold: max |Δ| / |ref| < "
          f"{KILL_GATE_THRESHOLD*100:.0f} % → ABORT ")
    print()

    results = {}
    for pid in performers:
        try:
            path = find_representative_bvh(pid)
        except FileNotFoundError as e:
            print(f"  [{pid}] ✗ {e}")
            continue
        offsets, names = extract_24joint_offsets(path)
        height = compute_height(offsets, names)
        results[pid] = {"path": path, "offsets": offsets,
                         "names": names, "height": height}
        print(f"  [{pid}] {path.name}  height = {height:.3f}  "
              f"(non-zero offsets: {(np.linalg.norm(offsets, axis=1) > 0).sum()}/24)")
    print()

    if args.reference not in results:
        print("[probe] reference performer missing; cannot run gate")
        sys.exit(2)

    ref = results[args.reference]
    print(f"=== Deviation vs reference {args.reference} "
          f"({ref['path'].name}) ===")
    max_rel_dev = 0.0
    for pid, r in results.items():
        if pid == args.reference:
            continue
        # per-joint Euclidean deviation, normalized by reference bone length
        ref_lens = np.linalg.norm(ref["offsets"], axis=1)
        cur_lens = np.linalg.norm(r["offsets"], axis=1)
        # relative deviation per bone (skip 0-length root)
        non_root = ref_lens > 1e-6
        rel_dev = np.abs(cur_lens - ref_lens) / np.maximum(ref_lens, 1e-9)
        rel_dev_nz = rel_dev[non_root]
        max_dev = float(rel_dev_nz.max())
        max_dev_bone_idx = int(np.argmax(rel_dev) if non_root.all()
                                else np.where(non_root)[0][np.argmax(rel_dev_nz)])
        max_dev_bone = r["names"][max_dev_bone_idx]
        height_ratio = r["height"] / ref["height"]
        max_rel_dev = max(max_rel_dev, max_dev)
        print(f"  {pid}: max |Δ|/|ref| = {max_dev*100:.2f} % "
              f"(worst bone: {max_dev_bone}); "
              f"height_ratio = {height_ratio:.3f}")
    print()
    print(f"=== KILL GATE ===")
    print(f"max relative deviation across all probed performers: "
          f"{max_rel_dev*100:.2f} %")
    if max_rel_dev < KILL_GATE_THRESHOLD:
        print(f"✗ FAIL — deviation < {KILL_GATE_THRESHOLD*100:.0f} %")
        print(" hypothesis (per-performer bone proportions differ "
              "meaningfully) is FALSIFIED at the raw-BVH level.")
        print("  Cached motion_train_quat.npz pipeline assumption is "
              "consistent with raw data; bone-retarget would be a no-op.")
        print(" Recommendation: ABORT; redirect Track-B budget "
              "to Track A backlog.")
        sys.exit(1)
    else:
        print(f"✓ PASS — deviation ≥ {KILL_GATE_THRESHOLD*100:.0f} %")
        print("  Per-performer skeletons are sufficiently distinct in raw "
              "BVH form. proceeds to Stage 1b (92-performer "
              "bone-proportion extraction).")
        sys.exit(0)


if __name__ == "__main__":
    main()
