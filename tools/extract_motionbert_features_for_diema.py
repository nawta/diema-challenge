"""exp081 — extract frozen MotionBERT-Lite features for DIEM-A clips.

For each DIEM-A clip in motion_train_quat.npz:
  1. Project BVH-24 → H36M-17 2D via diema.features.bvh_to_h36m17
  2. Feed through frozen DSTformer.forward(x, return_rep=True) → (T, 17, 512)
  3. Aggregate features (mean over T, optionally over J too)
  4. Save (N, 512) and (N, 17*512) variants for linear probe

Usage:
  conda run -n acii2026 python tools/extract_motionbert_features_for_diema.py \
      --motion-npz data/diema_challenge/processed/motion_train_quat.npz \
      --checkpoint data/motionbert/checkpoints/MB_pretrain_lite.bin \
      --output data/diema_challenge/processed/motionbert_features_b00_train.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MB_SRC = REPO_ROOT / "tmp" / "MotionBERT"
if str(MB_SRC) not in sys.path:
    sys.path.insert(0, str(MB_SRC))

from diema.features.bvh_to_h36m17 import project_bvh_to_h36m17


def build_motionbert_lite() -> torch.nn.Module:
    """Construct the MotionBERT-Lite DSTformer (matches MB_lite.yaml)."""
    from functools import partial

    from lib.model.DSTformer import DSTformer
    import torch.nn as nn

    return DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        att_fuse=True,
    )


def load_pretrained(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_pos" in ckpt:
            state = ckpt["model_pos"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    incompat = model.load_state_dict(state, strict=False)
    model_keys = set(model.state_dict().keys())
    missing = [k for k in incompat.missing_keys if not (k.startswith("head") or k.startswith("ts_attn"))]
    if missing:
        print(f"[load] WARN missing keys: {len(missing)} (first 5: {missing[:5]})")
    if incompat.unexpected_keys:
        print(f"[load] WARN unexpected: {len(incompat.unexpected_keys)} (first 5: {incompat.unexpected_keys[:5]})")
    loaded_fraction = (len(model_keys) - len(missing)) / max(len(model_keys), 1)
    print(f"[load] loaded {loaded_fraction:.1%} of model params from {checkpoint_path}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-frames", type=int, default=243)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_pretrained(build_motionbert_lite(), args.checkpoint, device)

    npz = np.load(args.motion_npz, allow_pickle=True)
    num_clips = int(npz["num_clips"])
    filenames = list(npz["filenames"])
    print(f"[extract] {num_clips} clips from {args.motion_npz}", flush=True)

    feats_pooled = np.zeros((num_clips, 512), dtype=np.float32)
    feats_per_joint = np.zeros((num_clips, 17, 512), dtype=np.float32)

    t0 = time.time()
    for start in range(0, num_clips, args.batch_size):
        end = min(start + args.batch_size, num_clips)
        batch = np.zeros((end - start, args.target_frames, 17, 3), dtype=np.float32)
        for j, i in enumerate(range(start, end)):
            rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
            jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
            batch[j] = project_bvh_to_h36m17(rp, jq, target_frames=args.target_frames)

        x = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            rep = model(x, return_rep=True)  # (B, T, 17, 512)
        rep_np = rep.cpu().numpy()
        feats_pooled[start:end] = rep_np.mean(axis=(1, 2))
        feats_per_joint[start:end] = rep_np.mean(axis=1)

        if (start // args.batch_size) % 5 == 0:
            print(f"  [{end}/{num_clips}] {time.time()-t0:.1f}s", flush=True)

    print(f"[extract] pooled shape {feats_pooled.shape}, per-joint shape {feats_per_joint.shape}")
    print(f"[extract] mean var per dim (pooled): {feats_pooled.var(axis=0).mean():.4f}")
    print(f"[extract] dead dims (pooled, std<1e-4): {(feats_pooled.std(axis=0)<1e-4).sum()}/512")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        filenames=filenames,
        features_pooled=feats_pooled,
        features_per_joint=feats_per_joint,
    )
    print(f"[extract] saved {args.output} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
