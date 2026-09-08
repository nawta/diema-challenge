"""exp086 — extract frozen PoseC3D (ResNet3dSlowOnly) features for DIEM-A.

For each DIEM-A clip:
  1. BVH-24 → COCO-17 Gaussian heatmap volume (17, 48, 64, 64)
     via diema.features.bvh_to_coco17_heatmap (PYSKL val_pipeline replica)
  2. Forward the frozen clean-room ResNet3dSlowOnly port
     (diema.models.posec3d.resnet3d_slowonly_port, weights from
      data/pyskl/checkpoints/poseconv3d_ntu60_xsub_joint.pth)
  3. Global average pool over (T', H', W') → (N, 512) pooled feature
  4. Save for sklearn linear probe (mirrors exp081/exp085).

Usage:
  conda run -n acii2026 python tools/extract_posec3d_features_for_diema.py \
      --motion-npz data/diema_challenge/processed/motion_train_quat.npz \
      --checkpoint data/pyskl/checkpoints/poseconv3d_ntu60_xsub_joint.pth \
      --output data/diema_challenge/processed/posec3d_ntu60xsub_train.npz
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from diema.features.bvh_to_coco17_heatmap import project_bvh_to_coco17_heatmap
from diema.models.posec3d.resnet3d_slowonly_port import (
    build_posec3d_backbone,
    load_backbone_checkpoint,
)


def _make_vol(args):
    rp, jq, clip_len, hw, sigma = args
    return project_bvh_to_coco17_heatmap(
        rp, jq, clip_len=clip_len, hw=hw, sigma=sigma
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motion-npz", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--hw", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_posec3d_backbone()
    rep = load_backbone_checkpoint(model, args.checkpoint)
    print(f"[load] {rep} from {args.checkpoint}")
    model.eval().to(device)
    for q in model.parameters():
        q.requires_grad_(False)

    npz = np.load(args.motion_npz, allow_pickle=True)
    num_clips = int(npz["num_clips"])
    filenames = list(npz["filenames"])
    print(f"[extract] {num_clips} clips from {args.motion_npz}", flush=True)

    feats = np.zeros((num_clips, model.feat_dim), dtype=np.float32)
    t0 = time.time()
    for start in range(0, num_clips, args.batch_size):
        end = min(start + args.batch_size, num_clips)
        jobs = []
        for i in range(start, end):
            rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
            jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
            jobs.append((rp, jq, args.clip_len, args.hw, args.sigma))
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                vols = list(ex.map(_make_vol, jobs))
        else:
            vols = [_make_vol(j) for j in jobs]
        x = torch.from_numpy(np.stack(vols, axis=0)).to(device)  # (B,17,T,H,W)
        with torch.no_grad():
            y = model(x)  # (B, 512, T', H', W')
        feats[start:end] = y.mean(dim=(2, 3, 4)).cpu().numpy()
        if (start // args.batch_size) % 20 == 0:
            print(f"  [{end}/{num_clips}] {time.time()-t0:.1f}s", flush=True)

    print(f"[extract] pooled {feats.shape}, mean-var/dim "
          f"{feats.var(axis=0).mean():.4f}, dead "
          f"{(feats.std(axis=0)<1e-4).sum()}/{model.feat_dim}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, filenames=filenames, features_pooled=feats)
    print(f"[extract] saved {args.output} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
