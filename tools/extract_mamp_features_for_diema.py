"""exp085 — extract frozen MAMP encoder features for DIEM-A clips.

For each DIEM-A clip in a motion_*_quat.npz:
  1. Project BVH-24 → NTU-25 3D via diema.features.bvh_to_ntu25
  2. Feed (N, C=3, T=120, V=25, M=1) through the frozen MAMP encoder
     (model.transformer.Transformer, encoder weights from pretrain ckpt)
  3. Aggregate the post-norm token grid (N, M, TP, VP, dim_feat):
     - pooled    = mean over (M, TP, VP)        → (N, 256)
       (matches ActionHeadLinprobe in tmp/MAMP/model/transformer.py)
     - per_joint = mean over (M, TP)            → (N, 25, 256)
  4. Save both variants for sklearn linear probe (mirrors exp081 MB).

Usage:
  conda run -n acii2026 python tools/extract_mamp_features_for_diema.py \
      --motion-npz data/diema_challenge/processed/motion_train_quat.npz \
      --checkpoint data/mamp/checkpoints/ntu60_xsub.pth \
      --output data/diema_challenge/processed/mamp_ntu60xsub_train.npz \
      --normalize unit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MAMP_SRC = REPO_ROOT / "tmp" / "MAMP"
if str(MAMP_SRC) not in sys.path:
    sys.path.insert(0, str(MAMP_SRC))

from diema.features.bvh_to_ntu25 import project_bvh_to_ntu25

# MAMP pretrain architecture (from ckpt['args'].model_args, ntu60_xsub):
MAMP_ARCH = dict(
    dim_in=3, dim_feat=256, depth=8, num_heads=8, mlp_ratio=4,
    num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
    qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
)


def build_mamp_encoder() -> torch.nn.Module:
    """Downstream MAMP Transformer (protocol='linprobe'); head unused."""
    from functools import partial

    import torch.nn as nn
    from model.transformer import Transformer

    model = Transformer(
        num_classes=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        protocol="linprobe",
        **MAMP_ARCH,
    )
    return model


def load_pretrained(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_model = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = model.state_dict()
    # Drop head keys (downstream-only) + any shape-mismatch keys
    for k in list(checkpoint_model.keys()):
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"[load] drop shape-mismatch {k}")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    enc_missing = [
        k for k in msg.missing_keys if not k.startswith("head.")
    ]
    if enc_missing:
        print(f"[load] WARN encoder missing keys: {len(enc_missing)} "
              f"(first 5: {enc_missing[:5]})")
    else:
        print("[load] all encoder weights loaded (only head.* missing — expected)")
    n_decoder_unexpected = sum(
        1 for k in msg.unexpected_keys if k.startswith("decoder") or k == "mask_token"
    )
    print(f"[load] ignored {n_decoder_unexpected} decoder/mask_token keys (expected)")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def forward_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Replicate model.transformer.Transformer.forward up to (not incl) head.

    x: (N, C=3, T=120, V=25, M=1) → returns (N, M, TP, VP, dim_feat).
    """
    N, C, T, V, M = x.shape
    x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
    x = model.joints_embed(x)
    NM, TP, VP, _ = x.shape
    x = x + model.pos_embed[:, :, :VP, :] + model.temp_embed[:, :TP, :, :]
    x = x.reshape(NM, TP * VP, -1)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    x = x.reshape(N, M, TP, VP, -1)
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-frames", type=int, default=120)
    parser.add_argument("--normalize", default="unit",
                        choices=["unit", "center", "none"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_pretrained(build_mamp_encoder(), args.checkpoint, device)

    npz = np.load(args.motion_npz, allow_pickle=True)
    num_clips = int(npz["num_clips"])
    filenames = list(npz["filenames"])
    print(f"[extract] {num_clips} clips from {args.motion_npz} "
          f"(normalize={args.normalize})", flush=True)

    feats_pooled = np.zeros((num_clips, MAMP_ARCH["dim_feat"]), dtype=np.float32)
    feats_per_joint = np.zeros(
        (num_clips, MAMP_ARCH["num_joints"], MAMP_ARCH["dim_feat"]), dtype=np.float32
    )

    t0 = time.time()
    for start in range(0, num_clips, args.batch_size):
        end = min(start + args.batch_size, num_clips)
        batch = np.zeros((end - start, args.target_frames, 25, 3), dtype=np.float32)
        for j, i in enumerate(range(start, end)):
            rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
            jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
            batch[j] = project_bvh_to_ntu25(
                rp, jq, target_frames=args.target_frames, normalize=args.normalize
            )
        # (B, T, V, 3) → (B, C=3, T, V, M=1)
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(-1).to(device)
        feat = forward_features(model, x)  # (B, M, TP, VP, D)
        feats_pooled[start:end] = feat.mean(dim=(1, 2, 3)).cpu().numpy()
        feats_per_joint[start:end] = feat.mean(dim=(1, 2)).cpu().numpy()

        if (start // args.batch_size) % 10 == 0:
            print(f"  [{end}/{num_clips}] {time.time()-t0:.1f}s", flush=True)

    print(f"[extract] pooled {feats_pooled.shape}, per-joint {feats_per_joint.shape}")
    print(f"[extract] pooled mean-var/dim {feats_pooled.var(axis=0).mean():.4f}, "
          f"dead dims {(feats_pooled.std(axis=0)<1e-4).sum()}/{MAMP_ARCH['dim_feat']}")

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
