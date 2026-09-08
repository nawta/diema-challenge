""" per-part projection-norm diagnostics.

See: (exp052 "part importance 保存")
Related:
- diema/models/conv1d_transformer/partalign_conv1d_tr_model.py (emits z_parts_raw)
- diema/training/part_align_trainer.py (trainer that drives z_parts into the loss)
- tools/export_part_importance.py (CLI wrapper over these helpers)
- docs/experiments.md (part-masking faithfulness, the twin lens on part importance)

Two complementary signals are computed per body part, for every sample whose
rationale was non-placeholder (part_mask=True):

1. ``norm_stats``      — mean / std of ``||z_parts_raw[:, p, :]||_2``
   across the val split. Paints "how much signal the model routed to this
   part on average".
2. ``per_emotion_norm`` — mean norm per ``(emotion, part)``. Reveals
   whether some emotions make the model lean on e.g. the head part more
   than others.
3. ``per_emotion_cosine`` — for each emotion, the cosine-sim between its
   mean normalized part vector and the pooled mean across all emotions.
   Low cosine = that emotion's part signal is distinctive; high = shared.

All outputs are numpy-friendly CPU tensors — the CLI wrapper is then a
thin serializer.
"""

from __future__ import annotations

from typing import Sequence

import torch


def compute_part_importance(
    z_parts_raw: torch.Tensor,
    part_mask: torch.Tensor,
    labels: torch.Tensor,
    part_names: Sequence[str],
    emotion_names: Sequence[str] | None = None,
) -> dict:
    """Summarize per-part and per-(emotion, part) projection magnitudes.

    Args:
        z_parts_raw: ``(N, P, D)`` pre-normalize projections.
        part_mask:   ``(N, P)`` bool. False = placeholder part (exclude).
        labels:      ``(N,)`` int64 emotion ids.
        part_names:  P-tuple / list with the canonical part order.
        emotion_names: optional label names; if None, ids are stringified.

    Returns dict with:
        part_names, emotion_names
        norm_stats:            list[{part, n, mean, std}]
        per_emotion_norm:      list[{emotion, part, n, mean, std}]
        per_emotion_cosine:    list[{emotion, part, cosine_to_global}]
    """
    if z_parts_raw.ndim != 3:
        raise ValueError(
            f"z_parts_raw must be 3-D (N, P, D), got {tuple(z_parts_raw.shape)}"
        )
    N, P, D = z_parts_raw.shape
    if part_mask.shape != (N, P):
        raise ValueError(
            f"part_mask shape {tuple(part_mask.shape)} does not match (N={N}, P={P})"
        )
    if labels.shape != (N,):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} does not match N={N}"
        )
    if len(part_names) != P:
        raise ValueError(f"part_names length {len(part_names)} does not match P={P}")

    z = z_parts_raw.detach().to(torch.float32)
    m = part_mask.bool()

    # Per-part norm stats over all valid (sample, part) cells.
    norms = z.norm(dim=-1)       # (N, P)
    norm_stats: list[dict] = []
    for p in range(P):
        valid = m[:, p]
        count = int(valid.sum().item())
        if count == 0:
            norm_stats.append({
                "part": part_names[p], "n": 0,
                "mean": 0.0, "std": 0.0,
            })
            continue
        vals = norms[valid, p]
        norm_stats.append({
            "part": part_names[p],
            "n": count,
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()) if count > 1 else 0.0,
        })

    # Per-emotion per-part norm stats.
    emo_unique = sorted({int(v) for v in labels.tolist()})
    emo_names = list(emotion_names) if emotion_names is not None else [str(e) for e in emo_unique]
    name_by_id = {e_id: emo_names[i] for i, e_id in enumerate(emo_unique)} \
        if emotion_names is not None and len(emo_names) == len(emo_unique) \
        else {e_id: str(e_id) for e_id in emo_unique}

    per_emotion_norm: list[dict] = []
    # Cache per-emotion per-part mean normalized vector for the cosine section.
    normed = torch.nn.functional.normalize(z, dim=-1)    # (N, P, D)
    per_emo_mean_normed: dict[int, torch.Tensor] = {}    # (P, D)

    for e_id in emo_unique:
        emo_mask = labels == e_id
        e_mean_normed = torch.zeros(P, D, dtype=torch.float32)
        for p in range(P):
            cell_mask = emo_mask & m[:, p]
            count = int(cell_mask.sum().item())
            if count == 0:
                per_emotion_norm.append({
                    "emotion": name_by_id[e_id], "part": part_names[p],
                    "n": 0, "mean": 0.0, "std": 0.0,
                })
                continue
            vals = norms[cell_mask, p]
            per_emotion_norm.append({
                "emotion": name_by_id[e_id],
                "part": part_names[p],
                "n": count,
                "mean": float(vals.mean().item()),
                "std": float(vals.std(unbiased=False).item()) if count > 1 else 0.0,
            })
            e_mean_normed[p] = normed[cell_mask, p].mean(dim=0)
        per_emo_mean_normed[e_id] = e_mean_normed

    # Global (cross-emotion) mean of normalized part vectors, per part.
    global_mean_normed = torch.zeros(P, D, dtype=torch.float32)
    for p in range(P):
        col_mask = m[:, p]
        if col_mask.any():
            global_mean_normed[p] = normed[col_mask, p].mean(dim=0)

    per_emotion_cosine: list[dict] = []
    for e_id in emo_unique:
        em_vec = per_emo_mean_normed[e_id]
        for p in range(P):
            # Cosine between per-emotion part mean and global part mean.
            denom = em_vec[p].norm().clamp_min(1e-12) * global_mean_normed[p].norm().clamp_min(1e-12)
            cos = float(((em_vec[p] * global_mean_normed[p]).sum() / denom).item())
            per_emotion_cosine.append({
                "emotion": name_by_id[e_id],
                "part": part_names[p],
                "cosine_to_global": cos,
            })

    return {
        "part_names": list(part_names),
        "emotion_ids": emo_unique,
        "emotion_names": [name_by_id[e] for e in emo_unique],
        "norm_stats": norm_stats,
        "per_emotion_norm": per_emotion_norm,
        "per_emotion_cosine": per_emotion_cosine,
    }
