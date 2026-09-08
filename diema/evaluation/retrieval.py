""" Motion ↔ Text retrieval metrics (exp054 probe).

See: (exp054 retrieval probe)
Related:
- diema/features/text_cache.py (ScenarioTextCache — global query target)
- diema/features/rationale_text_cache.py (6-part rationale target)
- tools/eval_motion_text_retrieval.py (CLI wrapper)

Pure-tensor API so tests can exercise every branch with synthetic data and
the CLI layer can feed it arbitrary checkpoint outputs.

Primary entry point :func:`motion_to_text_retrieval`:

    z_motion: (N, D)     L2-normalized or not
    t_text:   (M, D)     candidate targets (M may equal N or a larger pool)
    positive_map: list or LongTensor of length N — the ground-truth target
                  index for each query in ``t_text``
    labels:   optional (N,) LongTensor for same-class consistency
    groups:   optional (N,) LongTensor for cross-group clustering

Returns a ``dict[str, float]`` with ``recall_at_{1,5,10}``, ``medr``,
``same_class_topk_frac`` (if labels given), ``cross_group_topk_frac`` (if
groups given), and ``n_queries``.

All three metrics are implemented as tensor ops with **no side effects**
so they run on CPU or GPU, fp32 or bf16, without allocation surprises.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

__all__ = [
    "motion_to_text_retrieval",
    "recall_at_k",
    "median_rank",
    "same_class_topk_fraction",
    "cross_group_topk_fraction",
]


def _to_long_tensor(x, device: torch.device, name: str) -> torch.Tensor:
    """Coerce a list / tensor-like into a 1-D long tensor on ``device``.

    ``name`` is only used in the error message.
    """
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=torch.long)
    else:
        t = torch.as_tensor(list(x), device=device, dtype=torch.long)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(t.shape)}")
    return t


def _similarity_matrix(
    z: torch.Tensor, t: torch.Tensor, normalize: bool,
) -> torch.Tensor:
    """Return cosine similarity (if ``normalize``) or dot-product matrix.

    Shapes: ``z (N, D)``, ``t (M, D)`` → ``(N, M)``.
    """
    if z.ndim != 2 or t.ndim != 2:
        raise ValueError(
            f"expected 2-D tensors (N, D) and (M, D); got z={tuple(z.shape)} t={tuple(t.shape)}"
        )
    if z.shape[1] != t.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: z has D={z.shape[1]}, t has D={t.shape[1]}"
        )
    if normalize:
        z = F.normalize(z, dim=1)
        t = F.normalize(t, dim=1)
    return z @ t.t()


def recall_at_k(
    sim: torch.Tensor, positive: torch.Tensor, ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Fraction of queries whose positive is in top-k by similarity.

    Args:
        sim: (N, M) similarity matrix (bigger = closer).
        positive: (N,) long tensor of the correct target index in [0, M).
        ks: tuple of k values; keys are ``recall_at_{k}``.

    Ties are broken by argsort's stable default (lower index wins), which
    is fine for paper-facing reporting but should be flagged in the Go
    threshold if models tie-heavy (unlikely with 384-d embeddings).
    """
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2-D, got {tuple(sim.shape)}")
    N, M = sim.shape
    if positive.shape != (N,):
        raise ValueError(
            f"positive shape {tuple(positive.shape)} does not match sim[0]={N}"
        )
    # Sort descending. Indexing: top-k ids per row.
    order = sim.argsort(dim=1, descending=True)  # (N, M)
    out: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0 or k > M:
            raise ValueError(f"k={k} must be in (0, M={M}]")
        topk = order[:, :k]
        hit = (topk == positive.unsqueeze(1)).any(dim=1)
        out[f"recall_at_{k}"] = float(hit.float().mean().item())
    return out


def median_rank(
    sim: torch.Tensor, positive: torch.Tensor,
) -> float:
    """Median of the 1-indexed rank of the positive target over the batch."""
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2-D, got {tuple(sim.shape)}")
    N, _M = sim.shape
    if positive.shape != (N,):
        raise ValueError(
            f"positive shape {tuple(positive.shape)} does not match sim[0]={N}"
        )
    order = sim.argsort(dim=1, descending=True)  # (N, M)
    # Rank of positive: position in each row where order == positive[i], +1.
    row_eq = order == positive.unsqueeze(1)
    rank = row_eq.float().argmax(dim=1) + 1
    return float(rank.median().item())


def same_class_topk_fraction(
    sim: torch.Tensor, labels: torch.Tensor, k: int = 5,
) -> float:
    """Fraction of top-k neighbors that share the query's label.

    Uses the same similarity matrix — self-index is **excluded** if it
    happens to be in the top-k (diagonal filled with -inf before argsort).
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("same_class_topk_fraction needs a square sim matrix")
    if labels.shape != (sim.shape[0],):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match sim[0]={sim.shape[0]}"
        )
    # Mask self along diagonal so we don't trivially count the query itself.
    N = sim.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=sim.device)
    masked = sim.masked_fill(eye, float("-inf"))
    order = masked.argsort(dim=1, descending=True)
    topk = order[:, :k]
    neighbors_labels = labels[topk]                      # (N, k)
    query_labels = labels.unsqueeze(1)                   # (N, 1)
    return float((neighbors_labels == query_labels).float().mean().item())


def cross_group_topk_fraction(
    sim: torch.Tensor, groups: torch.Tensor, labels: torch.Tensor | None = None,
    k: int = 5,
) -> float:
    """Within samples that share the same label, fraction of top-k neighbors
    that come from a DIFFERENT group (e.g., different performer).

    Useful to detect trivial clustering by performer. When labels is None,
    this computes the global cross-group rate across all queries regardless
    of class. With labels, only same-label query-neighbor pairs contribute
    to the numerator (so the metric answers: "within a class, are
    neighbors diverse across groups?").
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("cross_group_topk_fraction needs a square sim matrix")
    N = sim.shape[0]
    if groups.shape != (N,):
        raise ValueError(
            f"groups shape {tuple(groups.shape)} must match sim[0]={N}"
        )
    eye = torch.eye(N, dtype=torch.bool, device=sim.device)
    masked = sim.masked_fill(eye, float("-inf"))
    order = masked.argsort(dim=1, descending=True)
    topk = order[:, :k]
    neighbor_groups = groups[topk]                       # (N, k)
    query_groups = groups.unsqueeze(1)                   # (N, 1)
    different_group = neighbor_groups != query_groups    # (N, k)

    if labels is None:
        return float(different_group.float().mean().item())

    if labels.shape != (N,):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match sim[0]={N}"
        )
    neighbor_labels = labels[topk]
    query_labels = labels.unsqueeze(1)
    same_label = neighbor_labels == query_labels         # (N, k)
    total = same_label.sum()
    if total.item() == 0:
        # No same-label neighbor pairs at all (possible with singleton classes);
        # returning 0 is safe but uninformative. Surface as 0.0.
        return 0.0
    numerator = (different_group & same_label).sum().float()
    return float((numerator / total.float()).item())


def motion_to_text_retrieval(
    z_motion: torch.Tensor,
    t_text: torch.Tensor,
    positive_map,
    *,
    ks: Iterable[int] = (1, 5, 10),
    labels: torch.Tensor | None = None,
    groups: torch.Tensor | None = None,
    same_class_k: int = 5,
    cross_group_k: int = 5,
    normalize: bool = True,
) -> dict[str, float]:
    """Batch-wise motion→text retrieval with optional diagnostic metrics.

    Args:
        z_motion: (N, D) motion projections.
        t_text:   (M, D) text candidate pool (M may equal N).
        positive_map: length-N list / tensor of indices into ``t_text``
            giving the correct target per query. Values in [-1, M); -1
            means "no positive" and the query is excluded from R@k / MedR
            (but still counted in same-class / cross-group if given).
        ks: Recall@k thresholds.
        labels: (N,) class labels for same-class consistency.
        groups: (N,) group labels for cross-group clustering (e.g., performer id).
        same_class_k / cross_group_k: neighborhood sizes.
        normalize: L2-normalize before computing similarity (default True).

    Returns:
        dict mapping ``recall_at_{k}``, ``medr``, optional
        ``same_class_top{k}_frac`` / ``cross_group_top{k}_frac``,
        ``n_queries``, ``n_valid_queries``.
    """
    device = z_motion.device
    sim = _similarity_matrix(z_motion, t_text, normalize=normalize)
    pos = _to_long_tensor(positive_map, device=device, name="positive_map")
    if pos.shape != (z_motion.shape[0],):
        raise ValueError(
            f"positive_map length {pos.numel()} does not match z_motion[0]={z_motion.shape[0]}"
        )

    valid_mask = pos >= 0
    n_valid = int(valid_mask.sum().item())
    out: dict[str, float] = {
        "n_queries": float(z_motion.shape[0]),
        "n_valid_queries": float(n_valid),
    }
    if n_valid > 0:
        sim_v = sim[valid_mask]
        pos_v = pos[valid_mask]
        out.update(recall_at_k(sim_v, pos_v, ks=ks))
        out["medr"] = median_rank(sim_v, pos_v)
    else:
        for k in ks:
            out[f"recall_at_{int(k)}"] = 0.0
        out["medr"] = float("nan")

    # Diagnostics use a symmetric N×N similarity (motion ↔ motion via text
    # anchor? No — we reuse the motion-to-text sim only if N == M and the
    # positives happen to form the diagonal. Otherwise, we compute a fresh
    # motion-to-motion similarity for the diagnostics). The spec calls for
    # same-emotion / cross-performer *retrieval consistency* on the val
    # sample, which is naturally a motion-motion neighborhood.
    if labels is not None or groups is not None:
        sim_mm = _similarity_matrix(z_motion, z_motion, normalize=normalize)
        if labels is not None:
            out[f"same_class_top{same_class_k}_frac"] = same_class_topk_fraction(
                sim_mm, labels, k=same_class_k,
            )
        if groups is not None:
            out[f"cross_group_top{cross_group_k}_frac"] = cross_group_topk_fraction(
                sim_mm, groups, labels=labels, k=cross_group_k,
            )
    return out
