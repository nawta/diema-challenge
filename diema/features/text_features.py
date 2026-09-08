"""Scenario text encoding for train-time text supervision.

At training time we can attach a frozen sentence-transformer embedding
of each sample's scenario description to the batch and use it as the
target side of an InfoNCE contrastive loss against a motion-side
projection. Inference never sees the text encoder — ``test_data.csv``
does not include scenario text, so the encoder is used strictly for
train-time supervision and is discarded when saving the final model.

This module provides:

  - :class:`ScenarioTextEncoder` — a thin wrapper around a frozen
    sentence-transformers model that caches embeddings keyed by the
    raw text (deduplication across the 7,992 train samples → 2,623
    unique scenarios).
  - :func:`load_text_cache` — read a ``.pt`` cache built by
    ``tools/build_scenario_cache.py``.
  - :func:`save_text_cache` — write a cache for downstream
    :class:`diema.data.dataset.MotionDataset` consumption.

Design notes:
    The sentence-transformers model is held on CPU by default to avoid
    taking a non-trivial slice of GPU memory during training (we only
    need it to *build* the cache, not to run every step). At train
    time, the dataset simply does a dict lookup on the cached
    ``torch.Tensor``.

Related: tools/build_scenario_cache.py (offline cache builder),
         diema/training/losses.py::info_nce_loss
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


DEFAULT_TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_DIM = 384


class ScenarioTextEncoder:
    """Frozen sentence-transformers wrapper with per-text memoization.

    Lazily loads the sentence-transformers model on first encode. The
    encoder is always placed in ``eval()`` mode and wrapped in
    ``torch.no_grad()`` during :meth:`encode` calls — callers should
    treat its outputs as detached constants.

    Args:
        model_name: sentence-transformers checkpoint name. Defaults to
            ``all-MiniLM-L6-v2`` (384-dim, English-only, <100 MB).
        device: device for the encoder. Defaults to CPU to keep GPU
            memory free for the training model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_ENCODER,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy
        self._cache: dict[str, torch.Tensor] = {}

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Imported here so `import diema.features.text_features` does
            # not pay the sentence-transformers dependency cost unless the
            # encoder is actually used.
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
            model.eval()
            self._model = model

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode a list of texts into a ``(N, dim)`` tensor.

        Texts that have already been encoded (by string equality) are
        served from the in-memory cache; the rest are batched through
        the model in a single call. The cache always stores
        **unnormalized** vectors — when ``normalize=True`` the caller
        gets an L2-normalized copy computed on the fly. This avoids
        stale cache entries when the same text is encoded under
        different ``normalize`` settings in a single process.

        Args:
            texts: list of raw scenario strings.
            normalize: if True, return L2-normalized rows so that
                ``(encode(a) * encode(b)).sum(-1)`` is cosine similarity.
                Defaults to True.

        Returns:
            Tensor of shape ``(len(texts), dim)`` on the encoder's device.
        """
        self._ensure_loaded()
        if len(texts) == 0:
            # torch.stack on an empty list throws; return a 0-row tensor
            # with the encoder's embedding dimension. Place it on the
            # encoder's device so callers can feed it into a collate /
            # contrastive loss without an extra ``.to(...)`` hop.
            return torch.empty(
                0, self.embedding_dim(),
                dtype=torch.float32, device=self.device,
            )
        # Partition into cached and missing.
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        results: list[torch.Tensor | None] = [None] * len(texts)
        for i, t in enumerate(texts):
            cached = self._cache.get(t)
            if cached is not None:
                results[i] = cached
            else:
                missing_texts.append(t)
                missing_positions.append(i)

        if missing_texts:
            assert self._model is not None
            # Cache the raw (unnormalized) embeddings. We apply
            # normalization below on the full result tensor so the
            # caller's ``normalize`` flag is always honored, regardless
            # of what previous callers requested.
            vecs = self._model.encode(
                missing_texts,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            # Ensure float32 regardless of the encoder's default dtype.
            vecs = vecs.to(torch.float32)
            for pos, text, vec in zip(missing_positions, missing_texts, vecs):
                self._cache[text] = vec
                results[pos] = vec

        assert all(r is not None for r in results), (
            "internal error: results tensor should be fully populated "
            "after cache lookup + missing-text encode"
        )
        stacked = torch.stack(results, dim=0)  # type: ignore[arg-type]
        if normalize:
            stacked = F.normalize(stacked, p=2, dim=1)
        return stacked

    def cache_snapshot(self) -> dict[str, torch.Tensor]:
        """Return a copy of the in-memory ``text → embedding`` cache.

        Used by :mod:`tools.build_scenario_cache` (and any other offline
        tool) instead of reaching into ``self._cache`` directly.
        """
        return dict(self._cache)

    def embedding_dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        # sentence-transformers 5.x renamed the accessor; fall back to
        # the old name for 4.x and earlier.
        fn = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or getattr(self._model, "get_sentence_embedding_dimension")
        return int(fn())


def save_text_cache(
    path: str | Path,
    cache: Mapping[str, torch.Tensor],
    meta: Mapping[str, str] | None = None,
) -> None:
    """Serialize a ``text → tensor`` cache for offline reuse.

    The stored file contains a dict with keys ``"embeddings"``
    (``dict[str, Tensor]``) and ``"meta"`` (a free-form ``dict[str, str]``).
    """
    payload = {
        "embeddings": {k: v.detach().cpu() for k, v in cache.items()},
        "meta": dict(meta or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_text_cache(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load a cache written by :func:`save_text_cache`.

    Returns:
        Tuple ``(embeddings, meta)``. ``embeddings`` maps raw scenario
        text to a ``torch.Tensor`` vector; ``meta`` contains arbitrary
        provenance metadata (model name, generation timestamp, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"text cache not found: {path}")
    # Use weights_only=False so we can deserialize the dict structure.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(
            f"text cache at {path} is missing the 'embeddings' key; "
            "rebuild with tools/build_scenario_cache.py"
        )
    embeddings = payload["embeddings"]
    meta = payload.get("meta", {})
    return dict(embeddings), dict(meta)


def extract_text_features(texts: list[str], model_name: str = DEFAULT_TEXT_ENCODER) -> torch.Tensor:
    """Convenience wrapper around a one-off :class:`ScenarioTextEncoder`.

    Prefer constructing the encoder once and reusing it if you are
    encoding many texts.
    """
    encoder = ScenarioTextEncoder(model_name=model_name)
    return encoder.encode(texts)
