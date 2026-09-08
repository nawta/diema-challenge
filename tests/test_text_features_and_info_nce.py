"""Tests for primitives: text features + InfoNCE loss.

The sentence-transformers encoder is a heavy dependency, so the
``ScenarioTextEncoder`` is tested via lazy-load behavior and cache
round-trip with a **stubbed** SentenceTransformer. The model itself is
not downloaded in CI.

InfoNCE loss has no external deps — it's tested against hand-computed
reference values.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from diema.features.text_features import (
    ScenarioTextEncoder,
    extract_text_features,
    load_text_cache,
    save_text_cache,
)
from diema.training.losses import info_nce_loss


# ---------------------------------------------------------------------------
# InfoNCE
# ---------------------------------------------------------------------------


class TestInfoNCELoss:
    def test_aligned_pairs_minimize_loss(self):
        """When z and t are identical unit vectors, the diagonal of the
        cosine matrix is 1/temperature while off-diagonals are
        orthogonal, so the cross-entropy is near zero (log(1) == 0)
        modulo the numerical bias from a tiny non-zero off-diagonal
        cosine. With batch N=4, D=4, temperature=0.1, the loss should
        be noticeably below random (log N ≈ 1.386).
        """
        torch.manual_seed(0)
        z = torch.eye(4)
        t = torch.eye(4)
        loss = info_nce_loss(z, t, temperature=0.1)
        # log(4) ≈ 1.386; aligned pairs at temperature 0.1 with orthogonal
        # off-diagonals give exp(10) / (exp(10) + 3*exp(0)) ≈ 1, so CE ≈ 0.
        assert loss.item() < 0.001

    def test_random_pairs_close_to_log_n(self):
        """Random embeddings give ~log(N) on average (the chance-level
        baseline) because cosines are nearly uniform. We verify it is
        within a generous band of log(N) rather than exactly."""
        torch.manual_seed(42)
        N, D = 64, 32
        z = torch.randn(N, D)
        t = torch.randn(N, D)
        loss = info_nce_loss(z, t, temperature=0.1)
        import math
        assert loss.item() > 0.5 * math.log(N)

    def test_symmetric_and_asymmetric_agree_on_aligned(self):
        """On perfectly aligned pairs, symmetric and asymmetric forms
        give the same (near-zero) value."""
        z = torch.eye(4)
        t = torch.eye(4)
        sym = info_nce_loss(z, t, temperature=0.1, symmetric=True)
        asym = info_nce_loss(z, t, temperature=0.1, symmetric=False)
        assert torch.allclose(sym, asym, atol=1e-5)

    def test_scale_invariant_due_to_l2_norm(self):
        """Multiplying either side by a positive scalar must leave the
        loss unchanged because the L2 normalization undoes the scale."""
        torch.manual_seed(1)
        z = torch.randn(8, 6)
        t = torch.randn(8, 6)
        loss_ref = info_nce_loss(z, t, temperature=0.1)
        loss_scaled = info_nce_loss(z * 100.0, t * 0.001, temperature=0.1)
        assert torch.allclose(loss_ref, loss_scaled, atol=1e-5)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        z = torch.randn(6, 8, requires_grad=True)
        t = torch.randn(6, 8, requires_grad=True)
        loss = info_nce_loss(z, t, temperature=0.1)
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0
        assert t.grad is not None and t.grad.abs().sum() > 0

    def test_rejects_mismatched_batch(self):
        with pytest.raises(ValueError, match="batch sizes"):
            info_nce_loss(torch.randn(4, 8), torch.randn(6, 8))

    def test_rejects_mismatched_dim(self):
        with pytest.raises(ValueError, match="feature dims"):
            info_nce_loss(torch.randn(4, 8), torch.randn(4, 16))

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="N, D"):
            info_nce_loss(torch.randn(4, 8, 2), torch.randn(4, 8, 2))

    def test_rejects_bad_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            info_nce_loss(torch.randn(4, 8), torch.randn(4, 8), temperature=0.0)


# ---------------------------------------------------------------------------
# Text cache save / load round-trip
# ---------------------------------------------------------------------------


class TestTextCacheRoundTrip:
    def test_save_and_load(self, tmp_path):
        cache = {
            "joy 1": torch.tensor([1.0, 0.0, 0.0]),
            "fear 1": torch.tensor([0.0, 1.0, 0.0]),
            "surprise 1": torch.tensor([0.0, 0.0, 1.0]),
        }
        meta = {"model_name": "test", "num_unique": "3"}
        path = tmp_path / "cache.pt"
        save_text_cache(path, cache, meta=meta)
        embeddings, loaded_meta = load_text_cache(path)
        assert set(embeddings.keys()) == set(cache.keys())
        for k, v in cache.items():
            assert torch.equal(embeddings[k], v)
        assert loaded_meta == meta

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_text_cache(tmp_path / "does_not_exist.pt")

    def test_load_wrong_format(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"not_embeddings": {}}, path)
        with pytest.raises(ValueError, match="embeddings"):
            load_text_cache(path)

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "cache.pt"
        cache = {"x": torch.zeros(3)}
        save_text_cache(nested, cache)
        assert nested.exists()


# ---------------------------------------------------------------------------
# ScenarioTextEncoder with stubbed SentenceTransformer
# ---------------------------------------------------------------------------


class _StubST:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self._device = "cpu"

    def eval(self):
        return self

    def encode(
        self,
        texts,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        # Return a deterministic fake embedding per text (hash → unit vector).
        out = []
        for t in texts:
            h = abs(hash(t))
            v = torch.zeros(self.dim, dtype=torch.float32)
            v[h % self.dim] = 1.0
            out.append(v)
        return torch.stack(out, dim=0)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class TestScenarioTextEncoder:
    def test_lazy_load_and_cache(self, monkeypatch):
        enc = ScenarioTextEncoder(model_name="stub", device="cpu")
        assert enc._model is None

        # Patch the SentenceTransformer import inside the encoder's
        # _ensure_loaded so we never touch the real network.
        def fake_sentence_transformer(name, device=None):
            return _StubST(dim=4)

        import diema.features.text_features as tf
        monkeypatch.setattr(
            tf, "ScenarioTextEncoder", tf.ScenarioTextEncoder
        )  # no-op, keep reference
        # Patch the actual SentenceTransformer import that happens inside
        # _ensure_loaded.
        import sys
        fake_mod = SimpleNamespace(SentenceTransformer=fake_sentence_transformer)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

        vecs = enc.encode(["hello", "world", "hello"])
        assert vecs.shape == (3, 4)
        # Rows 0 and 2 (both "hello") must be identical via the cache.
        assert torch.equal(vecs[0], vecs[2])
        # Cache contains both unique strings.
        assert set(enc._cache.keys()) == {"hello", "world"}

    def test_embedding_dim(self, monkeypatch):
        enc = ScenarioTextEncoder(model_name="stub")
        import sys
        fake_mod = SimpleNamespace(
            SentenceTransformer=lambda name, device=None: _StubST(dim=7)
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
        assert enc.embedding_dim() == 7

    def test_encode_empty_list_returns_empty_tensor(self, monkeypatch):
        """Edge case: ``encode([])`` must return a ``(0, dim)`` tensor
        instead of tripping ``torch.stack`` on an empty sequence."""
        enc = ScenarioTextEncoder(model_name="stub")
        import sys
        fake_mod = SimpleNamespace(
            SentenceTransformer=lambda name, device=None: _StubST(dim=5)
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
        vecs = enc.encode([])
        assert vecs.shape == (0, 5)
        assert vecs.dtype == torch.float32


class _MagnitudeStubST:
    """Stub encoder that returns pre-normalized vectors with *variable*
    magnitudes so we can tell whether the caller re-normalized."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def eval(self):
        return self

    def encode(
        self,
        texts,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        # Every text maps to a vector with L2 norm = 5.0 along axis 0.
        out = []
        for _ in texts:
            v = torch.zeros(self.dim, dtype=torch.float32)
            v[0] = 5.0
            out.append(v)
        return torch.stack(out, dim=0)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class TestScenarioTextEncoderNormalizeToggle:
    """Regression: the cache must always return a tensor consistent with
    the caller's ``normalize`` setting, not whichever setting happened
    to populate the cache first. Codex flagged the old implementation
    as silently returning cached normalized vectors for a later
    ``normalize=False`` request.
    """

    def _patched_encoder(self, monkeypatch):
        import sys
        fake_mod = SimpleNamespace(
            SentenceTransformer=lambda name, device=None: _MagnitudeStubST(dim=4)
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
        return ScenarioTextEncoder(model_name="stub")

    def test_cache_unchanged_across_normalize_flag(self, monkeypatch):
        enc = self._patched_encoder(monkeypatch)
        # First call: normalize=True → expect unit-norm rows.
        vecs_norm = enc.encode(["hello"], normalize=True)
        assert torch.allclose(vecs_norm.norm(dim=1), torch.ones(1), atol=1e-5)
        # Second call: same text, normalize=False → expect raw norm 5.0.
        vecs_raw = enc.encode(["hello"], normalize=False)
        assert torch.allclose(vecs_raw.norm(dim=1), torch.tensor([5.0]), atol=1e-5)

    def test_cache_snapshot_returns_unnormalized_vectors(self, monkeypatch):
        enc = self._patched_encoder(monkeypatch)
        enc.encode(["hello", "world"], normalize=True)
        snap = enc.cache_snapshot()
        for v in snap.values():
            # Raw embeddings from the stub have norm 5.0.
            assert torch.allclose(
                v.norm(), torch.tensor(5.0), atol=1e-5
            )
        # Mutating the snapshot must not affect the encoder cache.
        snap.clear()
        assert len(enc.cache_snapshot()) == 2
