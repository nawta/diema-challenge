"""Regression tests (2026-04-19).

See: internal review notes (2026-04-19).
Related:
- tools/explain_stability.py (Critical #1)
- diema/models/conv1d_transformer/textalign_conv1d_tr_model.py (Critical #2)
- tools/explain_part_masking.py (Critical #4, Warning #7)
- configs/leakage/forbidden_tokens.yaml (Warning #5)
- diema/training/losses_motion_text.py (Warning #6, #11)

Each test corresponds to a finding and would have failed before the fix.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


# ----------------------- Critical #1: explain_stability.py ------------------------


def test_stability_shared_noise_reproduces_across_calls() -> None:
    """Critical #1 — _sample_batch_noise must be deterministic under a fixed seed."""
    from tools.explain_stability import _sample_batch_noise

    # Build a trivial loader of 3 batches of shape (2, 6, 64, 25).
    class FakeLoader:
        def __iter__(self):
            for _ in range(3):
                yield (torch.zeros(2, 6, 64, 25), torch.zeros(2, dtype=torch.long), [""])

    a = _sample_batch_noise(FakeLoader(), torch.device("cpu"), noise_sigma=0.01, seed=42)
    b = _sample_batch_noise(FakeLoader(), torch.device("cpu"), noise_sigma=0.01, seed=42)
    assert a is not None and b is not None
    assert len(a) == len(b) == 3
    for i in range(3):
        assert torch.equal(a[i], b[i]), f"seed reproducibility broken at batch {i}"


def test_stability_local_generator_does_not_pollute_global_rng() -> None:
    """Critical #1 — the stability noise must not touch torch's global RNG."""
    from tools.explain_stability import _sample_batch_noise

    class FakeLoader:
        def __iter__(self):
            yield (torch.zeros(1, 6, 64, 25), torch.zeros(1, dtype=torch.long), [""])

    torch.manual_seed(1234)
    before = torch.randn(5)
    _ = _sample_batch_noise(FakeLoader(), torch.device("cpu"), noise_sigma=0.02, seed=99)
    torch.manual_seed(1234)
    after = torch.randn(5)
    assert torch.equal(before, after), \
        "_sample_batch_noise must use a local Generator, not touch torch.manual_seed"


def test_stability_zero_sigma_returns_none() -> None:
    from tools.explain_stability import _sample_batch_noise

    class FakeLoader:
        def __iter__(self):
            yield (torch.zeros(1, 6, 64, 25), torch.zeros(1, dtype=torch.long), [""])

    assert _sample_batch_noise(FakeLoader(), torch.device("cpu"), 0.0, seed=0) is None


# ------------- Critical #2: TextAlignConv1DTr_Model.from_config recursion --------


def test_textalign_from_config_rejects_self_as_backbone() -> None:
    """Critical #2 — using 'textalign_conv1d_tr' as backbone_name must be blocked."""
    from diema.models.conv1d_transformer.textalign_conv1d_tr_model import (
        TextAlignConv1DTr_Model,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="textalign_conv1d_tr",
            backbone_name="textalign_conv1d_tr",  # intentional misuse
            num_class=12, in_channels=6, dim=32, num_blocks=1, kernel_size=5,
            num_heads=2, mlp_ratio=2.0, drop_rate=0.0, late_dropout=0.0,
            late_dropout_start_step=0, pool_type="gap",
        ),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=[]),
    )
    with pytest.raises(ValueError, match="recurse"):
        TextAlignConv1DTr_Model.from_config(cfg)


def test_textalign_from_config_defaults_backbone_name() -> None:
    """Critical #2 — default backbone_name is conv1d_transformer; round-trip works."""
    from diema.models.conv1d_transformer.textalign_conv1d_tr_model import (
        TextAlignConv1DTr_Model,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="textalign_conv1d_tr",
            num_class=12, in_channels=6, dim=32, num_blocks=1, kernel_size=5,
            num_heads=2, mlp_ratio=2.0, drop_rate=0.0, late_dropout=0.0,
            late_dropout_start_step=0, pool_type="gap",
            text_dim=128,
        ),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=[]),
    )
    model = TextAlignConv1DTr_Model.from_config(cfg)
    x = torch.randn(2, 6, 64, 25)
    out = model(x)
    assert "logits" in out and "z_text" in out
    assert out["logits"].shape == (2, 12)
    assert out["z_text"].shape == (2, 128)


# --------------------- Critical #4: explain_part_masking clone -------------------


def test_part_masking_does_not_mutate_loader_batch() -> None:
    """Critical #4 — _forward_with_mask must clone x before zero-masking."""
    from tools.explain_part_masking import _forward_with_mask

    # A loader whose batches are shared tensors we can inspect.
    shared = torch.ones(2, 6, 64, 25)
    batches = [(shared, torch.zeros(2, dtype=torch.long), ["a", "b"])]

    class FakeLoader:
        def __iter__(self):
            for b in batches:
                yield b

    class ZeroModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict:
            return {"logits": torch.zeros(x.size(0), 12)}

    _forward_with_mask(ZeroModel(), FakeLoader(), torch.device("cpu"), (5, 6, 7, 8))
    # The underlying batch tensor must be unchanged.
    assert torch.all(shared == 1.0), \
        "_forward_with_mask mutated the loader's batch tensor in place"


def test_part_masking_empty_val_returns_typed_empty_arrays() -> None:
    """Warning #7 — empty val fold should not crash concat."""
    from tools.explain_part_masking import NUM_CLASS, _forward_with_mask

    class EmptyLoader:
        def __iter__(self):
            return iter([])

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return {"logits": torch.zeros(x.size(0), NUM_CLASS)}

    logits, labels = _forward_with_mask(
        ZeroModel(), EmptyLoader(), torch.device("cpu"), None
    )
    assert logits.shape == (0, NUM_CLASS)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


# ----------------- Warning #5: intensity regex matches real filenames ------------


def test_intensity_regex_matches_bare_tag_in_filename() -> None:
    """Warning #5 — fixed regex must catch `_L` / `_M` / `_H` bordered by `_`."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs/leakage/forbidden_tokens.yaml"
    forbidden = yaml.safe_load(yaml_path.read_text())
    pats = forbidden["filename_regex"]
    intensity_pat = next(p for p in pats if "(L|M|H)" in p)
    regex = re.compile(intensity_pat)

    # Real DIEM-A filenames that MUST match.
    for fn in ("JP_06_joy_1_L", "JP_06_joy_1_M", "JP_06_joy_1_H",
               "TW_33_anger_2_L", "a b L c d", "text_H_text"):
        assert regex.search(fn), f"intensity regex missed {fn!r}"


def test_intensity_regex_does_not_match_benign_words() -> None:
    """Regression — word-interior L/M/H must not trigger."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs/leakage/forbidden_tokens.yaml"
    forbidden = yaml.safe_load(yaml_path.read_text())
    pats = forbidden["filename_regex"]
    intensity_pat = next(p for p in pats if "(L|M|H)" in p)
    regex = re.compile(intensity_pat)
    for s in ("hello", "LOW_THRESHOLD", "HELPFUL"):
        assert not regex.search(s), f"intensity regex false-positive on {s!r}"


# ---------------- Warning #6: build_positive_mask excludes unknown (-1) ----------


def test_positive_mask_excludes_unknown_scenario() -> None:
    """Warning #6 — two -1 scenario_idx samples must not be paired as positives."""
    from diema.training.losses_motion_text import build_positive_mask

    scenario = torch.tensor([-1, -1, 0, 0])
    emotion = torch.tensor([0, 0, 0, 0])
    mask = build_positive_mask(scenario, emotion, policy="same_scenario_and_emotion")
    # Samples 0 and 1 both have scenario = -1; they must NOT be positives of each other.
    assert not mask[0, 1] and not mask[1, 0]
    # Diagonal preserved.
    assert mask[0, 0] and mask[1, 1]
    # Valid pair (scenario 0) IS positive.
    assert mask[2, 3] and mask[3, 2]


def test_positive_mask_excludes_unknown_emotion() -> None:
    from diema.training.losses_motion_text import build_positive_mask

    scenario = torch.tensor([0, 0, 0, 0])
    emotion = torch.tensor([-1, -1, 4, 4])
    mask = build_positive_mask(scenario, emotion, policy="same_emotion")
    assert not mask[0, 1] and not mask[1, 0]
    assert mask[2, 3] and mask[3, 2]


def test_positive_mask_exclude_unknown_opt_out() -> None:
    """exclude_unknown=False preserves the legacy (buggy) == comparison."""
    from diema.training.losses_motion_text import build_positive_mask

    scenario = torch.tensor([-1, -1, 0, 0])
    emotion = torch.tensor([0, 0, 0, 0])
    mask = build_positive_mask(
        scenario, emotion, policy="same_scenario", exclude_unknown=False,
    )
    # With exclusion off, the bug surfaces: -1 == -1 → treated as same scenario.
    assert mask[0, 1] and mask[1, 0]


# --------------- Warning #11: diagonal silent-fix emits a warning ---------------


def test_info_nce_warns_when_diagonal_missing() -> None:
    """Warning #11 — forcing missing diagonal should emit a UserWarning."""
    from diema.training.losses_motion_text import global_multi_positive_info_nce

    z = torch.randn(3, 8)
    t = torch.randn(3, 8)
    bad_mask = torch.zeros(3, 3, dtype=torch.bool)  # no diagonal at all
    with pytest.warns(UserWarning, match="diagonal"):
        global_multi_positive_info_nce(z, t, positive_mask=bad_mask)


def test_info_nce_no_warn_when_diagonal_complete() -> None:
    """No warning on well-formed masks."""
    import warnings
    from diema.training.losses_motion_text import global_multi_positive_info_nce

    z = torch.randn(3, 8)
    t = torch.randn(3, 8)
    good = torch.eye(3, dtype=torch.bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = global_multi_positive_info_nce(z, t, positive_mask=good)


# ----------- Warning #8: emotion_mask module is importable from diema ----------


def test_emotion_mask_is_in_diema_features() -> None:
    """Warning #8 — primitives live in diema.features.emotion_mask now."""
    from diema.features import emotion_mask

    for attr in ("MASK_TOKEN", "compile_emotion_mask_regex", "load_forbidden_tokens",
                 "mask_emotion_aliases"):
        assert hasattr(emotion_mask, attr), f"missing {attr}"


# -------------- Warning #10: ScenarioTextCache has public property -------------


def test_exp051_run_py_has_manual_best_ckpt_load() -> None:
    """PT 2.6 bug fix — run.py must load best.ckpt manually with weights_only=False.

    Regression guard for the UnpicklingError hit on 2026-04-20 when
    trainer.test(ckpt_path='best') went through PyTorch 2.6's new
    weights_only=True default. See commit a02a722.
    """
    import re
    run_py = Path(__file__).resolve().parents[1] / "experiments/exp051_tmr_scenario_global/run.py"
    text = run_py.read_text()
    # trainer.test must NOT be called with ckpt_path='best' anywhere — the docstring
    # may still mention the phrase as part of the comment, so we match the actual call.
    call = re.search(r"trainer\.test\([^)]*ckpt_path\s*=\s*['\"]best['\"]", text)
    assert call is None, (
        "run.py still calls trainer.test(ckpt_path='best') which breaks under "
        "PT 2.6 weights_only=True default"
    )
    assert "weights_only=False" in text, (
        "run.py no longer explicitly loads best.ckpt with weights_only=False"
    )
    assert "best_model_path" in text, (
        "run.py should extract best_model_path from the ModelCheckpoint callback"
    )


def test_scenario_text_cache_public_filename_to_idx(tmp_path: Path, monkeypatch) -> None:
    """Warning #10 — public read-only accessor instead of _filename_to_idx."""
    import pandas as pd
    from diema.features.text_features import save_text_cache
    from diema.features.text_cache import ScenarioTextCache

    # Minimal fake cache with a single scenario.
    emb = {"I realize it is my day off": torch.randn(384)}
    cache_path = tmp_path / "cache.pt"
    save_text_cache(cache_path, emb, meta={"model_name": "stub", "num_unique": "1"})
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "original_name": ["JP_06_joy_1_L.bvh"],
        "scenario": ["I realize it is my day off"],
    }).to_csv(csv_path, index=False)

    cache = ScenarioTextCache(cache_path=cache_path, csv_path=csv_path)
    pub = cache.filename_to_idx
    assert isinstance(pub, dict)
    assert pub == {"JP_06_joy_1_L": 0}
    # Mutating the returned dict must not alter internal state.
    pub["hack"] = 999
    assert "hack" not in cache.filename_to_idx
