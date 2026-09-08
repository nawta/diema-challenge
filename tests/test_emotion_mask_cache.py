""" masked-cache unit tests.

See: (emotion lexicon masking)
Related:
- tools/build_scenario_cache.py — _compile_emotion_mask_regex / mask_emotion_aliases
- configs/leakage/forbidden_tokens.yaml — G-12labels source of truth
"""
from __future__ import annotations

from pathlib import Path

from tools.build_scenario_cache import (
    MASK_TOKEN,
    _compile_emotion_mask_regex,
    _load_forbidden_tokens,
    mask_emotion_aliases,
)


def _regex():
    yaml_path = Path(__file__).resolve().parents[1] / "configs/leakage/forbidden_tokens.yaml"
    forbidden = _load_forbidden_tokens(yaml_path)
    return _compile_emotion_mask_regex(forbidden["emotion_aliases"])


def test_yaml_contains_12_canonical_classes() -> None:
    """G-12labels: the YAML must list all 12 DIEM-A class names verbatim."""
    yaml_path = Path(__file__).resolve().parents[1] / "configs/leakage/forbidden_tokens.yaml"
    forbidden = _load_forbidden_tokens(yaml_path)
    aliases = {a.lower() for a in forbidden["emotion_aliases"]}
    canonical = {
        "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
        "jealousy", "joy", "pride", "sadness", "shame", "surprise",
    }
    missing = canonical - aliases
    assert not missing, f"missing canonical classes: {missing}"


def test_mask_hits_base_class_name() -> None:
    regex = _regex()
    masked, hits = mask_emotion_aliases("The anger in his voice was clear", regex)
    assert hits == 1
    assert MASK_TOKEN in masked
    assert "anger" not in masked.lower()


def test_mask_is_case_insensitive() -> None:
    regex = _regex()
    masked, hits = mask_emotion_aliases("JOY is infectious", regex)
    assert hits == 1
    assert MASK_TOKEN in masked


def test_mask_word_boundary_does_not_cross_substrings() -> None:
    """'Surprisingly' must not match 'surprising' or 'surprise'."""
    regex = _regex()
    _, hits = mask_emotion_aliases("Surprisingly unusual weather", regex)
    assert hits == 0
    # But "surprised by" should match since 'surprised' is in the alias list
    _, h2 = mask_emotion_aliases("I was surprised by it", regex)
    assert h2 == 1


def test_mask_covers_derived_forms() -> None:
    regex = _regex()
    for text in (
        "He felt afraid of the dark",
        "She was grateful for the help",
        "A contemptuous look appeared",
        "They were ashamed",
    ):
        _, hits = mask_emotion_aliases(text, regex)
        assert hits >= 1, f"no mask hit on: {text!r}"


def test_no_mask_for_unrelated_text() -> None:
    regex = _regex()
    _, hits = mask_emotion_aliases(
        "I realize that it is my day off and I walk to the park", regex
    )
    assert hits == 0


def test_multiple_hits_in_one_sentence() -> None:
    regex = _regex()
    masked, hits = mask_emotion_aliases(
        "The joy and sadness of the moment competed", regex
    )
    assert hits == 2
    assert masked.count(MASK_TOKEN) == 2


def test_masked_text_is_deterministic() -> None:
    regex = _regex()
    text = "I feel grateful and proud"
    m1, _ = mask_emotion_aliases(text, regex)
    m2, _ = mask_emotion_aliases(text, regex)
    assert m1 == m2
