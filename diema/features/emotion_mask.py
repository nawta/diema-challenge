""" emotion-alias masking primitives.

Extracted from ``tools/build_scenario_cache.py`` so that ``diema`` (the
library) does not depend on ``tools`` (scripts). Both the cache builder
and :class:`diema.features.text_cache.ScenarioTextCache` import from here.

See: / configs/leakage/forbidden_tokens.yaml
"""

from __future__ import annotations

import re
from pathlib import Path

MASK_TOKEN = "[MASK]"


def load_forbidden_tokens(yaml_path: str | Path) -> dict:
    """Load the forbidden-token YAML used by / 3AD-5."""
    import yaml

    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"forbidden tokens yaml not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compile_emotion_mask_regex(aliases: list[str]) -> re.Pattern[str]:
    """Compile a single case-insensitive word-boundary regex for emotion aliases.

    Sort by length (long first) so multi-word aliases win over their substrings.
    """
    uniq = sorted({a.strip() for a in aliases if a.strip()}, key=len, reverse=True)
    escaped = [re.escape(a) for a in uniq]
    # Use \b on both sides so "surprisingly" does not match "surprising".
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def mask_emotion_aliases(
    text: str, regex: re.Pattern[str]
) -> tuple[str, int]:
    """Replace every word-boundary hit of an emotion alias with ``[MASK]``.

    Returns ``(masked_text, num_hits)``.
    """
    hits = 0

    def _sub(_: re.Match[str]) -> str:
        nonlocal hits
        hits += 1
        return MASK_TOKEN

    masked = regex.sub(_sub, text)
    return masked, hits
