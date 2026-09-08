""" / exp062 — body-motion DSL schema and rule-based extractor.

See: / docs/cards/prompt_card_3AD-4_v1.0.md (rationale source)
Related:
- tools/extract_rule_dsl.py (CLI wrapper)
- tools/probe_dsl_classifiability.py (downstream LogReg probe)
- output/rationale_cache/rationales/*.json (v1 rationale source)

The DSL discretizes free-form rationale text into low-cardinality categorical
features, on the hypothesis that a discrete representation gives a linear
probe a denser signal than a 768-d sentence-transformer embedding (see Phase
3AE Tier 0 redux: mpnet/TF-IDF/bge-large all stuck at 9-12% F1).

Schema (per plan):

GLOBAL (6 attributes, derived from `global_motion` field):
  - openness            : low / mid / high
  - verticality         : down / neutral / up
  - locomotion          : none / light / active
  - energy              : low / mid / high
  - tempo               : slow / variable / burst
  - asymmetry           : low / mid / high

PER-PART (5 attributes × 6 parts = 30 attributes):
  Parts: head, torso, left_arm, right_arm, left_leg, right_leg
  - action              : still / raise / lower / open / close / rotate / swing / step / retract
  - direction           : up / down / forward / backward / outward / inward / lateral
  - amplitude           : low / mid / high
  - tempo               : slow / sustained / burst
  - confidence          : 0 / 1 / 2

Total: 36 categorical attributes per clip. With N=7644 train, this is
statistically reasonable for a LogReg with one-hot encoding (~210 features).

Extractor strategy
------------------
The rule-based extractor uses **keyword matching with disjoint vocabularies**
per attribute. It returns the most-frequent matching keyword per category.
For attributes with no match, a default value is used (e.g., "mid" / "neutral").

This is intentionally simple — it acts as a deterministic baseline that the
later Qwen3 LM-based extractor must beat (gate: DSL probe ≥ free-form +2pt).
If the rule-based DSL probe is already at the 9-12% ceiling, the LM-based
extractor likely cannot break it either (the bottleneck is signal absence,
not feature engineering).
"""

from __future__ import annotations

import re
from collections import Counter

PART_NAMES: tuple[str, ...] = (
    "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
)

GLOBAL_ATTR_VALUES = {
    "openness":    ("low", "mid", "high"),
    "verticality": ("down", "neutral", "up"),
    "locomotion":  ("none", "light", "active"),
    "energy":      ("low", "mid", "high"),
    "tempo":       ("slow", "variable", "burst"),
    "asymmetry":   ("low", "mid", "high"),
}

PART_ATTR_VALUES = {
    "action":     ("still", "raise", "lower", "open", "close", "rotate", "swing", "step", "retract"),
    "direction":  ("up", "down", "forward", "backward", "outward", "inward", "lateral"),
    "amplitude":  ("low", "mid", "high"),
    "tempo":      ("slow", "sustained", "burst"),
    "confidence": ("0", "1", "2"),
}

PART_ATTRS_ORDER = ("action", "direction", "amplitude", "tempo", "confidence")
GLOBAL_ATTRS_ORDER = ("openness", "verticality", "locomotion", "energy", "tempo", "asymmetry")


# Keyword vocabularies — case-insensitive whole-word matching against the text.
# Order does not matter; keywords are summed across the field text.

_GLOBAL_VOCAB = {
    "openness": {
        "low":  ("close", "closed", "narrow", "narrowed", "contract", "contracted",
                "compact", "tight", "tucked", "huddle", "huddled", "fold", "folded",
                "shrink", "shrunk"),
        "high": ("open", "wide", "widen", "expand", "expanded", "outward",
                "spread", "stretch", "stretched", "extend", "extended"),
    },
    "verticality": {
        "down": ("down", "downward", "downwards", "lower", "lowered", "drop",
                "drops", "descend", "descending", "crouch", "crouched", "kneel",
                "kneeling", "bend", "bends", "bent", "bending", "lean forward",
                "stoop", "slump"),
        "up":   ("up", "upward", "upwards", "raise", "raised", "raising",
                "rise", "rising", "ascend", "ascending", "extend", "extended",
                "stand", "standing", "stood", "stretch", "stretched", "elevated",
                "lift", "lifted", "lifting"),
    },
    "locomotion": {
        "none":   ("still", "stationary", "stands still", "stand still", "in place",
                  "no distinctive", "fixed", "static", "remains in", "remains stationary",
                  "minimal motion", "no movement"),
        "active": ("walk", "walking", "step", "steps", "stepping", "march",
                  "marching", "run", "running", "stride", "striding", "gait",
                  "locomotion", "travel", "advance", "moves forward",
                  "moves backward"),
    },
    "energy": {
        "low":  ("slow", "subtle", "small", "minimal", "gentle", "gradual",
                "mild", "soft", "low amplitude", "low speed", "tiny", "slight"),
        "high": ("fast", "rapid", "large", "burst", "explosive", "vigorous",
                "intense", "swift", "sharp", "high amplitude", "high speed",
                "powerful", "abrupt", "sudden"),
    },
    "tempo": {
        "slow":  ("slow", "sustained", "gradual", "gentle", "steady",
                 "continuous", "throughout"),
        "burst": ("burst", "single burst", "sudden", "abrupt", "quick",
                 "swift", "sharp"),
    },
    "asymmetry": {
        "low":  ("symmetric", "symmetrically", "synchronous", "synchronously",
                "in sync", "in synchrony", "coordinated", "both", "matching",
                "together", "simultaneously", "mirror", "mirroring"),
        "high": ("alternating", "alternately", "alternates", "alternate",
                "opposite", "opposing", "asymmetric", "reciprocal", "reciprocally",
                "out of sync", "counter", "counter-phase", "opposing",
                "one then", "one before"),
    },
}

_PART_VOCAB = {
    "action": {
        "still":   ("still", "stationary", "no distinctive", "remains",
                   "fixed", "no motion", "stays in", "minimal motion"),
        "raise":   ("raise", "raised", "raising", "rise", "rising", "rises",
                   "lifting", "lift", "lifted", "lifts", "elevated",
                   "extends upward", "moves upward", "moves up"),
        "lower":   ("lower", "lowered", "lowering", "lowers", "drop", "drops",
                   "dropping", "descend", "descending", "descends",
                   "moves downward", "moves down", "extends downward"),
        "open":    ("open", "opens", "opening", "spread", "spreads", "widen",
                   "widens", "extend", "extends", "extending", "outward"),
        "close":   ("close", "closes", "closing", "narrow", "narrows",
                   "narrowing", "tighten", "tightens", "tightening", "tucked",
                   "inward", "withdraw", "withdraws", "fold", "folds"),
        "rotate":  ("rotate", "rotates", "rotating", "rotation", "twist",
                   "twists", "turn", "turns", "turning", "swivel"),
        "swing":   ("swing", "swings", "swinging", "sway", "swaying",
                   "oscillate", "oscillates", "oscillation", "oscillating",
                   "wave", "waves", "waving"),
        "step":    ("step", "steps", "stepping", "stride", "strides", "walk",
                   "walking", "walks", "gait", "march", "marches", "marching"),
        "retract": ("retract", "retracts", "retracting", "withdraw", "withdraws",
                   "pull back", "pulls back", "fold", "folds", "folding"),
        # NB: also include "bend / bent / bends / bending" splits below as
        # `lower` for legs and `close` for arms — too ambiguous to assign
        # uniformly, so we leave them out of the action vocab and let the
        # downstream signal carry it.
    },
    "direction": {
        "up":       ("upward", "upwards", "up,", "up ", "above", "vertical"),
        "down":     ("downward", "downwards", "down,", "down ", "below"),
        "forward":  ("forward", "forwards", "front", "frontward", "fronward"),
        "backward": ("backward", "backwards", "back", "rearward", "rearwards"),
        "outward":  ("outward", "outwards", "outside", "lateral"),
        "inward":   ("inward", "inwards", "inside"),
        "lateral":  ("lateral", "left and right", "left/right", "side to side",
                    "side-to-side", "left-right", "horizontal"),
    },
    "amplitude": {
        "low":  ("small", "minimal", "subtle", "slight", "tiny", "minor",
                "low amplitude", "small amplitude"),
        "mid":  ("medium", "moderate", "moderately", "mid", "middling",
                "medium amplitude", "moderate amplitude"),
        "high": ("large", "wide", "big", "extensive", "expansive", "great",
                "high amplitude", "large amplitude", "wide amplitude"),
    },
    "tempo": {
        "slow":      ("slow", "slowly", "gradual", "gradually", "leisurely",
                     "gentle", "low speed"),
        "sustained": ("sustained", "steady", "steadily", "continuous",
                     "continuously", "throughout", "consistent", "consistently",
                     "repeated", "repeats", "repeatedly", "rhythmic",
                     "rhythmically"),
        "burst":     ("fast", "rapid", "rapidly", "burst", "quick", "quickly",
                     "swift", "swiftly", "abrupt", "abruptly", "sudden",
                     "suddenly", "high speed"),
    },
}

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:sec|second|s|hz)\b", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"^\s*(no\s+distinctive\s+motion|none|n/?a|)\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(str(text)))


def _count_kw(text: str, keywords: tuple[str, ...]) -> int:
    """Count whole-word case-insensitive occurrences of any keyword in text.

    Uses regex word boundary; keywords with non-word characters (e.g., "down,")
    are matched literally as substrings instead of with \\b.
    """
    count = 0
    lower = text.lower()
    for kw in keywords:
        if not kw:
            continue
        if re.fullmatch(r"\w+", kw):
            # word-boundary match
            count += len(re.findall(rf"\b{re.escape(kw)}\b", lower))
        else:
            # substring match (kw contains punctuation / spaces)
            count += lower.count(kw.lower())
    return count


def _argmax_or_default(votes: dict[str, int], default: str) -> str:
    """Return the highest-voted value; ties broken by stable order; default if all zero."""
    if not votes:
        return default
    items = list(votes.items())
    # Stable sort: highest count first, original key order on ties
    items.sort(key=lambda kv: -kv[1])
    if items[0][1] == 0:
        return default
    return items[0][0]


def extract_global_attrs(global_motion_text: str) -> dict[str, str]:
    """6 global DSL attributes from the `global_motion` field."""
    text = (global_motion_text or "").strip()
    out: dict[str, str] = {}

    for attr in GLOBAL_ATTRS_ORDER:
        vocab = _GLOBAL_VOCAB.get(attr, {})
        # voting only over the polar values; the middle value is default
        votes = {v: _count_kw(text, kws) for v, kws in vocab.items()}
        if attr == "openness":
            out[attr] = _argmax_or_default(votes, default="mid")
        elif attr == "verticality":
            out[attr] = _argmax_or_default(votes, default="neutral")
        elif attr == "locomotion":
            # tri-valued: "none / light / active"; if no positive vote, default "light"
            out[attr] = _argmax_or_default(votes, default="light")
        elif attr == "energy":
            out[attr] = _argmax_or_default(votes, default="mid")
        elif attr == "tempo":
            out[attr] = _argmax_or_default(votes, default="variable")
        elif attr == "asymmetry":
            out[attr] = _argmax_or_default(votes, default="mid")
    return out


def extract_part_attrs(part_text: str) -> dict[str, str]:
    """5 per-part DSL attributes from one body-part text field."""
    text = (part_text or "").strip()
    if _is_placeholder(text):
        return {
            "action": "still",
            "direction": "up",  # arbitrary; confidence=0 carries the "no info" signal
            "amplitude": "low",
            "tempo": "slow",
            "confidence": "0",
        }

    out: dict[str, str] = {}

    # action
    action_votes = {
        v: _count_kw(text, kws) for v, kws in _PART_VOCAB["action"].items()
    }
    out["action"] = _argmax_or_default(action_votes, default="still")

    # direction
    dir_votes = {
        v: _count_kw(text, kws) for v, kws in _PART_VOCAB["direction"].items()
    }
    out["direction"] = _argmax_or_default(dir_votes, default="up")

    # amplitude
    amp_votes = {
        v: _count_kw(text, kws) for v, kws in _PART_VOCAB["amplitude"].items()
    }
    out["amplitude"] = _argmax_or_default(amp_votes, default="mid")

    # tempo (per-part)
    tempo_votes = {
        v: _count_kw(text, kws) for v, kws in _PART_VOCAB["tempo"].items()
    }
    out["tempo"] = _argmax_or_default(tempo_votes, default="sustained")

    # confidence: 0 if all attribute votes were zero, 2 if text is detailed
    # (≥ 12 alphabetic words and contains at least one numeric/temporal cue),
    # 1 otherwise.
    n_alpha_words = len(re.findall(r"[A-Za-z]{2,}", text))
    has_numeric = bool(_NUMERIC_RE.search(text))
    all_zero = (
        sum(action_votes.values()) == 0
        and sum(dir_votes.values()) == 0
        and sum(amp_votes.values()) == 0
        and sum(tempo_votes.values()) == 0
    )
    if all_zero or n_alpha_words < 4:
        out["confidence"] = "0"
    elif n_alpha_words >= 12 and has_numeric:
        out["confidence"] = "2"
    else:
        out["confidence"] = "1"
    return out


def extract_dsl(rationale: dict) -> dict[str, dict[str, str]]:
    """Top-level extractor — calls global + 6 per-part extractors and returns
    a flat dict shaped like::

        {
          "global": {"openness": "...", ...},
          "head":   {"action": "...", ...},
          "torso":  {...},
          ...
        }
    """
    global_text = str(rationale.get("global_motion", "") or "")
    out: dict[str, dict[str, str]] = {"global": extract_global_attrs(global_text)}
    for p in PART_NAMES:
        out[p] = extract_part_attrs(str(rationale.get(p, "") or ""))
    return out


def flatten_dsl(dsl: dict[str, dict[str, str]]) -> dict[str, str]:
    """Flatten the nested DSL into a single column-name → value dict.

    Column-name format: ``{section}__{attr}`` (e.g., ``global__openness``,
    ``head__action``). 36 columns total.
    """
    flat: dict[str, str] = {}
    for attr, val in dsl.get("global", {}).items():
        flat[f"global__{attr}"] = val
    for p in PART_NAMES:
        for attr, val in dsl.get(p, {}).items():
            flat[f"{p}__{attr}"] = val
    return flat


DSL_COLUMNS: tuple[str, ...] = tuple(
    [f"global__{a}" for a in GLOBAL_ATTRS_ORDER]
    + [f"{p}__{a}" for p in PART_NAMES for a in PART_ATTRS_ORDER]
)


def dsl_value_set(column: str) -> tuple[str, ...]:
    """Return the canonical value vocabulary for a flat DSL column."""
    section, attr = column.split("__", 1)
    if section == "global":
        return GLOBAL_ATTR_VALUES[attr]
    return PART_ATTR_VALUES[attr]
