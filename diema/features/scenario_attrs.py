""" / exp065 — 8-dim scenario attribute schema and rule-based extractor.

Related:
- tools/extract_rule_scenario_attrs.py (CLI wrapper)
- tools/probe_scenario_attrs_classifiability.py (downstream probe)

The 8 attributes (per plan) discretize a free-text scenario
("I broke up with my girlfriend", "I am eating my favorite ramen") into a
low-dimensional categorical representation. The plan calls for Qwen3 text
model extraction; this rule-based extractor acts as a deterministic
upper-bound proxy and a paper-grade reproducibility baseline.

Schema
------
Each attribute is 3-valued (24 one-hot features total):

- ``valence``               : neg / neutral / pos
- ``arousal``               : low / mid / high
- ``dominance``             : low / mid / high
- ``sociality``             : low / mid / high
- ``agency``                : self / mixed / other
- ``approach_avoidance``    : avoid / neutral / approach
- ``energy``                : low / mid / high
- ``openness``              : closed / neutral / open

Reference numbers
-----------------
- Random (1/12): 8.33% F1
- Scenario text mpnet linear probe (`docs/analysis/exp065_scenario_probe.md`):
  **53.09% ± 0.78%** — this is the upper bound for any function of the
  scenario text. The 24-dim one-hot is a 30-50× compression vs raw 768-d
  mpnet, so a substantial drop is expected; the question is whether enough
  signal survives to make an auxiliary head useful (gate: F1 +0.20pt
  downstream OR qualitative explainability gain).
"""

from __future__ import annotations

import re

ATTR_VALUES: dict[str, tuple[str, ...]] = {
    "valence":            ("neg", "neutral", "pos"),
    "arousal":            ("low", "mid", "high"),
    "dominance":          ("low", "mid", "high"),
    "sociality":          ("low", "mid", "high"),
    "agency":             ("self", "mixed", "other"),
    "approach_avoidance": ("avoid", "neutral", "approach"),
    "energy":             ("low", "mid", "high"),
    "openness":           ("closed", "neutral", "open"),
}

ATTR_ORDER: tuple[str, ...] = tuple(ATTR_VALUES.keys())

# Keyword vocabularies. The scenarios in train_data.csv are short situational
# descriptions ("I am eating my favorite ramen", "I farted in an elevator"),
# so the keyword approach favors high-signal lexical cues.

_VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "valence": {
        "neg": (
            "broke", "broken", "break up", "broken up", "lost", "lose", "losing",
            "pain", "painful", "hurt", "hurts", "hurting", "fail", "failed",
            "fails", "failure", "wrong", "wrongly", "unreasonable", "criticized",
            "criticize", "criticism", "rejected", "reject", "rejection",
            "terrible", "horrible", "awful", "bad", "worse", "worst",
            "sick", "death", "die", "died", "dying", "kill", "killed",
            "rude", "insult", "insulted", "complain", "complaint", "complaints",
            "betrayed", "betray", "abandon", "abandoned", "abandons", "ignored",
            "ignore", "ignores", "isolation", "alone", "lonely", "scared",
            "afraid", "fearful", "ghost", "haunted", "spider", "snake", "rat",
            "rats", "fly", "flies", "bee", "bees", "wasp", "wasps", "insect",
            "trash", "garbage", "smell", "smelly", "filthy", "dirt", "dirty",
            "fart", "farted", "farts", "vomit", "vomited", "ash", "blood",
            "lying", "lied", "lie", "cheat", "cheated", "stole", "steal",
            "stolen", "robbed", "robbery", "embarrassed", "embarrassing",
            "shame", "shameful", "ashamed", "guilty", "guilt",
            "stress", "stressed", "stressful", "exhausted", "tired",
            "complaint", "complaints", "blamed", "blame", "blames",
            "envy", "envious", "jealous", "jealousy", "regret", "regretted",
            "in front of", "argue", "argued", "fight", "fought",
        ),
        "pos": (
            "favorite", "wonderful", "amazing", "great", "good", "excellent",
            "love", "loved", "loves", "like", "liked", "likes", "enjoy",
            "enjoyed", "enjoying", "happy", "delicious", "delight", "delighted",
            "wallet", "returned", "thanks", "thank", "thanked", "thanking",
            "gratitude", "grateful", "achievement", "achieved", "accomplished",
            "accomplish", "celebrate", "celebrated", "celebration",
            "won", "win", "winning", "success", "successful", "succeed",
            "praise", "praised", "compliment", "complimented", "complimentary",
            "perfect", "perfectly", "best", "beautiful", "beautifully",
            "promotion", "promoted", "promote", "day off", "vacation",
            "holiday", "treat", "treats", "gift", "gifted", "present", "presents",
            "lucky", "luckily", "fortunate", "fortunately", "happy", "joyful",
            "joyous", "proud", "pride", "honored", "honors", "honor",
            "free time", "relax", "relaxed", "relaxing", "warm", "warmth",
            "kind", "kindness", "friendly", "lovely",
        ),
    },
    "arousal": {
        "low":  (
            "quiet", "calm", "peaceful", "still", "relaxing", "relaxed",
            "alone", "rest", "resting", "sleep", "sleeping", "boring",
            "routine", "normal", "ordinary", "in a quiet", "free time",
            "day off", "alone time", "in solitude", "sit", "sitting",
            "watching", "watching tv", "reading", "thinking",
        ),
        "high": (
            "suddenly", "sudden", "appears", "appeared", "appearing",
            "rushes", "rushed", "rushing", "rush", "scared", "scream",
            "screamed", "screaming", "yelled", "yelling", "yell",
            "shocked", "surprised", "amazed", "amazing", "exciting",
            "excited", "burst", "explode", "exploded", "exploding",
            "loud", "loudly", "fast", "noise", "noisy", "intense", "intensely",
            "panic", "panicked", "panicking", "startled", "startle",
            "startling", "alarm", "alarmed", "fight", "fought", "argue",
            "argued", "shouting", "shout", "shouted", "running", "ran",
            "ghost", "haunted", "horror", "scary", "terrifying",
            "discovered", "discover", "found out", "find out", "realize",
            "realized", "realizing",
        ),
    },
    "dominance": {
        "low": (
            "received", "given", "gave me", "gave us", "happened to me",
            "powerless", "helpless", "victim", "criticized", "ignored",
            "rejected", "scolded", "scold", "blamed", "blame", "told off",
            "ordered", "forced", "pushed", "yelled at", "shouted at",
            "betrayed", "abandoned", "lost",
        ),
        "high": (
            "i did", "i broke", "i won", "i achieved", "i decided",
            "i refused", "i confronted", "i succeeded", "i defeated",
            "i scolded", "i corrected", "i taught", "lead", "leading",
            "command", "control", "controlled", "insist", "insisted",
            "demand", "demanded",
        ),
    },
    "sociality": {
        "low":  (
            "alone", "by myself", "in private", "no one", "nobody",
            "in my room", "deserted", "empty", "isolation", "isolated",
            "in solitude",
        ),
        "high": (
            "people", "friend", "friends", "colleagues", "family",
            "stranger", "strangers", "everyone", "crowd", "crowded",
            "we", "us", "social", "party", "meeting", "gathering",
            "audience", "public",
        ),
    },
    "agency": {
        "self":  (
            "i made", "i caused", "i did", "i broke", "i confronted",
            "i decided", "i took", "i achieved", "i hit", "i farted",
            "i secretly", "i ate", "i won", "i cheated", "i stole",
            "i lied", "i forgot", "i ignored", "i refused",
        ),
        "other": (
            "someone", "they", "she", "he", "my friend", "my colleague",
            "my boss", "my partner", "my sister", "my brother", "my mom",
            "my dad", "my mother", "my father", "the person", "people",
            "stranger", "strangers", "another person", "the other",
        ),
    },
    "approach_avoidance": {
        "avoid":     (
            "fly", "flies", "bee", "bees", "wasp", "spider", "insect",
            "snake", "rat", "rats", "mouse", "haunted", "ghost", "scary",
            "afraid", "scared", "ran away", "ran from", "escape", "escaped",
            "avoid", "avoided", "back away", "stepped back", "withdrew",
            "withdraw", "hide", "hid", "shut", "covered", "block",
        ),
        "approach":  (
            "approach", "approaching", "go toward", "went toward",
            "want to", "to grab", "to get", "to receive", "embrace",
            "embraced", "hug", "hugged", "shake hands", "shook hands",
            "greet", "greeted", "welcome", "welcomed", "invite", "invited",
            "i ate", "eating", "drink", "drinking", "to take",
        ),
    },
    "energy": {
        "low":  (
            "quiet", "still", "alone", "rest", "resting", "calm", "peaceful",
            "slowly", "slow", "gentle", "subtle", "boring", "normal",
            "routine",
        ),
        "high": (
            "suddenly", "sudden", "fast", "rapid", "rapidly", "rush",
            "rushed", "rushing", "loud", "intense", "burst", "explode",
            "scream", "shout", "yell", "yelling", "shouting", "running",
            "ran",
        ),
    },
    "openness": {
        "closed": (
            "alone", "by myself", "private", "in my room", "deserted",
            "in solitude", "isolation", "isolated", "shy", "embarrassed",
            "ashamed", "withdraw", "withdrew", "hide", "hid", "secretly",
        ),
        "open":   (
            "publicly", "in public", "everyone", "audience", "crowd",
            "crowded", "open", "outdoors", "outside", "in front of",
            "on stage", "on the street", "in the park",
        ),
    },
}


def _count_kw(text: str, keywords: tuple[str, ...]) -> int:
    """Whole-word case-insensitive count for word-only keywords; substring
    for keywords with spaces / punctuation."""
    lower = text.lower()
    count = 0
    for kw in keywords:
        if not kw:
            continue
        if re.fullmatch(r"\w+", kw):
            count += len(re.findall(rf"\b{re.escape(kw)}\b", lower))
        else:
            count += lower.count(kw.lower())
    return count


def _argmax_or_default(votes: dict[str, int], default: str) -> str:
    if not votes:
        return default
    items = list(votes.items())
    items.sort(key=lambda kv: -kv[1])
    if items[0][1] == 0:
        return default
    return items[0][0]


_DEFAULTS = {
    "valence": "neutral",
    "arousal": "mid",
    "dominance": "mid",
    "sociality": "mid",
    "agency": "mixed",
    "approach_avoidance": "neutral",
    "energy": "mid",
    "openness": "neutral",
}


def extract_attrs(scenario_text: str) -> dict[str, str]:
    """Map a single scenario string to the 8-attribute dict."""
    text = (scenario_text or "").strip()
    out: dict[str, str] = {}
    for attr in ATTR_ORDER:
        vocab = _VOCAB.get(attr, {})
        votes = {v: _count_kw(text, kws) for v, kws in vocab.items()}
        out[attr] = _argmax_or_default(votes, default=_DEFAULTS[attr])
    return out


# Stable column ordering for one-hot encoding.
ONEHOT_COLUMNS: tuple[tuple[str, str], ...] = tuple(
    (attr, val) for attr in ATTR_ORDER for val in ATTR_VALUES[attr]
)
ONEHOT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{attr}=={val}" for attr, val in ONEHOT_COLUMNS
)
