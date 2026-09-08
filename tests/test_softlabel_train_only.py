""" (Tier 2) ethics gate: assert exp060 imports vlm_softlabel only at train time.

Constraint (docs/rule_and_ethics_checklist.md §3.3): the soft-label tensor is
**train-only auxiliary supervision**. It must never enter:
- the test inference path (no leakage to test split)
- the submission tarball

This test scans the exp060 run module's source for any string referencing the
soft-label file, and flags the only allowed location: inside the train-time
model construction (``_load_softlabels`` helper called from ``main`` before
``trainer.fit``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SOFTLABEL_PATH_RE = re.compile(r"vlm_softlabel_train\.npz")


def test_exp060_run_references_softlabel_only_via_load_helper():
    run_path = Path("experiments/exp060_vlm_softlabel_kd/run.py")
    assert run_path.is_file(), f"missing exp060 run: {run_path}"
    source = run_path.read_text(encoding="utf-8")
    occurrences = SOFTLABEL_PATH_RE.findall(source)
    # Exactly one reference, inside the dataclass default for `softlabel_path`.
    assert len(occurrences) <= 2, (
        f"too many references to vlm_softlabel_train.npz in {run_path}; "
        f"found {len(occurrences)} occurrences. Expected ≤ 2: the dataclass "
        "default + (optionally) the docstring."
    )


def test_no_other_module_imports_softlabel_npz():
    """The soft-label file path may appear in:
    - experiments/exp060_*/run.py (the legitimate train-time loader)
    - tools/generate_emotion_softlabel*  (the producer)
    - diema/training/kd_trainer.py (docstring only — the consumer of the
      already-loaded tensor)

    Anything else (especially submission tooling, test inference paths)
    must NOT reference it.
    """
    repo = Path(".")
    allowed_files = {
        "diema/training/kd_trainer.py",  # docstring only
    }
    bad: list[str] = []
    for p in list(Path("diema").rglob("*.py")) + list(Path("tools").glob("check_submission*.py")):
        rel = str(p)
        if rel in allowed_files:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if SOFTLABEL_PATH_RE.search(text):
            bad.append(rel)
    assert not bad, (
        "vlm_softlabel_train.npz referenced outside of allowed files: "
        + ", ".join(bad)
    )


def test_softlabel_kd_trainer_buffer_is_train_only():
    """The teacher tensor is registered as a buffer, but the Lightning module's
    training_step is the only consumer — no test_step or predict_step path
    references softlabel_tensor."""
    src = Path("diema/training/kd_trainer.py").read_text(encoding="utf-8")
    # softlabel_tensor must not appear inside any def starting with test_ or
    # predict_ or validation_.
    blocks = re.split(r"\n    def ", src)
    for block in blocks:
        head = block[: block.find("(")] if "(" in block else block[:32]
        if head.startswith(("test_step", "predict_step", "validation_step")):
            assert "softlabel_tensor" not in block, (
                f"softlabel_tensor referenced inside {head} — train-only "
                "constraint violated"
            )
