"""Regression test: every experiment run.py must use Lightning's
``ckpt_path="best"`` sentinel (or ``trainer.checkpoint_callback.best_model_path``)
when calling ``trainer.test``.

The stale-checkpoint bug: if a previous smoke run wrote ``best.ckpt`` into
the output dir, Lightning's ``ModelCheckpoint`` avoids clobbering it by
writing the new best to ``best-v1.ckpt``. Any run.py that hardcodes
``ckpt_path=str(output_dir / "best.ckpt")`` will then load the stale
smoke-run model for the final test evaluation — the reported test_acc
and test_f1 are silently wrong, but the ``OOFPredictionCallback`` output
is still correct (it was written during training from the best val batch).

This test forbids the bad pattern from creeping back. See commit
ea7300f.
"""

from __future__ import annotations

from pathlib import Path


EXPECTED_GOOD_MARKER = 'ckpt_path="best"'
BAD_MARKER = 'best_ckpt = output_dir / "best.ckpt"'


def _all_run_scripts() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    return sorted((repo_root / "experiments").glob("exp*/run.py"))


def test_run_scripts_present():
    # At least the experiments referenced should be present so we
    # know the loop is non-trivial.
    paths = _all_run_scripts()
    assert len(paths) >= 10, f"expected >= 10 exp*/run.py files, found {len(paths)}"


def test_no_run_script_uses_stale_best_ckpt_pattern():
    offenders: list[str] = []
    for path in _all_run_scripts():
        text = path.read_text(encoding="utf-8")
        if BAD_MARKER in text:
            offenders.append(str(path))
    assert not offenders, (
        "The following run.py files still use the stale best.ckpt path "
        "which loads a leftover smoke-run checkpoint during trainer.test. "
        "Switch to `ckpt_path=\"best\"` (Lightning sentinel) or "
        "`trainer.checkpoint_callback.best_model_path`:\n  - "
        + "\n  - ".join(offenders)
    )


def test_every_run_script_loads_best_sentinel_or_callback():
    """Every run.py that calls trainer.test must either use the "best"
    sentinel or read best_model_path from the ModelCheckpoint callback.

    The callback check accepts any form that mentions both
    ``checkpoint_callback`` and ``best_model_path`` — so either dot
    access (``trainer.checkpoint_callback.best_model_path``) or
    getattr-based access (``getattr(trainer.checkpoint_callback,
    "best_model_path", ...)``) passes.
    """
    bad: list[str] = []
    for path in _all_run_scripts():
        text = path.read_text(encoding="utf-8")
        if "trainer.test(" not in text:
            # Scripts that don't run trainer.test are fine.
            continue
        uses_sentinel = EXPECTED_GOOD_MARKER in text
        uses_callback = (
            "checkpoint_callback" in text and "best_model_path" in text
        )
        if uses_sentinel or uses_callback:
            continue
        bad.append(str(path))
    assert not bad, (
        "The following run.py files call trainer.test but do not use the "
        "`ckpt_path=\"best\"` sentinel nor read "
        "`trainer.checkpoint_callback.best_model_path`:\n  - "
        + "\n  - ".join(bad)
    )
