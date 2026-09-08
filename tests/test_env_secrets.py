""" smoke tests for secret management.

See: docs/rule_and_ethics_checklist.md §2
Related: tools/env_check.py (fail-fast runtime check)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_env_is_gitignored() -> None:
    """`.env` must be covered by a .gitignore pattern (direct or glob)."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        ".env is not gitignored. Add '.env' to .gitignore "
        "before running scripts."
    )


def test_env_never_committed() -> None:
    """Guard against historical accidental `.env` commits."""
    result = subprocess.run(
        ["git", "log", "--all", "--full-history", "--", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert not result.stdout.strip(), (
        ".env appears in git history. Rotate secrets and consult the user before "
        "running `git filter-repo`."
    )


def test_env_example_exists() -> None:
    """`.env.example` must document required secrets so contributors know what to set."""
    example = PROJECT_ROOT / ".env.example"
    assert example.exists, ".env.example missing."
    content = example.read_text()
    for var in ("GEMINI_API_KEY", "RATIONALE_HMAC_SALT"):
        assert var in content, f"{var} missing from .env.example"


def test_env_file_not_tracked() -> None:
    """`.env` must not appear in `git ls-files`."""
    result = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert not result.stdout.strip(), ".env is tracked by git."
