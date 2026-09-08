""" fail-fast environment check for external-API secrets and mocap dependencies.

See: docs/rule_and_ethics_checklist.md — secret management policy
Related:
- tools/generate_part_rationales_gemini.py — consumer of GEMINI_API_KEY
- diema/features/rationale_cache.py — consumer of RATIONALE_HMAC_SALT

This script performs fail-fast checks so that /3AE scripts refuse to start
when required secrets are missing or when optional mocap libraries are not installed.

Usage:
    python tools/env_check.py                    # check all
    python tools/env_check.py --secrets          # secrets only (fast, no imports)
    python tools/env_check.py --mocap            # mocap libs only
    python tools/env_check.py --gemini           # verify GEMINI_API_KEY reaches Google
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRET_VARS = ("GEMINI_API_KEY", "RATIONALE_HMAC_SALT")
RULE_DOC = "docs/rule_and_ethics_checklist.md"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids adding python-dotenv dependency for a fail-fast script."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def check_gitignore() -> tuple[bool, str]:
    """ blocking gate: `.env` must be pattern-aware gitignored."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-v", ".env"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "git is not installed"
    if result.returncode != 0:
        return False, (
            ".env is NOT covered by .gitignore. Add '.env' (or a covering pattern) "
            "and re-run. Do not proceed to "
        )
    return True, result.stdout.strip()


def check_git_history_clean() -> tuple[bool, str]:
    """Check `.env` was never committed historically."""
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--full-history", "--", ".env"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "git is not installed"
    if result.stdout.strip():
        return False, (
            ".env appears in git history. IMMEDIATELY rotate the API key on Google AI "
            "Studio, then ask the user before running `git filter-repo`."
        )
    return True, "no prior .env commits"


def check_secrets() -> list[tuple[str, bool, str]]:
    """Return (var_name, ok, message) tuples for each required secret."""
    _load_dotenv(PROJECT_ROOT / ".env")
    results: list[tuple[str, bool, str]] = []
    for var in SECRET_VARS:
        value = os.environ.get(var, "").strip()
        if not value:
            results.append(
                (var, False, f"{var} is not set. See {RULE_DOC}.")
            )
        else:
            results.append((var, True, f"{var} is set (len={len(value)})"))
    return results


def check_mocap_libs() -> list[tuple[str, bool, str]]:
    """Verify optional mocap libraries for / 3AF."""
    checks: list[tuple[str, bool, str]] = []
    for module, purpose in (
        ("ezc3d", " C3D parser"),
        ("aitviewer", " motion renderer"),
        ("google.generativeai", " Gemini API"),
        ("ftlangdetect", " language detection"),
    ):
        try:
            __import__(module)
            checks.append((module, True, purpose))
        except ImportError as exc:
            checks.append((module, False, f"{purpose} — not installed ({exc})"))
    return checks


def check_gemini_reachable() -> tuple[bool, str]:
    """Optional: verify GEMINI_API_KEY is accepted by the API (one network call)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False, "GEMINI_API_KEY not set"
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return False, "google-generativeai not installed"
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
    except Exception as exc:  # pragma: no cover — depends on remote API
        return False, f"Gemini API rejected the key or network failed: {exc}"
    return True, f"Gemini API reachable ({len(models)} models listed)"


def _print(results: list[tuple[str, bool, str]], title: str) -> int:
    print(f"\n=== {title} ===")
    fails = 0
    for name, ok, msg in results:
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}: {msg}")
        if not ok:
            fails += 1
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--secrets", action="store_true", help="check env secrets only")
    parser.add_argument("--mocap", action="store_true", help="check mocap libraries only")
    parser.add_argument("--gemini", action="store_true", help="verify GEMINI_API_KEY online")
    args = parser.parse_args()

    run_all = not (args.secrets or args.mocap or args.gemini)
    fails = 0

    if run_all or args.secrets:
        ok, msg = check_gitignore()
        fails += _print([(".env gitignored", ok, msg)], " git hygiene")
        ok2, msg2 = check_git_history_clean()
        fails += _print([(".env absent from git log", ok2, msg2)], " git history")
        fails += _print(check_secrets, " required secrets")

    if run_all or args.mocap:
        fails += _print(check_mocap_libs, " mocap libraries")

    if args.gemini:
        ok, msg = check_gemini_reachable()
        fails += _print([("Gemini reachable", ok, msg)], " Gemini API key (online)")

    if fails:
        print(f"\n{fails} check(s) failed. Fix before proceeding to /3AE.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
