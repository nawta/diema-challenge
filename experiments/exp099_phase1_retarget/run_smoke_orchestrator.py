"""run_smoke_orchestrator.py — Stage 3 driver with auto-halt gates.

Runs 7 SkateFormer fold-runs (3 modes × ≤ 3 folds), parses each fold's
metrics.json after completion, and aborts the remaining runs if any
mode crosses an underperformance floor.

Gate logic (per user decision "Halt if any mode underperforms"):

  rotation_6d:
    - floor at fold 0: best val_f1 ≥ 0.18 (~2.1× chance baseline);
      if this fails the model itself is broken → halt EVERYTHING.
    - record val_f1_rotation_6d_fold0 as the baseline.

  jointpos_honest fold 0:
    - floor: best val_f1 ≥ val_f1_rotation_6d_fold0 - 0.05 (5 pp).
    - fail → skip remaining honest folds + jointpos_flat (joint_pos
      architecture is unlikely to recover at fold 1/2 if it failed
      at fold 0).
    - record val_f1_jointpos_honest_fold0.

  jointpos_flat fold 0:
    - floor: same (val_f1_rotation_6d_fold0 - 0.05).
    - fail → just record, no further runs (last in sequence).

Run order:
  1. rotation_6d fold 0     (baseline)
  2. rotation_6d fold 1
  3. rotation_6d fold 2
  4. jointpos_honest fold 0 (primary candidate)
  5. jointpos_honest fold 1
  6. jointpos_honest fold 2
  7. jointpos_flat fold 0   (side-by-side check)

Writes:
- `experiments/exp099_phase1_retarget/stage3_smoke_results.json`
- progress in stdout (each run's start/end + val_f1)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "output" / "artifacts" / "exp099_phase1_smoke"
SCRIPT = REPO_ROOT / "experiments" / "exp099_phase1_retarget" / "run_smoke.py"
RESULTS = (REPO_ROOT / "experiments" / "exp099_phase1_retarget"
           / "stage3_smoke_results.json")

ROTATION_6D_FLOOR = 0.18      # hard floor: model itself broken if below
JOINTPOS_FLOOR_GAP = 0.05     # joint_pos must be ≥ rotation_6d_fold0 - 5 pp

PYTHON_BIN = "python"

RUNS = [
    ("rotation_6d",     0),
    ("rotation_6d",     1),
    ("rotation_6d",     2),
    ("jointpos_honest", 0),
    ("jointpos_honest", 1),
    ("jointpos_honest", 2),
    ("jointpos_flat",   0),
]


def best_val_f1(metrics_json: Path) -> float | None:
    if not metrics_json.exists():
        return None
    data = json.loads(metrics_json.read_text())
    history = data.get("history", [])
    f1s = [h.get("val_f1") for h in history if h.get("val_f1") is not None]
    return float(max(f1s)) if f1s else None


def run_single(mode: str, fold: int) -> tuple[float | None, dict]:
    """Spawn run_smoke.py and return (best_val_f1, full metrics dict)."""
    output_dir = ARTIFACTS / f"{mode}_fold_{fold:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON_BIN, str(SCRIPT),
           f"mode={mode}", f"fold={fold}", "dry_run=False"]
    print(f"[{time.strftime('%H:%M:%S')}] launching: {' '.join(cmd)}",
          flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] returncode={result.returncode}, "
          f"elapsed={elapsed/60:.1f} min", flush=True)
    if result.returncode != 0:
        return None, {"returncode": result.returncode}

    metrics_path = output_dir / "metrics.json"
    val_f1 = best_val_f1(metrics_path)
    full = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    print(f"[{time.strftime('%H:%M:%S')}] {mode} fold {fold} best val_f1 = "
          f"{val_f1:.4f}" if val_f1 is not None else
          f"[{time.strftime('%H:%M:%S')}] {mode} fold {fold} NO METRICS",
          flush=True)
    return val_f1, full


def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting 7-run smoke "
          f"orchestrator", flush=True)
    results: dict = {"runs": [], "halted": False, "halt_reason": None}
    val_f1_baseline = None
    val_f1_honest_fold0 = None

    for mode, fold in RUNS:
        # Gate checks BEFORE launching
        if results["halted"]:
            results["runs"].append({"mode": mode, "fold": fold,
                                    "status": "skipped (halted)"})
            continue

        # Skip jointpos modes if rotation_6d_fold0 broken
        if mode.startswith("jointpos") and val_f1_baseline is None:
            results["halted"] = True
            results["halt_reason"] = (
                "rotation_6d_fold_00 has no baseline val_f1; skip joint_pos")
            results["runs"].append({"mode": mode, "fold": fold,
                                    "status": "skipped (no baseline)"})
            continue

        # Skip remaining jointpos_honest folds if fold0 underperformed
        if (mode == "jointpos_honest" and fold > 0
                and val_f1_honest_fold0 is not None
                and val_f1_honest_fold0 < val_f1_baseline - JOINTPOS_FLOOR_GAP):
            results["runs"].append({"mode": mode, "fold": fold,
                                    "status": "skipped (honest fold0 below floor)",
                                    "honest_fold0": val_f1_honest_fold0,
                                    "rot6d_baseline": val_f1_baseline})
            continue

        # Launch
        val_f1, full = run_single(mode, fold)
        row = {"mode": mode, "fold": fold, "val_f1": val_f1,
               "test_results": full.get("test_results", {})}

        # Apply per-row gates
        if mode == "rotation_6d" and fold == 0:
            if val_f1 is None or val_f1 < ROTATION_6D_FLOOR:
                results["halted"] = True
                results["halt_reason"] = (
                    f"rotation_6d fold0 val_f1={val_f1} < floor "
                    f"{ROTATION_6D_FLOOR}: model itself broken")
                row["status"] = "FAIL (architecture broken)"
            else:
                val_f1_baseline = val_f1
                row["status"] = "PASS (baseline established)"
        elif mode == "jointpos_honest" and fold == 0:
            val_f1_honest_fold0 = val_f1
            if val_f1 is None:
                row["status"] = "FAIL (no metrics)"
            elif val_f1 < val_f1_baseline - JOINTPOS_FLOOR_GAP:
                row["status"] = (f"FAIL ({val_f1:.4f} < "
                                 f"{val_f1_baseline:.4f} - "
                                 f"{JOINTPOS_FLOOR_GAP})")
            else:
                row["status"] = "PASS"
        elif mode == "jointpos_flat" and fold == 0:
            if val_f1 is None:
                row["status"] = "FAIL (no metrics)"
            elif val_f1 < val_f1_baseline - JOINTPOS_FLOOR_GAP:
                row["status"] = (f"INFO (below honest gate; "
                                 f"{val_f1:.4f} < "
                                 f"{val_f1_baseline:.4f} - "
                                 f"{JOINTPOS_FLOOR_GAP})")
            else:
                row["status"] = "PASS"
        else:
            row["status"] = "INFO (non-gate fold)"

        results["runs"].append(row)
        # Persist after every run
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=2))

    results["baseline_val_f1_rotation_6d_fold0"] = val_f1_baseline
    results["honest_fold0_val_f1"] = val_f1_honest_fold0
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.strftime('%H:%M:%S')}] orchestrator done; "
          f"results → {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
