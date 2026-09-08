"""Environment configuration for path management.

See: docs/decisions.md — パス管理方針
Related: utils/io.py (I/O helpers), tools/* (pipeline scripts)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default data root: data/diema_challenge
# Override with DIEMA_DATA_ROOT environment variable if needed.
_DEFAULT_DATA_ROOT = Path("data/diema_challenge")


def _project_root() -> Path:
    """Locate the project root by walking up from this file."""
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """Resolve data root from env var or default."""
    return Path(os.environ.get("DIEMA_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))


@dataclass
class EnvConfig:
    """Project-wide path configuration.

    All paths are absolute, derived from project root.
    Data paths point to data/diema_challenge by default.
    Override data root via DIEMA_DATA_ROOT env var.
    """

    project_root: Path = field(default_factory=_project_root)
    data_root: Path = field(default_factory=_data_root)

    @property
    def input_dir(self) -> Path:
        return self.data_root

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def oof_dir(self) -> Path:
        return self.project_root / "output" / "oof"

    @property
    def predictions_dir(self) -> Path:
        return self.project_root / "output" / "predictions"

    @property
    def submissions_dir(self) -> Path:
        return self.project_root / "output" / "submissions"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "output" / "artifacts"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    def exp_output_dir(self, exp_name: str) -> Path:
        return self.artifacts_dir / exp_name
