"""Logging setup with stdout + file handlers.

See: docs/decisions.md — ログ管理方針
Related: — に作業ログ
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None, filename: str = "run.log") -> logging.Logger:
    """Create a logger with stdout and optional file handler.

    Args:
        name: logger name (typically __name__)
        log_dir: directory for log file. If None, file handler is skipped.
        filename: log file name within log_dir.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s : %(levelname)s - %(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / filename, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
