"""I/O helper functions for data loading and saving.

See: diema_challenge_implementation_spec.md §6.1 — データ前処理
Related: diema/data/preprocess.py, tools/prepare_dataset.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def load_npz(path: str | Path) -> dict:
    """Load a NumPy .npz file.

    Returns a dict-like NpzFile. Use allow_pickle=True for object arrays.
    """
    return np.load(path, allow_pickle=True)


def save_npz(path: str | Path, **arrays) -> None:
    """Save arrays to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_pickle(path: str | Path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str | Path) -> None:
    """Save an object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str | Path, indent: int = 2) -> None:
    """Save an object to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
