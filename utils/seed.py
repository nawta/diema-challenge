"""Reproducibility utilities.

See: diema_challenge_implementation_spec.md §6.5 — seed 固定
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch (CPU+CUDA), and cuDNN determinism.
    Also sets PYTHONHASHSEED for hash-based randomness.

    Args:
        seed: integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
