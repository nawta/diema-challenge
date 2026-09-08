"""Keypoint Pool MLP: MLP-only 2-stream (joint + bone) with lateral connections.

Reference: ASL Signs 12th place solution (DONJYARAHOI)
  https://www.kaggle.com/competitions/asl-signs/writeups/donjyarahoi-12th-place-solution-mlp-based-structur
Related paper: https://arxiv.org/abs/2211.01367 (Two-Stream Network)
"""

from diema.models.keypoint_pool_mlp.keypoint_pool_mlp_model import KeypointPoolMLP_Model

__all__ = ["KeypointPoolMLP_Model"]
