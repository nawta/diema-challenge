"""Squeezeformer + RoPE — ASL Fingerspelling 1st place adaptation.

17-layer Improved Squeezeformer encoder with LLaMA-style RoPE attention,
adapted for DIEM-A 12-class emotion classification.

Reference:
  - Squeezeformer paper: https://arxiv.org/abs/2206.00888
  - ASL FS 1st place (Dieter / Christof Henkel):
      https://www.kaggle.com/competitions/asl-fingerspelling/writeups/darragh-dieter-1st-place-solution-improved-squeeze
  - Repo: https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution

"""

from diema.models.squeezeformer.squeezeformer_model import Squeezeformer_Model

__all__ = ["Squeezeformer_Model"]
