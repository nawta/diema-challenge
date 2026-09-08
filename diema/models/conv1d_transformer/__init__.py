"""Conv1D + Transformer hybrid (hoyso48 / ASL Signs 1st place).

Reference: https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with

"""

from diema.models.conv1d_transformer.conv1d_transformer_model import Conv1DTransformer_Model
from diema.models.conv1d_transformer.region_aware_conv1d_transformer_model import (
    RegionAwareConv1DTransformer_Model,
)
from diema.models.conv1d_transformer.textalign_conv1d_tr_model import TextAlignConv1DTr_Model
from diema.models.conv1d_transformer.c3d_latefusion_model import C3DLateFusion_Model

__all__ = [
    "Conv1DTransformer_Model",
    "RegionAwareConv1DTransformer_Model",
    "TextAlignConv1DTr_Model",
    "C3DLateFusion_Model",
]
