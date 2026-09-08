"""Conformer: Convolution-augmented Transformer for DIEM-A emotion classification.

Shallow 5-layer Conformer adapted from the ASL Fingerspelling 11th place solution
(bliao / Baohao Liao & Shaomu Tan). Reference: https://github.com/BaohaoLiao/aslfrv4

Paper: "Conformer: Convolution-augmented Transformer for Speech Recognition"
       https://arxiv.org/abs/2005.08100

"""

from diema.models.conformer.conformer_model import Conformer_Model

__all__ = ["Conformer_Model"]
