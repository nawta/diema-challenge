"""Shallow 5-layer Conformer for DIEM-A 12-class emotion classification.

Direct port of the ASL Fingerspelling 11th place solution architecture
(bliao / Baohao Liao & Shaomu Tan) from the Kaggle writeup:
    https://www.kaggle.com/competitions/asl-fingerspelling/writeups/\
baohao-shaomu-the-11th-place-shallow-encoder-decod
    https://github.com/BaohaoLiao/aslfrv4

Paper base: "Conformer: Convolution-augmented Transformer for Speech Recognition"
            (Gulati et al., 2020, https://arxiv.org/abs/2005.08100)

DIEM-A adaptations:
  - Input is a skeleton tensor (N, C=6, T=64, V=25) instead of
    landmark sequences (N, 130_keypoints, T=384).
  - Tokenization: we embed per (t, v) with a small Conv2d stem, then
    aggregate over the V dimension (mean-pool) so that each clip becomes
    a sequence of T tokens (one per frame). This matches the native
    Conformer assumption of "one token per time step" while still letting
    the stem fuse joint information locally.
  - Encoder only: the original bliao recipe uses a 3-layer Transformer
    decoder for CTC/CE joint loss. For DIEM-A classification we drop the
    decoder and use GAP over T + a Linear head for 12 classes.
  - conv_kernel_size=15 (not 31). DIEM-A clips are only 64 frames, so the
    original 31-tap depthwise conv covers ~half the clip; 15 is a better
    local receptive field for T=64.
  - Fully pure-PyTorch: no fairseq / torchaudio / timm dependency.

BaseModel contract:
    Input : (N, C, T, V) = (N, 6, 64, 25)
    Output: {"logits": (N, num_class=12)}

Related:
  - diema/models/conformer/conformer_block.py  (ConformerBlock, Macaron FFN, ConvolutionModule)
  - diema/models/skateformer/partition_attention.py  (MultiHeadSelfAttention — reused)
  - diema/models/ctrgcn/ctrgcn_model.py  (from_config pattern reference)

"""

import math

import torch
import torch.nn as nn

from diema.models.base import BaseModel
from diema.models.conformer.conformer_block import ConformerBlock


class TokenEmbedding(nn.Module):
    """Per-(t, v) token embedding stem.

    Uses a small Conv2d (kernel 3x1) for local temporal smoothing and a 1x1
    lift to the model dimension, followed by BatchNorm. Identical in spirit
    to the SkateFormer stem.
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, d_model, kernel_size=(3, 1), padding=(1, 0),
        )
        self.bn = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V) -> (N, d_model, T, V)
        return self.bn(self.conv(x))


class Conformer_Model(BaseModel):
    """Shallow Conformer encoder for DIEM-A skeleton emotion classification.

    Args:
        num_class: number of output classes (12 for DIEM-A)
        in_channels: input feature channels (6 for rotation_6d)
        num_layers: number of ConformerBlock layers (default 5, bliao 11th)
        hidden_dim: model dimension / token embedding dim (default 384)
        mlp_dim: Macaron FFN hidden dimension (default 1024)
        num_heads: MHSA heads (must divide hidden_dim) (default 6)
        conv_kernel_size: depthwise conv kernel in ConvolutionModule
            (default 15, reduced from paper's 31 for clip_length=64)
        dropout: dropout for FFN / conv / proj (default 0.1)
        attn_dropout: attention weight dropout (default 0.0)
        drop_path: stochastic depth *max* rate; layers linearly interpolate
            from 0 to drop_path across depth (default 0.1)
        num_nodes: V dimension, used only for initialising the positional
            / joint embeddings; default 25 (DIEM-A)
        max_seq_len: maximum T supported by the learned temporal positional
            embedding (default 256, same as SkateFormer)
    """

    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        num_layers: int = 5,
        hidden_dim: int = 384,
        mlp_dim: int = 1024,
        num_heads: int = 6,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
        num_nodes: int = 25,
        max_seq_len: int = 256,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Stem: (N, C, T, V) -> (N, d_model, T, V)
        self.embed = TokenEmbedding(in_channels, hidden_dim)

        # Learned temporal positional embedding (one token per frame after
        # V-pooling). Size (1, T_max, d_model), broadcast over batch.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.input_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Linearly-scaled DropPath schedule across the stack.
        if num_layers > 1:
            dpr = [drop_path * i / (num_layers - 1) for i in range(num_layers)]
        else:
            dpr = [drop_path]

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=hidden_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        # ConformerBlock already ends with a LayerNorm (`final_norm`), so we
        # don't add a second back-to-back normalization at the encoder
        # output.
        self.head_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        N, C, T, V = x.size()
        assert T <= self.max_seq_len, (
            f"Input T={T} exceeds max_seq_len={self.max_seq_len}"
        )

        # Stem: (N, C, T, V) -> (N, d_model, T, V)
        h = self.embed(x)

        # Aggregate over V (one token per frame).
        # (N, d_model, T, V) -> (N, d_model, T) -> (N, T, d_model)
        h = h.mean(dim=3)
        h = h.transpose(1, 2).contiguous()

        # Add learned temporal positional embedding (broadcast over batch).
        h = h + self.pos_embed[:, :T, :]
        h = self.input_drop(h)

        # Run Conformer blocks (each ends with its own LayerNorm).
        for blk in self.blocks:
            h = blk(h)

        # GAP over T -> (N, d_model)
        h = h.mean(dim=1)
        h = self.head_drop(h)
        logits = self.fc(h)
        return {"logits": logits}

    @property
    def output_dim(self) -> int:
        return self.num_class

    @classmethod
    def from_config(cls, config) -> "Conformer_Model":
        return cls(
            num_class=config.model.num_class,
            in_channels=getattr(config.model, "in_channels", 6),
            num_layers=getattr(config.model, "num_layers", 5),
            hidden_dim=getattr(config.model, "hidden_dim", 384),
            mlp_dim=getattr(config.model, "mlp_dim", 1024),
            num_heads=getattr(config.model, "num_heads", 6),
            conv_kernel_size=getattr(config.model, "conv_kernel_size", 15),
            dropout=getattr(config.model, "dropout", 0.1),
            attn_dropout=getattr(config.model, "attn_dropout", 0.0),
            drop_path=getattr(config.model, "drop_path", 0.1),
            num_nodes=getattr(config.skeleton, "num_nodes", 25),
            max_seq_len=getattr(config.model, "max_seq_len", 256),
        )


from diema.models.registry import register_model  # noqa: E402

register_model("conformer", Conformer_Model)
