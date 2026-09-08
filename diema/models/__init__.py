"""Model registry and auto-import for all model architectures.

See: diema_challenge_implementation_spec.md §6.4 — モデル層
Related: diema/models/stgcn/ (STGCN++), diema/models/heads.py
"""

from diema.models.registry import get_model, list_models, register_model, build_model

# Auto-register all model architectures
import diema.models.stgcn.stgcn_model  # noqa: F401 — registers "stgcn"
import diema.models.ctrgcn.ctrgcn_model  # noqa: F401 — registers "ctrgcn"
import diema.models.ctrgcn.ctrgcn_subset_model  # noqa: F401 — registers "ctrgcn_subset"
import diema.models.skateformer.skateformer_model  # noqa: F401 — registers "skateformer"
import diema.models.protogcn.protogcn_model  # noqa: F401 — registers "protogcn"
import diema.models.conv1d_transformer.conv1d_transformer_model  # noqa: F401 — registers "conv1d_transformer"
import diema.models.conv1d_transformer.joint_masked_conv1d_transformer_model  # noqa: F401 — registers "joint_masked_conv1d_transformer"
import diema.models.conv1d_transformer.region_aware_conv1d_transformer_model  # noqa: F401 — registers "region_aware_conv1d_transformer"
import diema.models.conv1d_transformer.supcon_conv1d_tr_model  # noqa: F401 — registers "supcon_conv1d_tr"
import diema.models.conv1d_transformer.moe_head_model  # noqa: F401 — registers "style_moe_conv1d_tr"
import diema.models.dual_head.dual_head_model  # noqa: F401 — registers "dual_head"
import diema.models.keypoint_pool_mlp.keypoint_pool_mlp_model  # noqa: F401 — registers "keypoint_pool_mlp"
import diema.models.conformer.conformer_model  # noqa: F401 — registers "conformer"
import diema.models.squeezeformer.squeezeformer_model  # noqa: F401 — registers "squeezeformer"
import diema.models.motionbert.motionbert_model  # noqa: F401 — registers "motionbert"

__all__ = ["get_model", "list_models", "register_model", "build_model"]
