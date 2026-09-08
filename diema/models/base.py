"""Base model protocol for all emotion recognition models.

Every model must subclass BaseModel to ensure a consistent interface
for training, CLI, and config system.

See: diema_challenge_implementation_spec.md §6.4
Related: diema/models/registry.py, diema/training/train.py

Ported from: an internal baseline
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all emotion recognition models.

    Contract:
      - Input: always (N, C, T, V) tensor — batch, channels, frames, joints.
        The model reshapes internally if it needs a different layout.
      - Output: always a dict with at least {"logits": Tensor (N, num_class)}.
        Models may add extra keys:
          - "aux_losses": dict of {name: value} — added to main loss
          - "attention": any tensor — for visualization only
    """

    def _validate_input(self, x):
        """Guard against silent dimension-order bugs."""
        assert x.ndim == 4, (
            f"Expected input of shape (N, C, T, V), got {x.shape}. "
            f"All models receive (batch, channels, frames, joints)."
        )

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C, T, V)
        Returns:
            dict with at least {"logits": Tensor (N, num_class)}
        """
        ...

    @property
    @abstractmethod
    def output_dim(self):
        """Return the number of output classes."""
        ...

    @classmethod
    def from_config(cls, config):
        """Instantiate a model from a config namespace.

        Subclasses should override if they need custom config mapping.
        """
        return cls(
            num_class=config.model.num_class,
            edge_index=config.skeleton.inward_edges,
            num_nodes=config.skeleton.num_nodes,
            in_channels=config.model.in_channels,
            dropout=getattr(config.model, "dropout", 0.5),
        )
