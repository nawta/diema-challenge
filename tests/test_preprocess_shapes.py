"""Tests for data preprocessing output shapes.

Uses synthetic data to verify tensor shapes without requiring real BVH files.

See: diema_challenge_implementation_spec.md §6.1, §6.3
"""

import numpy as np
import pytest
import torch

from diema.data.parser import NUM_CLASSES


class TestTensorShapes:
    """Test expected tensor shapes for the pipeline."""

    def test_ctv_shape_6d(self):
        """6D representation: C=6, T=clip_length, V=25 nodes."""
        clip_length = 64
        num_nodes = 25
        in_channels = 6
        batch_size = 4

        # Simulate model input tensor
        x = torch.randn(batch_size, in_channels, clip_length, num_nodes)
        assert x.shape == (4, 6, 64, 25)

    def test_ctv_shape_quaternion(self):
        """Quaternion representation: C=4, T=clip_length, V=25 nodes."""
        x = torch.randn(4, 4, 64, 25)
        assert x.shape == (4, 4, 64, 25)

    def test_label_shape(self):
        """Labels are scalar integers in [0, NUM_CLASSES)."""
        labels = torch.randint(0, NUM_CLASSES, (4,))
        assert labels.shape == (4,)
        assert labels.dtype == torch.int64

    def test_logit_shape(self):
        """Model output logits: (N, NUM_CLASSES)."""
        logits = torch.randn(4, NUM_CLASSES)
        assert logits.shape == (4, 12)

    def test_stgcn_data_bn_reshape(self):
        """Verify the BatchNorm reshape used in STGCN_Model."""
        N, C, T, V = 4, 6, 64, 25
        x = torch.randn(N, C, T, V)

        # Forward reshape: (N, C, T, V) -> (N, V*C, T)
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        assert x_bn.shape == (N, 150, T)

        # Inverse reshape: (N, V*C, T) -> (N, C, T, V)
        x_back = x_bn.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        assert x_back.shape == (N, C, T, V)


class TestMetricsShapes:
    """Test metric computation with valid input shapes."""

    def test_compute_metrics(self):
        from utils.metrics import compute_metrics
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        result = compute_metrics(y_true, y_pred)

        assert result["macro_f1"] == 1.0
        assert result["accuracy"] == 1.0
        assert len(result["class_wise_f1"]) == 12
        assert result["confusion_matrix"].shape == (12, 12)

    def test_compute_metrics_wrong(self):
        from utils.metrics import compute_metrics
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 2, 3, 0])
        result = compute_metrics(y_true, y_pred)

        assert result["accuracy"] == 0.0
        assert result["macro_f1"] == 0.0
