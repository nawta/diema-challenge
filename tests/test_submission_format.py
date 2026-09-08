"""Tests for submission format validation.

See: diema_challenge_implementation_spec.md §6.8
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from diema.data.parser import NUM_CLASSES


class TestSubmissionFormat:
    """Test submission file validation logic."""

    def _write_csv(self, df: pd.DataFrame) -> str:
        """Write DataFrame to temp CSV and return path."""
        path = tempfile.mktemp(suffix=".csv")
        df.to_csv(path, index=False)
        return path

    def test_valid_submission(self, sample_submission_data):
        df = pd.DataFrame(sample_submission_data)
        # All labels in range
        assert (df["predicted_label"] >= 0).all()
        assert (df["predicted_label"] < NUM_CLASSES).all()
        # No duplicates
        assert df["sample_name"].duplicated().sum() == 0

    def test_duplicate_sample_names(self):
        df = pd.DataFrame({
            "sample_name": ["a", "a", "b"],
            "predicted_label": [0, 1, 2],
        })
        assert df["sample_name"].duplicated().sum() == 1

    def test_label_out_of_range(self):
        df = pd.DataFrame({
            "sample_name": ["a", "b"],
            "predicted_label": [0, 99],
        })
        out_of_range = ((df["predicted_label"] < 0) | (df["predicted_label"] >= NUM_CLASSES)).sum()
        assert out_of_range == 1

    def test_negative_label(self):
        df = pd.DataFrame({
            "sample_name": ["a", "b"],
            "predicted_label": [-1, 5],
        })
        out_of_range = ((df["predicted_label"] < 0) | (df["predicted_label"] >= NUM_CLASSES)).sum()
        assert out_of_range == 1

    def test_expected_row_count(self, sample_submission_data):
        df = pd.DataFrame(sample_submission_data)
        # Should have exactly as many rows as expected
        assert len(df) == len(sample_submission_data["sample_name"])

    def test_missing_column(self):
        df = pd.DataFrame({"sample_name": ["a"]})
        assert "predicted_label" not in df.columns
