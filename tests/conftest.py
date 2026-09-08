"""Shared test fixtures for DIEM-A challenge tests.

Provides dummy data generators that don't require actual BVH files.
"""

import pytest

from diema.data.parser import EMOTION_TO_IDX


@pytest.fixture
def sample_filenames():
    """List of valid DIEM-A filename stems covering various patterns."""
    return [
        "JP_01_anger_1_H",
        "JP_01_anger_2_M",
        "JP_01_joy_1_L",
        "JP_02_contempt_1_H",
        "JP_02_disgust_2_M",
        "JP_03_fear_1_H",
        "JP_03_surprise_2_L",
        "TW_10_sadness_1_M",
        "TW_10_gratitude_2_H",
        "TW_10_guilt_1_L",
        "TW_11_jealousy_1_M",
        "TW_11_shame_2_H",
        "TW_12_pride_1_M",
        "TW_12_anger_2_L",
        "JP_04_joy_1_H",
        "JP_04_contempt_2_M",
        "JP_05_disgust_1_L",
        "JP_05_fear_2_H",
        "TW_13_surprise_1_M",
        "TW_13_sadness_2_L",
    ]


@pytest.fixture
def augmented_filenames():
    """List of augmented filename stems."""
    return [
        "JP_01_anger_1_H_aug00",
        "JP_01_anger_1_H_aug01",
        "TW_10_sadness_1_M_aug00",
    ]


@pytest.fixture
def all_emotions():
    """List of all 12 emotion labels."""
    return list(EMOTION_TO_IDX.keys())


@pytest.fixture
def emotion_to_idx():
    """Emotion-to-index mapping."""
    return EMOTION_TO_IDX.copy()


@pytest.fixture
def sample_submission_data():
    """Sample submission data for validation tests."""
    return {
        "sample_name": [f"JP_01_anger_{i}_H" for i in range(10)],
        "predicted_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "predicted_emotion": [
            "anger", "contempt", "disgust", "fear", "joy",
            "sadness", "surprise", "jealousy", "shame", "guilt",
        ],
    }
