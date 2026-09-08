"""Tests for DIEM-A filename parser.

See: diema/data/parser.py
"""

import pytest

from diema.data.parser import (
    DiemaFileInfo,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
    NUM_CLASSES,
    emotion_to_label,
    is_valid_emotion,
    parse_emotion,
    parse_filename,
    parse_performer_id,
)


class TestParseFilename:
    """Test parse_filename with various filename patterns."""

    def test_basic_japanese(self):
        info = parse_filename("JP_06_anger_2_M")
        assert info.nationality == "JP"
        assert info.performer_num == "06"
        assert info.emotion == "anger"
        assert info.scenario == "2"
        assert info.intensity == "M"
        assert info.performer_id == "JP_06"
        assert info.emotion_idx == 0
        assert not info.is_augmented

    def test_basic_taiwanese(self):
        info = parse_filename("TW_30_joy_1_H")
        assert info.nationality == "TW"
        assert info.performer_num == "30"
        assert info.emotion == "joy"
        assert info.scenario == "1"
        assert info.intensity == "H"
        assert info.performer_id == "TW_30"

    def test_augmented_filename(self):
        info = parse_filename("JP_06_anger_2_M_aug00")
        assert info.nationality == "JP"
        assert info.performer_num == "06"
        assert info.emotion == "anger"
        assert info.augment_suffix == "aug00"
        assert info.is_augmented

    def test_all_emotions(self, all_emotions):
        for emotion in all_emotions:
            info = parse_filename(f"JP_01_{emotion}_1_M")
            assert info.emotion == emotion
            assert info.emotion_idx == EMOTION_TO_IDX[emotion]

    def test_all_intensities(self):
        for intensity in ["L", "M", "H"]:
            info = parse_filename(f"JP_01_anger_1_{intensity}")
            assert info.intensity == intensity

    def test_short_filename_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_filename("JP_06")

    def test_three_parts_raises(self):
        with pytest.raises(ValueError, match="at least 5"):
            parse_filename("JP_06_anger")

    def test_frozen_dataclass(self):
        info = parse_filename("JP_06_anger_2_M")
        with pytest.raises(AttributeError):
            info.nationality = "TW"


class TestParsePerformerId:
    def test_basic(self):
        assert parse_performer_id("JP_06_anger_2_M") == "JP_06"

    def test_taiwanese(self):
        assert parse_performer_id("TW_30_joy_1_H") == "TW_30"

    def test_augmented(self):
        assert parse_performer_id("JP_06_anger_2_M_aug00") == "JP_06"


class TestParseEmotion:
    def test_basic(self):
        assert parse_emotion("JP_06_anger_2_M") == "anger"

    def test_all_emotions(self, all_emotions):
        for emotion in all_emotions:
            assert parse_emotion(f"JP_01_{emotion}_1_M") == emotion


class TestEmotionMapping:
    def test_num_classes(self):
        assert NUM_CLASSES == 12

    def test_bidirectional_mapping(self):
        for emotion, idx in EMOTION_TO_IDX.items():
            assert IDX_TO_EMOTION[idx] == emotion

    def test_indices_are_contiguous(self):
        indices = sorted(EMOTION_TO_IDX.values())
        assert indices == list(range(NUM_CLASSES))


class TestEmotionToLabel:
    def test_valid(self):
        assert emotion_to_label("JP_06_anger_2_M") == 0
        assert emotion_to_label("TW_30_joy_1_H") == 4

    def test_invalid_raises(self):
        with pytest.raises(KeyError):
            emotion_to_label("JP_06_nonexistent_2_M")


class TestIsValidEmotion:
    def test_valid(self):
        assert is_valid_emotion("JP_06_anger_2_M") is True

    def test_invalid(self):
        assert is_valid_emotion("JP_06_nonexistent_2_M") is False
