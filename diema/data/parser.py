"""DIEM-A filename parser and emotion label mapping.

Parses the DIEM-A Challenge filename convention to extract metadata.
This is the single source of truth for emotion↔index mapping and
filename structure.

See: diema_challenge_implementation_spec.md §3.3 — 命名規則
Related: diema/data/splits.py (performer_id for LPO), diema/data/preprocess.py

Filename format:
    {nationality}_{performerID}_{emotion}_{scenario}_{intensity}
    Example: JP_06_anger_2_M → nationality=JP, performer_id=JP_06, emotion=anger, scenario=2, intensity=M

Augmented filenames (from the internal baseline preprocessing) append _augNN:
    JP_06_anger_2_M_aug00 → same metadata, augmentation suffix ignored
"""

from dataclasses import dataclass


# Official 12-class emotion-to-index mapping (matches configs/emo_to_idx_12.txt)
EMOTION_TO_IDX: dict[str, int] = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "sadness": 5,
    "surprise": 6,
    "jealousy": 7,
    "shame": 8,
    "guilt": 9,
    "gratitude": 10,
    "pride": 11,
}

IDX_TO_EMOTION: dict[int, str] = {v: k for k, v in EMOTION_TO_IDX.items()}

NUM_CLASSES = len(EMOTION_TO_IDX)


@dataclass(frozen=True)
class DiemaFileInfo:
    """Parsed metadata from a DIEM-A filename."""

    nationality: str      # e.g., "JP", "TW"
    performer_num: str    # e.g., "06", "30"
    emotion: str          # e.g., "anger"
    scenario: str         # e.g., "2"
    intensity: str        # e.g., "M", "H", "L"
    augment_suffix: str   # e.g., "", "aug00"

    @property
    def performer_id(self) -> str:
        """Unique performer identifier for LPO splits."""
        return f"{self.nationality}_{self.performer_num}"

    @property
    def emotion_idx(self) -> int:
        """Emotion label index. Raises KeyError if emotion not in mapping."""
        return EMOTION_TO_IDX[self.emotion]

    @property
    def is_augmented(self) -> bool:
        return self.augment_suffix != ""


def parse_filename(filename_stem: str) -> DiemaFileInfo:
    """Parse a DIEM-A filename stem into structured metadata.

    Args:
        filename_stem: filename without extension, e.g. 'JP_06_anger_2_M'
            or 'TW_30_joy_2_M_aug02'

    Returns:
        DiemaFileInfo with all parsed fields.

    Raises:
        ValueError: if filename doesn't match expected format (< 5 parts)
    """
    parts = filename_stem.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Cannot parse DIEM-A filename '{filename_stem}': "
            f"expected at least 5 underscore-separated parts "
            f"(nationality_performerID_emotion_scenario_intensity), "
            f"got {len(parts)}"
        )

    nationality = parts[0]
    performer_num = parts[1]
    emotion = parts[2]
    scenario = parts[3]
    intensity = parts[4]

    # Remaining parts are augmentation suffix (e.g., "aug00")
    augment_suffix = "_".join(parts[5:]) if len(parts) > 5 else ""

    return DiemaFileInfo(
        nationality=nationality,
        performer_num=performer_num,
        emotion=emotion,
        scenario=scenario,
        intensity=intensity,
        augment_suffix=augment_suffix,
    )


def parse_performer_id(filename_stem: str) -> str:
    """Extract performer ID from a DIEM-A filename stem.

    Lightweight version for split generation — avoids full parsing overhead.

    Args:
        filename_stem: filename without extension

    Returns:
        Performer ID string, e.g., 'JP_06'
    """
    parts = filename_stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def parse_emotion(filename_stem: str) -> str:
    """Extract emotion string from a DIEM-A filename stem.

    Args:
        filename_stem: filename without extension

    Returns:
        Emotion string, e.g., 'anger'
    """
    parts = filename_stem.split("_")
    return parts[2]


def emotion_to_label(filename_stem: str) -> int:
    """Extract emotion label index from a DIEM-A filename stem.

    Args:
        filename_stem: filename without extension

    Returns:
        Integer emotion index (0-11)

    Raises:
        KeyError: if emotion not in EMOTION_TO_IDX
    """
    emotion = parse_emotion(filename_stem)
    return EMOTION_TO_IDX[emotion]


def is_valid_emotion(filename_stem: str) -> bool:
    """Check if the file's emotion is in the 12-class mapping."""
    emotion = parse_emotion(filename_stem)
    return emotion in EMOTION_TO_IDX
