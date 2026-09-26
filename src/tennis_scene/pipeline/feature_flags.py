"""Feature eligibility and historical scene labels; execution order is declared by component IO."""

from collections.abc import Mapping
from enum import StrEnum


class Stage(StrEnum):
    """Pipeline stages."""

    COURT_KP = "court_kp"
    GVHMR = "gvhmr"
    BALL_DETECTION = "ball_detection"
    PLCS = "plcs"
    BLCS = "blcs"
    PERSON_OBSERVATIONS = "person_observations"
    CAMERA_GEOMETRY = "camera_geometry"
    PLAYER_RECONSTRUCTION = "player_reconstruction"
    BALL_RECONSTRUCTION = "ball_reconstruction"


def validate_requested_features(enabled: Mapping[str, bool]) -> None:
    for required in ("court_kp", "camera_geometry"):
        if not enabled[required]:
            raise ValueError(f"Required reconstruction feature is disabled: {required}")
    for child, parent in (("player_reconstruction", "person_observations"), ("ball_reconstruction", "ball_detection"),
                          ("gvhmr", "player_reconstruction")):
        if enabled[child] and not enabled[parent]:
            raise ValueError(f"Feature {child} requires missing dependency {parent}")
