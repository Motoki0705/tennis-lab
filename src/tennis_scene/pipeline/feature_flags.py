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
    PLCS_REID = "plcs_reid"
    COURT_SIDE = "court_side"
    CAMERA_GEOMETRY = "camera_geometry"
    PLAYER_RECONSTRUCTION = "player_reconstruction"
    BALL_RECONSTRUCTION = "ball_reconstruction"


def validate_requested_features(enabled: Mapping[str, bool]) -> None:
    for required in ("court_kp", "court_side", "camera_geometry"):
        if not enabled[required]:
            raise ValueError(f"Required reconstruction feature is disabled: {required}")
    for child, parent in (("plcs_reid", "person_observations"), ("court_side", "person_observations"),
                          ("player_reconstruction", "plcs_reid"), ("ball_reconstruction", "ball_detection"),
                          ("gvhmr", "player_reconstruction")):
        if enabled[child] and not enabled[parent]:
            raise ValueError(f"Feature {child} requires missing dependency {parent}")
