"""Feature eligibility; execution order is declared by component IO."""

from collections.abc import Mapping


def validate_requested_features(enabled: Mapping[str, bool]) -> None:
    for required in ("court_kp", "camera_geometry"):
        if not enabled[required]:
            raise ValueError(f"Required reconstruction feature is disabled: {required}")
    for child, parent in (("player_reconstruction", "person_observations"), ("ball_reconstruction", "ball_detection"),
                          ("gvhmr", "player_reconstruction")):
        if enabled[child] and not enabled[parent]:
            raise ValueError(f"Feature {child} requires missing dependency {parent}")
