"""Optional output features; execution order is declared by component IO.

Court detection, calibration and camera alignment always run: every
reconstruction needs cameras. Only the features below can be disabled.
"""

from collections.abc import Mapping

OPTIONAL_FEATURES = ("person_observations", "ball_detection", "player_reconstruction", "ball_reconstruction", "gvhmr")
FEATURE_DEPENDENCIES = (
    ("player_reconstruction", "person_observations"),
    ("ball_reconstruction", "ball_detection"),
    ("gvhmr", "player_reconstruction"),
)


def validate_requested_features(enabled: Mapping[str, bool]) -> None:
    if set(enabled) != set(OPTIONAL_FEATURES):
        raise ValueError(f"Feature flags must be exactly {OPTIONAL_FEATURES}, got {sorted(enabled)}")
    for child, parent in FEATURE_DEPENDENCIES:
        if enabled[child] and not enabled[parent]:
            raise ValueError(f"Feature {child} requires missing dependency {parent}")
