"""Court geometry helpers shared by the tennis simulator.

This module re-exports from src.utils.geometry.court for backward compatibility.

Note:
    New code should import from src.utils.geometry directly.

"""

# Re-export all court geometry
from src.utils.geometry.court import (
    # Dimensions
    BASELINE_CLEAR,
    CENTER_MARK_LENGTH,
    COURT_LENGTH,
    DOUBLES_WIDTH,
    FENCE_HEIGHT,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    NET_POST_OFFSET_X,
    SERVICE_LINE_DISTANCE,
    SIDELINE_CLEAR,
    SINGLES_WIDTH,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    # Functions and classes
    Camera,
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
    sample_camera_position_on_fence,
)

__all__ = [
    # Dimensions
    "COURT_LENGTH",
    "HALF_LENGTH",
    "SINGLES_WIDTH",
    "HALF_SINGLES_WIDTH",
    "DOUBLES_WIDTH",
    "HALF_DOUBLES_WIDTH",
    "SERVICE_LINE_DISTANCE",
    "CENTER_MARK_LENGTH",
    "NET_HEIGHT_CENTER",
    "NET_HEIGHT_POST",
    "NET_POST_OFFSET_X",
    "BASELINE_CLEAR",
    "SIDELINE_CLEAR",
    "FENCE_HEIGHT",
    "X_MIN",
    "X_MAX",
    "Y_MIN",
    "Y_MAX",
    # Functions and classes
    "court_keypoints_3d",
    "Camera",
    "make_look_at_camera",
    "project_points",
    "sample_camera_position_on_fence",
]
