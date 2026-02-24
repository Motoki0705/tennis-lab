"""Court Detection (CourtKP20) module.

This module provides court keypoint detection from tennis images.
It detects the 20 keypoints defined in `src/utils/geometry/court.court_keypoints_3d()`.

Keypoint specification (CourtKP20):
    0..3:  far/near doubles corners
    4..7:  far/near singles corners
    8..11: service line endpoints
    12,13: service T (far, near)
    14:    net center (ground)
    15..18: net posts (base/top, left/right)
    19:    center strap top
"""

from src.tasks.court_detection.models.court_keypoint_model import CourtKeypointModel

__all__ = ["CourtKeypointModel"]
