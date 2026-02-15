"""Court geometry for 3D rendering in the WebUI.

We reuse the repo's canonical court definition in `src/utils/geometry/court.py`
to keep visualization and simulation in the same coordinate system.
"""

from __future__ import annotations

from src.blcs.generate_dataset.api_server.schemas import CourtGeometryResponse
from src.utils.schema.court import court_keypoints_3d


def build_court_geometry_response() -> CourtGeometryResponse:
    kp = court_keypoints_3d()  # (20, 3)

    # Line segments defined over CourtKP20 indices.
    # The intent is to create a "recognizable tennis court" with minimal lines.
    #
    # CourtKP20 index reference (see `court_keypoints_3d` docstring):
    # - Doubles corners: 0..3
    # - Singles corners: 4..7
    # - Service line endpoints: 8..11
    # - Service T: 12, 13
    # - Net center (ground): 14
    # - Net posts: 15..18
    # - Center strap top: 19
    segments = [
        # Doubles rectangle
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        # Singles rectangle
        [4, 6],
        [6, 7],
        [7, 5],
        [5, 4],
        # Service lines (far / near)
        [8, 9],
        [10, 11],
        # Singles sidelines between service lines (service box sides)
        [8, 10],
        [9, 11],
        # Center service lines (to net center ground)
        [12, 14],
        [13, 14],
        # Net posts (verticals)
        [15, 16],
        [17, 18],
        # Net top (approx curve as 2 segments via center strap top)
        [16, 19],
        [19, 18],
    ]

    return CourtGeometryResponse(
        keypoints=kp.tolist(),
        segments=segments,
    )

