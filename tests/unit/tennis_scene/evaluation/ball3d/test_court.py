from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.tennis_scene.evaluation.ball3d.court import court_views_from_diagnostics
from tests.support.ball3d_evaluation import court_diagnostics, make_inputs


@pytest.mark.parametrize("mutation", ["missing", "camera", "frame_count", "selection"])
def test_view_rejects_missing_or_misbound_raw_court_observations(mutation):
    view = make_inputs().views[1]
    with pytest.raises(ValueError, match="court|Court|camera axis"):
        if mutation == "missing":
            replace(view, court=cast(Any, None))
        elif mutation == "camera":
            replace(view, court=replace(view.court, camera_id="far"))
        elif mutation == "frame_count":
            replace(view, court=replace(view.court, source_num_frames=5))
        else:
            selected = view.court.selected_for_homography.copy()
            selected[0, 8] = (
                True  # This raw near-camera keypoint is explicitly invalid.
            )
            replace(view.court, selected_for_homography=selected)


def test_sparse_source_court_frames_are_not_broadcast_and_missing_lines_fail_on_request():
    inputs = make_inputs()
    doc = court_diagnostics(inputs)
    del doc["cameras"][1]["frames"][0]["raw_line_observations"]
    views = court_views_from_diagnostics(
        doc, inputs.camera_ids, inputs.source_size_wh, inputs.num_frames
    )
    np.testing.assert_array_equal(views[1].frame_indices, [0])
    assert views[1].keypoints_px.shape == (1, 14, 2)
    np.testing.assert_array_equal(views[1].keypoints_px[0, 0], [68.25, 79.5])
    assert views[1].lines is None and views[1].line_missing_reason
    with pytest.raises(ValueError, match="near.*LINE observations unavailable"):
        views[1].require_lines()
    with pytest.raises(ValueError, match="near.*LINE observations unavailable"):
        court_views_from_diagnostics(
            doc,
            inputs.camera_ids,
            inputs.source_size_wh,
            inputs.num_frames,
            require_lines=True,
        )
