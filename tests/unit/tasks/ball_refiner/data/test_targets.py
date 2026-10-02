"""Amodal labels differ from detector visibility, including ambiguous frames."""

import json

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.targets import TargetReason, project_store_targets
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def test_single_ball_policy_does_not_invent_negative_or_estimated_targets(tmp_path):
    write_store_clip(tmp_path, "chat/clip", [
        frame(0, ball()), frame(1, ball("out_of_frame", None)), frame(2),
        frame(3, ball("unresolved", None)), frame(4, ball("interpolated")),
        frame(5, ball("occlusion_estimated")), frame(6, ball(), annotated=False),
        frame(7, ball(), ball("out_of_frame", None, track="second")),
    ])
    store = BallFrameStore(tmp_path)
    result = project_store_targets(store, store.clips[0])
    assert result.reason.tolist() == [0, 1, 3, 7, 5, 6, 2, 4]
    assert result.position_valid.tolist() == [True] + [False] * 7
    assert result.presence_valid.tolist() == [True, True] + [False] * 6
    assert result.presence.tolist() == [True] + [False] * 7
    target = result.target(0, 8)
    assert target.weight.tolist() == [[1, 1, 0, 0, 0, 0, 0, 0]]
    np.testing.assert_allclose(result.uv[0], [10 / 63, 20 / 47])
    np.testing.assert_equal(result.uv[4], result.uv[0])  # reference only
    assert np.isnan(result.uv[7]).all()  # no choice between multiple balls
    assert sum(result.counts().values()) == 8
    for start, stop in [(-1, 1), (0, 0), (0, 9)]:
        with pytest.raises(ValueError, match="window"):
            result.target(start, stop)


def test_source_resize_and_actual_pts_are_recovered_without_nominal_fps(tmp_path):
    write_store_clip(tmp_path, "tracknet/clip", [frame(i, ball()) for i in (1000, 1002, 1007)], size=(64, 48))
    metadata = tmp_path / "metadata.json"
    document = json.loads(metadata.read_text())
    document["clips"][0].update(source_width=128, source_height=96)
    metadata.write_text(json.dumps(document))
    store = BallFrameStore(tmp_path)
    result = project_store_targets(store, store.clips[0])
    np.testing.assert_allclose(result.uv[0], [20 / 127, 40 / 95])
    np.testing.assert_allclose(result.timestamps_seconds, [0, 2 / 30, 7 / 30])
    assert result.pts.tolist() == [1000, 1002, 1007]
    assert result.frame_index.tolist() == [0, 1, 2]
    assert result.reason.tolist() == [TargetReason.OBSERVED] * 3


def test_outside_endpoint_grid_is_rejected_instead_of_clipped(tmp_path):
    write_store_clip(tmp_path, "tracknet/clip", [frame(0, ball(xy=(63.5, 20)))])
    store = BallFrameStore(tmp_path)
    with pytest.raises(ValueError, match="endpoint grid"):
        project_store_targets(store, store.clips[0])
