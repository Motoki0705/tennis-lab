from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.visualization.reveal.contracts import (
    RevealOutput,
    RevealViewpoint,
)
from src.synthetic_data_generation.visualization.reveal.viewpoint import (
    build_reveal_camera,
    look_at_camera_to_frame,
)

OUTPUT = RevealOutput(width=64, height=48, fps=30.0, crf=17)


def _viewpoint(**overrides: object) -> RevealViewpoint:
    values: dict[str, object] = {
        "anchor_court_id": "court-000",
        "eye_court_m": (16.0, -24.0, 14.0),
        "target_court_m": (-7.5, 0.0, 0.0),
        "up_court": (0.0, 0.0, 1.0),
        "horizontal_fov_deg": 60.0,
    }
    values.update(overrides)
    return RevealViewpoint(**values)  # type: ignore[arg-type]


def _court_to_nht(adapter, layout, court_id: str, point) -> NDArray[np.float64]:
    court = next(item for item in layout.courts if item.court_instance_id == court_id)
    metric = court.scene_from_court.apply(np.asarray(point, dtype=np.float64))
    similarity = np.asarray(adapter.nht_scene_from_metric_scene).reshape(4, 4)
    result: NDArray[np.float64] = similarity[:3, :3] @ metric + similarity[:3, 3]
    return result


def test_look_at_pose_is_opencv_with_image_up_along_world_up() -> None:
    pose = look_at_camera_to_frame(
        np.zeros(3), np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))
    )

    np.testing.assert_allclose(pose[:3, 2], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(pose[:3, 0], (0.0, -1.0, 0.0))
    np.testing.assert_allclose(pose[:3, 1], (0.0, 0.0, -1.0))
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_camera_centre_and_target_map_through_court_and_similarity(
    adapter, layout
) -> None:
    viewpoint = _viewpoint()
    camera = build_reveal_camera(viewpoint, layout, adapter, OUTPUT)

    np.testing.assert_allclose(
        camera.camera_to_scene[:3, 3],
        _court_to_nht(adapter, layout, "court-000", viewpoint.eye_court_m),
    )
    target = _court_to_nht(adapter, layout, "court-000", viewpoint.target_court_m)
    pixels, depth = camera.project(target[None])
    np.testing.assert_allclose(pixels[0], camera.intrinsics[:2, 2], atol=1e-9)
    assert depth[0] > 0.0


def test_points_above_the_target_project_higher_in_the_image(adapter, layout) -> None:
    viewpoint = _viewpoint()
    camera = build_reveal_camera(viewpoint, layout, adapter, OUTPUT)
    target = np.asarray(viewpoint.target_court_m)
    points = np.stack(
        [
            _court_to_nht(adapter, layout, "court-000", target),
            _court_to_nht(adapter, layout, "court-000", target + (0.0, 0.0, 2.0)),
        ]
    )

    pixels, _ = camera.project(points)

    assert pixels[1, 1] < pixels[0, 1]


def test_horizontal_fov_sets_the_focal_length(adapter, layout) -> None:
    camera = build_reveal_camera(
        _viewpoint(horizontal_fov_deg=90.0), layout, adapter, OUTPUT
    )

    assert camera.intrinsics[0, 0] == pytest.approx(32.0)
    assert camera.intrinsics[1, 1] == pytest.approx(32.0)


def test_unknown_anchor_court_is_rejected(adapter, layout) -> None:
    with pytest.raises(ValueError, match="Unknown anchor court"):
        build_reveal_camera(
            _viewpoint(anchor_court_id="court-999"), layout, adapter, OUTPUT
        )


def test_viewpoint_rejects_degenerate_directions() -> None:
    with pytest.raises(ValueError, match="differ"):
        _viewpoint(target_court_m=(16.0, -24.0, 14.0))
    with pytest.raises(ValueError, match="parallel"):
        _viewpoint(eye_court_m=(0.0, 0.0, 10.0), target_court_m=(0.0, 0.0, 0.0))
