"""Tests for integrated CourtKP reference-frame preparation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import src.tennis_scene.pipeline.court_reference as court_reference_module
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tennis_scene.pipeline.court_reference import (
    CourtReferenceRuntimeConfig,
    court_footpoint_polygon_px,
    prepare_court_reference,
    reference_metadata,
)
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    CourtConfig,
    court_keypoints_3d,
)


def _camera_fit(half_turn: bool) -> dict[str, Any]:
    return {
        "camera_center_court_m": [0.0, 12.0 if half_turn else -12.0, 5.0],
        "K": np.eye(3).tolist(),
        "R": np.eye(3).tolist(),
        "t": [0.0, 0.0, 1.0],
        "rmse_px": 0.0,
        "calibration": "test",
    }


def test_camera_view_reference_preserves_keypoints_and_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half_turns = (False, False, True)
    calls: list[bool] = []

    def fake_fit(
        keypoints: np.ndarray,
        size: tuple[int, int],
        half_turn: bool,
    ) -> dict[str, Any]:
        assert keypoints.shape == (14, 2)
        assert size == (1920, 1080)
        calls.append(half_turn)
        return _camera_fit(half_turn)

    monkeypatch.setattr(court_reference_module, "fit_camera", fake_fit)
    keypoints: NDArray[np.float32] = np.zeros((3, 2, 14, 2), dtype=np.float32)
    visibility: NDArray[np.float32] = np.ones((3, 2, 14), dtype=np.float32)
    for camera in range(3):
        keypoints[camera, :, :, 0] = np.arange(14) + 20 * camera

    visibility[2, 1, [1, 7]] = 0
    context = prepare_court_reference(
        camera_ids=("cam0", "cam1", "cam2"),
        keypoints=keypoints,
        visibility=visibility,
        contract=resolve_court_keypoint_contract("camera_view_v2"),
        config=CourtReferenceRuntimeConfig(
            reference_camera="cam0",
            view_half_turns=half_turns,
        ),
        size=(1920, 1080),
        frame_index=0,
    )

    assert calls == list(half_turns)
    np.testing.assert_array_equal(context.keypoints[:2], keypoints[:2])
    np.testing.assert_array_equal(
        context.keypoints[2],
        keypoints[2],
    )
    np.testing.assert_array_equal(context.visibility[:2], visibility[:2])
    np.testing.assert_array_equal(context.visibility[2], visibility[2])
    assert context.selection is not None
    assert context.document is not None
    assert context.provenance.reference_camera_id == "cam0"
    assert context.document["view_half_turns"] == [False, False, True]

    plcs_metadata = reference_metadata(context.selection, 2, "plcs")
    blcs_metadata = reference_metadata(context.selection, 1, "blcs")
    assert plcs_metadata.reference_view_index.tolist() == [0, 0]
    assert blcs_metadata.reference_view_index.tolist() == [0]


def test_camera_view_reference_rejects_invisible_calibration_point() -> None:
    visibility: NDArray[np.float32] = np.ones((3, 2, 14), dtype=np.float32)
    visibility[1, 0, 4] = 0
    with pytest.raises(ValueError, match="all 14 finite visible points"):
        prepare_court_reference(
            camera_ids=("cam0", "cam1", "cam2"),
            keypoints=np.full((3, 2, 14, 2), 0.5, dtype=np.float32),
            visibility=visibility,
            contract=resolve_court_keypoint_contract("camera_view_v2"),
            config=CourtReferenceRuntimeConfig(
                reference_camera="cam0",
                view_half_turns=(False, False, True),
            ),
            size=(1920, 1080),
            frame_index=0,
        )


def test_physical_reference_rejects_unused_camera_view_declarations() -> None:
    with pytest.raises(ValueError, match="physical_v1 forbids"):
        prepare_court_reference(
            camera_ids=("cam0",),
            keypoints=np.full((1, 2, 14, 2), 0.5, dtype=np.float32),
            visibility=np.ones((1, 2, 14), dtype=np.float32),
            contract=resolve_court_keypoint_contract("physical_v1"),
            config=CourtReferenceRuntimeConfig(
                reference_camera="cam0",
                view_half_turns=(False,),
            ),
            size=(1920, 1080),
            frame_index=0,
        )


def test_court_footpoint_polygon_projects_explicit_physical_margins() -> None:
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    keypoints = (physical * 10.0 + 200.0) / 400.0

    polygon = court_footpoint_polygon_px(
        keypoints,
        size=(400, 400),
        sideline_margin_m=1.0,
        baseline_margin_m=2.0,
    )

    expected = np.asarray(
        [
            (-HALF_DOUBLES_WIDTH - 1.0, HALF_LENGTH + 2.0),
            (HALF_DOUBLES_WIDTH + 1.0, HALF_LENGTH + 2.0),
            (HALF_DOUBLES_WIDTH + 1.0, -HALF_LENGTH - 2.0),
            (-HALF_DOUBLES_WIDTH - 1.0, -HALF_LENGTH - 2.0),
        ]
    ) * 10.0 + 200.0
    actual = np.asarray(polygon)
    distances = np.linalg.norm(expected[:, None] - actual[None], axis=-1)
    assert actual.shape == (4, 2)
    assert len(set(distances.argmin(axis=1))) == 4
    assert float(distances.min(axis=1).max()) < 1e-4


@pytest.mark.parametrize("baseline_margin", [1.0, 30.0])
@pytest.mark.parametrize("half_turn", [False, True])
def test_court_footpoint_region_matches_world_bounds_even_past_camera(
    baseline_margin: float,
    half_turn: bool,
) -> None:
    # The camera plane is world y=-33.33. A 30m margin crosses it, although
    # the court itself remains entirely visible in this perspective image.
    homography: np.ndarray = np.array(
        [[50.0, 10.0, 640.0], [0.0, -5.0, 360.0], [0.0, 0.03, 1.0]]
    )
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    projected = cv2.perspectiveTransform(physical[None], homography)[0]
    keypoints = projected / [1280, 720]
    view = build_court_view_record(
        camera_id="cam0",
        camera_center_court_m=[0.0, 12.0 if half_turn else -12.0, 5.0],
        contract=resolve_court_keypoint_contract("camera_view_v2"),
    )
    keypoints = keypoints[np.asarray(view.semantic_to_physical[:14])]
    polygon = np.asarray(court_footpoint_polygon_px(
        keypoints, size=(1280, 720), sideline_margin_m=1.0,
        baseline_margin_m=baseline_margin,
    ), dtype=np.float32)
    assert np.isfinite(polygon).all()
    assert (polygon >= -1e-6).all()
    assert (polygon <= [1280 + 1e-6, 720 + 1e-6]).all()
    assert cv2.isContourConvex(polygon)
    assert cv2.pointPolygonTest(polygon, (640.0, 360.0), False) >= 0

    # Independent membership oracle: map test pixels back to the court plane.
    pixels = np.random.default_rng(0).uniform([0, 0], [1280, 720], size=(500, 2))
    homogeneous = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(homography).T
    world = homogeneous[:, :2] / homogeneous[:, 2:]
    expected = (
        (homogeneous[:, 2] > 0)
        & (abs(world[:, 0]) <= HALF_DOUBLES_WIDTH + 1.0)
        & (abs(world[:, 1]) <= HALF_LENGTH + baseline_margin)
    )
    actual = np.array([
        cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
        for x, y in pixels
    ])
    np.testing.assert_array_equal(actual, expected)
