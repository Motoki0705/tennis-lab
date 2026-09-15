"""Tests for integrated CourtKP reference-frame preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import src.tennis_scene.pipeline.court_reference as court_reference_module
from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tennis_scene.pipeline.court_reference import (
    CourtReferenceRuntimeConfig,
    court_footpoint_polygon_px,
    prepare_court_reference,
    reference_metadata,
)
from src.utils.schema.court import (
    COURT_KP20_HALF_TURN_INDEX,
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


def test_camera_view_reference_aligns_keypoints_and_visibility(
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
    keypoints: NDArray[np.float32] = np.zeros(
        (3, 2, 14, 2), dtype=np.float32
    )
    visibility: NDArray[np.float32] = np.ones(
        (3, 2, 14), dtype=np.float32
    )
    for camera in range(3):
        keypoints[camera, :, :, 0] = np.arange(14) + 20 * camera

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
        keypoints[2][:, COURT_KP20_HALF_TURN_INDEX[:14]],
    )
    np.testing.assert_array_equal(context.visibility, visibility)
    assert context.selection is not None
    assert context.document is not None
    assert context.provenance.reference_camera_id == "cam0"
    assert context.document["view_half_turns"] == [False, False, True]

    plcs_metadata = reference_metadata(context.selection, 2, "plcs")
    blcs_metadata = reference_metadata(context.selection, 1, "blcs")
    assert plcs_metadata.reference_view_index.tolist() == [0, 0]
    assert blcs_metadata.reference_view_index.tolist() == [0]


def test_camera_view_reference_rejects_invisible_calibration_point() -> None:
    visibility: NDArray[np.float32] = np.ones(
        (3, 2, 14), dtype=np.float32
    )
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
    keypoints = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]

    polygon = court_footpoint_polygon_px(
        keypoints,
        size=(1, 1),
        sideline_margin_m=1.0,
        baseline_margin_m=2.0,
    )

    np.testing.assert_allclose(
        polygon,
        [
            (-HALF_DOUBLES_WIDTH - 1.0, HALF_LENGTH + 2.0),
            (HALF_DOUBLES_WIDTH + 1.0, HALF_LENGTH + 2.0),
            (HALF_DOUBLES_WIDTH + 1.0, -HALF_LENGTH - 2.0),
            (-HALF_DOUBLES_WIDTH - 1.0, -HALF_LENGTH - 2.0),
        ],
        atol=1e-4,
    )
