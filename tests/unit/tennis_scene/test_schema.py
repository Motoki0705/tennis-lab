"""Metre-valued SceneResult and normalization-provenance contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.base.data import (
    CourtCoordinateContractMismatchError,
    MissingCourtCoordinateMetadataError,
)
from src.tennis_scene.schema import (
    SCENE_RESULT_POSITION_UNIT,
    SceneResult,
    attach_scene_result_court_coordinate_provenance,
    validate_scene_result_court_coordinate_provenance,
)
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


def _scene() -> SceneResult:
    return SceneResult(
        num_frames=1,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.zeros((1, 1, 14, 2), dtype=np.float32),
        court_vis=np.ones((1, 1, 14), dtype=np.float32),
        player_position=np.array([[[2.0, -4.0, 1.0]]], dtype=np.float32),
        player_yaw=np.zeros((1, 1), dtype=np.float32),
        smpl_body_pose=np.zeros((1, 1, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 1, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        ball_3d=np.array([[1.0, 3.0, 2.0]], dtype=np.float32),
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_attaching_provenance_never_changes_scene_result_meter_arrays(
    version: str,
) -> None:
    scene = _scene()
    original_player = scene.player_position.copy()
    assert scene.ball_3d is not None
    original_ball = scene.ball_3d.copy()
    contract = resolve_court_coordinate_normalization(version)

    attach_scene_result_court_coordinate_provenance(scene, contract)

    assert SCENE_RESULT_POSITION_UNIT == "m"
    np.testing.assert_array_equal(scene.player_position, original_player)
    np.testing.assert_array_equal(scene.ball_3d, original_ball)
    metadata = validate_scene_result_court_coordinate_provenance(scene, contract)
    assert metadata is not None
    assert metadata.version == version


def test_metadata_free_scene_result_is_legacy_v1_only() -> None:
    scene = _scene()
    assert (
        validate_scene_result_court_coordinate_provenance(
            scene,
            resolve_court_coordinate_normalization("v1"),
        )
        is None
    )
    with pytest.raises(MissingCourtCoordinateMetadataError, match="legacy v1 only"):
        validate_scene_result_court_coordinate_provenance(
            scene,
            resolve_court_coordinate_normalization("v2"),
        )


def test_scene_result_provenance_runtime_mismatch_is_rejected() -> None:
    scene = _scene()
    attach_scene_result_court_coordinate_provenance(
        scene,
        resolve_court_coordinate_normalization("v1"),
    )

    with pytest.raises(CourtCoordinateContractMismatchError, match="does not match runtime"):
        validate_scene_result_court_coordinate_provenance(
            scene,
            resolve_court_coordinate_normalization("v2"),
        )
