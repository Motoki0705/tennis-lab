"""Tests for strict CourtKP result persistence."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from tests.unit.tennis_scene.pipeline.config_factories import make_court_kp_config


class _TypedCourtPredictor:
    def predict(self, image: np.ndarray, *, postprocess: str, heads: tuple[str, ...]):
        from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
            geometry_prediction,
        )

        assert image.shape == (8, 12, 3)
        assert postprocess == "hybrid" and heads == ("kp", "line")
        return geometry_prediction()


def test_result_rejects_missing_visibility_and_frame_indices() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        CourtKPResult.from_dict({"keypoints": [[[[0.0, 0.0]]]]})


def test_result_rejects_non_object_diagnostics() -> None:
    with pytest.raises(TypeError, match="diagnostics must be an object"):
        CourtKPResult.from_dict(
            {
                "keypoints": [[[[0.0, 0.0]]]],
                "visibility": [[[1.0]]],
                "frame_indices": [0],
                "diagnostics": [],
            }
        )


def test_predict_frame_consumes_typed_task_prediction(tmp_path) -> None:
    module = CourtKPModule(make_court_kp_config(tmp_path))
    module._predictor = _TypedCourtPredictor()  # type: ignore[assignment]

    keypoints, valid, diagnostic = module._predict_frame_geometry(
        np.zeros((8, 12, 3), dtype=np.uint8)
    )

    assert keypoints.shape == (14, 2)
    np.testing.assert_array_equal(keypoints[:2], [[-2, 3], [15, 3]])
    assert keypoints.dtype == np.float32
    assert valid.sum() == 12
    assert diagnostic["selected_keypoints"] == [3, 4, 8, 9]


def test_result_allows_only_invisible_out_of_image_geometry() -> None:
    points: np.ndarray = np.full((1, 1, 14, 2), 0.5, dtype=np.float32)
    points[0, 0, 0] = [-0.2, 1.1]
    valid: np.ndarray = np.ones((1, 1, 14), dtype=np.float32)
    valid[0, 0, 0] = 0
    result = CourtKPResult(points, valid, np.array([0], dtype=np.int32))
    assert result.validate()[0]
    result.visibility[0, 0, 0] = 1
    assert not result.validate()[0]


def test_camera_view_checkpoint_preserves_local_order_for_reference_alignment(
    tmp_path,
) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    prediction = replace(
        geometry_prediction(),
        keypoint_schema="synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
    )
    module = CourtKPModule(make_court_kp_config(tmp_path))
    module._predictor = cast(
        Any, SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    )
    actual, visible, diagnostic = module._predict_frame_geometry(
        np.zeros((8, 12, 3), dtype=np.uint8)
    )
    points, validity = prediction.downstream_keypoints()
    np.testing.assert_array_equal(actual, points)
    np.testing.assert_array_equal(visible, validity)
    assert diagnostic["output_keypoint_contract"] == "camera_view_v2"


def test_camera_view_checkpoint_never_silently_uses_physical_identity(tmp_path) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    prediction = replace(
        geometry_prediction(),
        keypoint_schema="synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
    )
    module = CourtKPModule(
        replace(make_court_kp_config(tmp_path), output_keypoint_contract="physical_v1")
    )
    module._predictor = cast(
        Any, SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    )
    with pytest.raises(ValueError, match="camera_view_v2 downstream"):
        module._predict_frame_geometry(np.zeros((8, 12, 3), dtype=np.uint8))


@pytest.mark.parametrize("saved_contract", [None, "physical_v1", "camera_view_v2"])
def test_explicit_saved_artifacts_bypass_model_but_validate_recorded_contract(
    tmp_path, saved_contract: str | None
) -> None:
    artifact = tmp_path / "saved.json"
    saved = CourtKPResult(
        np.full((1, 2, 14, 2), 0.5, dtype=np.float32),
        np.ones((1, 2, 14), dtype=np.float32),
        np.arange(2, dtype=np.int32),
        diagnostics=None
        if saved_contract is None
        else {"output_keypoint_contract": saved_contract},
    )
    saved.save(artifact)
    config = replace(
        make_court_kp_config(tmp_path),
        source="load",
        load_path=artifact,
        output_keypoint_contract="physical_v1",
    )
    module = CourtKPModule(config)
    module.load()
    assert module._predictor is None
    if saved_contract == "camera_view_v2":
        with pytest.raises(ValueError, match="contract does not match"):
            module.process([tmp_path / "video.mp4"])
    else:
        actual = module.process([tmp_path / "video.mp4"])
        np.testing.assert_array_equal(actual.keypoints, saved.keypoints)
        np.testing.assert_array_equal(actual.visibility, saved.visibility)


def test_model_sequence_keeps_geometry_and_masks_failure_without_refitting(
    tmp_path, monkeypatch
) -> None:
    import src.tasks.court_detection.geometry as geometry
    import src.tennis_scene.pipeline.components.court_kp as component
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    predictions = iter(
        [geometry_prediction(), geometry_prediction(status="joint_optimization_failed")]
    )
    predictor = SimpleNamespace(
        predict=lambda *args, **kwargs: next(predictions),
        checkpoint_identity={"checkpoint_sha256": "fixture"},
    )
    module = CourtKPModule(make_court_kp_config(tmp_path))
    module._predictor = cast(Any, predictor)
    packets = [
        SimpleNamespace(
            frame=np.zeros((8, 12, 3), dtype=np.uint8), index=i, original_size=(12, 8)
        )
        for i in range(2)
    ]
    monkeypatch.setattr(
        component, "OpenCVVideoFrameReader", lambda *args, **kwargs: packets
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A second homography fit must not run")

    monkeypatch.setattr(geometry, "refine_court_keypoints_with_homography", forbidden)
    result = module._process_model_video([tmp_path / "video.mp4"], max_frames=None)
    assert result.validate()[0]
    assert result.keypoints[0, 0, 0, 0] < 0
    assert result.visibility[0, 0].sum() == 12
    assert not result.visibility[0, 1].any()
    assert not result.keypoints[0, 1].any()
    path = tmp_path / "result.json"
    result.save(path)
    loaded = CourtKPResult.load(path)
    np.testing.assert_array_equal(loaded.keypoints, result.keypoints)
    assert loaded.diagnostics == result.diagnostics


@pytest.mark.parametrize(
    "saved_contract,declared,accepted",
    [
        (None, None, False),
        (None, "camera_view_v2", True),
        ("camera_view_v2", None, True),
        ("physical_v1", "camera_view_v2", False),
    ],
)
def test_camera_view_saved_inputs_require_recorded_or_declared_contract(
    tmp_path, saved_contract: str | None, declared: str | None, accepted: bool
) -> None:
    artifact = tmp_path / "saved.json"
    saved = CourtKPResult(
        np.full((1, 1, 14, 2), 0.5, dtype=np.float32),
        np.ones((1, 1, 14), dtype=np.float32),
        np.array([0], dtype=np.int32),
        diagnostics=None
        if saved_contract is None
        else {"output_keypoint_contract": saved_contract},
    )
    saved.save(artifact)
    original = artifact.read_bytes()
    config = replace(
        make_court_kp_config(tmp_path),
        source="load",
        load_path=artifact,
        load_keypoint_contract=declared,
    )
    module = CourtKPModule(config)
    if accepted:
        result = module.process([tmp_path / "video.mp4"])
        np.testing.assert_array_equal(result.keypoints, saved.keypoints)
        assert result.diagnostics is not None
        assert result.diagnostics["output_keypoint_contract"] == "camera_view_v2"
    else:
        with pytest.raises(ValueError, match="contract"):
            module.process([tmp_path / "video.mp4"])
    assert artifact.read_bytes() == original


def test_legacy_load_declaration_cannot_override_the_runtime_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="must match"):
        replace(
            make_court_kp_config(tmp_path),
            source="load",
            load_path=tmp_path / "saved.json",
            load_keypoint_contract="physical_v1",
        )
