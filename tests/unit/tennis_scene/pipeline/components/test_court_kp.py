"""Tests for strict CourtKP result persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from tests.unit.tennis_scene.pipeline.config_factories import make_court_kp_config


def test_pipeline_rejects_a_kp_cap_above_eight_before_model_loading(tmp_path) -> None:
    config = make_court_kp_config(tmp_path)
    with pytest.raises(ValueError, match="cap between 4 and 8"):
        replace(config, postprocess=replace(config.postprocess, max_kp=9))


class _TypedCourtPredictor:
    def predict(self, image: np.ndarray, *, postprocess: str, heads: tuple[str, ...]):
        from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
            geometry_prediction,
        )

        assert image.shape == (8, 12, 3)
        assert postprocess == "hybrid" and heads == ("kp", "line")
        return geometry_prediction()


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


def test_non_camera_view_checkpoint_schema_is_rejected(tmp_path) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    prediction = replace(geometry_prediction(), keypoint_schema="physical_kp14:gaussian_max_v1")
    module = CourtKPModule(make_court_kp_config(tmp_path))
    module._predictor = cast(Any, SimpleNamespace(predict=lambda *args, **kwargs: prediction))
    with pytest.raises(ValueError, match="camera-view KP14 checkpoint schema"):
        module._predict_frame_geometry(np.zeros((8, 12, 3), dtype=np.uint8))


def _run_frame_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction: Any, *, region_search: bool = False,
                    size: tuple[int, int] = (30, 20)) -> tuple[CourtKPResult, list[Any], CourtKPModule]:
    import src.tennis_scene.pipeline.components.court_kp as component
    from src.tasks.court_detection.inference.regions import (
        CourtRegionSearchConfig,
        CourtRegionSelection,
    )
    from src.tennis_scene.pipeline.components.court_kp import CourtDetectionInput
    from src.tennis_scene.pipeline.contracts import SourceVideo

    calls: list[Any] = []
    config = make_court_kp_config(tmp_path)
    if region_search:
        config = replace(config, region_search=CourtRegionSearchConfig(enabled=True))
    module = CourtKPModule(config)
    def predict(image: np.ndarray, **kwargs: Any) -> Any:
        calls.append(("predict", image.shape))
        return prediction

    def read(path: Path, index: int) -> SimpleNamespace:
        calls.append(("read", index))
        return SimpleNamespace(frame=np.ones((20, 30, 3), np.uint8), index=index, original_size=size)

    def select(*args: Any) -> CourtRegionSelection:
        calls.append(("region",))
        return CourtRegionSelection((5, 4, 17, 12), ())

    predictor = SimpleNamespace(predict=predict, checkpoint_identity={"checkpoint_sha256": "fixture"})
    monkeypatch.setattr(module, "load", lambda: setattr(module, "_predictor", predictor))
    monkeypatch.setattr(component, "read_video_frame", read)
    monkeypatch.setattr(component, "select_court_region", select)
    video = SourceVideo("cam0", tmp_path / "cam0.mp4", "media_hash", 1010, 59.94, 30, 20)
    return module.process(CourtDetectionInput(video)), calls, module


def test_component_infers_only_frame_zero_and_normalizes_by_last_pixel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    prediction = geometry_prediction()
    result, calls, module = _run_frame_zero(tmp_path, monkeypatch, prediction)
    assert calls == [("read", 0), ("predict", (20, 30, 3))]
    points, _ = prediction.downstream_keypoints()
    np.testing.assert_allclose(result.keypoints[0, 0], points / [29, 19], rtol=1e-6)
    assert result.keypoints.shape == (1, 1, 14, 2) and result.frame_indices.tolist() == [0]
    assert result.diagnostics is not None
    assert result.diagnostics["observed_frame_indices"] == [0]
    assert result.diagnostics["source_frame_count"] == 1010
    assert result.diagnostics["temporal_policy"] == "static_first_frame"
    assert result.diagnostics["checkpoint"] == {"checkpoint_sha256": "fixture"}
    assert result.diagnostics["cameras"][0]["frames"][0]["frame_index"] == 0
    assert not module.is_loaded


def test_failed_geometry_is_invisible_and_never_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.tasks.court_detection.geometry as geometry
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A second homography fit must not run")

    monkeypatch.setattr(geometry, "refine_court_keypoints_with_homography", forbidden)
    result, _, _ = _run_frame_zero(tmp_path, monkeypatch, geometry_prediction(status="joint_optimization_failed"))
    assert not result.visibility.any() and not result.keypoints.any()


def test_region_search_runs_on_frame_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    result, calls, _ = _run_frame_zero(tmp_path, monkeypatch, geometry_prediction(), region_search=True)
    assert calls[:2] == [("read", 0), ("region",)]
    assert calls[2] == ("predict", (8, 12, 3))  # the selected region crop
    assert result.diagnostics is not None
    assert result.diagnostics["cameras"][0]["region_selection"]["frame_index"] == 0


def test_frame_size_must_match_the_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )

    with pytest.raises(ValueError, match="disagrees with source"):
        _run_frame_zero(tmp_path, monkeypatch, geometry_prediction(), size=(32, 20))
