"""Single forward, explicit geometry, raw-head compatibility and masked output."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.geometry.confidence_homography import (
    ConfidenceHomographyResult,
)
from src.tasks.court_detection.geometry.hybrid_homography import (
    DEFAULT_HYBRID_CONFIG,
    HybridHomographyConfig,
    HybridHomographyResult,
)
from src.tasks.court_detection.inference.contracts import CourtPrediction
from src.tasks.court_detection.inference.predictor import (
    CourtKeypointPredictor,
    CourtPredictor,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelIOError,
    CourtModelSpec,
)
from src.tasks.court_detection.models.pose_head import CourtModelOutput
from src.utils.schema.court import GROUND_COURT_KP_NAMES
from tests.unit.tasks.court_detection.inference.test_pose_output_predictors import (
    _bundle,
    _loss_config,
    _StaticPoseModel,
)


def _predictor(
    hybrid_config: HybridHomographyConfig = DEFAULT_HYBRID_CONFIG,
) -> CourtPredictor:
    bundle = _bundle()
    model = _StaticPoseModel(bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(bundle, 3, 32), loss_config=_loss_config()
    )
    return CourtPredictor(
        bind_model_io(model, adapter), torch.device("cpu"), hybrid_config=hybrid_config
    )


@pytest.mark.parametrize("cap", [9, 14, 100])
def test_downstream_kp_cap_is_enforced_before_loading_or_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    import src.tasks.court_detection.inference.predictor as module

    config = replace(DEFAULT_HYBRID_CONFIG, max_kp=cap)
    load = Mock(side_effect=AssertionError("Invalid settings must fail before loading"))
    monkeypatch.setattr(module, "load_court_checkpoint", load)
    with pytest.raises(ValueError, match="cap between 4 and 8"):
        CourtPredictor.load_from_checkpoint(
            tmp_path / "model.ckpt", device="cpu", hybrid_config=config
        )
    load.assert_not_called()
    with pytest.raises(ValueError, match="cap between 4 and 8"):
        _predictor(config)


def geometry_prediction(*, status: str = "ok") -> CourtPrediction:
    points = np.tile([4.0, 3.0], (14, 1))
    points[0] = [-2, 3]
    points[1] = [15, 3]
    selected: np.ndarray = np.zeros(14, dtype=bool)
    selected[[3, 4, 8, 9]] = True
    baseline = ConfidenceHomographyResult(
        np.eye(3), points, np.arange(14), selected, selected, np.zeros(14), "ok"
    )
    geometry = HybridHomographyResult(
        np.eye(3) if status == "ok" else None,
        points if status == "ok" else np.full((14, 2), np.nan),
        selected if status == "ok" else np.zeros(14, dtype=bool),
        np.zeros(14) if status == "ok" else np.full(14, np.nan),
        status,
        baseline,
        None,
        None,
        3,
        1,
        None,
    )
    raw = CourtKeypointPrediction(
        torch.full((14, 1, 2), 6.0),
        torch.full((14, 1), 0.8),
        torch.ones((14, 1), dtype=torch.bool),
        torch.zeros((14, 4, 6)),
    )
    return CourtPrediction(
        {"kp": raw},
        (8, 12),
        (4, 6),
        geometry,
        "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
    )


def test_raw_multihead_request_runs_one_forward_and_preserves_all_heads() -> None:
    predictor = _predictor()
    forward = Mock(wraps=predictor.model.forward)
    predictor.model.forward = forward
    result = predictor.predict(torch.zeros(1, 3, 5, 6), postprocess="none")
    assert forward.call_count == 1
    assert set(result.raw_heads) == {"kp", "seg", "line"}
    assert result.homography is None
    assert result.original_size_hw == result.native_size_hw == (5, 6)
    raw = result.raw_heads["kp"]
    assert isinstance(raw, CourtKeypointPrediction)
    assert raw.heatmaps.shape == (1, 5, 6)
    with pytest.raises(ValueError, match="requires explicit"):
        result.downstream_keypoints()


@pytest.mark.parametrize(
    "options",
    [
        {"heads": ("semantic_line",), "postprocess": "none"},
        {"heads": ("line",)},
        {"postprocess": "invalid"},
    ],
)
def test_invalid_head_or_geometry_requests_fail_before_forward(
    options: dict[str, Any],
) -> None:
    predictor = _predictor()
    forward = Mock(wraps=predictor.model.forward)
    predictor.model.forward = forward
    with pytest.raises(CourtModelIOError):
        predictor.predict(torch.zeros(1, 3, 5, 6), **options)
    forward.assert_not_called()


def test_noncanonical_kp_cannot_silently_enter_hybrid() -> None:
    with pytest.raises(CourtModelIOError, match="ordered KP14"):
        _predictor().predict(torch.zeros(1, 3, 5, 6))


@pytest.mark.parametrize("pose_long_side", [False, True])
def test_hybrid_receives_one_forward_original_pixel_kp_and_native_line(
    monkeypatch: pytest.MonkeyPatch,
    pose_long_side: bool,
) -> None:
    import src.tasks.court_detection.inference.predictor as module

    targets = dict(_bundle().targets)
    targets["kp"] = CourtTargetSpec(
        "kp",
        "test_ordered_kp14",
        14,
        GROUND_COURT_KP_NAMES,
        torch.float32,
        False,
    )
    bundle = CourtTargetBundleSpec(targets)

    class OrderedModel(_StaticPoseModel):
        calls = 0

        def forward(
            self, image: torch.Tensor, *args: Any, **kwargs: Any
        ) -> CourtModelOutput:
            self.calls += 1
            output = super().forward(image, *args, **kwargs)
            logits = dict(output.dense_logits)
            logits["kp"] = image.new_full((1, 14, *image.shape[-2:]), -8)
            logits["kp"][:, :, 10, 10] = 8
            if pose_long_side:
                logits["kp"][:, :, -1, -1] = (
                    12  # Padding must not consume the only peak.
                )
            return CourtModelOutput(logits, output.pose)

    model = OrderedModel(bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(bundle, 3, 32, pose_long_side=pose_long_side),
        loss_config=_loss_config(),
    )
    predictor = CourtPredictor(
        bind_model_io(model, adapter), torch.device("cpu"), subpixel_refine=False
    )
    captured: dict[str, Any] = {}

    def fit(
        template: np.ndarray,
        points: np.ndarray,
        scores: np.ndarray,
        line: np.ndarray,
        **kwargs: Any,
    ) -> HybridHomographyResult:
        captured.update(
            template=template, points=points, scores=scores, line=line, **kwargs
        )
        result = geometry_prediction().homography
        assert result is not None
        return result

    monkeypatch.setattr(module, "estimate_hybrid_homography", fit)
    result = predictor.predict(Image.new("RGB", (31, 17)))
    assert model.calls == 1
    assert captured["image_size_hw"] == (17, 31)
    native_h, native_w = captured["line"].shape
    np.testing.assert_allclose(
        captured["points"],
        np.tile(
            [10 * 31 / 32, 10 * 31 / 32]
            if pose_long_side
            else [10 / (native_w - 1) * 30, 10 / (native_h - 1) * 16],
            (14, 1),
        ),
        atol=1e-5,
    )
    if pose_long_side:
        assert result.native_size_hw == (18, 32)
        assert captured["line"].shape == (18, 32)
    raw = CourtKeypointPredictor(
        bind_model_io(model, adapter), torch.device("cpu"), subpixel_refine=False
    ).predict(Image.new("RGB", (31, 17)))
    expected = result.raw_heads["kp"]
    assert isinstance(expected, CourtKeypointPrediction)
    torch.testing.assert_close(raw.keypoints, expected.keypoints)
    assert bool(raw.valid.all())
    assert captured["edges"].shape == (9, 2)
    assert captured["config"].max_kp == 8
    assert (
        np.max(np.abs(captured["template"])) < 12
    )  # metres, not the paper's raster pixels
    assert np.all(captured["line"] == 0.5)
    raw = result.raw_heads["kp"]
    assert isinstance(raw, CourtKeypointPrediction)
    np.testing.assert_array_equal(captured["points"], raw.keypoints[:, 0])


def test_fitted_visibility_is_not_the_selected_observation_mask() -> None:
    prediction = geometry_prediction()
    points, valid = prediction.downstream_keypoints()
    assert prediction.homography is not None
    assert prediction.homography.selected.sum() == 4
    assert valid.sum() == 12
    assert not valid[:2].any()
    np.testing.assert_array_equal(points[:2], [[-2, 3], [15, 3]])
    raw = prediction.raw_heads["kp"]
    assert isinstance(raw, CourtKeypointPrediction)
    assert torch.all(raw.keypoints == 6)


def test_failed_geometry_is_zero_masked_and_json_safe_without_fallback() -> None:
    prediction = geometry_prediction(status="insufficient_line_support")
    points, valid = prediction.downstream_keypoints()
    assert np.isfinite(points).all() and not points.any() and not valid.any()
    diagnostic = prediction.geometry_diagnostics()
    assert diagnostic["status"] == "insufficient_line_support"
    assert diagnostic["homography_court_metres_to_image_pixels"] is None
    assert diagnostic["selected_keypoints"] == []
    json.dumps(diagnostic, allow_nan=False)
    with pytest.raises(ValueError, match="requires explicit"):
        replace(prediction, homography=None).downstream_keypoints()
