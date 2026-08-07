"""Tests for strict CourtKP result persistence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.court_detection.model_io import CourtKeypointPrediction
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from tests.unit.tennis_scene.pipeline.config_factories import make_court_kp_config


class _TypedCourtPredictor:
    def predict(self, image: np.ndarray) -> CourtKeypointPrediction:
        assert image.shape == (8, 12, 3)
        return CourtKeypointPrediction(
            keypoints=torch.arange(28, dtype=torch.float32).reshape(14, 2),
            scores=torch.linspace(0.1, 0.9, 14),
            heatmaps=torch.zeros((14, 2, 3)),
        )


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

    keypoints, scores = module._predict_frame_pixels(
        np.zeros((8, 12, 3), dtype=np.uint8)
    )

    np.testing.assert_array_equal(
        keypoints,
        np.arange(28, dtype=np.float32).reshape(14, 2),
    )
    np.testing.assert_allclose(scores, np.linspace(0.1, 0.9, 14))
    assert keypoints.dtype == np.float32
    assert scores.dtype == np.float32
