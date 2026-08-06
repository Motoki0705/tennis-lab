"""Tests for court keypoint predictor decoding contracts."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.model_io.adapters import CourtKeypointModelIO
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


class _StaticLogitModel(CourtHierarchicalModel):
    def __init__(self, logits: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.num_classes = logits.shape[1]
        self.register_buffer("_logits", logits)

    def forward(
        self,
        image: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert feature_1 is None
        assert feature_2 is None
        assert feature_3 is None
        assert feature_4 is None
        logits = cast(torch.Tensor, self._logits)
        return logits.expand(image.shape[0], -1, -1, -1)


def _gaussian_probability_heatmap(
    *,
    height: int,
    width: int,
    center_xy: tuple[float, float],
    sigma: float,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    center_x, center_y = center_xy
    dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return 0.95 * torch.exp(-dist2 / (2.0 * sigma * sigma))


def _predictor(
    logits: torch.Tensor, *, subpixel_refine: bool
) -> CourtKeypointPredictor:
    model = _StaticLogitModel(logits)
    adapter = CourtKeypointModelIO(
        CourtModelSpec(
            task="kp",
            in_channels=3,
            output_channels=logits.shape[1],
            short_side=32,
        ),
        focal_gamma=2.0,
    )
    adapter.validate_model_pair(model)
    return CourtKeypointPredictor(
        bind_model_io(model, adapter),
        torch.device("cpu"),
        subpixel_refine=subpixel_refine,
    )


def test_predict_returns_sigmoid_peak_scores_and_argmax_pixels() -> None:
    probs = torch.full((1, 5, 6), 0.05)
    probs[0, 2, 3] = 0.9
    logits = torch.logit(probs).unsqueeze(0)
    predictor = _predictor(logits, subpixel_refine=False)

    result = predictor.predict(torch.zeros(1, 3, 5, 6))

    assert result.keypoints.shape == (1, 2)
    assert result.scores.shape == (1,)
    torch.testing.assert_close(result.keypoints, torch.tensor([[3.0, 2.0]]))
    torch.testing.assert_close(result.scores, torch.tensor([0.9]))
    torch.testing.assert_close(result.heatmaps, logits[0])


def test_predict_uses_once_selected_subpixel_refinement() -> None:
    true_center = torch.tensor([[2.35, 3.4]])
    probs = _gaussian_probability_heatmap(
        height=7,
        width=7,
        center_xy=(float(true_center[0, 0]), float(true_center[0, 1])),
        sigma=1.15,
    )
    logits = torch.logit(probs.clamp(1.0e-6, 0.999)).unsqueeze(0).unsqueeze(0)
    argmax = (
        _predictor(logits, subpixel_refine=False)
        .predict(torch.zeros(1, 3, 7, 7))
        .keypoints
    )
    refined = (
        _predictor(logits, subpixel_refine=True)
        .predict(torch.zeros(1, 3, 7, 7))
        .keypoints
    )

    assert torch.linalg.vector_norm(refined - true_center) < torch.linalg.vector_norm(
        argmax - true_center
    )
    torch.testing.assert_close(refined, true_center, atol=0.05, rtol=0.0)
