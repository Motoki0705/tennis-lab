"""Tests for court keypoint predictor decoding contracts."""

from __future__ import annotations

import torch
from torch import nn

from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor


class _StaticLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(image.shape[0], -1, -1, -1)


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


def test_predict_returns_sigmoid_peak_scores_and_argmax_pixels() -> None:
    probs = torch.full((1, 5, 6), 0.05)
    probs[0, 2, 3] = 0.9
    logits = torch.logit(probs).unsqueeze(0)
    predictor = CourtKeypointPredictor(
        model=_StaticLogitModel(logits),
        device=torch.device("cpu"),
        subpixel_refine=False,
    )

    result = predictor.predict(torch.zeros(1, 3, 5, 6))

    assert result["keypoints"].shape == (1, 2)
    assert result["scores"].shape == (1,)
    torch.testing.assert_close(result["keypoints"], torch.tensor([[3.0, 2.0]]))
    torch.testing.assert_close(result["scores"], torch.tensor([0.9]))


def test_predict_subpixel_refine_can_be_disabled_per_call() -> None:
    true_center = torch.tensor([[2.35, 3.4]])
    probs = _gaussian_probability_heatmap(
        height=7,
        width=7,
        center_xy=(float(true_center[0, 0]), float(true_center[0, 1])),
        sigma=1.15,
    )
    logits = torch.logit(probs.clamp(1.0e-6, 0.999)).unsqueeze(0).unsqueeze(0)
    predictor = CourtKeypointPredictor(
        model=_StaticLogitModel(logits),
        device=torch.device("cpu"),
        subpixel_refine=True,
    )

    argmax = predictor.predict(
        torch.zeros(1, 3, 7, 7),
        subpixel_refine=False,
    )["keypoints"]
    refined = predictor.predict(torch.zeros(1, 3, 7, 7))["keypoints"]

    assert torch.linalg.vector_norm(refined - true_center) < torch.linalg.vector_norm(
        argmax - true_center
    )
    torch.testing.assert_close(refined, true_center, atol=0.05, rtol=0.0)
