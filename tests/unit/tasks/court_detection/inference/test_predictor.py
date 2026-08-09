"""Tests for multi-peak Court keypoint predictor decoding."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import (
    CourtModelSpec,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=1,
                channel_names=("symmetric_pair",),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


class _StaticLogitModel(CourtHierarchicalModel):
    def __init__(
        self,
        logits: torch.Tensor,
        bundle: CourtTargetBundleSpec,
    ) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle
        self.register_buffer("_logits", logits)

    def forward(
        self,
        image: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert all(
            value is None
            for value in (feature_1, feature_2, feature_3, feature_4)
        )
        logits = cast(torch.Tensor, self._logits)
        return {"kp": logits.expand(image.shape[0], -1, -1, -1)}


def _predictor(
    logits: torch.Tensor,
    *,
    subpixel_refine: bool,
    max_peaks: int = 1,
) -> CourtKeypointPredictor:
    bundle = _bundle()
    model = _StaticLogitModel(logits, bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(
            target_bundle=bundle,
            in_channels=3,
            short_side=32,
        ),
        loss_config=CourtLossConfig(
            seg_ce_weight=1.0,
            seg_dice_weight=1.0,
            kp_focal_gamma=2.0,
            line_bce_weight=1.0,
            line_dice_weight=1.0,
            line_pos_weight=1.0,
        ),
    )
    adapter.validate_model_pair(model)
    return CourtKeypointPredictor(
        bind_model_io(model, adapter),
        torch.device("cpu"),
        subpixel_refine=subpixel_refine,
        max_peaks=max_peaks,
    )


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
    distance_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return 0.95 * torch.exp(-distance_squared / (2.0 * sigma * sigma))


def test_predict_returns_peak_axis_and_scores() -> None:
    probabilities = torch.full((1, 5, 6), 0.001)
    probabilities[0, 2, 3] = 0.9
    logits = torch.logit(probabilities).unsqueeze(0)
    predictor = _predictor(logits, subpixel_refine=False)

    result = predictor.predict(torch.zeros(1, 3, 5, 6))

    assert result.keypoints.shape == (1, 1, 2)
    assert result.scores.shape == (1, 1)
    assert result.valid.tolist() == [[True]]
    torch.testing.assert_close(
        result.keypoints[:, 0],
        torch.tensor([[3.0, 2.0]]),
    )
    torch.testing.assert_close(result.scores[:, 0], torch.tensor([0.9]))


def test_predict_uses_selected_subpixel_refinement() -> None:
    true_center = torch.tensor([[2.35, 3.4]])
    probabilities = _gaussian_probability_heatmap(
        height=7,
        width=7,
        center_xy=(float(true_center[0, 0]), float(true_center[0, 1])),
        sigma=1.15,
    )
    logits = torch.logit(probabilities.clamp(1.0e-6, 0.999)).unsqueeze(0).unsqueeze(0)
    argmax = (
        _predictor(logits, subpixel_refine=False)
        .predict(torch.zeros(1, 3, 7, 7))
        .keypoints[:, 0]
    )
    refined = (
        _predictor(logits, subpixel_refine=True)
        .predict(torch.zeros(1, 3, 7, 7))
        .keypoints[:, 0]
    )

    assert torch.linalg.vector_norm(refined - true_center) < (
        torch.linalg.vector_norm(argmax - true_center)
    )
    torch.testing.assert_close(
        refined,
        true_center,
        atol=0.05,
        rtol=0.0,
    )
