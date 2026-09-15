"""Tests for multi-peak Court keypoint predictor decoding."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
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
        patch_valid_mask: torch.Tensor | None = None,
    ) -> dict[CourtTargetKind, torch.Tensor]:
        assert all(
            value is None
            for value in (
                feature_1,
                feature_2,
                feature_3,
                feature_4,
                patch_valid_mask,
            )
        )
        logits = cast(torch.Tensor, self._logits)
        return {"kp": logits.expand(image.shape[0], -1, -1, -1)}


def _predictor(
    logits: torch.Tensor,
    *,
    subpixel_refine: bool,
    max_peaks: int | None = None,
) -> CourtKeypointPredictor:
    bundle = _bundle()
    model = _StaticLogitModel(logits, bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(
            target_bundle=bundle,
            in_channels=3,
            short_side=32,
        ),
        loss_config=_loss_config(),
    )
    adapter.validate_model_pair(model)
    bound = bind_model_io(model, adapter)
    if max_peaks is None:
        return CourtKeypointPredictor(
            bound,
            torch.device("cpu"),
            subpixel_refine=subpixel_refine,
        )
    return CourtKeypointPredictor(
        bound,
        torch.device("cpu"),
        subpixel_refine=subpixel_refine,
        max_peaks=max_peaks,
    )


def _loss_config() -> CourtLossConfig:
    return CourtLossConfig.from_mapping(
        {
            "seg": {"ce_weight": 1.0, "dice_weight": 1.0, "weight": 1.0},
            "kp": {"focal_gamma": 2.0, "weight": 1.0},
            "line": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "pos_weight": 1.0,
                "weight": 1.0,
            },
            "pose": {
                "enabled": False,
                "translation_weight": 0.0,
                "rotation_weight": 0.0,
                "focal_weight": 0.0,
            },
            "consistency": {
                "enabled": False,
                "weight": 0.0,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": 0.0,
                "gradient_flow": "both",
            },
        }
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


def _two_peak_probabilities(
    *,
    primary_xy: tuple[float, float] = (4.0, 4.0),
    secondary_xy: tuple[float, float] = (12.0, 12.0),
    primary: float = 0.9,
    secondary: float = 0.6,
    sigma: float = 0.9,
) -> torch.Tensor:
    """Return one channel holding a strong main peak and a weak distant peak."""
    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=torch.float32),
        torch.arange(16, dtype=torch.float32),
        indexing="ij",
    )

    def _gaussian(center_xy: tuple[float, float], amplitude: float) -> torch.Tensor:
        center_x, center_y = center_xy
        distance_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
        return amplitude * torch.exp(-distance_squared / (2.0 * sigma * sigma))

    return torch.maximum(
        _gaussian(primary_xy, primary),
        _gaussian(secondary_xy, secondary),
    )


def _two_peak_logits() -> torch.Tensor:
    return torch.logit(_two_peak_probabilities())[None, None]


def _single_pixel_probability(value: float) -> torch.Tensor:
    probabilities = torch.full((1, 1, 16, 16), 1.0e-6)
    probabilities[0, 0, 3, 3] = value
    return probabilities


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


def test_default_decode_returns_only_the_main_peak() -> None:
    predictor = _predictor(_two_peak_logits(), subpixel_refine=False)

    assert predictor.max_peaks == 1
    assert predictor.peak_threshold == pytest.approx(0.05)
    assert predictor.nms_kernel == 7
    result = predictor.predict(torch.zeros(1, 3, 16, 16))

    assert result.keypoints.shape == (1, 1, 2)
    assert result.scores.shape == (1, 1)
    assert result.valid.tolist() == [[True]]
    torch.testing.assert_close(result.keypoints[:, 0], torch.tensor([[4.0, 4.0]]))
    assert float(result.scores[0, 0].item()) == pytest.approx(0.9, abs=1e-3)


def test_explicit_multi_peak_opt_in_keeps_both_peaks_in_score_order() -> None:
    predictor = _predictor(
        _two_peak_logits(),
        subpixel_refine=False,
        max_peaks=2,
    )

    result = predictor.predict(torch.zeros(1, 3, 16, 16))

    assert result.keypoints.shape == (1, 2, 2)
    assert result.valid.tolist() == [[True, True]]
    torch.testing.assert_close(result.keypoints[0, 0], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(result.keypoints[0, 1], torch.tensor([12.0, 12.0]))
    assert float(result.scores[0, 0].item()) > float(result.scores[0, 1].item())


def test_decode_settings_are_read_only_views_of_the_decoder_config() -> None:
    predictor = _predictor(
        _two_peak_logits(),
        subpixel_refine=False,
        max_peaks=2,
    )
    config = predictor.decoder_config

    assert predictor.max_peaks == config.max_peaks == 2
    assert predictor.peak_threshold == config.threshold
    assert predictor.nms_kernel == config.nms_kernel
    attribute = "max_peaks"
    with pytest.raises(AttributeError):
        setattr(predictor, attribute, 3)
    attribute = "decoder_config"
    with pytest.raises(AttributeError):
        setattr(predictor, attribute, config)


def test_channel_below_the_default_threshold_is_invalid_with_zero_score() -> None:
    probabilities = torch.full((1, 1, 16, 16), 0.049)
    predictor = _predictor(torch.logit(probabilities), subpixel_refine=False)

    result = predictor.predict(torch.zeros(1, 3, 16, 16))

    assert result.valid.tolist() == [[False]]
    assert float(result.scores[0, 0].item()) == 0.0


def test_threshold_boundary_needs_local_contrast_to_become_a_peak() -> None:
    at_threshold = _predictor(
        torch.logit(torch.full((1, 1, 16, 16), 0.05)),
        subpixel_refine=False,
    ).predict(torch.zeros(1, 3, 16, 16))
    above_threshold = _predictor(
        torch.logit(_single_pixel_probability(0.051)),
        subpixel_refine=False,
    ).predict(torch.zeros(1, 3, 16, 16))

    assert at_threshold.valid.tolist() == [[False]]
    assert float(at_threshold.scores[0, 0].item()) == 0.0
    assert above_threshold.valid.tolist() == [[True]]
    assert float(above_threshold.scores[0, 0].item()) == pytest.approx(0.051, abs=1e-6)
