"""Tests for multi-peak Court keypoint predictor decoding."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference.predictor import (
    CourtKeypointPredictor,
    CourtPosePredictor,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelIOError,
    CourtModelOutput,
    CourtModelSpec,
)
from src.tasks.court_detection.model_io.images import prepare_court_pose_image
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.pose_head import CourtRawPoseOutput


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
        loss_config=_loss_config(),
    )
    adapter.validate_model_pair(model)
    return CourtKeypointPredictor(
        bind_model_io(model, adapter),
        torch.device("cpu"),
        subpixel_refine=subpixel_refine,
        max_peaks=max_peaks,
    )


def _loss_config(*, pose: bool = False) -> CourtLossConfig:
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
                "enabled": pose,
                "translation_weight": 1.0 if pose else 0.0,
                "rotation_weight": 1.0 if pose else 0.0,
                "focal_weight": 1.0 if pose else 0.0,
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


class _StaticPoseModel(CourtHierarchicalModel):
    def __init__(
        self,
        logits: torch.Tensor,
        raw_pose: torch.Tensor,
        bundle: CourtTargetBundleSpec,
    ) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle
        self.register_buffer("_logits", logits)
        self.register_buffer("_raw_pose", raw_pose)

    def forward(
        self,
        image: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
        patch_valid_mask: torch.Tensor | None = None,
    ) -> CourtModelOutput:
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
        raw_pose = cast(torch.Tensor, self._raw_pose)
        return CourtModelOutput(
            dense_logits=MappingProxyType(
                {"kp": logits.expand(image.shape[0], -1, -1, -1)}
            ),
            pose=CourtRawPoseOutput(
                raw_pose.expand(image.shape[0], -1)
            ),
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


def test_pose_predictor_returns_source_pixel_pose_and_dense_output() -> None:
    bundle = _bundle()
    probabilities = torch.full((1, 1, 4, 8), 0.001)
    probabilities[0, 0, 2, 4] = 0.9
    logits = torch.logit(probabilities)
    raw_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.log(128.0)]]
    )
    model = _StaticPoseModel(logits, raw_pose, bundle)
    adapter = CourtPoseModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=8),
        loss_config=_loss_config(pose=True),
    )
    predictor = CourtPosePredictor(
        cast(Any, bind_model_io(model, adapter)),
        torch.device("cpu"),
        patch_size=4,
        subpixel_refine=False,
        max_peaks=1,
    )

    result = predictor.predict(np.zeros((10, 20, 3), dtype=np.uint8))

    torch.testing.assert_close(result.pose.translation_m, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(result.pose.rotation, torch.eye(3).unsqueeze(0))
    torch.testing.assert_close(result.pose.focal_px, torch.tensor([320.0]))
    assert result.pose.translation_m.device.type == "cpu"
    keypoints = result.dense["kp"]
    assert isinstance(keypoints, CourtKeypointPrediction)
    torch.testing.assert_close(
        keypoints.keypoints[:, 0],
        torch.tensor([[4.0 / 7.0 * 19.0, 2.0 / 3.0 * 9.0]]),
    )


def test_pose_predictor_rejects_dense_only_adapter() -> None:
    bundle = _bundle()
    model = _StaticLogitModel(torch.zeros(1, 1, 4, 8), bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=8),
        loss_config=_loss_config(),
    )

    with pytest.raises(CourtModelIOError, match="pose-enabled checkpoint"):
        CourtPosePredictor(
            cast(Any, bind_model_io(model, adapter)),
            torch.device("cpu"),
            patch_size=4,
        )


def test_pose_preprocessing_uses_isotropic_long_side_and_patch_padding() -> None:
    image: NDArray[np.uint8] = np.arange(
        7 * 10 * 3,
        dtype=np.uint8,
    ).reshape(7, 10, 3)

    prepared = prepare_court_pose_image(
        image,
        long_side=8,
        patch_size=4,
        device=torch.device("cpu"),
    )

    assert prepared.original_size_hw == (7, 10)
    assert prepared.content_size_hw == (6, 8)
    assert prepared.model_size_hw == (8, 8)
    assert prepared.source_to_model_scale == pytest.approx(0.8)
    assert prepared.images.shape == (1, 3, 8, 8)
    torch.testing.assert_close(
        prepared.images[:, :, 6:],
        prepared.images[:, :, 5:6].expand(-1, -1, 2, -1),
    )


def test_keypoint_predictor_rejects_pose_adapter_with_different_geometry() -> None:
    bundle = _bundle()
    probabilities = torch.full((1, 1, 4, 8), 0.001)
    probabilities[0, 0, 2, 4] = 0.9
    raw_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.log(128.0)]]
    )
    model = _StaticPoseModel(torch.logit(probabilities), raw_pose, bundle)
    adapter = CourtPoseModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=8),
        loss_config=_loss_config(pose=True),
    )
    with pytest.raises(CourtModelIOError, match="pose-safe image geometry"):
        CourtKeypointPredictor(
            cast(Any, bind_model_io(model, adapter)),
            torch.device("cpu"),
            subpixel_refine=False,
            max_peaks=1,
        )


def test_keypoint_predictor_decodes_typed_dense_checkpoint_output() -> None:
    bundle = _bundle()
    probabilities = torch.full((1, 1, 4, 8), 0.001)
    probabilities[0, 0, 2, 4] = 0.9
    raw_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.log(128.0)]]
    )
    model = _StaticPoseModel(torch.logit(probabilities), raw_pose, bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(bundle, in_channels=3, short_side=8),
        loss_config=_loss_config(),
    )
    predictor = CourtKeypointPredictor(
        cast(Any, bind_model_io(model, adapter)),
        torch.device("cpu"),
        subpixel_refine=False,
        max_peaks=1,
    )

    result = predictor.predict(torch.zeros(1, 3, 4, 8))

    torch.testing.assert_close(result.keypoints[:, 0], torch.tensor([[4.0, 2.0]]))
