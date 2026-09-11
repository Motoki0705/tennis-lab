"""Regression tests for predictors backed by pose-enabled Court models."""

from __future__ import annotations

import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.configuration import CourtLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference.mask_predictor import (
    CourtLinePredictor,
    CourtSegPredictor,
)
from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.pose_head import (
    CourtModelOutput,
    CourtRawPoseOutput,
)


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=1,
                channel_names=("center",),
                target_dtype=torch.float32,
                precomputed=False,
            ),
            "seg": CourtTargetSpec(
                kind="seg",
                schema="test_seg",
                output_channels=2,
                channel_names=("background", "court"),
                target_dtype=torch.int64,
                precomputed=True,
            ),
            "line": CourtTargetSpec(
                kind="line",
                schema="test_line",
                output_channels=1,
                channel_names=("line",),
                target_dtype=torch.float32,
                precomputed=True,
            ),
        }
    )


class _StaticPoseModel(CourtHierarchicalModel):
    def __init__(self, bundle: CourtTargetBundleSpec) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle

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
        batch_size, _, height, width = image.shape
        return CourtModelOutput(
            dense_logits={
                "kp": image.new_zeros((batch_size, 1, height, width)),
                "seg": image.new_zeros((batch_size, 2, height, width)),
                "line": image.new_zeros((batch_size, 1, height, width)),
            },
            pose=CourtRawPoseOutput(image.new_zeros((batch_size, 10))),
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
                "enabled": True,
                "translation_weight": 1.0,
                "rotation_weight": 1.0,
                "focal_weight": 1.0,
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


def test_all_predictors_accept_typed_pose_model_output() -> None:
    bundle = _bundle()
    model = _StaticPoseModel(bundle)
    adapter = CourtModelIOAdapter(
        CourtModelSpec(target_bundle=bundle, in_channels=3, short_side=32),
        loss_config=_loss_config(),
    )
    adapter.validate_model_pair(model)
    bound = bind_model_io(model, adapter)
    device = torch.device("cpu")
    image = torch.zeros(1, 3, 5, 6)

    keypoints = CourtKeypointPredictor(
        bound,
        device,
        subpixel_refine=False,
        max_peaks=1,
    ).predict(image)
    segmentation = CourtSegPredictor(bound, device).predict(image)
    lines = CourtLinePredictor(bound, device).predict(image)

    assert keypoints.heatmaps.shape == (1, 5, 6)
    assert segmentation.mask.shape == (5, 6)
    assert lines.probability.shape == (5, 6)
