"""CourtKP integration tests for pose-enabled Court checkpoints."""

from __future__ import annotations

import numpy as np
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
from src.tasks.court_detection.model_io.contracts import CourtModelSpec
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.pose_head import (
    CourtModelOutput,
    CourtRawPoseOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from tests.unit.tennis_scene.pipeline.config_factories import make_court_kp_config


class _StaticPoseEnabledCourtModel(CourtHierarchicalModel):
    def __init__(
        self,
        bundle: CourtTargetBundleSpec,
        kp_logits: torch.Tensor,
    ) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle
        self.register_buffer("_kp_logits", kp_logits)

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
        return CourtModelOutput(
            dense_logits={
                "kp": self._kp_logits.expand(image.shape[0], -1, -1, -1)
            },
            pose=CourtRawPoseOutput(image.new_zeros((image.shape[0], 10))),
        )


def _pose_enabled_predictor() -> CourtKeypointPredictor:
    bundle = CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="ordered_kp14_test",
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )
    probabilities = torch.full((1, 14, 8, 12), 0.001)
    probabilities[:, :, 3, 4] = 0.99
    probabilities[:, :, 6, 10] = 0.08
    model = _StaticPoseEnabledCourtModel(bundle, torch.logit(probabilities))
    adapter = CourtModelIOAdapter(
        CourtModelSpec(target_bundle=bundle, in_channels=3, short_side=8),
        loss_config=CourtLossConfig.from_mapping(
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
        ),
    )
    return CourtKeypointPredictor(
        bind_model_io(model, adapter),
        torch.device("cpu"),
        subpixel_refine=False,
    )


def test_tennis_scene_consumes_only_kp14_from_pose_enabled_model(tmp_path) -> None:
    predictor = _pose_enabled_predictor()
    module = CourtKPModule(make_court_kp_config(tmp_path))
    module._predictor = predictor

    keypoints, scores, valid = module._predict_frame_pixels(
        np.zeros((8, 12, 3), dtype=np.uint8)
    )

    assert predictor.max_peaks == 1
    assert keypoints.shape == (14, 2)
    assert scores.shape == (14,)
    assert valid.shape == (14,)
    np.testing.assert_allclose(
        keypoints,
        np.broadcast_to(np.array([4.0, 3.0], dtype=np.float32), (14, 2)),
    )
    np.testing.assert_allclose(scores, np.full(14, 0.99, dtype=np.float32))
    np.testing.assert_array_equal(valid, np.ones(14, dtype=bool))
