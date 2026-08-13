from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.base.data.court_peaks import (
    COURT_SEMANTIC_CLASS_NAMES,
    CourtPeakFrame,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io import PLCSTrackQueryIOAdapter


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        camera_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        temporal_attention_mask: Tensor,
        reference_view_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            detection_mask,
            court_kp,
            court_vis,
            frame_mask,
            camera_state_valid,
            spatial_attention_mask,
            temporal_attention_mask,
            reference_view_mask,
        )
        batch, _, frames = human_kp.shape[:3]
        rotation = torch.tensor([0.0, 1.0], device=human_kp.device)
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": rotation.expand(batch, frames, 2, -1),
            "presence_logits": torch.tensor([2.0, -2.0], device=human_kp.device).expand(
                batch, frames, -1
            ),
        }


class _FixedKP7TrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        court_peak_uv: Tensor,
        court_peak_score: Tensor,
        court_peak_covariance: Tensor,
        court_peak_valid: Tensor,
        player_anchor: Tensor,
        player_features: Tensor,
        frame_mask: Tensor,
        camera_state_valid: Tensor,
        spatial_attention_mask: Tensor,
        temporal_attention_mask: Tensor,
        reference_view_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            detection_mask,
            court_peak_uv,
            court_peak_score,
            court_peak_covariance,
            court_peak_valid,
            player_anchor,
            player_features,
            frame_mask,
            camera_state_valid,
            spatial_attention_mask,
            temporal_attention_mask,
            reference_view_mask,
        )
        batch, _, frames = human_kp.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": torch.ones(batch, frames, 2, 2, device=human_kp.device),
            "presence_logits": torch.ones(batch, frames, 2, device=human_kp.device),
        }


def _court_peak_frames(frames: int) -> list[CourtPeakFrame]:
    return [
        CourtPeakFrame(
            batch_index=0,
            view_index=0,
            frame_index=frame_index,
            keypoints_pixels=torch.full((7, 2, 2), 5.0),
            scores=torch.ones(7, 2),
            covariance_pixels=torch.eye(2)
            .reshape(1, 1, 2, 2)
            .expand(7, 2, 2, 2),
            valid=torch.ones(7, 2, dtype=torch.bool),
            image_size_hw=(11, 11),
            semantic_class_names=COURT_SEMANTIC_CLASS_NAMES,
        )
        for frame_index in range(frames)
    ]


def test_predictor_returns_cpu_lifecycle_and_yaw_outputs() -> None:
    predictor = PLCSTrackingPredictor(
        model=_FixedTrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedTrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
            mask_invisible_observations=True,
        ),
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        joint_visibility=torch.ones(*shape, 17, dtype=torch.bool),
        detection_score=torch.ones(*shape),
        detection_mask=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
        reference_view_index=torch.zeros(1, dtype=torch.long),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=True,
    )

    assert result["position_meters"].shape == (1, 3, 2, 3)
    assert result["presence"][..., 0].all()
    assert not result["presence"][..., 1].any()
    torch.testing.assert_close(
        result["yaw_radians"],
        torch.full((1, 3, 2), torch.pi / 2),
    )
    assert all(value.device.type == "cpu" for value in result.values())


def test_predictor_accepts_indexed_court_predictor_frames_directly() -> None:
    predictor = PLCSTrackingPredictor(
        model=_FixedKP7TrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedKP7TrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
            mask_invisible_observations=True,
            court_observation_profile="kp7_reference",
        ),
        device=torch.device("cpu"),
    )
    shape = (1, 1, 2, 1)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        joint_visibility=torch.ones(*shape, 17, dtype=torch.bool),
        detection_score=torch.ones(*shape),
        detection_mask=torch.ones(*shape, dtype=torch.bool),
        court_peak_frames=_court_peak_frames(2),
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        view_mask=torch.ones(1, 1, dtype=torch.bool),
        reference_view_index=torch.zeros(1, dtype=torch.long),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=False,
    )

    assert result["position"].shape == (1, 2, 2, 3)
    assert result["presence"].all()
