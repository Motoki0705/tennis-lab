"""Evaluation controls must not receive camera GT or overcount event matches."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.blcs.ablation.evaluation import event_diagnostic, geometric_baseline
from src.utils.projection.camera_projector import make_look_at_camera, project_points
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import denormalize_court_position


def test_geometry_recovers_trajectory_and_gaps_using_only_2d_inputs() -> None:
    court = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14]
    position = torch.tensor([[0.2 * i, 0.4 * i, 1.0 + 0.1 * i] for i in range(7)])
    court_uv, ball_uv = [], []
    for center in ((5.0, -22.0, 6.0), (-5.0, 22.0, 6.0)):
        camera = make_look_at_camera(
            center, look_at=(0, 0, 0), image_size=(1920, 1080), hfov_deg=60
        )
        court_uv.append(project_points(camera, court)[0] / torch.tensor([1920, 1080]))
        ball_uv.append(project_points(camera, position)[0] / torch.tensor([1920, 1080]))
    observed = torch.ones(1, 2, 7, dtype=torch.bool)
    observed[:, :, 3] = False
    batch = {
        "ball_uv": torch.stack(ball_uv)[None],
        "ball_vis": observed,
        "court_kp": torch.stack(court_uv)[None, :, None].expand(1, 2, 7, 14, 2),
        "court_vis": torch.ones(1, 2, 7, 14),
        "camera_w": torch.full((1, 2), 1920),
        "camera_h": torch.full((1, 2), 1080),
    }
    prediction, receipt = geometric_baseline(batch)
    np.testing.assert_allclose(
        denormalize_court_position(prediction.position)[0], position, atol=2e-4
    )
    assert receipt["triangulated_frames"] == 6
    assert receipt["interpolated_frames"] == 1
    assert receipt["calibration_uses_ground_truth_camera"] is False
    # Unusable 2D court coverage cannot be rescued by privileged camera fields.
    batch["court_vis"].zero_()
    batch["camera_R"] = torch.eye(3).expand(1, 2, 3, 3)
    with pytest.raises(ValueError, match="Fewer than two cameras"):
        geometric_baseline(batch)


def test_event_matching_maximizes_support_before_timing_accuracy() -> None:
    predicted: NDArray[np.bool_] = np.zeros(8, dtype=np.bool_)
    target: NDArray[np.bool_] = np.zeros(8, dtype=np.bool_)
    predicted[[3, 5]] = True
    target[[1, 4]] = True
    result = event_diagnostic(predicted, target, 0.1)
    assert result["true_positive"] == 2
    assert result["f1"] == 1
    assert result["timing_mean_s"] == pytest.approx(0.15)
    assert not result["partition_failed"]
    predicted[6] = True
    result = event_diagnostic(predicted, target, 0.1)
    assert result["f1"] == pytest.approx(0.8)
    assert result["partition_failed"]


def test_no_events_have_undefined_f1_and_timing() -> None:
    result = event_diagnostic(np.zeros(8, dtype=bool), np.zeros(8, dtype=bool), 0.1)
    assert result["f1"] is None
    assert result["timing_mean_s"] is None
    assert result["true_positive"] == 0
