from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.ball_detection.data.augmentation.video import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.evaluation.coordinates import CoordinateMetrics
from src.tasks.ball_detection.training.temporal.packs import supervised_loss_sum


def test_coordinate_gradients_use_total_observed_frames_and_keep_empty_children() -> (
    None
):
    prediction = torch.full((4, 32, 2), 0.4, requires_grad=True)
    target = torch.full_like(prediction, 0.6)
    mask = torch.zeros((4, 32), dtype=torch.bool)
    mask[0, :2] = True
    whole, count = supervised_loss_sum(
        prediction,
        dict(uv=target, position_valid=mask),
        sigma_ratio=0.012,
        gamma=2.0,
        coordinates=True,
    )
    (whole / count).backward()
    expected = prediction.grad.clone()
    prediction.grad = None
    teachers = 0
    for i in range(4):
        value, n = supervised_loss_sum(
            prediction[i : i + 1],
            dict(uv=target[i : i + 1], position_valid=mask[i : i + 1]),
            sigma_ratio=0.012,
            gamma=2.0,
            coordinates=True,
        )
        value.backward()
        teachers += n
    torch.testing.assert_close(prediction.grad / teachers, expected, rtol=0, atol=0)
    assert teachers == 2


def test_edge_only_pack_retains_known_gt_and_audits_ineligible_occlusion() -> None:
    cfg = AugmentationConfig.load(
        Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    )
    cfg = replace(cfg, camera_probability=0.0, occlusion_probability=1.0)
    rgb = torch.zeros((1, 128, 3, 24, 24), dtype=torch.uint8)
    valid = torch.zeros((1, 128), dtype=torch.bool)
    valid[0, 0] = valid[0, -1] = True
    batch = dict(
        uv=torch.full((1, 128, 2), 0.5),
        position_valid=valid,
        timestamps=torch.arange(128)[None].float() / 30,
        clip_id=["edge"],
        start=[0],
        frame_step=[1],
    )
    output, changed, audit = VideoAugmenter(cfg, 42)(rgb, batch, epoch=0)
    assert torch.equal(rgb, output) and torch.equal(changed["position_valid"], valid)
    assert audit["occlusion_requested"].tolist() == [True]
    assert audit["occlusion_eligible"].tolist() == [False]
    assert audit["occlusion_applied"].tolist() == [False]


def test_coordinate_owner_uses_actual_temporal_window_center() -> None:
    batch = dict(
        clip_id=["clip", "clip"],
        frame_step=[1, 1],
        start=[0, 32],
        common_evaluation=[True, True],
        source=["tracknet", "tracknet"],
        frame_indices=torch.stack((torch.arange(64), torch.arange(32, 96))),
        position_valid=torch.zeros((2, 64), dtype=torch.bool),
    )
    batch["position_valid"][0, 32] = batch["position_valid"][1, 0] = True
    errors = torch.zeros((2, 64))
    errors[0, 32] = 1.0
    errors[1, 0] = 9.0
    metrics = CoordinateMetrics((1,))
    metrics.add(batch, errors)
    assert metrics.frames[(1, "clip", 32)][0][1] == 0
    assert metrics.frames[(1, "clip", 32)][1] == 1.0
