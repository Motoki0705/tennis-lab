from __future__ import annotations

import torch

from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.utils.data.augmentation import (
    apply_burst_visibility_dropout,
    inject_false_positive_observations,
)


def _sample() -> dict[str, torch.Tensor]:
    return {
        "ball_uv": torch.tensor(
            [
                [
                    [0.10, 0.10],
                    [0.20, 0.20],
                    [0.30, 0.30],
                    [0.40, 0.40],
                    [0.50, 0.50],
                    [0.60, 0.60],
                ]
            ],
            dtype=torch.float32,
        ),
        "ball_vis": torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]),
        "ball_mask": torch.ones(1, 6),
        "court_kp": torch.full((1, 6, 2, 2), 0.5),
        "court_vis": torch.ones(1, 6, 2),
        "position_3d": torch.zeros(6, 3),
        "velocity_3d": torch.zeros(6, 3),
        "seq_len": torch.tensor(6),
        "camera_R": torch.eye(3).unsqueeze(0),
        "camera_C": torch.zeros(1, 3),
        "camera_f": torch.ones(1),
        "camera_cx": torch.zeros(1),
        "camera_cy": torch.zeros(1),
        "camera_w": torch.ones(1),
        "camera_h": torch.ones(1),
    }


def test_blcs_augmentation_preserves_clean_targets_for_noisy_inputs() -> None:
    torch.manual_seed(7)
    sample = _sample()
    augmentation = BLCSBallObservationAugmentation(
        {
            "enabled": True,
            "preserve_clean_targets": True,
            "uv_scale": {"enabled": False},
            "gaussian_noise": {"enabled": False},
            "visibility_dropout": {"enabled": False},
            "temporal_jitter": {
                "enabled": True,
                "prob": 1.0,
                "jitter_std": 0.01,
                "drift_std": 0.0,
            },
            "burst_dropout": {"enabled": False},
            "false_positive": {"enabled": False},
            "edge_degradation": {"enabled": False},
            "speed_conditioned": {"enabled": False},
        }
    )

    out = augmentation.forward(sample)

    assert torch.equal(out["ball_uv_target"], sample["ball_uv"])
    assert torch.equal(out["ball_vis_target"], sample["ball_vis"])
    assert not torch.equal(out["ball_uv"], sample["ball_uv"])
    assert torch.equal(out["position_3d"], sample["position_3d"])
    assert torch.equal(out["camera_R"], sample["camera_R"])


def test_burst_dropout_creates_contiguous_missing_span() -> None:
    generator = torch.Generator().manual_seed(3)
    visibility = torch.ones(1, 8)

    out = apply_burst_visibility_dropout(
        visibility,
        prob=1.0,
        min_len=3,
        max_len=3,
        max_bursts=1,
        generator=generator,
    )

    dropped = (out == 0).nonzero(as_tuple=False)[:, -1]
    assert dropped.numel() == 3
    assert torch.equal(dropped, torch.arange(int(dropped[0]), int(dropped[0]) + 3))


def test_false_positive_injection_only_fills_invisible_frames() -> None:
    generator = torch.Generator().manual_seed(11)
    uv = torch.zeros(1, 4, 2)
    visibility = torch.tensor([[1.0, 0.0, 0.0, 1.0]])

    out_uv, out_vis = inject_false_positive_observations(
        uv,
        visibility,
        false_positive_prob=1.0,
        generator=generator,
    )

    assert torch.equal(out_vis, torch.ones_like(visibility))
    assert torch.equal(out_uv[:, 0], uv[:, 0])
    assert torch.equal(out_uv[:, 3], uv[:, 3])
    assert torch.all((out_uv[:, 1:3] >= 0.0) & (out_uv[:, 1:3] <= 1.0))
    assert not torch.equal(out_uv[:, 1:3], uv[:, 1:3])
