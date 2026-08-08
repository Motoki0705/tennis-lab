"""Production-device policy tests for the PLCS stage boundary."""

from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.plcs.handler import (
    PLCSObjectRequest,
    PLCSStageParameters,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionCategory


def _parameters(model_root: Path, *, device: str) -> PLCSStageParameters:
    return PLCSStageParameters(
        seed=7,
        split="train",
        scene_splits={"B00": "train"},
        objects=(
            PLCSObjectRequest(
                category=MotionCategory.RUNNING,
                start_frame=0,
                anchor_position_court_m=(0.0, 0.0, 0.0),
                yaw_radians=0.0,
            ),
        ),
        smplh_model_root=model_root,
        gaussian_count=32,
        smplh_batch_size=8,
        device=device,
    )


def test_production_parameters_reject_cpu_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit CUDA"):
        _parameters(tmp_path, device="cpu")

    configured = _parameters(tmp_path, device="cuda:0")
    assert configured.smplh_batch_size == 8
