"""Boundary tests for required PLCS motion metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler


def test_motion_archive_requires_explicit_framerate(tmp_path: Path) -> None:
    archive = tmp_path / "motion.npz"
    np.savez(
        archive,
        poses=np.zeros((2, 156), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.array("neutral"),
    )
    sampler = object.__new__(MotionSampler)

    with pytest.raises(ValueError, match="missing required mocap_framerate"):
        sampler.load_motion(archive)
