"""Boundary tests for required PLCS motion metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    load_amass_motion_clip,
)
from src.tasks.plcs.motion import (
    Coco17MotionClip,
    MotionSourceKind,
    save_motion_clip,
)


def _common_clip(*, source_id: str, fps: float) -> Coco17MotionClip:
    frames = 3
    return Coco17MotionClip(
        source_id=source_id,
        source_path=f"/dataset/{source_id}.mp4",
        source_kind=MotionSourceKind.GVHMR,
        category="tennis",
        gender="neutral",
        fps=fps,
        timestamps_s=np.arange(frames, dtype=np.float64) / fps,
        joints_3d_m=np.zeros((frames, 17, 3), dtype=np.float32),
        root_translation_m=np.zeros((frames, 3), dtype=np.float32),
        root_rotation=np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
        joint_confidence=np.ones((frames, 17), dtype=np.float32),
        frame_valid=np.ones(frames, dtype=np.bool_),
        provenance={"camera_id": "cam1"},
    )


def test_motion_archive_requires_explicit_framerate(tmp_path: Path) -> None:
    archive = tmp_path / "motion.npz"
    np.savez(
        archive,
        poses=np.zeros((2, 156), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.array("neutral"),
    )
    with pytest.raises(ValueError, match="missing fields.*mocap_framerate"):
        load_amass_motion_clip(archive)


def test_sampler_loads_common_coco17_motion_without_smpl_assets(
    tmp_path: Path,
) -> None:
    clip = _common_clip(
        source_id="video_000/clip_000/cam1",
        fps=59.94,
    )
    artifact = save_motion_clip(clip, tmp_path / "cam1.motion.npz")
    config = OmegaConf.create(
        {
            "motion_sources": {
                "tennis": {
                    "format": "coco17_motion_v1",
                    "paths": [str(tmp_path)],
                    "weight": 1.0,
                }
            }
        }
    )

    sampler = MotionSampler(
        config,
        smplh_model_path=tmp_path / "intentionally-absent-smplh",
        coco17_regressor_path=tmp_path / "intentionally-absent-regressor.npy",
    )
    loaded = sampler.load_motion(artifact, category="tennis")

    assert sampler.get_available_categories() == ["tennis"]
    assert sampler.get_category_file_count("tennis") == 1
    assert loaded.metadata() == clip.metadata()
    np.testing.assert_array_equal(loaded.joints_3d_m, clip.joints_3d_m)


def test_sampler_filters_by_native_fps_without_resampling(tmp_path: Path) -> None:
    clip_30 = _common_clip(source_id="motion_30", fps=30.0)
    clip_60 = _common_clip(source_id="motion_60", fps=60.0)
    save_motion_clip(clip_30, tmp_path / "30.motion.npz")
    save_motion_clip(clip_60, tmp_path / "60.motion.npz")
    config = OmegaConf.create(
        {
            "motion_sources": {
                "tennis": {
                    "format": "coco17_motion_v1",
                    "paths": [str(tmp_path)],
                    "weight": 1.0,
                }
            }
        }
    )
    sampler = MotionSampler(
        config,
        smplh_model_path=tmp_path / "absent-smplh",
        coco17_regressor_path=tmp_path / "absent-regressor.pt",
    )

    selected = sampler.sample_motion(required_fps=30.0)

    assert selected.source_id == "motion_30"
    assert selected.fps == 30.0
    with pytest.raises(RuntimeError, match="No configured motion matches"):
        sampler.sample_motion(required_fps=28.0)


@pytest.mark.parametrize("weight", [True, float("inf")])
def test_sampler_rejects_non_numeric_or_non_finite_weight(
    tmp_path: Path,
    weight: object,
) -> None:
    save_motion_clip(
        _common_clip(source_id="motion", fps=30.0),
        tmp_path / "motion.motion.npz",
    )
    config = OmegaConf.create(
        {
            "motion_sources": {
                "tennis": {
                    "format": "coco17_motion_v1",
                    "paths": [str(tmp_path)],
                    "weight": weight,
                }
            }
        }
    )

    expected_error = TypeError if weight is True else ValueError
    with pytest.raises(expected_error, match="weight must be"):
        MotionSampler(
            config,
            smplh_model_path=tmp_path / "absent-smplh",
            coco17_regressor_path=tmp_path / "absent-regressor.pt",
        )
