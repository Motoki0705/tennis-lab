"""ACCAD source validation and native-rate sampling without resampling."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    PLCSMotionClip,
    load_amass_motion_clip,
)
from src.tasks.plcs.motion import Coco17MotionClip, MotionSourceKind
from src.tasks.plcs.motion.sources import AccadCoco17Adapter


def _archive(path: Path, fps: float) -> None:
    np.savez(
        path,
        poses=np.zeros((3, 156), dtype=np.float32),
        trans=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.array("neutral"),
        mocap_framerate=np.array(fps),
    )


def _config(
    root: Path, *, weight: object = 1.0, source_format: str = "amass_smplh_v1"
) -> DictConfig:
    return OmegaConf.create(
        {
            "motion_sources": {
                "walking": {
                    "format": source_format,
                    "paths": [str(root)],
                    "weight": weight,
                }
            }
        }
    )


class _Adapter:
    def convert(self, source: PLCSMotionClip) -> Coco17MotionClip:
        frames = source.frame_count
        return Coco17MotionClip(
            source_id=Path(source.source_path).stem,
            source_path=source.source_path,
            source_kind=MotionSourceKind.ACCAD,
            category=source.category.value,
            gender=source.gender,
            fps=source.fps,
            timestamps_s=np.arange(frames, dtype=np.float64) / source.fps,
            joints_3d_m=np.zeros((frames, 17, 3), dtype=np.float32),
            root_translation_m=np.zeros((frames, 3), dtype=np.float32),
            root_rotation=np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
            joint_confidence=np.ones((frames, 17), dtype=np.float32),
            frame_valid=np.ones(frames, dtype=np.bool_),
            provenance={},
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


def test_sampler_filters_by_native_fps_without_resampling(tmp_path: Path) -> None:
    _archive(tmp_path / "30_poses.npz", 30.0)
    _archive(tmp_path / "60_poses.npz", 60.0)
    sampler = MotionSampler(
        _config(tmp_path),
        smplh_model_path=tmp_path / "smplh",
        coco17_regressor_path=tmp_path / "regressor.pt",
        accad_adapter=cast(AccadCoco17Adapter, _Adapter()),
    )
    selected = sampler.sample_motion(required_fps=30.0)
    assert selected.source_id == "30_poses"
    assert selected.fps == 30.0
    assert selected.frame_count == 3
    assert sampler.get_category_file_count("walking") == 2
    with pytest.raises(RuntimeError, match="No configured motion matches"):
        sampler.sample_motion(required_fps=28.0)


@pytest.mark.parametrize("weight", [True, float("inf")])
def test_sampler_rejects_non_numeric_or_non_finite_weight(
    tmp_path: Path, weight: object
) -> None:
    expected_error = TypeError if weight is True else ValueError
    with pytest.raises(expected_error, match="weight must be"):
        MotionSampler(
            _config(tmp_path, weight=weight),
            smplh_model_path=tmp_path / "absent-smplh",
            coco17_regressor_path=tmp_path / "absent-regressor.pt",
        )


def test_sampler_rejects_retired_motion_artifact_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        MotionSampler(
            _config(tmp_path, source_format="coco17_motion_v1"),
            smplh_model_path=tmp_path / "absent-smplh",
            coco17_regressor_path=tmp_path / "absent-regressor.pt",
        )
