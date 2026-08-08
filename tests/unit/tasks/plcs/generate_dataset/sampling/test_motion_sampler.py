"""Tests for lossless full-frame ACCAD/AMASS PLCS clips."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    ACCADMotionLibrary,
    MotionCategory,
    PLCSMotionClip,
    load_amass_motion_clip,
)

_FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True)
class _AMASSArrays:
    poses: _FloatArray
    trans: _FloatArray
    betas: _FloatArray
    gender: NDArray[np.str_]
    mocap_framerate: NDArray[np.float64]


def _arrays(
    frame_count: int,
    *,
    dtype: type[np.float32] | type[np.float64],
) -> _AMASSArrays:
    poses: _FloatArray
    trans: _FloatArray
    betas: _FloatArray
    if dtype is np.float32:
        poses = np.arange(frame_count * 156, dtype=np.float32).reshape(
            frame_count, 156
        )
        trans = np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3)
        betas = np.arange(16, dtype=np.float32)
    else:
        poses = np.arange(frame_count * 156, dtype=np.float64).reshape(
            frame_count, 156
        )
        trans = np.arange(frame_count * 3, dtype=np.float64).reshape(frame_count, 3)
        betas = np.arange(16, dtype=np.float64)
    return _AMASSArrays(
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array("neutral"),
        mocap_framerate=np.array(120.0),
    )


def _save_arrays(
    path: Path,
    arrays: _AMASSArrays,
    *,
    include_framerate: bool = True,
) -> None:
    if include_framerate:
        np.savez(
            path,
            poses=arrays.poses,
            trans=arrays.trans,
            betas=arrays.betas,
            gender=arrays.gender,
            mocap_framerate=arrays.mocap_framerate,
        )
        return
    np.savez(
        path,
        poses=arrays.poses,
        trans=arrays.trans,
        betas=arrays.betas,
        gender=arrays.gender,
    )


def test_motion_clip_preserves_dtype_order_and_every_smplh_component(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_poses.npz"
    arrays = _arrays(7, dtype=np.float64)
    _save_arrays(source, arrays)

    clip = PLCSMotionClip.from_amass_arrays(
        source_path=source,
        category="running",
        gender="neutral",
        fps=120.0,
        poses=arrays.poses,
        trans=arrays.trans,
        betas=arrays.betas,
    )

    assert clip.frame_count == 7
    assert clip.category is MotionCategory.RUNNING
    assert clip.body_pose_axis_angle.shape == (7, 63)
    assert clip.global_orient_axis_angle.shape == (7, 3)
    assert clip.right_hand_pose_axis_angle.shape == (7, 45)
    assert clip.left_hand_pose_axis_angle.shape == (7, 45)
    assert clip.root_translation_m.shape == (7, 3)
    assert clip.full_pose_axis_angle().dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(clip.full_pose_axis_angle(), arrays.poses)
    np.testing.assert_array_equal(clip.root_translation_m, arrays.trans)
    assert not clip.body_pose_axis_angle.flags.writeable


def test_motion_clip_rejects_frame_mismatch_and_unknown_category(
    tmp_path: Path,
) -> None:
    arrays = _arrays(3, dtype=np.float32)
    with pytest.raises(ValueError, match="same T"):
        PLCSMotionClip.from_amass_arrays(
            source_path=tmp_path / "motion.npz",
            category="walking",
            gender="neutral",
            fps=30.0,
            poses=arrays.poses,
            trans=np.zeros((2, 3), dtype=np.float32),
            betas=arrays.betas,
        )
    with pytest.raises(ValueError, match="running, walking, or general"):
        PLCSMotionClip.from_amass_arrays(
            source_path=tmp_path / "motion.npz",
            category="tennis",
            gender="neutral",
            fps=30.0,
            poses=arrays.poses,
            trans=arrays.trans,
            betas=arrays.betas,
        )


def test_motion_library_selects_each_category_deterministically(tmp_path: Path) -> None:
    category_paths: dict[str, tuple[Path, ...]] = {}
    for category in MotionCategory:
        paths = []
        for index in range(2):
            path = tmp_path / f"{category.value}-{index}_poses.npz"
            _save_arrays(path, _arrays(4 + index, dtype=np.float32))
            paths.append(path)
        category_paths[category.value] = tuple(paths)
    library = ACCADMotionLibrary.from_category_paths(category_paths)

    for category in MotionCategory:
        selected_path = library.select_path(category, seed=19)
        first = library.select(category, seed=19)
        second = library.select(category, seed=19)
        assert first.source_path == str(selected_path)
        assert first.source_path == second.source_path
        assert first.category is category
        assert first.frame_count in {4, 5}


def test_motion_archive_requires_explicit_framerate(tmp_path: Path) -> None:
    archive = tmp_path / "motion_poses.npz"
    _save_arrays(
        archive,
        _arrays(2, dtype=np.float32),
        include_framerate=False,
    )

    with pytest.raises(ValueError, match="missing fields.*mocap_framerate"):
        load_amass_motion_clip(archive)
