from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.plcs.generate_dataset.sampling.motion_sampler import (
    MotionSampler,
    MotionSequence,
)


def _load_plcs_motion_sampler_config() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    paths_cfg = OmegaConf.load(repo_root / "src/plcs/configs/paths/default.yaml")
    motion_sources_cfg = OmegaConf.load(
        repo_root / "src/plcs/configs/motion_sources/default.yaml"
    )
    cfg = OmegaConf.create(
        {
            "motion_sources": motion_sources_cfg,
            "smplh_model_path": paths_cfg.smplh_model_path,
        }
    )
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]


@pytest.mark.local_data  # type: ignore[misc]
def test_motion_sampler_indexes_sources_and_files_from_default_configs() -> None:
    cfg = _load_plcs_motion_sampler_config()
    sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg["smplh_model_path"],
        device="cpu",
    )

    assert set(sampler._motion_sources.keys()) == {"running", "walking", "general"}
    for category, source_cfg in sampler._motion_sources.items():
        assert source_cfg.paths == cfg["motion_sources"][category]["paths"]
        assert source_cfg.weight == pytest.approx(cfg["motion_sources"][category]["weight"])

    assert set(sampler._motion_files.keys()) == {"running", "walking", "general"}
    for category, files in sampler._motion_files.items():
        assert files, f"Expected some motion files indexed for {category}"
        assert all(isinstance(p, Path) for p in files)
        assert all(p.suffix == ".npz" for p in files)
        assert all(p.name.endswith("_poses.npz") for p in files)
        for root_str in cfg["motion_sources"][category]["paths"]:
            root = Path(root_str)
            if not root.exists():
                pytest.skip(f"Local dataset path missing: {root}")
            if root.is_dir():
                assert any(p.is_relative_to(root) for p in files)
            else:
                assert root in files


@pytest.mark.local_data  # type: ignore[misc]
def test_infer_category_from_nonexistent_path_returns_general() -> None:
    cfg = _load_plcs_motion_sampler_config()
    sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg["smplh_model_path"],
        device="cpu",
    )

    nonexistent = Path("this/path/does/not/exist/some_poses.npz")
    assert sampler._infer_category_from_path(nonexistent) == "general"


@pytest.mark.local_data  # type: ignore[misc]
def test_load_motion_output_is_expected() -> None:
    cfg = _load_plcs_motion_sampler_config()
    sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg["smplh_model_path"],
        device="cpu",
    )

    motion_path = Path("data/ACCAD/Female1General_c3d/A1 - Stand_poses.npz")
    if not motion_path.exists():
        pytest.skip(f"Local motion file missing: {motion_path}")

    motion = sampler.load_motion(motion_path, max_frames=2)
    assert isinstance(motion, MotionSequence)
    assert motion.source_path.endswith(str(motion_path))
    assert motion.category == "general"
    assert motion.gender in {"female", "male"}
    assert motion.fps > 0

    assert motion.poses.dtype == np.float32
    assert motion.trans.dtype == np.float32
    assert motion.betas.dtype == np.float32
    assert motion.poses.shape == (2, 156)
    assert motion.trans.shape == (2, 3)
    assert motion.num_frames == 2
    assert motion.joints_3d is None


@pytest.mark.local_data  # type: ignore[misc]
def test_compute_joints_3d_shape_and_translation_effect() -> None:
    cfg = _load_plcs_motion_sampler_config()
    sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg["smplh_model_path"],
        device="cpu",
    )

    motion_path = Path("data/ACCAD/Female1General_c3d/A1 - Stand_poses.npz")
    if not motion_path.exists():
        pytest.skip(f"Local motion file missing: {motion_path}")
    motion = sampler.load_motion(motion_path, max_frames=2)

    joints = sampler.compute_joints_3d(motion, batch_size=2)
    assert motion.joints_3d is not None
    assert joints.shape[0] == motion.num_frames
    assert joints.ndim == 3 and joints.shape[2] == 3
    assert np.isfinite(joints).all()

    motion_no_trans = MotionSequence(
        source_path=motion.source_path,
        category=motion.category,
        gender=motion.gender,
        fps=motion.fps,
        poses=motion.poses.copy(),
        trans=np.zeros_like(motion.trans),
        betas=motion.betas.copy(),
    )
    joints_no_trans = sampler.compute_joints_3d(motion_no_trans, batch_size=2)

    delta = joints - joints_no_trans
    expected = np.broadcast_to(motion.trans[:, None, :], delta.shape)
    np.testing.assert_allclose(delta, expected, atol=1e-4, rtol=1e-4)
