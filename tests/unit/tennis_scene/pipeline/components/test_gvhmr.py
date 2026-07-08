"""Tests for the tennis_scene GVHMR component (serialization / config)."""

from pathlib import Path

import numpy as np

from src.tennis_scene.pipeline.components.gvhmr import GVHMRConfig, GVHMRResult


def make_result(with_vertices: bool = True) -> GVHMRResult:
    rng = np.random.default_rng(0)
    P, T = 2, 4
    return GVHMRResult(
        smpl_body_pose=rng.normal(size=(P, T, 63)).astype(np.float32),
        smpl_global_orient=rng.normal(size=(P, T, 3)).astype(np.float32),
        smpl_betas=rng.normal(size=(P, 10)).astype(np.float32),
        smpl_vertices_local=(
            rng.normal(size=(P, T, 16, 3)).astype(np.float32) if with_vertices else None
        ),
        human_kp_2d=rng.normal(size=(P, T, 17, 2)).astype(np.float32),
        human_kp_vis=rng.random(size=(P, T, 17)).astype(np.float32),
        bbx_xys=rng.normal(size=(P, T, 3)).astype(np.float32),
        track_ids=np.array([3, 7], dtype=np.int32),
    )


class TestGVHMRResultRoundTrip:
    def test_save_load_roundtrip(self, tmp_path: Path):
        result = make_result()
        path = tmp_path / "gvhmr_result.json"
        result.save(path)
        loaded = GVHMRResult.load(path)

        np.testing.assert_allclose(loaded.smpl_body_pose, result.smpl_body_pose, rtol=1e-6)
        np.testing.assert_allclose(loaded.smpl_betas, result.smpl_betas, rtol=1e-6)
        np.testing.assert_allclose(
            loaded.smpl_vertices_local, result.smpl_vertices_local, rtol=1e-6
        )
        np.testing.assert_allclose(loaded.human_kp_2d, result.human_kp_2d, rtol=1e-6)
        np.testing.assert_allclose(loaded.bbx_xys, result.bbx_xys, rtol=1e-6)
        assert loaded.track_ids is not None
        np.testing.assert_array_equal(loaded.track_ids, result.track_ids)

    def test_optional_fields_survive_roundtrip(self, tmp_path: Path):
        result = make_result(with_vertices=False)
        result.track_ids = None
        path = tmp_path / "gvhmr_result.json"
        result.save(path)
        loaded = GVHMRResult.load(path)
        assert loaded.smpl_vertices_local is None
        assert loaded.track_ids is None


class TestGVHMRConfig:
    def test_defaults_point_to_ckpt_symlinks(self):
        config = GVHMRConfig(gvhmr_checkpoint="ckpt/gvhmr/gvhmr_siga24_release.ckpt")
        assert str(config.yolo_checkpoint).startswith("ckpt/")
        assert str(config.vitpose_checkpoint).startswith("ckpt/")
        assert str(config.hmr2_checkpoint).startswith("ckpt/")
        assert config.track_selection == "interactive"
        assert config.subprocess_mode is False
