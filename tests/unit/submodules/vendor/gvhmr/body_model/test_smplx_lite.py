"""Tests for the vendored SmplxLite body model using a synthetic SMPL-X npz."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.submodules.configuration import BundledModelAssetPaths
from src.submodules.vendor.gvhmr.body_model.smplx_lite import (
    SmplxLite,
    batch_rigid_transform_v2,
    resolve_smplx_model_file,
)

NUM_VERTS = 40
NUM_JOINTS = 55


def _bundled_assets(path: Path) -> BundledModelAssetPaths:
    return BundledModelAssetPaths(
        hmr2_mean_params=path,
        smplx_to_smpl=path,
        smpl_coco17_regressor=path,
        smplx_verts437=path,
        smpl_neutral_joint_regressor=path,
    )


@pytest.fixture()
def synthetic_smplx_npz(tmp_path: Path) -> Path:
    """A structurally valid, tiny SMPL-X model file."""
    rng = np.random.default_rng(0)
    parents = np.arange(-1, NUM_JOINTS - 1)  # simple chain
    kintree_table = np.stack([parents, np.arange(NUM_JOINTS)], axis=0)
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(
        path,
        shapedirs=rng.normal(size=(NUM_VERTS, 3, 400)).astype(np.float64) * 0.01,
        v_template=rng.normal(size=(NUM_VERTS, 3)).astype(np.float64),
        J_regressor=(np.ones((NUM_JOINTS, NUM_VERTS)) / NUM_VERTS).astype(np.float64),
        posedirs=rng.normal(size=(NUM_VERTS, 3, (NUM_JOINTS - 1) * 9)).astype(np.float64) * 0.001,
        weights=(np.ones((NUM_VERTS, NUM_JOINTS)) / NUM_JOINTS).astype(np.float64),
        kintree_table=kintree_table.astype(np.int64),
        f=np.zeros((10, 3), dtype=np.int64),
        hands_meanl=np.zeros(45, dtype=np.float64),
        hands_meanr=np.zeros(45, dtype=np.float64),
    )
    return path


class TestSmplxLite:
    def test_forward_shapes(self, synthetic_smplx_npz: Path):
        model = SmplxLite(
            model_path=synthetic_smplx_npz,
            bundled_assets=_bundled_assets(synthetic_smplx_npz),
        )
        B, L = 2, 3
        verts = model(
            body_pose=torch.randn(B, L, 63) * 0.1,
            betas=torch.randn(B, L, 10) * 0.1,
            global_orient=torch.randn(B, L, 3) * 0.1,
            transl=torch.randn(B, L, 3),
        )
        assert verts.shape == (B, L, NUM_VERTS, 3)
        assert torch.isfinite(verts).all()

    def test_zero_pose_zero_betas_recovers_template(self, synthetic_smplx_npz: Path):
        model = SmplxLite(
            model_path=synthetic_smplx_npz,
            bundled_assets=_bundled_assets(synthetic_smplx_npz),
        )
        transl = torch.tensor([[[1.0, 2.0, 3.0]]])
        verts = model(
            body_pose=torch.zeros(1, 1, 63),
            betas=torch.zeros(1, 1, 10),
            global_orient=torch.zeros(1, 1, 3),
            transl=transl,
        )
        template = cast(torch.Tensor, model.v_template)
        expected = template[None, None] + transl[..., None, :]
        torch.testing.assert_close(verts, expected, atol=1e-5, rtol=0)

    def test_get_skeleton_shape(self, synthetic_smplx_npz: Path):
        model = SmplxLite(
            model_path=synthetic_smplx_npz,
            bundled_assets=_bundled_assets(synthetic_smplx_npz),
        )
        skeleton = model.get_skeleton(torch.zeros(2, 10))
        assert skeleton.shape == (2, NUM_JOINTS, 3)
        joint_template = cast(torch.Tensor, model.J_template)
        torch.testing.assert_close(skeleton, joint_template.expand(2, -1, -1))

    def test_missing_model_file_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="smpl-x.is.tue.mpg.de"):
            resolve_smplx_model_file(tmp_path)


class TestBatchRigidTransform:
    def test_identity_rotations_keep_joints(self):
        gen = torch.Generator().manual_seed(1)
        joints = torch.randn(2, NUM_JOINTS, 3, generator=gen)
        rot_mats = torch.eye(3).expand(2, NUM_JOINTS, 3, 3).contiguous()
        parents = torch.arange(-1, NUM_JOINTS - 1)

        posed_joints, rel_transforms = batch_rigid_transform_v2(rot_mats, joints, parents)
        torch.testing.assert_close(posed_joints, joints, atol=1e-5, rtol=0)
        # With identity rotations the relative transforms carry no translation.
        torch.testing.assert_close(
            rel_transforms[..., :3, 3],
            torch.zeros(2, NUM_JOINTS, 3),
            atol=1e-5,
            rtol=0,
        )
