"""Licensed-asset round trip for outputs of the real clip experiment."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.submodules.configuration import BundledModelAssetPaths
from src.submodules.models.gvhmr.mesh_recovery import SmplVertexReconstructor
from src.tennis_scene.motion_alignment.similarity import SimilarityTransform
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP


@pytest.mark.local_data
def test_exported_scaled_smpl_reconstructs_the_same_transformed_mesh():
    asset_dir = os.environ.get("TENNIS_LAB_LOCAL_ASSET_ROOT")
    result_dir = os.environ.get("TENNIS_LAB_ALIGNMENT_RUN_DIR")
    if asset_dir is None or result_dir is None:
        pytest.skip("Set TENNIS_LAB_LOCAL_ASSET_ROOT and TENNIS_LAB_ALIGNMENT_RUN_DIR")
    assert asset_dir is not None and result_dir is not None
    root, results = Path(asset_dir).resolve(), Path(result_dir).resolve()
    vendor = root / "src/submodules/vendor/gvhmr"
    assets = BundledModelAssetPaths(
        hmr2_mean_params=vendor / "hmr2/smpl_mean_params.npz",
        smplx_to_smpl=vendor / "body_model/data/smplx2smpl_sparse.pt",
        smpl_coco17_regressor=vendor / "body_model/data/smpl_coco17_J_regressor.pt",
        smplx_verts437=vendor / "body_model/data/smplx_verts437.pt",
        smpl_neutral_joint_regressor=vendor / "body_model/data/smpl_neutral_J_regressor.pt",
    )
    torch.set_num_threads(4)
    reconstructor = SmplVertexReconstructor(root / "third_party/GVHMR/inputs/checkpoints/body_models", device="cpu", bundled_assets=assets)
    metrics = json.loads((results / "metrics.json").read_text())
    for player, record in metrics["players"].items():
        with np.load(results / "inputs" / f"{record['camera_id']}.gvhmr.npz") as raw:
            indices = [0, len(raw["transl"]) // 2, len(raw["transl"]) - 1]
            original = reconstructor.reconstruct({key: torch.from_numpy(raw[key][indices]) for key in ("body_pose", "betas", "global_orient", "transl")}).numpy()
        for mode in ("fixed", "free"):
            fit = record[mode]["transform"]
            transform = SimilarityTransform(fit["scale"], np.deg2rad(fit["yaw_deg"]), np.asarray(fit["translation_m"]))
            expected = transform.apply(original.astype(np.float64) @ SMPL_Y_UP_TO_COURT_Z_UP.T)
            with np.load(results / f"player_{player}_{mode}.npz") as exported:
                parameters = {key: torch.from_numpy(exported[key][indices]) for key in ("body_pose", "betas", "global_orient")}
                parameters["transl"] = torch.zeros((len(indices), 3))
                reconstructed = reconstructor.reconstruct(parameters).numpy()
                actual = float(exported["body_scale"]) * reconstructed + exported["transl"][indices, None]
            # float32 LBS plus topology conversion; tolerance is 0.03 mm.
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=0)
