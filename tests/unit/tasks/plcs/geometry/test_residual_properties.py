"""Preserved geometry and inference properties after PLCS layer relocation."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from numpy.testing import assert_allclose

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.configuration_contracts import ResidualModelConfig
from src.tasks.plcs.data.residual_types import CameraRig
from src.tasks.plcs.geometry.residual_features import (
    InsufficientGeometryError,
    prepare_geometry,
)
from src.tasks.plcs.models.triangulation_residual import GeometricResidualModel
from src.tasks.plcs.training.residual_lightning_module import ResidualLightningModule
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def compose_recipe():
    with initialize_config_dir(config_dir=str(Path("src/tasks/plcs/configs").resolve()), version_base="1.3"):
        return compose(config_name="train_triangulation_residual")

def camera_fixture():
    centers = np.array([[-8.0, -18.0, 3.0], [8.0, -18.0, 3.0], [0.0, 18.0, 3.0]])
    rotations = []
    for center in centers:
        z = np.array([0.0, 0.0, 1.0]) - center
        z /= np.linalg.norm(z)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        rotations.append(np.stack((x, np.cross(z, x), z)))
    r = np.array(rotations)
    k = np.tile(
        np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
        (3, 1, 1),
    )
    return CameraRig(
        k, r, -np.einsum("vij,vj->vi", r, centers), np.tile([1920, 1080], (3, 1))
    )



def world_fixture(joints=17):
    x = np.random.default_rng(5).uniform(
        [-0.3, -8.0, 0.2], [0.3, -7.5, 1.8], (1, joints, 3)
    )
    return np.repeat(x, 16, axis=0) + np.arange(16)[:, None, None] * np.array(
        [0.01, 0, 0]
    )



def geometry_fixture(joints=17):
    rig = camera_fixture()
    world = world_fixture(joints)
    p, _ = project_multiview(world, rig.matrices)
    p = p.transpose(2, 0, 1, 3)
    court, _ = project_multiview(
        court_keypoints_3d(STANDARD_COURT_CONFIG).numpy(), rig.matrices
    )
    return rig, world, p, court[:14].transpose(1, 0, 2)



@pytest.mark.parametrize("joints,root", [(17, (11, 12))])
def test_clean_geometry_recovers_world_and_reprojection(joints, root):
    rig, world, p, court = geometry_fixture(joints)
    g = prepare_geometry(
        p,
        np.ones(p.shape[:-1]),
        court,
        np.ones((3, 14)),
        rig,
        root_indices=root,
        fps=60.0,
    )
    assert g.features.shape == (3, 16, 18 * joints + 61)
    assert g.init_valid.all()
    assert_allclose(g.init_world_m, world, atol=1e-6)
    assert_allclose(
        g.root_init_m[:, None] + g.relative_init_m, g.init_world_m, atol=1e-6
    )
    assert_allclose(g.relative_init_m[:, root].mean(axis=1), 0, atol=1e-6)
    assert_allclose(g.residual_uv, 0, atol=1e-6)
    assert_allclose(np.diff(g.time_positions), 0.5)



def test_missing_points_stay_masked_and_use_only_observed_seed():
    rig, world, p, court = geometry_fixture()
    score = np.ones(p.shape[:-1])
    score[:, 5:9] = 0
    score[:, :, 9] = 0
    p[score == 0] = np.nan
    g = prepare_geometry(
        p,
        score,
        court,
        np.ones((3, 14)),
        rig,
        root_indices=(11, 12),
        fps=30.0,
    )
    assert not g.init_valid[5:9].any()
    assert not g.init_valid[:, 9].any()
    assert_allclose(g.init_world_m[:, 9], g.root_init_m, atol=1e-6)
    assert_allclose(g.init_world_m[:, 0], world[:, 0], atol=1e-6)
    assert np.isfinite(g.features).all()
    with pytest.raises(InsufficientGeometryError):
        prepare_geometry(
            p,
            np.zeros_like(score),
            court,
            np.ones((3, 14)),
            rig,
            root_indices=(11, 12),
            fps=30.0,
        )



@pytest.mark.parametrize(
    "invalid_contract",
    [
        "feature_rank",
        "feature_width",
        "mask_shape",
        "mask_dtype",
        "time_shape",
        "device",
    ],
)
def test_training_rejects_invalid_tensor_contract_before_forward(
    invalid_contract, monkeypatch
):
    from unittest.mock import Mock

    cfg = compose_recipe()
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    module = ResidualLightningModule(cfg)
    batch = {
        "features": torch.zeros(2, 3, 8, 367),
        "view_valid": torch.ones(2, 3, 8, dtype=torch.bool),
        "time_positions": torch.zeros(2, 8),
    }
    if invalid_contract == "feature_rank":
        batch["features"] = batch["features"][0]
    elif invalid_contract == "feature_width":
        batch["features"] = batch["features"][..., :-1]
    elif invalid_contract == "mask_shape":
        batch["view_valid"] = batch["view_valid"][:, :-1]
    elif invalid_contract == "mask_dtype":
        batch["view_valid"] = batch["view_valid"].float()
    elif invalid_contract == "time_shape":
        batch["time_positions"] = batch["time_positions"][:, :-1]
    else:
        batch["view_valid"] = batch["view_valid"].to("meta")
    forward = Mock(side_effect=AssertionError("Invalid batch reached model forward"))
    monkeypatch.setattr(module.model, "forward", forward)
    with pytest.raises(ValueError):
        module._step(batch, "train")
    forward.assert_not_called()



@pytest.mark.parametrize("task,joints", [("plcs", 17)])
def test_sliding_window_zero_residual_is_identity(task, joints):
    from src.tasks.plcs.inference.residual_predictor import predict_geometry

    rig, world, p, court = geometry_fixture(joints)
    indices = (11, 12) if task == "plcs" else (0,)
    geometry = prepare_geometry(
        p,
        np.ones(p.shape[:-1]),
        court,
        np.ones((3, 14)),
        rig,
        root_indices=indices,
        fps=60.0,
    )
    model = GeometricResidualModel(
        ResidualModelConfig(
            "plcs_triangulation_residual", 32, 1, 4, 64, "swiglu", 0.0, 10000.0
        ),
    )
    result = predict_geometry(
        model, geometry, window_size=7, device=torch.device("cpu")
    )
    for value in result.values():
        assert value.shape[0] == len(world)
        assert_allclose(value, 0, atol=0)
    invalid = replace(geometry, view_valid=geometry.view_valid.astype(np.float32))
    with pytest.raises(ValueError, match="view/time contract"):
        predict_geometry(model, invalid, window_size=7, device=torch.device("cpu"))



@pytest.mark.parametrize("ffn_type", ["swiglu", "mlp"])
def test_config_selects_and_executes_both_camera_and_time_ffns(ffn_type):
    from src.utils.models.components.ffn_layers import MLP, SwiGLU

    config = compose_recipe()
    config.model.ffn_type = ffn_type
    model_config = replace(
        validate_residual_config(config).model,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = GeometricResidualModel(model_config).eval()
    expected_type = MLP if ffn_type == "mlp" else SwiGLU
    calls = []
    for block in (*model.camera_layers, *model.time_layers):
        assert isinstance(block.ffn, expected_type)
        block.ffn.register_forward_hook(
            lambda module, args, output: calls.append(output)
        )
    with torch.no_grad():
        model.root_head[-1].weight.normal_(std=0.01)
        output = model(
            torch.randn(1, 3, 8, 367),
            torch.ones(1, 3, 8, dtype=torch.bool),
            torch.arange(8)[None].float(),
        )
    assert len(calls) == 2
    assert torch.isfinite(output["root_residual"]).all()
    assert torch.count_nonzero(output["root_residual"]) > 0

