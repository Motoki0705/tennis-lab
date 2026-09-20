"""Independent geometry, gauge, missing-data, task, and camera-mask properties."""

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from numpy.testing import assert_allclose

from src.tasks.base.triangulation_residual.configuration import (
    FeatureConfig,
    ModelConfig,
    validate_config,
)
from src.tasks.base.triangulation_residual.contracts import CameraRig
from src.tasks.base.triangulation_residual.corruption import corrupt_observations
from src.tasks.base.triangulation_residual.geometry import (
    InsufficientGeometryError,
    prepare_geometry,
)
from src.tasks.base.triangulation_residual.model import (
    GeometricResidualModel,
    reconstruct_world,
)
from src.tasks.base.triangulation_residual.training import (
    ResidualLightningModule,
    ResidualTrainingRunner,
)
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


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


@pytest.mark.parametrize("joints,root", [(17, (11, 12)), (1, (0,))])
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
        feature_config=FeatureConfig("raw", 1.0),
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
        feature_config=FeatureConfig("raw", 1.0),
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
            feature_config=FeatureConfig("raw", 1.0),
        )


@pytest.mark.parametrize("task,joints", [("plcs", 17), ("blcs", 1)])
def test_zero_residual_identity_camera_permutation_and_padding(task, joints):
    cfg = ModelConfig(
        f"{task}_triangulation_residual_v1", 32, 2, 4, 64, "swiglu", 0.0, 10000.0
    )
    model = GeometricResidualModel(task, cfg).eval()
    x = torch.randn(2, 3, 8, 18 * joints + 61)
    valid = torch.ones((2, 3, 8), dtype=torch.bool)
    time = torch.arange(8, dtype=torch.float32)[None].expand(2, -1)
    o = model(x, valid, time)
    root = torch.randn(2, 8, 3)
    rel = torch.randn(2, 8, joints, 3)
    if task == "blcs":
        rel.zero_()
    world, _, _ = reconstruct_world(o, root, rel, task=task)
    torch.testing.assert_close(world, root[:, :, None] + rel)
    for head in (model.root_head, model.relative_head):
        if head is not None:
            torch.nn.init.normal_(head[-1].weight, std=0.01)
    before = model(x, valid, time)
    after = model(x[:, [2, 0, 1]], valid[:, [2, 0, 1]], time)
    padded = model(
        torch.cat((x, torch.randn(2, 1, 8, x.shape[-1]) * 100), dim=1),
        torch.cat((valid, torch.zeros(2, 1, 8, dtype=torch.bool)), dim=1),
        time,
    )
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(before[key], padded[key], atol=1e-5, rtol=1e-5)
    if task == "plcs":
        torch.testing.assert_close(
            before["relative_residual"][:, :, [11, 12]].mean(2),
            torch.zeros(2, 8, 3),
            atol=1e-7,
            rtol=0,
        )
    sum(v.square().sum() for v in before.values()).backward()
    assert torch.isfinite(model.root_head[-1].weight.grad).all()


def compose_recipe(task, version=1):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        return compose(
            config_name="train_triangulation_residual" + ("_v2" if version == 2 else "")
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

    cfg = compose_recipe("blcs")
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    module = ResidualLightningModule(cfg)
    batch = {
        "features": torch.zeros(2, 3, 8, 79),
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


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_noise_keeps_true_camera_separate_and_clean_branch_exact(task):
    c = validate_config(compose_recipe(task))
    cfg = replace(c.corruption, clean_probability=1.0, hard_probability=0.0)
    rig = camera_fixture()
    world = world_fixture(c.joints)
    original = rig.K.copy()
    clean = corrupt_observations(world, rig, np.random.default_rng(8), cfg)
    assert_allclose(clean.estimated_rig.matrices, clean.true_rig.matrices)
    assert_allclose(rig.K, original)
    cfg = replace(cfg, clean_probability=0.0, camera_rotation_std_deg=1.0)
    noisy = corrupt_observations(world, rig, np.random.default_rng(8), cfg)
    assert not np.allclose(noisy.estimated_rig.matrices, noisy.true_rig.matrices)
    assert not np.shares_memory(noisy.estimated_rig.K, noisy.true_rig.K)


def test_checkpoint_semantics_reject_legacy_or_other_task():
    cfg = compose_recipe("plcs")
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    module = ResidualLightningModule(cfg)
    checkpoint: dict[str, Any] = {"hyper_parameters": {"config": cfg}}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    checkpoint["geometric_residual_contract"]["root_definition"] = "smpl_translation"
    with pytest.raises(ValueError, match="semantics"):
        module.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("task,joints", [("plcs", 17), ("blcs", 1)])
def test_supervised_residuals_reconstruct_gt_and_padding_does_not_change_loss(
    task, joints
):
    from src.tasks.base.triangulation_residual.contracts import CleanResidualScene
    from src.tasks.base.triangulation_residual.data import (
        ResidualDataset,
        collate_residual,
    )
    from src.tasks.base.triangulation_residual.losses import residual_loss

    cfg = validate_config(compose_recipe(task))
    scene = CleanResidualScene(
        "fixture",
        world_fixture(joints).astype(np.float32),
        30.0,
        camera_fixture(),
        "fixture",
    )
    ds = ResidualDataset([Path("fixture")], lambda _: scene, cfg, "val")
    batch = collate_residual([ds[0]])
    target = batch["target_world"]
    gt_root = target[:, :, cfg.root_indices].mean(2)
    output = {
        "root_residual" if task == "plcs" else "position_residual": gt_root
        - batch["root_init"]
    }
    if task == "plcs":
        output["relative_residual"] = (
            target - gt_root[:, :, None] - batch["relative_init"]
        )
    loss, _, world = residual_loss(output, batch, cfg.loss, task)
    assert loss < 1e-5
    torch.testing.assert_close(world, target, atol=2e-6, rtol=1e-6)
    corrupted = {k: v.clone() for k, v in output.items()}
    for value in corrupted.values():
        value[:, 16:] += 1000
    masked_loss, _, _ = residual_loss(corrupted, batch, cfg.loss, task)
    torch.testing.assert_close(loss, masked_loss)


@pytest.mark.parametrize("task,joints", [("plcs", 17), ("blcs", 1)])
def test_sliding_window_zero_residual_is_identity(task, joints):
    from src.tasks.base.triangulation_residual.inference import predict_geometry

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
        feature_config=FeatureConfig("raw", 1.0),
    )
    model = GeometricResidualModel(
        task,
        ModelConfig(
            f"{task}_triangulation_residual_v1", 32, 1, 4, 64, "swiglu", 0.0, 10000.0
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


def test_runner_tests_best_checkpoint_explicitly(tmp_path):
    from unittest.mock import Mock

    from pytorch_lightning.callbacks import ModelCheckpoint

    cfg = compose_recipe("blcs")
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    cfg.paths.output_root = str(tmp_path)
    module = ResidualLightningModule(cfg)
    module.residual_config.runtime.run.output_dir.mkdir(parents=True, exist_ok=True)
    cb = ModelCheckpoint(monitor="val/world_mpjpe_m")
    cb.best_model_path = str(tmp_path / "best.ckpt")
    cb.best_model_score = torch.tensor(0.1)
    Path(cb.best_model_path).touch()
    trainer = Mock()
    trainer.test.return_value = []
    dm = Mock()
    ResidualTrainingRunner().test_after_fit(trainer, module, dm, [cb])
    trainer.test.assert_called_once_with(
        module, datamodule=dm, ckpt_path=cb.best_model_path, weights_only=False
    )


def test_infeasible_camera_subset_is_replaced_without_changing_target(monkeypatch):
    import src.tasks.base.triangulation_residual.data as data_module
    from src.tasks.base.triangulation_residual.contracts import CleanResidualScene

    cfg = validate_config(compose_recipe("blcs"))
    cfg = replace(
        cfg,
        data=replace(cfg.data, min_views=2, max_views=2),
        corruption=replace(
            cfg.corruption,
            clean_probability=1.0,
            hard_probability=0.0,
            focal_scale_min=1.0,
            focal_scale_max=1.0,
        ),
    )
    scene = CleanResidualScene(
        "fixed",
        world_fixture(1).astype(np.float32),
        30.0,
        camera_fixture(),
        "fixed-source",
    )
    seen = []
    original = data_module.corrupt_observations

    def unavailable_first_subset(world, rig, rng, corruption):
        seen.append(tuple(sorted(tuple(center) for center in np.round(rig.centers, 5))))
        sample = original(world, rig, rng, corruption)
        if len(seen) == 1:
            sample.scores[:] = 0
            sample.observations_px[:] = np.nan
        return sample

    monkeypatch.setattr(data_module, "corrupt_observations", unavailable_first_subset)
    dataset = data_module.ResidualDataset(
        [Path("fixed")], lambda _: scene, cfg, "train"
    )
    sample = dataset[0]
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert sample["geometry_attempts"].item() == 2
    assert sample["corruption_rounds"].item() == 1
    assert sample["scene_id"] == "fixed"
    assert_allclose(sample["target_world"][:16], scene.world_m)


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_own_checkpoint_serialization_loads_dictconfig_explicitly(tmp_path, task):
    from lightning_fabric.utilities.cloud_io import _load

    cfg = compose_recipe(task)
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    module = ResidualLightningModule(cfg)
    checkpoint = {
        "hyper_parameters": {"config": cfg},
        "state_dict": module.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "own.ckpt"
    torch.save(checkpoint, path)
    restored = _load(path, map_location="cpu", weights_only=False)
    rebuilt = ResidualLightningModule(restored["hyper_parameters"]["config"])
    rebuilt.on_load_checkpoint(restored)
    rebuilt.load_state_dict(restored["state_dict"], strict=True)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key])


@pytest.mark.parametrize("joints,root", [(17, (11, 12)), (1, (0,))])
def test_asinh_changes_only_residual_features_and_preserves_missingness(joints, root):
    rig, _, observations, court = geometry_fixture(joints)
    observations[0, ..., 0] += 25
    observations[1, ..., 1] -= 25
    observations[2, ..., 0] -= 15
    scores = np.ones(observations.shape[:-1])
    scores[:, 5:9] = 0
    if joints == 17:
        scores[:, :, 9] = 0
    observations[scores == 0] = np.nan

    def geometry(encoding, scale):
        return prepare_geometry(
            observations,
            scores,
            court,
            np.ones((3, 14)),
            rig,
            root_indices=root,
            fps=30.0,
            feature_config=FeatureConfig(encoding, scale),
        )

    raw = geometry("raw", 1.0)
    encoded = geometry("asinh", 0.01)
    wider = geometry("asinh", 0.02)
    residual_slice = slice(4 * joints, 6 * joints)
    for result in (encoded, wider):
        np.testing.assert_array_equal(
            result.features[..., : 4 * joints], raw.features[..., : 4 * joints]
        )
        np.testing.assert_array_equal(
            result.features[..., 6 * joints :], raw.features[..., 6 * joints :]
        )
        for field in fields(raw):
            if field.name != "features":
                np.testing.assert_array_equal(
                    getattr(result, field.name), getattr(raw, field.name)
                )
    residual = raw.residual_uv.reshape(3, 16, -1)
    np.testing.assert_array_equal(raw.features[..., residual_slice], residual)
    values = encoded.features[..., residual_slice]
    assert_allclose(values, np.arcsinh(residual / 0.01), rtol=2e-7, atol=1e-8)
    assert_allclose(
        wider.features[..., residual_slice],
        np.arcsinh(residual / 0.02),
        rtol=2e-7,
        atol=1e-8,
    )
    np.testing.assert_array_equal(values[residual == 0], 0)
    np.testing.assert_array_equal(np.sign(values), np.sign(residual))
    assert (residual > 0).any() and (residual < 0).any()
    assert not np.allclose(values, residual / 0.01)
    assert not np.allclose(values, 2 * wider.features[..., residual_slice])
    assert not raw.init_valid[5:9].any()


@pytest.mark.parametrize("task", ["plcs", "blcs"])
@pytest.mark.parametrize("version", [1, 2])
def test_feature_ablation_preserves_dataset_rng_seeds_and_targets(task, version):
    from src.tasks.base.triangulation_residual.contracts import CleanResidualScene
    from src.tasks.base.triangulation_residual.data import ResidualDataset

    config = validate_config(compose_recipe(task, version))
    scene = CleanResidualScene(
        "paired",
        world_fixture(config.joints).astype(np.float32),
        30.0,
        camera_fixture(),
        "paired-source",
    )
    encoded_config = replace(config, features=FeatureConfig("asinh", 0.01))
    datasets = [
        ResidualDataset([Path("paired")], lambda _: scene, cfg, "train")
        for cfg in (config, encoded_config)
    ]
    for dataset in datasets:
        dataset.set_epoch(3)
    raw, encoded = (dataset[0] for dataset in datasets)
    assert raw.keys() == encoded.keys()
    for key in raw:
        if key == "scene_id":
            assert raw[key] == encoded[key]
        elif key != "features":
            torch.testing.assert_close(raw[key], encoded[key], rtol=0, atol=0)
    joints = config.joints
    torch.testing.assert_close(
        raw["features"][..., : 4 * joints],
        encoded["features"][..., : 4 * joints],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        raw["features"][..., 6 * joints :],
        encoded["features"][..., 6 * joints :],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        raw["features"][..., 4 * joints : 6 * joints],
        encoded["features"][..., 4 * joints : 6 * joints],
    )


@pytest.mark.parametrize("ffn_type", ["swiglu", "mlp"])
def test_config_selects_and_executes_both_camera_and_time_ffns(ffn_type):
    from src.utils.models.components.ffn_layers import MLP, SwiGLU

    config = compose_recipe("blcs")
    config.model.ffn_type = ffn_type
    model_config = replace(
        validate_config(config).model,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    model = GeometricResidualModel("blcs", model_config).eval()
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
            torch.randn(1, 3, 8, 79),
            torch.ones(1, 3, 8, dtype=torch.bool),
            torch.arange(8)[None].float(),
        )
    assert len(calls) == 2
    assert torch.isfinite(output["position_residual"]).all()
    assert torch.count_nonzero(output["position_residual"]) > 0


@pytest.mark.parametrize("task,joints", [("plcs", 17), ("blcs", 1)])
@pytest.mark.parametrize("encoding", ["legacy", "raw", "asinh"])
def test_official_clip_inference_uses_checkpoint_features_and_records_migration(
    tmp_path, monkeypatch, task, joints, encoding
):
    import json
    from contextlib import nullcontext

    from omegaconf import OmegaConf

    import src.tasks.base.triangulation_residual.inference as inference
    from src.tasks.base.triangulation_residual.contracts import RealResidualScene

    cfg = compose_recipe(task)
    cfg.model.hidden_dim = 32
    cfg.model.num_heads = 4
    cfg.model.ffn_dim = 64
    cfg.model.num_layers = 1
    if encoding == "asinh":
        cfg.features.residual_encoding = "asinh"
        cfg.features.residual_scale = 0.01
    module = ResidualLightningModule(cfg)
    checkpoint = {
        "hyper_parameters": {"config": OmegaConf.to_container(cfg, resolve=True)},
        "state_dict": module.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    if encoding == "legacy":
        marker = checkpoint["geometric_residual_contract"]
        marker["schema_version"] = 1
        del marker["ffn_type"], marker["features"]
        embedded = checkpoint["hyper_parameters"]["config"]
        del embedded["model"]["ffn_type"], embedded["features"]
    path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, path)
    original_bytes = path.read_bytes()
    rig, _, observations, court = geometry_fixture(joints)
    observations[0, ..., 0] += 25
    scene = RealResidualScene(
        observations,
        np.ones(observations.shape[:-1]),
        court,
        np.ones((3, 14)),
        rig,
        30.0,
        {"source": "fixture"},
    )
    monkeypatch.setattr(
        f"src.tasks.{task}.triangulation_residual.real_clip.load_real_clip",
        lambda _: [scene] if task == "plcs" else scene,
    )
    geometries = []

    def capture_geometry(*args, **kwargs):
        assert kwargs["feature_config"] == module.residual_config.features
        result = prepare_geometry(*args, **kwargs)
        geometries.append(result)
        return result

    monkeypatch.setattr(inference, "prepare_geometry", capture_geometry)
    expectation = (
        pytest.warns(UserWarning, match="Migrating legacy")
        if encoding == "legacy"
        else nullcontext()
    )
    with expectation:
        inference.evaluate_clip(
            task,
            path,
            tmp_path / "clip",
            tmp_path / "result",
            device="cpu",
            render=False,
        )
    assert path.read_bytes() == original_bytes
    metadata = json.loads((tmp_path / "result/predictions.metadata.json").read_text())
    assert metadata["contract"]["schema_version"] == 2
    assert metadata["contract"]["features"]["residual_encoding"] == (
        "asinh" if encoding == "asinh" else "raw"
    )
    if encoding == "legacy":
        assert (
            metadata["checkpoint_migration"]["source_contract"]
            == checkpoint["geometric_residual_contract"]
        )
    else:
        assert metadata["checkpoint_migration"] is None
    assert len(geometries) == 1
    with np.load(tmp_path / "result/predictions.npz") as result:
        np.testing.assert_array_equal(
            result["reprojection_residual_uv"][0], geometries[0].residual_uv
        )
        np.testing.assert_array_equal(
            result["preprojected_uv"][0], geometries[0].reprojected_uv
        )
