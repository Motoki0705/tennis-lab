"""BLCS family registration: real axial heads, physics loss and scene loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_model
from src.tasks.blcs.model_io.factory import compose_blcs_model_io
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)
from tests.support.model_contracts.inputs import (
    camera_observations,
    compose_tiny,
    write_splits,
)
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily, save_checkpoint
from tests.support.physics.ball_record import simulated_record

CONFIG_DIR = Path("src/tasks/blcs/configs")
SELECTORS = {
    "multiview_axial_small": "blcs_multiview_axial",
    "multiview_axial_base": "blcs_multiview_axial",
    "multiview_axial_large": "blcs_multiview_axial",
    "multiview_axial_xlarge": "blcs_multiview_axial",
}


def _compose(root: Path, gan: bool) -> DictConfig:
    return compose_tiny(
        "blcs",
        root,
        gan,
        [
            "model=multiview_axial_small",
            "model.camera_layers_per_stage=[1]",
            "model.time_layers_per_stage=[1]",
            "model.time_global_stage_mask=[false]",
            "model.max_num_cameras=2",
            "model.physics_head=true",
            "loss=physics",
            "loss.physics_start_step=0",
            "loss.physics_warmup_steps=0",
            "loss.parameters_weight=0.1",
            "loss.event_weight=0.1",
        ],
    )


def _write_dataset(root: Path) -> None:
    writer = BLCSDatasetWriter(root)
    for index, (frames, views) in enumerate(((3, 1), (5, 2))):
        position, record = simulated_record(frames)
        position_tensor = torch.from_numpy(position)
        velocity = torch.from_numpy(record.velocity)
        cameras = []
        for camera in range(views):
            obs = camera_observations(frames, camera, 1)
            cameras.append(
                CameraData(
                    camera_params=obs["camera_params"],
                    ball_uv=obs["uv"][:, 0],
                    ball_vis=obs["vis"][:, 0],
                    ball_visibility_ratio=1.0 - 1.0 / frames,
                    court_kp_uv=obs["court_kp_uv"],
                    court_kp_vis=obs["court_kp_vis"],
                    court_visibility_count=19.0,
                )
            )
        writer.save_scene(
            BLCSSceneData(
                scene_id=f"scene_{index}",
                initial_from_cell=0,
                initial_from_side="near",
                rally_length=1,
                end_reason="contract",
                winner_side=None,
                shots=[],
                ball_pos_world=position_tensor,
                ball_pos_norm=normalize_court_position(position_tensor),
                ball_vel_world=velocity,
                ball_vel_norm=normalize_court_velocity(velocity),
                cameras=cameras,
                num_cameras_sampled=views,
                physics_record=record,
                fps_out=30,
                sim_fps=120,
                physics_config_dict={},
                court_config_dict={},
                num_balls=1,
            )
        )
    writer.save_meta_json()
    write_splits(root)


def _build(config: DictConfig) -> tuple[Any, Any]:
    _write_dataset(Path(config.paths.data_root) / "sceneset")
    return _components(config)


def _components(config: DictConfig) -> tuple[Any, Any]:
    composition = compose_blcs_training(config, generator_config=None)
    return composition.lightning_module, composition.datamodule


def _restore(config: DictConfig, path: Path) -> BLCSLightningModule:
    saved = torch.load(path, map_location="cpu", weights_only=False)[
        "hyper_parameters"
    ]["config"]
    return cast(
        BLCSLightningModule,
        BLCSLightningModule.load_from_checkpoint(
            path,
            model_io=compose_blcs_model_io(saved),
            map_location="cpu",
            strict=True,
            weights_only=False,
        ),
    )


def _batch(config: DictConfig) -> Any:
    samples = []
    root = Path(config.paths.data_root) / "sceneset"
    for index, (frames, views) in enumerate(((3, 1), (5, 2))):
        exact = OmegaConf.merge(
            config,
            {
                "data": {
                    "seq_len_range": [frames, frames],
                    "num_views_range": [views, views],
                }
            },
        )
        split = f"exact_{index}.txt"
        (root / split).write_text(f"scene_{index}\n")
        dataset = BallTrajectoryDataset(
            scene_dir=root, split_file=split, config=exact, augment=False
        )
        samples.append(dataset[0])
    return collate_multiview_trajectories(samples)


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    # All heads are supervised in this physics profile; no known-unused names.
    generator = names_under(module, "model")
    contracts = [
        OptimizerContract(
            "generator",
            generator,
            {
                "embedding": names_under(module, "model.group_embed"),
                "axial": names_under(module, "model.stages"),
                "position": names_under(module, "model.output_head.position"),
                "velocity": names_under(module, "model.output_head.velocity"),
                "physics/event": names_under(module, "model.physics_head"),
            },
        )
    ]
    if module.gan_enabled:
        discriminator = names_under(module, "discriminator")
        contracts.append(
            OptimizerContract(
                "discriminator",
                discriminator,
                {
                    "trunk": names_under(module, "discriminator.blocks"),
                    "head": names_under(module, "discriminator.head"),
                },
            )
        )
    return ParameterContract(
        frozenset(dict(module.named_parameters())), frozenset(), frozenset()
    ), tuple(contracts)


def _assert_batch(batch: Any) -> None:
    assert batch["ball_uv"].shape == (2, 2, 5, 2)
    assert batch["ball_uv"].dtype == torch.float32
    pad = batch["padding_mask"]
    assert pad.dtype == torch.bool
    assert not pad[0, 0, :3].any() and pad[0, 0, 3:].all()
    assert pad[0, 1].all() and not pad[1].any()
    assert not batch["ball_vis"][:, :, 1].any()  # missing observation, still supervised
    assert batch["position_3d"].shape == (2, 5, 3)


def _assert_prediction(output: Any) -> None:
    for tensor, shape in (
        (output.position, (2, 5, 3)),
        (output.velocity, (2, 5, 3)),
        (output.physics_parameters, (2, 5, 7)),
        (output.event_logits, (2, 5, 1)),
    ):
        assert tensor is not None and tensor.shape == shape
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()


def _supervised_result(module: Any, batch: Any) -> Any:
    result = module._compute_supervised_result(batch, "val")
    # Ground truth supervises missing observations, while added time padding is excluded.
    assert result["prepared"].loss_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    return result


def _predict(module: Any, batch: Any) -> Any:
    normalized = module.model_io.run(batch)
    predictor = BLCSPredictor(module.model_io, torch.device("cpu"))
    physical = predictor.predict_batch(batch, denormalize=True)
    assert physical.coordinates_in_metres
    assert physical.physics_parameters is not None and physical.event_logits is not None
    return dict(
        normalized=normalized,
        physical=physical,
        padding_mask=batch["padding_mask"],
        ball_vis=batch["ball_vis"],
    )


def _predict_artifact(path: Path, batch: Any) -> Any:
    binding, _ = load_checkpoint_model(path)
    binding.model.eval()
    return _predict(SimpleNamespace(model_io=binding), batch)


AXIAL = ModelFamily(
    name="blcs_multiview_axial",
    compose=_compose,
    build=_build,
    rebuild=_components,
    training_state=lambda module: (
        module.gan_training.state_dict() if module.gan_training is not None else None
    ),
    batch=_batch,
    supervised_result=_supervised_result,
    predict=_predict,
    predict_artifact=_predict_artifact,
    save=save_checkpoint,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
