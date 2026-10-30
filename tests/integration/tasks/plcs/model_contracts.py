"""PLCS family registration: direct/temporal canonical heads and real loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.schema.court_normalization import normalize_court_position
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

CONFIG_DIR = Path("src/tasks/plcs/configs")
SELECTORS = {
    "multiview_axial_small": "plcs_multiview_axial",
    "multiview_axial_base": "plcs_multiview_axial",
    "multiview_axial_large": "plcs_multiview_axial",
    "multiview_axial_xlarge": "plcs_multiview_axial",
    "multiview_axial_base_temporal_pose": "plcs_multiview_axial",
}


def _compose(root: Path, gan: bool, *, temporal: bool = False) -> DictConfig:
    return compose_tiny(
        "plcs",
        root,
        gan,
        [
            f"model={'multiview_axial_base_temporal_pose' if temporal else 'multiview_axial_small'}",
            "model.max_views=2",
            "data.min_cameras=1",
            "data.augmentation.frame_rate.enabled=false",
            "loss=all_outputs_beta01",
        ],
    )


def _write_dataset(root: Path) -> None:
    contract = resolve_court_keypoint_contract("physical_v1")
    writer = PLCSDatasetWriter(root, court_keypoint_contract=contract)
    writer.save_meta_json()
    for index, (frames, views) in enumerate(((3, 1), (5, 2))):
        time: np.ndarray = np.arange(frames, dtype=np.float32)
        position_m = np.stack(
            (0.2 + time * 0.1, -2.0 + time * 0.3, np.ones(frames, dtype=np.float32)),
            axis=-1,
        )
        canonical: np.ndarray = np.linspace(
            -0.5, 0.8, frames * 17 * 3, dtype=np.float32
        ).reshape(frames, 17, 3)
        # Asymmetric moving joints, with a root/yaw that cannot mask a detached head.
        yaw = time * 0.1 + 0.3
        rotation = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
        world = canonical.copy()
        world[..., 0] = (
            canonical[..., 0] * rotation[:, :1] - canonical[..., 1] * rotation[:, 1:]
        )
        world[..., 1] = (
            canonical[..., 0] * rotation[:, 1:] + canonical[..., 1] * rotation[:, :1]
        )
        world += position_m[:, None]
        cameras = []
        for camera in range(views):
            obs = camera_observations(frames, camera, 17)
            params = obs["camera_params"]
            cameras.append(
                CameraData(
                    camera_params=params,
                    human_kp_uv=obs["uv"],
                    human_kp_vis=obs["vis"],
                    court_kp_uv=np.broadcast_to(
                        obs["court_kp_uv"], (frames, 20, 2)
                    ).copy(),
                    court_kp_vis=np.broadcast_to(
                        obs["court_kp_vis"], (frames, 20)
                    ).copy(),
                    human_visibility_ratio=1.0 - 1.0 / frames,
                    court_visibility_count=19.0,
                    court_view=build_court_view_record(
                        camera_id=f"camera_{camera}",
                        camera_center_court_m=params["C"],
                        contract=contract,
                    ),
                )
            )
        writer.save_scene(
            SceneData(
                meta={
                    "scene_id": f"scene_{index}",
                    "motion_source": f"contract_{index}",
                    "motion_category": "contract",
                    "gender": "neutral",
                    "fps": 30.0,
                    "num_frames": frames,
                    "initial_position": [0.2, -2.0],
                    "initial_yaw": 0.3,
                    "num_cameras_sampled": views,
                },
                position=normalize_court_position(position_m),
                rotation=rotation,
                canonical_pose_3d=canonical,
                human_kp_3d=world,
                cameras=cameras,
                num_persons=1,
                court_keypoint_contract=contract,
            )
        )
    writer.save_meta_json()
    write_splits(root)


def _build(config: DictConfig) -> tuple[Any, Any]:
    _write_dataset(Path(config.paths.data_root) / "sceneset")
    return build_plcs_lightning_module(config), build_plcs_datamodule(config)


def _restore(config: DictConfig, path: Path) -> PLCSLightningModule:
    return PLCSLightningModule.load_from_checkpoint(
        path,
        config=config,
        map_location="cpu",
        strict=True,
        weights_only=False,
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
        samples.append(
            SceneDataset(scene_dir=root, split_file=split, config=exact, augment=False)[
                0
            ]
        )
    return collate_plcs_batch(samples)


def _parameters(module: Any) -> tuple[ParameterContract, tuple[OptimizerContract, ...]]:
    generator = names_under(module, "model")
    groups = {
        "embedding": names_under(module, "model.group_embed"),
        "camera attention": names_under(module, "model.camera_layers"),
        "time attention": names_under(module, "model.time_layers"),
        "position": names_under(module, "model.position_head"),
        "yaw": names_under(module, "model.rotation_head"),
    }
    if module.model.canonical_pose_readout == "temporal_decomposition":
        groups["static canonical"] = names_under(
            module, "model.canonical_pose_head.static_head"
        )
        groups["motion canonical"] = names_under(
            module, "model.canonical_pose_head.motion_head"
        )
    else:
        groups["canonical"] = names_under(module, "model.canonical_pose_head")
    contracts = [OptimizerContract("generator", generator, groups)]
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
    assert batch["human_kp"].shape == (2, 2, 5, 17, 2)
    assert batch["human_kp"].dtype == torch.float32
    pad = batch["padding_mask"]
    assert pad.dtype == torch.bool
    assert not pad[0, 0, :3].any() and pad[0, 0, 3:].all()
    assert pad[0, 1].all() and not pad[1].any()
    assert not batch["human_vis"][:, :, 1].any()
    assert batch["human_kp_3d"].shape == (2, 5, 17, 3)


def _assert_prediction(output: Any) -> None:
    for tensor, shape in (
        (output.position, (2, 5, 3)),
        (output.rotation, (2, 5, 2)),
        (output.canonical_pose, (2, 5, 17, 3)),
    ):
        assert tensor is not None and tensor.shape == shape
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()


def _supervised_result(module: Any, batch: Any) -> Any:
    result = module._compute_supervised_result(batch, "val")
    assert (~result["prepared"].target_padding_mask.all(dim=1)).tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    return result


AXIAL = ModelFamily(
    name="plcs_multiview_axial",
    compose=_compose,
    build=_build,
    batch=_batch,
    supervised_result=_supervised_result,
    predict=lambda module, batch: module.model_io.run(batch),
    save=save_checkpoint,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
TEMPORAL = ModelFamily(
    name="plcs_multiview_axial/temporal_decomposition",
    compose=lambda root, gan: _compose(root, gan, temporal=True),
    build=_build,
    batch=_batch,
    supervised_result=AXIAL.supervised_result,
    predict=AXIAL.predict,
    save=AXIAL.save,
    restore=_restore,
    parameters=_parameters,
    assert_batch=_assert_batch,
    assert_prediction=_assert_prediction,
)
