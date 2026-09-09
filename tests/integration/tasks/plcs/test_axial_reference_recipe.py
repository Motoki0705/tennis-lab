"""Recipe composition, strict reference boundary and checkpoint roundtrip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.tasks.plcs.model_io.axial_reference import (
    validate_axial_reference_checkpoint,
    write_axial_reference_checkpoint,
)
from src.tasks.plcs.model_io.factory import build_plcs_model_io

ROOT = Path(__file__).resolve().parents[4]


def recipe() -> DictConfig:
    with initialize_config_dir(
        config_dir=str(ROOT / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        return compose(
            config_name="train_axial_reference",
            overrides=[
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.rope_dim=8",
                "model.num_layers=1",
                "model.ffn_dim=64",
            ],
        )


def test_recipe_and_reference_adapter() -> None:
    config = recipe()
    assert config.model.canonical_pose_readout == "temporal_decomposition"
    assert config.loss.rotation_weight == config.loss.angle_weight == 0.1
    assert config.loss.reprojection_weight == 1.0
    assert config.training.compile.enabled
    assert config.run.seed == 42
    runtime = PLCSTrainingConfig.from_config(config)
    binding = build_plcs_model_io(runtime)
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = tuple(
        build_court_view_record(
            camera_id=f"camera_{i}",
            camera_center_court_m=(10.0, -20.0 if i < 2 else 20.0, 3.0),
            contract=contract,
        )
        for i in range(4)
    )
    provenance = build_reference_frame_provenance(views, reference_camera_id="camera_2")
    batch: dict[str, Any] = dict(
        human_kp=torch.rand(1, 4, 8, 17, 2),
        court_kp=torch.rand(1, 4, 8, 14, 2),
        human_vis=torch.ones(1, 4, 8, 17),
        court_vis=torch.ones(1, 4, 8, 14),
        padding_mask=torch.zeros(1, 4, 8, dtype=torch.bool),
        court_keypoint_metadata=court_keypoint_contract_document(contract),
        court_reference_provenance=provenance,
        reference_view_index=torch.tensor([2]),
        position=torch.zeros(1, 8, 3),
        rotation=torch.tensor([1.0, 0.0]).expand(1, 8, 2),
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            tuple(v.camera_id for v in views)
        ),
        selected_views=views,
        reference_camera_id="camera_2",
    )
    batch.update(
        view_camera_ids=torch.tensor([[0, 1, 2, 3]]),
        reference_camera_id=torch.tensor([2]),
        reference_from_physical=torch.tensor(
            provenance.reference_from_physical
        ).unsqueeze(0),
        physical_from_reference=torch.tensor(
            provenance.physical_from_reference
        ).unsqueeze(0),
        reference_view_selection=(selection,),
        stable_camera_id_table=(selection.stable_camera_id_table,),
    )
    prepared = binding.adapter.prepare_training_batch(batch)
    output = binding.model(**prepared.call.kwargs)
    assert output["canonical_pose"].shape == (1, 8, 17, 3)
    batch["reference_view_index"] = torch.tensor([1])
    with pytest.raises(ValueError, match="provenance"):
        binding.adapter.build_call(batch)
    batch["reference_view_index"] = torch.tensor([2])
    batch["padding_mask"][0, 2, 3] = True
    with pytest.raises(ValueError, match="reference context"):
        binding.adapter.build_call(batch)
    del batch["reference_view_index"]
    with pytest.raises(ValueError, match="reference_view_index"):
        binding.adapter.build_call(batch)


def test_checkpoint_contract_rejects_wrong_family_or_missing_marker() -> None:
    checkpoint: dict[str, object] = {}
    name = "plcs_multiview_axial_reference"
    with pytest.raises(ValueError, match="checkpoint contract"):
        validate_axial_reference_checkpoint(checkpoint, model_name=name)
    write_axial_reference_checkpoint(checkpoint, model_name=name)
    validate_axial_reference_checkpoint(checkpoint, model_name=name)
    with pytest.raises(ValueError, match="exact reference model"):
        validate_axial_reference_checkpoint(
            checkpoint, model_name="plcs_multiview_axial"
        )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_generation_recipe_selects_four_corners(task: str) -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        config = compose(config_name="generate_dataset_camera_view_v2")
    assert list(config.camera.fixed_camera_indices) == [0, 1, 2, 3]
    assert config.run.output_dir == f"{task}/single_object_camera_view_v2"
    if task == "plcs":
        runtime = PLCSGenerationConfig.from_config(config)
        assert runtime.split_group == "motion_source"
    else:
        build_generator_config(config)


def test_motion_group_splits_keep_sources_disjoint(tmp_path: Path) -> None:
    import json

    from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter

    writer = PLCSDatasetWriter(
        tmp_path,
        court_keypoint_contract=resolve_court_keypoint_contract("camera_view_v2"),
    )
    writer.scene_records = [
        dict(file=f"scene_{i}", motion_source=f"motion_{i // 3}") for i in range(60)
    ]
    writer.save_motion_group_splits(val_ratio=0.1, test_ratio=0.1, seed=42)
    info = json.loads((tmp_path / "split_info.json").read_text())
    assert info["n_scenes"] == {"train": 48, "val": 6, "test": 6}
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for file in (tmp_path / f"{split}.txt").read_text().splitlines():
            source = f"motion_{int(file.split('_')[1]) // 3}"
            assert source not in seen or seen[source] == split
            seen[source] = split
    assert len(seen) == 20


@pytest.mark.parametrize("field,value", [("target_frame_contract", "physical_court_v1"), ("axial_rope_contract", "time_camera_role_v1"), ("reference_selector_mode", "disabled")])
def test_recipe_rejects_mismatched_reference_semantics(field: str, value: str) -> None:
    config = recipe()
    config.model[field] = value
    with pytest.raises(ValueError, match="axial reference contract"):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("indices", [[0, 0, 1, 2], [0, 1], [-1, 0, 1, 2]])
def test_recipe_rejects_invalid_sampling_pool(indices: list[int]) -> None:
    config = recipe()
    config.data.camera_candidates = indices
    with pytest.raises(ValueError, match="camera"):
        PLCSTrainingConfig.from_config(config)


def test_lightning_checkpoint_roundtrip_and_weight_initialization(tmp_path: Path) -> None:
    import pytorch_lightning as pl

    from src.tasks.plcs.training.lightning_module import PLCSLightningModule
    from src.tasks.plcs.training.runner import PLCSTrainingRunner

    config = recipe()
    module = PLCSLightningModule(config)
    checkpoint: dict[str, Any] = {
        "state_dict": module.state_dict(),
        "hyper_parameters": {"config": config},
        "pytorch-lightning_version": pl.__version__,
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "reference.ckpt"
    torch.save(checkpoint, path)
    restored = PLCSLightningModule.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    assert type(restored.model) is type(module.model)
    for key, tensor in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)
    config.paths.checkpoint_root = str(tmp_path)
    config.run.init_weights = path.name
    runtime = PLCSTrainingRunner().validate_runtime_config(config)
    PLCSTrainingRunner().maybe_load_init_weights(runtime, restored)
    del checkpoint["axial_reference"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="checkpoint contract"):
        PLCSLightningModule.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="checkpoint contract"):
        PLCSTrainingRunner().maybe_load_init_weights(runtime, restored)
