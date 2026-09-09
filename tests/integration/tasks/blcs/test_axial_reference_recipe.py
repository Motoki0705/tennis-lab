"""BLCS reference selection, padding, compilation and persistence contracts."""

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
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tasks.blcs.model_io.factory import compose_blcs_trajectory_model_io
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.utils.models import precompute_freqs_cis_nd

ROOT = Path(__file__).resolve().parents[4]


def recipe(layers: int = 1) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(ROOT / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        return compose(
            config_name="train_axial_reference",
            overrides=[
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.rope_dim=8",
                f"model.num_layers={layers}",
                "model.ffn_dim=64",
                "model.dropout=0.0",
                f"model.camera_layers_per_stage={[1] * layers}",
                f"model.time_layers_per_stage={[1] * layers}",
                f"model.time_global_stage_mask={str([False] * layers).lower()}",
                "model.max_seq_len=128",
            ],
        )


def batch(views: int = 4, reference: int = 2) -> dict[str, Any]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    records = tuple(
        build_court_view_record(
            camera_id=f"cam_{i}",
            camera_center_court_m=(10.0, 20.0 if i < 2 else -20.0, 3.0),
            contract=contract,
        )
        for i in range(views)
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            tuple(f"cam_{i}" for i in range(4))
        ),
        selected_views=records,
        reference_camera_id=f"cam_{reference}",
    )
    fields = {
        k: v.unsqueeze(0)
        for k, v in selection.to_tensor_fields(dtype=torch.float32).items()
    }
    fields["physical_from_reference"] = fields["reference_from_physical"].transpose(
        -1, -2
    )
    return dict(
        ball_uv=torch.rand(1, views, 8, 2),
        ball_vis=torch.ones(1, views, 8, dtype=torch.bool),
        court_kp=torch.rand(1, views, 8, 14, 2),
        court_vis=torch.ones(1, views, 8, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, views, 8, dtype=torch.bool),
        reference_view_selection=(selection,),
        stable_camera_id_table=(selection.stable_camera_id_table,),
        court_reference_provenance=(selection.provenance,),
        **fields,
    )


def test_recipe_and_selector_frequencies() -> None:
    config = recipe()
    validate_training_boundary(config)
    assert config.data.num_court_kp == config.model.num_court_tokens == 14
    assert config.court_keypoints.selector == "camera_view_v2"
    assert config.data.batch_size == 16
    assert config.run.seed == 42 and config.training.compile.enabled
    assert config.loss.reprojection_weight == 1
    assert config.data.num_views_range == [3, 4]
    model = compose_blcs_trajectory_model_io(config).model
    for reference in range(4):
        positions = model._build_token_positions(seq_len=128, n_cams=4)
        selector = (torch.arange(4) != reference)[None, :, None].expand(128, 4, 1)
        expected = precompute_freqs_cis_nd(
            8,
            torch.cat((positions, selector.long()), -1),
            base=(10000.0, 1000.0, 1000.0),
        )
        torch.testing.assert_close(model.token_freqs_cis[reference], expected)


def test_selected_reference_readout_matches_embedding() -> None:
    binding = compose_blcs_trajectory_model_io(recipe(0))
    model = binding.model.eval()
    inputs = batch()
    call = binding.build_call(inputs)
    with torch.no_grad():
        expected = model.group_embed(
            inputs["court_kp"][:, 2].reshape(8, 14, 2),
            inputs["ball_uv"][:, 2].reshape(8, 2),
            inputs["ball_vis"][:, 2].reshape(8),
        ).reshape(1, 8, 32)
        expected = model.output_head(model.final_norm(expected))["position"]
        actual = binding.execute_call(call)["position"]
    torch.testing.assert_close(actual, expected)


def test_padding_isolation_and_finite_backward() -> None:
    binding = compose_blcs_trajectory_model_io(recipe())
    model = binding.model.eval()
    inputs = batch(3)
    call = dict(binding.build_call(inputs).kwargs)
    expected = model(**call)["position"]
    padded = dict(call)
    for key in ("ball_uv", "ball_vis", "court_kp", "court_vis", "padding_mask"):
        value = call[key]
        extra = torch.full_like(value[:, :1], True if key == "padding_mask" else 0)
        padded[key] = torch.cat((value, extra), dim=1)
    padded["ball_uv"][:, 3] = 999
    actual = model(**padded)["position"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("mutation", ["missing", "mismatch", "padding", "two_views"])
def test_reference_boundary_rejects_invalid_identity(mutation: str) -> None:
    binding = compose_blcs_trajectory_model_io(recipe())
    inputs = batch(2, 1) if mutation == "two_views" else batch()
    if mutation == "missing":
        del inputs["reference_view_index"]
    if mutation == "mismatch":
        inputs["reference_view_index"] = torch.tensor([1])
    if mutation == "padding":
        inputs["padding_mask"][0, 2, 3] = True
    with pytest.raises((ValueError, TypeError)):
        binding.build_call(inputs)


def test_compiled_three_view_forward_matches_eager() -> None:
    binding = compose_blcs_trajectory_model_io(recipe())
    model = binding.model.eval()
    inputs = binding.build_call(batch(3)).kwargs
    with torch.no_grad():
        expected = model(**inputs)
        actual = torch.compile(model, backend="eager", fullgraph=True)(**inputs)
    torch.testing.assert_close(actual["position"], expected["position"])


def test_checkpoint_roundtrip_and_marker_rejection(tmp_path: Path) -> None:
    config = recipe()
    binding = compose_blcs_trajectory_model_io(config)
    module = BLCSLightningModule(config, model_io=binding)
    checkpoint: dict[str, Any] = {
        "hyper_parameters": {"config": config},
        "state_dict": module.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    runtime = load_checkpoint_runtime(path)
    restored = BLCSLightningModule(
        config, model_io=compose_blcs_trajectory_model_io(runtime.config)
    )
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint["axial_reference"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="axial reference checkpoint"):
        load_checkpoint_runtime(path)
    with pytest.raises(ValueError, match="axial reference checkpoint"):
        restored.on_load_checkpoint(checkpoint)


def test_invisible_out_of_frame_uv_is_valid_but_visible_uv_is_rejected() -> None:
    binding = compose_blcs_trajectory_model_io(recipe())
    inputs = batch()
    inputs["ball_uv"][0, 0, 0] = torch.tensor([-0.2, 1.2])
    inputs["ball_vis"][0, 0, 0] = False
    binding.build_call(inputs)
    inputs.update(
        position_3d=torch.zeros(1, 8, 3),
        velocity_3d=torch.zeros(1, 8, 3),
        camera_R=torch.eye(3).expand(1, 4, 3, 3),
        camera_C=torch.zeros(1, 4, 3),
        camera_f=torch.ones(1, 4),
        camera_cx=torch.ones(1, 4),
        camera_cy=torch.ones(1, 4),
        camera_w=torch.ones(1, 4),
        camera_h=torch.ones(1, 4),
        ball_uv_target=inputs["ball_uv"].clone(),
        ball_vis_target=inputs["ball_vis"].clone(),
    )
    binding.adapter.build_training_batch(inputs)
    inputs["ball_vis_target"][0, 0, 0] = True
    with pytest.raises(ValueError, match="target_uv"):
        binding.adapter.build_training_batch(inputs)
    inputs["ball_vis_target"][0, 0, 0] = False
    inputs["ball_uv_target"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        binding.adapter.build_training_batch(inputs)
    inputs["ball_vis"][0, 0, 0] = True
    with pytest.raises(ValueError, match="ball_uv"):
        binding.build_call(inputs)


def test_kp14_recipe_rejects_twenty_point_model_inputs() -> None:
    binding = compose_blcs_trajectory_model_io(recipe())
    inputs = batch()
    inputs["court_kp"] = torch.rand(1, 4, 8, 20, 2)
    inputs["court_vis"] = torch.ones(1, 4, 8, 20, dtype=torch.bool)
    with pytest.raises(ValueError, match="14"):
        binding.build_call(inputs)
