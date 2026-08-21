"""Shape and validation tests for the SLCS fusion model."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from src.tasks.slcs.models.slcs_model import SLCSFusionModel


def _model(
    *,
    num_shared_layers: int,
    num_position_layers: int = 0,
    num_rotation_layers: int = 0,
) -> SLCSFusionModel:
    return SLCSFusionModel(
        hidden_dim=32,
        num_shared_layers=num_shared_layers,
        num_position_layers=num_position_layers,
        num_rotation_layers=num_rotation_layers,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        rope_dim=8,
        rope_theta_time=10000.0,
        rope_theta_entity=10000.0,
        attention_type="mha",
        ffn_type="swiglu",
        num_players=2,
        num_court_kp=14,
        max_seq_len=8,
        invisible_init_std=0.02,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
        dino_patch_downsample_factor=1,
        dino_cross_attn_every=1,
        log_b_min=-6.0,
        log_b_max=3.0,
    )


def _inputs() -> dict[str, torch.Tensor]:
    batch, players, frames, joints, court_kp = 2, 2, 8, 17, 14
    return {
        "player_kp": torch.rand(batch, players, frames, joints, 2),
        "player_kp_vis": torch.ones(batch, players, frames, joints),
        "player_valid": torch.ones(batch, players, frames, dtype=torch.bool),
        "ball_uv": torch.rand(batch, frames, 2),
        "ball_vis": torch.ones(batch, frames, dtype=torch.bool),
        "court_kp": torch.rand(batch, frames, court_kp, 2),
        "court_vis": torch.ones(batch, frames, court_kp),
        "padding_mask": torch.zeros(batch, frames, dtype=torch.bool),
        "dino_tokens": torch.rand(batch, 2, 12, 8),
        "dino_frame_idx": torch.tensor([[0, 7], [0, 7]]),
        "dino_padding_mask": torch.zeros(batch, 2, dtype=torch.bool),
    }


def test_forward_shapes_and_finite_rotation_pairs() -> None:
    model = _model(num_shared_layers=2)
    output = model(**_inputs())
    assert output["player_position"].shape == (2, 2, 8, 3)
    assert output["player_rotation"].shape == (2, 2, 8, 2)
    assert output["ball_position"].shape == (2, 8, 3)
    assert output["player_position_log_b"].shape == (2, 2, 8)
    assert output["ball_position_log_b"].shape == (2, 8)
    assert torch.isfinite(output["player_rotation"]).all()
    assert (torch.linalg.vector_norm(output["player_rotation"], dim=-1) > 0).all()


def test_fully_split_trunks_isolate_position_and_rotation_gradients() -> None:
    model = _model(
        num_shared_layers=0,
        num_position_layers=1,
        num_rotation_layers=1,
    )
    model(**_inputs())["player_position"].sum().backward()

    assert any(
        parameter.grad is not None for parameter in model.position_entity_layers.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.rotation_entity_layers.parameters()
    )


def test_all_shared_configuration_has_no_task_trunk_parameters() -> None:
    model = _model(
        num_shared_layers=2,
        num_position_layers=0,
        num_rotation_layers=0,
    )

    assert len(model.entity_layers) == 2
    assert len(model.position_entity_layers) == 0
    assert len(model.rotation_entity_layers) == 0
    assert not isinstance(model.final_norm, nn.Identity)
    assert isinstance(model.position_final_norm, nn.Identity)
    assert isinstance(model.rotation_final_norm, nn.Identity)


def test_public_forward_accepts_only_raw_observations_and_padding_masks() -> None:
    assert list(inspect.signature(_model(num_shared_layers=1).forward).parameters) == [
        "player_kp",
        "player_kp_vis",
        "player_valid",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "dino_tokens",
        "dino_frame_idx",
        "dino_padding_mask",
    ]


def test_batch_with_all_dino_padding_is_finite() -> None:
    model = _model(num_shared_layers=1)
    inputs = _inputs()
    inputs["dino_tokens"].zero_()
    inputs["dino_padding_mask"].fill_(True)

    output = model(**inputs)

    assert torch.isfinite(output["player_position"]).all()


def test_padding_values_cannot_change_real_frame_outputs() -> None:
    torch.manual_seed(12)
    model = _model(num_shared_layers=2).eval()
    inputs = _inputs()
    inputs["padding_mask"][:, -2:] = True
    inputs["player_valid"][:, :, -2:] = False
    inputs["player_kp_vis"][:, :, -2:] = 0.0
    inputs["ball_vis"][:, -2:] = False
    inputs["court_vis"][:, -2:] = 0.0
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["player_kp"][:, :, -2:] = 10_000.0
    changed["ball_uv"][:, -2:] = -10_000.0
    changed["court_kp"][:, -2:] = 10_000.0

    with torch.no_grad():
        baseline = model(**inputs)
        modified = model(**changed)

    real_frames = ~inputs["padding_mask"]
    real_player_frames = real_frames.unsqueeze(1).expand(2, 2, 8)
    for key in (
        "player_position",
        "player_rotation",
        "player_position_log_b",
        "player_rotation_log_b",
    ):
        torch.testing.assert_close(
            baseline[key][real_player_frames], modified[key][real_player_frames]
        )
    for key in ("ball_position", "ball_position_log_b"):
        torch.testing.assert_close(
            baseline[key][real_frames], modified[key][real_frames]
        )


def test_dino_padding_values_cannot_change_outputs() -> None:
    torch.manual_seed(23)
    model = _model(num_shared_layers=1).eval()
    inputs = _inputs()
    inputs["dino_padding_mask"][:, 1] = True
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["dino_tokens"][:, 1] = 10_000.0

    with torch.no_grad():
        baseline = model(**inputs)
        modified = model(**changed)

    for key in baseline:
        torch.testing.assert_close(baseline[key], modified[key])
