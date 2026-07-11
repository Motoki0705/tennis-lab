"""Shape and validation tests for the SLCS fusion model."""

from __future__ import annotations

import pytest
import torch

from src.tasks.slcs.models.slcs_model import SLCSFusionModel


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
        "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
        "dino_tokens": torch.rand(batch, 2, 12, 8),
        "dino_frame_idx": torch.tensor([[0, 7], [0, 7]]),
        "dino_valid": torch.ones(batch, 2, dtype=torch.bool),
    }


def test_forward_shapes_and_finite_rotation_pairs() -> None:
    model = SLCSFusionModel(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_seq_len=8,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
        dino_cross_attn_every=1,
    )
    output = model(**_inputs())
    assert output["player_position"].shape == (2, 2, 8, 3)
    assert output["player_rotation"].shape == (2, 2, 8, 2)
    assert output["ball_position"].shape == (2, 8, 3)
    assert output["player_position_log_b"].shape == (2, 2, 8)
    assert output["ball_position_log_b"].shape == (2, 8)
    assert torch.isfinite(output["player_rotation"]).all()
    assert (torch.linalg.vector_norm(output["player_rotation"], dim=-1) > 0).all()


def test_sparse_dino_inputs_are_all_or_none() -> None:
    model = SLCSFusionModel(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=8,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
    )
    inputs = _inputs()
    inputs.pop("dino_frame_idx")
    with pytest.raises(ValueError, match="provided together"):
        model(**inputs)
