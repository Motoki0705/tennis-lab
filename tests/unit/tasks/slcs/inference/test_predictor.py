"""Unit tests for the typed SLCS predictor boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.model_io import (
    SLCSDecodedOutput,
    SLCSModelIOAdapter,
    SLCSModelIOSpec,
    SLCSRawOutput,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.tasks.slcs.training.lightning_module import SLCSLightningModule


def _model() -> SLCSFusionModel:
    return SLCSFusionModel(
        hidden_dim=32,
        num_shared_layers=1,
        num_position_layers=0,
        num_rotation_layers=0,
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
        max_seq_len=4,
        invisible_init_std=0.02,
        dino_embed_dim=8,
        dino_grid_h=3,
        dino_grid_w=4,
        dino_patch_downsample_factor=1,
        dino_cross_attn_every=1,
        log_b_min=-6.0,
        log_b_max=3.0,
    )


def _adapter() -> SLCSModelIOAdapter:
    return SLCSModelIOAdapter(
        SLCSModelIOSpec(
            num_players=2,
            num_court_kp=14,
            max_seq_len=4,
            dino_num_tokens=12,
            dino_encoded_num_tokens=12,
            dino_embed_dim=8,
            log_b_min=-6.0,
            log_b_max=3.0,
        )
    )


def _batch() -> dict[str, Tensor]:
    batch_size, players, frames, joints, court = 1, 2, 4, 17, 14
    return {
        "player_kp": torch.rand(batch_size, players, frames, joints, 2),
        "player_kp_vis": torch.ones(batch_size, players, frames, joints),
        "player_valid": torch.ones(
            batch_size, players, frames, dtype=torch.bool
        ),
        "ball_uv": torch.rand(batch_size, frames, 2),
        "ball_vis": torch.ones(batch_size, frames, dtype=torch.bool),
        "court_kp": torch.rand(batch_size, frames, court, 2),
        "court_vis": torch.ones(batch_size, frames, court),
        "frame_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "dino_tokens": torch.rand(batch_size, 2, 12, 8),
        "dino_frame_idx": torch.tensor([[0, frames - 1]]),
        "dino_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "target_player_position": torch.rand(batch_size, players, frames, 3),
        "target_player_rotation": torch.nn.functional.normalize(
            torch.rand(batch_size, players, frames, 2), dim=-1
        ),
        "target_player_valid": torch.ones(
            batch_size, players, frames, dtype=torch.bool
        ),
        "target_player_weight": torch.ones(batch_size, players, frames),
        "target_ball_position": torch.rand(batch_size, frames, 3),
        "target_ball_valid": torch.ones(batch_size, frames, dtype=torch.bool),
        "target_ball_weight": torch.ones(batch_size, frames),
    }


class _TypedLightningFixture(nn.Module):
    """Minimal typed surface consumed by ``SLCSPredictor``."""

    def __init__(
        self,
        model: SLCSFusionModel,
        adapter: SLCSModelIOAdapter,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_adapter = adapter
        self.model_io: BoundModelIO[
            Mapping[str, object], SLCSRawOutput, SLCSDecodedOutput
        ] = bind_model_io(model, adapter)


def test_predictor_returns_typed_detached_cpu_output_and_targets() -> None:
    model = _model()
    adapter = _adapter()
    adapter.validate_model(model)
    lightning_fixture = _TypedLightningFixture(model, adapter)
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    predictor = SLCSPredictor(
        cast(SLCSLightningModule, lightning_fixture),
        torch.device("cpu"),
    )

    output, targets = predictor.predict_with_targets(_batch())

    assert predictor.model is model
    assert isinstance(output, SLCSDecodedOutput)
    assert output.player_position.shape == (1, 2, 4, 3)
    assert targets.target_ball_position.shape == (1, 4, 3)
    assert output.player_position.device.type == "cpu"
    assert targets.target_ball_position.device.type == "cpu"
    assert not output.player_position.requires_grad
    assert len(calls) == 1
