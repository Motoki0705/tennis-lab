"""Boundary tests for the sole SLCS model I/O adapter."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    ModelOutputContractError,
    bind_model_io,
)
from src.tasks.slcs.model_io import (
    SLCSDecodedOutput,
    SLCSModelIOAdapter,
    SLCSModelIOSpec,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel


def _model(*, num_players: int = 2) -> SLCSFusionModel:
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
        num_players=num_players,
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


def _adapter() -> SLCSModelIOAdapter:
    return SLCSModelIOAdapter(
        SLCSModelIOSpec(
            num_players=2,
            num_court_kp=14,
            max_seq_len=8,
            dino_num_tokens=12,
            dino_encoded_num_tokens=12,
            dino_embed_dim=8,
            log_b_min=-6.0,
            log_b_max=3.0,
        )
    )


def _batch(*, frames: int = 8) -> dict[str, Tensor]:
    batch_size, players, joints, court = 2, 2, 17, 14
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
        "dino_frame_idx": torch.tensor([[0, frames - 1], [0, frames - 1]]),
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


def test_valid_batch_runs_once_and_returns_typed_decode() -> None:
    model = _model()
    adapter = _adapter()
    adapter.validate_model(model)
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))

    output = bind_model_io(model, adapter).run(_batch())

    assert isinstance(output, SLCSDecodedOutput)
    assert output.player_position.shape == (2, 2, 8, 3)
    assert len(calls) == 1


def test_all_invalid_dino_slots_are_an_explicit_boundary_case() -> None:
    batch = _batch(frames=1)
    batch["dino_tokens"] = torch.zeros(2, 1, 12, 8)
    batch["dino_frame_idx"] = torch.zeros(2, 1, dtype=torch.int64)
    batch["dino_valid"] = torch.zeros(2, 1, dtype=torch.bool)

    call = _adapter().build_call(batch)
    dino_attn_mask = call.kwargs["dino_attn_mask"]
    dino_batch_has_evidence = call.kwargs["dino_batch_has_evidence"]
    assert dino_attn_mask is not None
    assert dino_batch_has_evidence is not None
    assert not dino_batch_has_evidence.any()
    assert dino_attn_mask[:, :, 0].all()
    assert not dino_attn_mask[:, :, 1:].any()

    output = bind_model_io(_model(), _adapter()).run(batch)

    assert torch.isfinite(output.ball_position).all()


def test_adapter_prepares_padded_self_attention_masks() -> None:
    batch = _batch()
    batch["frame_mask"][:, -2:] = False
    batch["player_kp_vis"][:, :, -2:] = 0.0
    batch["player_valid"][:, :, -2:] = False
    batch["ball_vis"][:, -2:] = False
    batch["dino_valid"][:, 1] = False

    call = _adapter().build_call(batch)

    entity_mask = call.kwargs["entity_attn_mask"]
    time_mask = call.kwargs["time_attn_mask"]
    assert entity_mask is not None
    assert time_mask is not None
    entity_mask = entity_mask.reshape(2, 8, 3, 3)
    time_mask = time_mask.reshape(2, 3, 8, 8)
    assert entity_mask[:, :6].all()
    assert entity_mask[:, 6:, :, 0].all()
    assert not entity_mask[:, 6:, :, 1:].any()
    assert time_mask[:, :, :, :6].all()
    assert not time_mask[:, :, :, 6:].any()


def _drop_player_key(batch: dict[str, Tensor]) -> None:
    del batch["player_kp"]


def _wrong_player_dtype(batch: dict[str, Tensor]) -> None:
    batch["player_kp"] = batch["player_kp"].to(torch.float64)


def _wrong_dino_rank(batch: dict[str, Tensor]) -> None:
    batch["dino_tokens"] = batch["dino_tokens"][0]


def _wrong_dino_width(batch: dict[str, Tensor]) -> None:
    batch["dino_tokens"] = torch.rand(2, 2, 12, 7)


def _out_of_window_dino(batch: dict[str, Tensor]) -> None:
    batch["dino_frame_idx"][0, 1] = 8


def _non_monotonic_dino(batch: dict[str, Tensor]) -> None:
    batch["dino_frame_idx"][0] = torch.tensor([4, 3])


def _pixel_coordinate_input(batch: dict[str, Tensor]) -> None:
    batch["ball_uv"][0, 0] = torch.tensor([640.0, 360.0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_player_key, "missing"),
        (_wrong_player_dtype, "torch.float32"),
        (_wrong_dino_rank, "rank 4"),
        (_wrong_dino_width, "axis 3"),
        (_out_of_window_dino, "must lie"),
        (_non_monotonic_dino, "strictly increasing"),
        (_pixel_coordinate_input, "normalized UV"),
    ],
)
def test_invalid_model_input_fails_before_forward(
    mutate: Callable[[dict[str, Tensor]], None], message: str
) -> None:
    model = _model()
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    batch = _batch()
    mutate(batch)

    with pytest.raises(ModelInputContractError, match=message):
        bind_model_io(model, _adapter()).run(batch)

    assert not calls


def test_invalid_training_target_fails_before_forward() -> None:
    model = _model()
    adapter = _adapter()
    binding = bind_model_io(model, adapter)
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    batch = _batch()
    batch["target_ball_position"][0, 0, 0] = torch.nan

    binding.build_call(batch)
    with pytest.raises(ModelInputContractError, match="non-finite"):
        adapter.build_training_targets(batch)

    assert not calls


def test_nonzero_weight_for_invalid_target_fails_before_forward() -> None:
    model = _model()
    adapter = _adapter()
    calls: list[object] = []
    model.register_forward_pre_hook(lambda *_: calls.append(object()))
    batch = _batch()
    batch["target_ball_valid"][0, 0] = False

    with pytest.raises(ModelInputContractError, match="must be zero"):
        adapter.build_training_targets(batch)

    assert not calls


def test_same_model_class_with_incompatible_dimensions_is_rejected() -> None:
    adapter = _adapter()
    with pytest.raises(ModelAdapterMismatchError, match="dimensions"):
        adapter.validate_model(_model(num_players=3))


def test_invalid_raw_output_is_rejected_by_decode() -> None:
    adapter = _adapter()
    raw = _model()(**adapter.build_call(_batch()).kwargs)
    del raw["ball_position"]

    with pytest.raises(ModelOutputContractError, match="missing"):
        adapter.decode_output(raw)  # type: ignore[arg-type]
