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


def _adapter(version: str = "v1") -> SLCSModelIOAdapter:
    from src.utils.schema.court_normalization import (
        resolve_court_coordinate_normalization,
    )

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
        ),
        court_coordinate_normalization=resolve_court_coordinate_normalization(
            version
        ),
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
        "padding_mask": torch.zeros(batch_size, frames, dtype=torch.bool),
        "dino_tokens": torch.rand(batch_size, 2, 12, 8),
        "dino_frame_idx": torch.tensor([[0, frames - 1], [0, frames - 1]]),
        "dino_padding_mask": torch.zeros(batch_size, 2, dtype=torch.bool),
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


def test_all_dino_padding_is_a_finite_raw_model_boundary_case() -> None:
    batch = _batch(frames=1)
    batch["dino_tokens"] = torch.zeros(2, 1, 12, 8)
    batch["dino_frame_idx"] = torch.zeros(2, 1, dtype=torch.int64)
    batch["dino_padding_mask"] = torch.ones(2, 1, dtype=torch.bool)

    call = _adapter().build_call(batch)
    torch.testing.assert_close(
        call.kwargs["dino_padding_mask"], batch["dino_padding_mask"]
    )

    output = bind_model_io(_model(), _adapter()).run(batch)

    assert torch.isfinite(output.ball_position).all()


def test_adapter_forwards_only_raw_observations_and_padding_masks() -> None:
    batch = _batch()
    batch["padding_mask"][:, -2:] = True
    batch["player_kp_vis"][:, :, -2:] = 0.0
    batch["player_valid"][:, :, -2:] = False
    batch["ball_vis"][:, -2:] = False
    batch["court_vis"][:, -2:] = 0.0
    batch["dino_padding_mask"][:, 1] = True

    call = _adapter().build_call(batch)

    assert set(call.kwargs) == {
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
    }
    torch.testing.assert_close(call.kwargs["padding_mask"], batch["padding_mask"])
    torch.testing.assert_close(
        call.kwargs["dino_padding_mask"], batch["dino_padding_mask"]
    )


def test_padding_masks_require_contiguous_suffixes() -> None:
    batch = _batch()
    batch["padding_mask"][0, 3] = True
    batch["player_kp_vis"][0, :, 3] = 0.0
    batch["player_valid"][0, :, 3] = False
    batch["ball_vis"][0, 3] = False
    batch["court_vis"][0, 3] = 0.0

    with pytest.raises(ModelInputContractError, match="contiguous padding suffix"):
        _adapter().build_call(batch)

    batch = _batch()
    batch["dino_padding_mask"][0, 0] = True
    with pytest.raises(ModelInputContractError, match="contiguous padding suffix"):
        _adapter().build_call(batch)


def test_dino_sample_cannot_reference_a_padded_frame() -> None:
    batch = _batch()
    batch["padding_mask"][:, -1] = True
    batch["player_kp_vis"][:, :, -1] = 0.0
    batch["player_valid"][:, :, -1] = False
    batch["ball_vis"][:, -1] = False
    batch["court_vis"][:, -1] = 0.0

    with pytest.raises(ModelInputContractError, match="padded frame"):
        _adapter().build_call(batch)


@pytest.mark.parametrize(
    ("legacy_key", "replacement_key"),
    [
        ("frame_mask", "padding_mask"),
        ("dino_valid", "dino_padding_mask"),
    ],
)
def test_legacy_mask_keys_are_rejected(
    legacy_key: str, replacement_key: str
) -> None:
    batch = _batch()
    batch[legacy_key] = batch.pop(replacement_key)

    with pytest.raises(ModelInputContractError, match="legacy or adapter-prepared"):
        _adapter().build_call(batch)


@pytest.mark.parametrize(
    "prepared_key",
    [
        "entity_attn_mask",
        "time_attn_mask",
        "dino_attn_mask",
        "dino_batch_has_evidence",
    ],
)
def test_adapter_prepared_mask_inputs_are_rejected(prepared_key: str) -> None:
    batch = _batch()
    batch[prepared_key] = torch.ones(1, dtype=torch.bool)

    with pytest.raises(ModelInputContractError, match="legacy or adapter-prepared"):
        _adapter().build_call(batch)


def test_observation_and_target_validity_remain_independent_from_padding() -> None:
    batch = _batch()
    batch["player_kp_vis"][:, 0, 2] = 0.0
    batch["player_valid"][:, 0, 2] = False
    batch["ball_vis"][:, 3] = False
    batch["target_player_valid"][:, 1, 4] = False
    batch["target_player_weight"][:, 1, 4] = 0.0
    batch["target_ball_valid"][:, 5] = False
    batch["target_ball_weight"][:, 5] = 0.0

    call = _adapter().build_call(batch)
    targets = _adapter().build_training_targets(batch)

    call_padding_mask = call.kwargs["padding_mask"]
    call_player_valid = call.kwargs["player_valid"]
    call_ball_vis = call.kwargs["ball_vis"]
    assert isinstance(call_padding_mask, Tensor)
    assert isinstance(call_player_valid, Tensor)
    assert isinstance(call_ball_vis, Tensor)
    assert not call_padding_mask.any()
    assert not call_player_valid[:, 0, 2].any()
    assert not call_ball_vis[:, 3].any()
    assert targets.player_mask[:, 0, 2].all()
    assert not targets.player_mask[:, 1, 4].any()
    assert not targets.ball_mask[:, 5].any()


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


def test_v2_adapter_decodes_positions_and_scalar_uncertainty_to_meters() -> None:
    adapter = _adapter("v2")
    normalized = torch.ones(1, 2, 3, 3)
    output = SLCSDecodedOutput(
        player_position=normalized,
        player_rotation=torch.tensor([1.0, 0.0]).expand(1, 2, 3, 2),
        player_position_log_b=torch.zeros(1, 2, 3),
        player_rotation_log_b=torch.zeros(1, 2, 3),
        ball_position=torch.ones(1, 3, 3),
        ball_position_log_b=torch.zeros(1, 3),
    )

    physical = adapter.to_physical(output)

    torch.testing.assert_close(
        physical.player_position_meters,
        torch.full((1, 2, 3, 3), 11.885),
    )
    torch.testing.assert_close(
        physical.ball_position_meters,
        torch.full((1, 3, 3), 11.885),
    )
    torch.testing.assert_close(
        physical.player_position_sigma_m,
        torch.full((1, 2, 3), 11.885),
    )
    torch.testing.assert_close(
        physical.ball_position_sigma_m,
        torch.full((1, 3), 11.885),
    )
