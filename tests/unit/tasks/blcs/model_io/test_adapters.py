"""Boundary tests for BLCS model/adapter composition and typed decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    ModelOutputContractError,
    bind_model_io,
)
from src.tasks.blcs.model_io import (
    AxialTrajectoryModelIOAdapter,
    MultiViewTrajectoryModelIOAdapter,
    SingleTrajectoryModelIOAdapter,
    TrackQueryModelIOAdapter,
    compose_blcs_model_io,
)
from src.tasks.blcs.models import BLCSModel


class _CountingBLCSModel(BLCSModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.calls = 0

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor,
        court_vis: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        del court_kp, ball_vis, court_vis, attention_mask
        self.calls += 1
        return {
            "position": torch.zeros(
                ball_uv.shape[0], ball_uv.shape[1], 3, device=ball_uv.device
            )
        }


def _single_adapter() -> SingleTrajectoryModelIOAdapter:
    return SingleTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=8,
        predict_velocity=False,
        input_profile="single",
        max_num_cameras=None,
    )


def _single_batch(*, court_tokens: int = 14) -> dict[str, Tensor]:
    return {
        "ball_uv": torch.zeros(2, 3, 2),
        "court_kp": torch.zeros(2, court_tokens, 2),
        "ball_vis": torch.ones(2, 3, dtype=torch.bool),
        "ball_mask": torch.ones(2, 3, dtype=torch.bool),
        "court_vis": torch.ones(2, court_tokens, dtype=torch.bool),
    }


def _model_mapping(name: str) -> dict[str, object]:
    if name == "blcs":
        return {
            "name": name,
            "io": {"input_profile": "single"},
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 32,
            "ffn_type": "swiglu",
            "dropout": 0.0,
            "max_seq_len": 8,
            "invisible_init_std": 0.02,
            "rope_dim": 4,
            "rope_theta": 10000.0,
            "rope_theta_time": 10000.0,
            "rope_theta_camera": 1000.0,
            "rope_theta_type": 1000.0,
            "predict_velocity": False,
            "num_court_tokens": 14,
        }
    if name == "blcs_multiview":
        return {
            "name": name,
            "io": {"input_profile": "multiview"},
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 32,
            "ffn_type": "swiglu",
            "dropout": 0.0,
            "rope_dim": 4,
            "rope_theta": 10000.0,
            "rope_theta_time": 10000.0,
            "rope_theta_camera": 1000.0,
            "rope_theta_type": 1000.0,
            "max_seq_len": 8,
            "max_num_cameras": 3,
            "predict_velocity": False,
            "num_court_tokens": 14,
            "invisible_init_std": 0.02,
            "query_init_std": 0.02,
        }
    if name == "blcs_multiview_axial":
        return {
            "name": name,
            "io": {"input_profile": "multiview"},
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 4,
            "attention_type": "mha",
            "num_kv_heads": None,
            "ffn_dim": 32,
            "ffn_type": "swiglu",
            "dropout": 0.0,
            "rope_dim": 4,
            "rope_theta_time": 10000.0,
            "rope_theta_camera": 1000.0,
            "max_seq_len": 8,
            "max_num_cameras": 3,
            "predict_velocity": False,
            "invisible_init_std": 0.02,
            "num_court_tokens": 14,
            "time_window_radius": 2,
            "camera_layers_per_stage": [1],
            "time_layers_per_stage": [1],
            "time_global_stage_mask": [False],
        }
    if name == "blcs_track_query":
        return {
            "name": name,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_stages": 1,
            "ffn_dim": 32,
            "num_queries": 2,
            "rope_dim": 4,
            "dropout": 0.0,
            "role_rope_enabled": True,
            "mask_invisible_observations": True,
            "invisible_init_std": 0.02,
            "court_observation_profile": "kp14_reference_baseline",
            "kp7_camera_rope_enabled": False,
            "observation_fusion": "linear",
        }
    raise AssertionError(f"Unexpected test model name: {name}")


def test_invalid_input_fails_before_the_bound_model_is_called() -> None:
    model = _CountingBLCSModel()
    binding = bind_model_io(model, _single_adapter())

    with pytest.raises(ModelInputContractError, match="court_kp"):
        binding.run(_single_batch(court_tokens=13))

    assert model.calls == 0


def test_valid_input_runs_once_and_returns_typed_decode() -> None:
    model = _CountingBLCSModel()
    binding = bind_model_io(model, _single_adapter())

    prediction = binding.run(_single_batch())

    assert model.calls == 1
    assert prediction.position.shape == (2, 3, 3)
    assert prediction.velocity is None


def test_binding_rejects_a_mismatched_model_adapter_pair() -> None:
    with pytest.raises(ModelAdapterMismatchError, match="SingleTrajectory"):
        bind_model_io(nn.Linear(2, 3), _single_adapter())


@pytest.mark.parametrize("violation", ["missing", "dtype", "range", "mask"])
def test_single_adapter_rejects_required_dtype_and_semantic_violations(
    violation: str,
) -> None:
    batch = _single_batch()
    if violation == "missing":
        del batch["ball_mask"]
    elif violation == "dtype":
        batch["ball_uv"] = batch["ball_uv"].long()
    elif violation == "range":
        batch["ball_uv"][0, 0, 0] = 1.5
    else:
        batch["ball_vis"] = batch["ball_vis"].float()
        batch["ball_vis"][0, 0] = 0.5

    with pytest.raises(ModelInputContractError):
        _single_adapter().build_call(batch)


def test_trajectory_decode_requires_exact_output_keys_and_shape() -> None:
    adapter = _single_adapter()

    with pytest.raises(ModelOutputContractError, match="output keys"):
        adapter.decode_output(
            {
                "position": torch.zeros(1, 2, 3),
                "unexpected": torch.zeros(1),
            }
        )
    with pytest.raises(ModelOutputContractError, match=r"\(B,T,3\)"):
        adapter.decode_output({"position": torch.zeros(1, 2, 4)})
    with pytest.raises(ModelOutputContractError, match="floating dtype"):
        adapter.decode_output(
            {"position": torch.zeros(1, 2, 3, dtype=torch.int64)}
        )


def test_multiview_adapter_normalizes_static_court_and_rejects_camera_overflow() -> None:
    adapter = MultiViewTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=8,
        predict_velocity=False,
        input_profile="multiview",
        max_num_cameras=2,
    )
    batch = {
        "ball_uv": torch.zeros(1, 2, 3, 2),
        "ball_vis": torch.ones(1, 2, 3, dtype=torch.bool),
        "ball_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 14, dtype=torch.bool),
    }

    call = adapter.build_call(batch)

    court_kp = call.kwargs["court_kp"]
    assert isinstance(court_kp, Tensor)
    assert court_kp.shape == (1, 2, 3, 14, 2)
    overflow = dict(batch)
    overflow["ball_uv"] = torch.zeros(1, 3, 3, 2)
    overflow["ball_vis"] = torch.ones(1, 3, 3, dtype=torch.bool)
    overflow["ball_mask"] = torch.ones(1, 3, 3, dtype=torch.bool)
    with pytest.raises(ModelInputContractError, match="max_num_cameras"):
        adapter.build_call(overflow)


def test_multiview_boundary_prepares_empty_row_safe_attention_before_forward() -> None:
    binding = compose_blcs_model_io(
        {
            "model": _model_mapping("blcs_multiview"),
            "tracking_metrics": {
                "presence_threshold": 0.5,
                "duplicate_distance": 0.05,
            },
        }
    )
    batch = {
        "ball_uv": torch.zeros(1, 2, 3, 2),
        "ball_vis": torch.zeros(1, 2, 3, dtype=torch.bool),
        "ball_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 3, 14, 2),
        "court_vis": torch.zeros(1, 2, 3, 14, dtype=torch.bool),
    }

    call = binding.build_call(batch)
    query_mask = call.kwargs["query_attention_mask"]
    cross_mask = call.kwargs["cross_attention_mask"]
    frame_token_valid = call.kwargs["frame_token_valid"]
    assert isinstance(query_mask, Tensor)
    assert isinstance(cross_mask, Tensor)
    assert isinstance(frame_token_valid, Tensor)
    assert query_mask.any(dim=-1).all()
    assert cross_mask.any(dim=-1).all()
    assert not frame_token_valid.any()

    decoded = binding.run(batch)
    assert torch.isfinite(decoded.position).all()


def test_track_query_adapter_rejects_semantics_and_decodes_presence_once() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.6,
        mask_invisible_observations=True,
    )
    batch = {
        "ball_uv": torch.zeros(1, 1, 2, 2, 2),
        "ball_score": torch.ones(1, 1, 2, 2),
        "ball_visible": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 1, 2, 14, 2),
        "court_vis": torch.ones(1, 1, 2, 14, dtype=torch.bool),
        "frame_mask": torch.tensor([[True, False]]),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
        "reference_view_index": torch.zeros(1, dtype=torch.long),
    }
    with pytest.raises(ModelInputContractError, match="padded view or frame"):
        adapter.build_call(batch)

    logits = torch.tensor([[[-1.0, 1.0]]])
    prediction = adapter.decode_output(
        {"position": torch.zeros(1, 1, 2, 3), "presence_logits": logits}
    )
    assert not prediction.presence[..., 0].any()
    assert prediction.presence[..., 1].all()


def test_track_query_training_rejects_reference_orientation_mismatch() -> None:
    """The declared reference view and oriented targets must be one contract."""
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=True,
    )
    source_position = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, -0.5, 0.6]]]],
    ).expand(1, 2, 2, 3).clone()
    source_velocity = torch.tensor(
        [[[[0.01, 0.02, 0.03], [0.04, -0.05, 0.06]]]],
    ).expand_as(source_position).clone()
    batch = {
        "ball_uv": torch.zeros(1, 2, 2, 2, 2),
        "ball_score": torch.ones(1, 2, 2, 2),
        "ball_visible": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
        # View 1 is on the +Y side, so its reference orientation sign is -1.
        "camera_center": torch.tensor(
            [[[0.0, -20.0, 3.0], [0.0, 20.0, 3.0]]]
        ),
        "reference_view_index": torch.tensor([1]),
        "orientation_sign": torch.tensor([1.0]),
        "target_position": source_position.clone(),
        "source_target_position": source_position,
        "target_velocity": source_velocity.clone(),
        "source_target_velocity": source_velocity,
        "target_presence": torch.ones(1, 2, 2, dtype=torch.bool),
        "target_instance_id": torch.tensor([[[0, 1], [0, 1]]]),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ModelInputContractError):
        adapter.build_training_batch(batch)


def test_track_query_training_rejects_ambiguous_reference_at_one_meter_margin() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=True,
    )
    source_position = torch.zeros(1, 2, 2, 3)
    source_velocity = torch.zeros_like(source_position)
    batch = {
        "ball_uv": torch.zeros(1, 1, 2, 1, 2),
        "ball_score": torch.ones(1, 1, 2, 1),
        "ball_visible": torch.ones(1, 1, 2, 1, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 2, 1, dtype=torch.bool),
        "court_kp": torch.zeros(1, 1, 2, 14, 2),
        "court_vis": torch.ones(1, 1, 2, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
        "camera_center": torch.tensor([[[0.0, 0.499, 0.0]]]),
        "reference_view_index": torch.tensor([0]),
        "orientation_sign": torch.tensor([-1.0]),
        "target_position": source_position.clone(),
        "source_target_position": source_position,
        "target_velocity": source_velocity.clone(),
        "source_target_velocity": source_velocity,
        "target_presence": torch.ones(1, 2, 2, dtype=torch.bool),
        "target_instance_id": torch.zeros(1, 2, 2, dtype=torch.long),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ModelInputContractError, match="orientation-ambiguous"):
        adapter.build_training_batch(batch)


def test_kp7_track_query_input_and_output_contracts_fail_closed() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=True,
        court_observation_profile="kp7_reference",
    )
    batch = {
        "ball_uv": torch.zeros(1, 2, 2, 2, 2),
        "ball_score": torch.ones(1, 2, 2, 2),
        "ball_visible": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 2, 2, dtype=torch.bool),
        "court_peak_uv": torch.zeros(1, 2, 2, 7, 1, 2),
        "court_peak_score": torch.ones(1, 2, 2, 7, 1),
        "court_peak_covariance": torch.eye(2)
        .view(1, 1, 1, 1, 1, 2, 2)
        .expand(1, 2, 2, 7, 1, 2, 2),
        "court_peak_valid": torch.ones(1, 2, 2, 7, 1, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
        "reference_view_index": torch.tensor([0]),
    }
    batch["court_peak_covariance"][0, 0, 0, 0, 0, 0, 1] = 0.5

    with pytest.raises(ModelInputContractError, match="symmetric"):
        adapter.build_call(batch)
    with pytest.raises(ModelOutputContractError, match="finite"):
        adapter.decode_output(
            {
                "position": torch.full((1, 2, 2, 3), float("nan")),
                "presence_logits": torch.zeros(1, 2, 2),
            }
        )


@pytest.mark.parametrize(
    ("name", "adapter_type"),
    [
        ("blcs", SingleTrajectoryModelIOAdapter),
        ("blcs_multiview", MultiViewTrajectoryModelIOAdapter),
        ("blcs_multiview_axial", AxialTrajectoryModelIOAdapter),
        ("blcs_track_query", TrackQueryModelIOAdapter),
    ],
)
def test_composition_root_selects_and_binds_one_matching_adapter(
    name: str,
    adapter_type: type[object],
) -> None:
    config: Mapping[str, object] = {
        "model": _model_mapping(name),
        "tracking_metrics": {
            "presence_threshold": 0.5,
            "duplicate_distance": 0.05,
        },
    }

    binding = compose_blcs_model_io(config)

    assert isinstance(binding.adapter, adapter_type)
    typed_adapter = cast("Any", binding.adapter)
    assert type(binding.model) is typed_adapter.model_type
