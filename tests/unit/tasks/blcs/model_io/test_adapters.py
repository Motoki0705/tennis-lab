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
            "num_stages": 4,
            "ffn_dim": 32,
            "num_queries": 2,
            "rope_dim": 4,
            "dropout": 0.0,
            "role_rope_enabled": True,
            "mask_invisible_observations": True,
            "invisible_init_std": 0.02,
            "observation_fusion": "linear",
            "mhc": {
                "coefficient_dim": 8,
                "sinkhorn_iters": 5,
                "eps": 1e-6,
                "residual_identity_bias": 4.0,
                "update_scale_init": 0.0,
            },
            "cswa": {
                "compression_ratio": 2,
                "window_radius": 1,
                "backend": "reference",
            },
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
        "ball_visible": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 1, 2, 14, 2),
        "court_vis": torch.ones(1, 1, 2, 14, dtype=torch.bool),
        "frame_mask": torch.tensor([[True, False]]),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    with pytest.raises(ModelInputContractError, match="candidate_mask"):
        adapter.build_call(batch)

    logits = torch.tensor([[[-1.0, 1.0]]])
    prediction = adapter.decode_output(
        {"position": torch.zeros(1, 1, 2, 3), "presence_logits": logits}
    )
    assert not prediction.presence[..., 0].any()
    assert prediction.presence[..., 1].all()


def test_track_query_adapter_requires_exact_q_and_mask_implications() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    batch = {
        "ball_uv": torch.zeros(1, 1, 2, 2, 2),
        "ball_visible": torch.zeros(1, 1, 2, 2, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "court_kp": torch.zeros(1, 1, 2, 14, 2),
        "court_vis": torch.ones(1, 1, 2, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    call = adapter.build_call(batch)
    camera_valid = call.kwargs["camera_state_valid"]
    object_raw = call.kwargs["object_temporal_state_valid"]
    query_raw = call.kwargs["query_temporal_state_valid"]
    spatial_dense = call.kwargs["spatial_attention_mask"]
    object_dense = call.kwargs["object_temporal_attention_mask"]
    query_dense = call.kwargs["query_temporal_attention_mask"]
    assert isinstance(camera_valid, Tensor)
    assert isinstance(object_raw, Tensor)
    assert isinstance(query_raw, Tensor)
    assert isinstance(spatial_dense, Tensor)
    assert isinstance(object_dense, Tensor)
    assert isinstance(query_dense, Tensor)
    assert camera_valid.shape == (1, 1, 2, 2)
    assert object_raw.shape == (1, 2)
    assert query_raw.shape == (2, 2)
    assert spatial_dense.shape == (2, 4, 4)
    assert object_dense.shape == (1, 2, 2)
    assert query_dense.shape == (2, 2, 2)
    assert camera_valid.all()

    for wrong_width in (1, 3):
        wrong_width_batch = dict(batch)
        wrong_width_batch["ball_uv"] = torch.zeros(1, 1, 2, wrong_width, 2)
        wrong_width_batch["ball_visible"] = torch.zeros(
            1, 1, 2, wrong_width, dtype=torch.bool
        )
        wrong_width_batch["candidate_mask"] = torch.zeros(
            1, 1, 2, wrong_width, dtype=torch.bool
        )
        with pytest.raises(ModelInputContractError, match="model.num_queries"):
            adapter.build_call(wrong_width_batch)

    inconsistent = dict(batch)
    inconsistent["ball_visible"] = batch["ball_visible"].clone()
    inconsistent["candidate_mask"] = batch["candidate_mask"].clone()
    inconsistent["ball_visible"][0, 0, 0, 1] = True
    inconsistent["candidate_mask"][0, 0, 0, 1] = False
    with pytest.raises(ModelInputContractError, match="assigned candidate"):
        adapter.build_call(inconsistent)


def test_track_query_adapter_accepts_shared_cross_view_lifecycle_assignment() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    shared_assignment = torch.tensor([True, False])
    candidate_mask = shared_assignment.reshape(1, 1, 1, 2).expand(1, 2, 1, 2)
    batch = {
        "ball_uv": torch.zeros(1, 2, 1, 2, 2),
        "ball_visible": torch.tensor(
            [[[[False, False]], [[True, False]]]], dtype=torch.bool
        ),
        "candidate_mask": candidate_mask,
        "court_kp": torch.zeros(1, 2, 1, 14, 2),
        "court_vis": torch.ones(1, 2, 1, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 1, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    call = adapter.build_call(batch)
    call_candidate_mask = call.kwargs["candidate_mask"]
    camera_state_valid = call.kwargs["camera_state_valid"]

    assert isinstance(call_candidate_mask, Tensor)
    assert isinstance(camera_state_valid, Tensor)
    assert torch.equal(call_candidate_mask, candidate_mask)
    assert torch.equal(camera_state_valid, candidate_mask)


def test_track_query_adapter_checks_lifecycle_per_batch_frame_and_query() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=3,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    view_mask = torch.tensor([[True, True, False], [False, True, True]])
    frame_mask = torch.tensor([[True, True], [True, False]])
    candidate_mask = torch.zeros(2, 3, 2, 3, dtype=torch.bool)
    candidate_mask[0, :2, 0] = torch.tensor([True, False, True])
    candidate_mask[0, :2, 1] = torch.tensor([False, True, False])
    candidate_mask[1, 1:, 0] = torch.tensor([False, True, True])
    ball_visible = torch.zeros_like(candidate_mask)
    ball_visible[0, 1, 0, 2] = True
    ball_visible[1, 2, 0, 1] = True
    batch = {
        "ball_uv": torch.zeros(2, 3, 2, 3, 2),
        "ball_visible": ball_visible,
        "candidate_mask": candidate_mask,
        "court_kp": torch.zeros(2, 3, 2, 14, 2),
        "court_vis": torch.ones(2, 3, 2, 14, dtype=torch.bool),
        "frame_mask": frame_mask,
        "view_mask": view_mask,
    }

    call = adapter.build_call(batch)

    call_candidate_mask = call.kwargs["candidate_mask"]
    camera_state_valid = call.kwargs["camera_state_valid"]
    assert isinstance(call_candidate_mask, Tensor)
    assert isinstance(camera_state_valid, Tensor)
    assert torch.equal(call_candidate_mask, candidate_mask)
    assert torch.equal(camera_state_valid, candidate_mask)


def test_track_query_adapter_rejects_different_assignments_across_valid_views() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    batch = {
        "ball_uv": torch.zeros(1, 2, 1, 2, 2),
        "ball_visible": torch.zeros(1, 2, 1, 2, dtype=torch.bool),
        "candidate_mask": torch.tensor(
            [[[[True, False]], [[False, True]]]], dtype=torch.bool
        ),
        "court_kp": torch.zeros(1, 2, 1, 14, 2),
        "court_vis": torch.ones(1, 2, 1, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 1, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ModelInputContractError, match="same lifecycle assignment"):
        adapter.build_call(batch)


@pytest.mark.parametrize("num_valid_views", [0, 1])
def test_track_query_adapter_ignores_padded_views_in_lifecycle_comparison(
    num_valid_views: int,
) -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    view_mask = torch.arange(2).unsqueeze(0) < num_valid_views
    candidate_mask = torch.zeros(1, 2, 1, 2, dtype=torch.bool)
    if num_valid_views == 1:
        candidate_mask[0, 0, 0] = torch.tensor([True, False])
    batch = {
        "ball_uv": torch.zeros(1, 2, 1, 2, 2),
        "ball_visible": torch.zeros(1, 2, 1, 2, dtype=torch.bool),
        "candidate_mask": candidate_mask,
        "court_kp": torch.zeros(1, 2, 1, 14, 2),
        "court_vis": torch.ones(1, 2, 1, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 1, dtype=torch.bool),
        "view_mask": view_mask,
    }

    call = adapter.build_call(batch)
    call_candidate_mask = call.kwargs["candidate_mask"]

    assert isinstance(call_candidate_mask, Tensor)
    assert torch.equal(call_candidate_mask, candidate_mask)


def test_track_query_adapter_ignores_padded_frame_in_lifecycle_comparison() -> None:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
        mask_invisible_observations=False,
    )
    shared_assignment = torch.tensor([True, False])
    candidate_mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    candidate_mask[:, :, 0] = shared_assignment
    batch = {
        "ball_uv": torch.zeros(1, 2, 2, 2, 2),
        "ball_visible": torch.zeros(1, 2, 2, 2, dtype=torch.bool),
        "candidate_mask": candidate_mask,
        "court_kp": torch.zeros(1, 2, 2, 14, 2),
        "court_vis": torch.ones(1, 2, 2, 14, dtype=torch.bool),
        "frame_mask": torch.tensor([[True, False]]),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    call = adapter.build_call(batch)
    call_candidate_mask = call.kwargs["candidate_mask"]

    assert isinstance(call_candidate_mask, Tensor)
    assert torch.equal(call_candidate_mask, candidate_mask)


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
    assert isinstance(binding.model, cast("Any", binding.adapter).model_type)
