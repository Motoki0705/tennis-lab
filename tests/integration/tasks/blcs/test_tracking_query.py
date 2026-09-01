"""CPU integration tests for canonical BLCS tracking queries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.model_io import compose_blcs_track_query_model_io
from src.tasks.blcs.models import BLCSTrackQueryModel

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _config(
    *,
    backend: str = "reference",
    hidden_dim: int = 16,
    ffn_dim: int = 32,
    rope_dim: int = 4,
    compression_ratio: int = 2,
) -> object:
    overrides = [
        f"model.hidden_dim={hidden_dim}",
        "model.num_heads=4",
        f"model.ffn_dim={ffn_dim}",
        f"model.rope_dim={rope_dim}",
        "model.num_queries=2",
        "model.num_stages=4",
        "model.dropout=0.0",
        "model.mhc.coefficient_dim=8",
        "model.mhc.sinkhorn_iters=5",
        f"model.cswa.compression_ratio={compression_ratio}",
        "model.cswa.window_radius=1",
        f"model.cswa.backend={backend}",
    ]
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train_tracking", overrides=overrides)


def _batch() -> dict[str, torch.Tensor]:
    candidate_shape = (1, 2, 4, 2)
    return {
        "ball_uv": torch.rand(*candidate_shape, 2),
        "ball_vis": torch.ones(candidate_shape, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 4, 14, 2),
        "court_vis": torch.ones(1, 2, 4, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 4, dtype=torch.bool),
    }


def _reference_config() -> object:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_tracking",
            overrides=[
                "model=tracking_query_reference",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_queries=2",
                "model.num_stages=4",
                "model.dropout=0.0",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
                "model.cswa.backend=reference",
            ],
        )


def _reference_batch() -> dict[str, object]:
    result: dict[str, object] = dict(_batch())
    result.update(
        {
            "reference_view_index": torch.tensor([1], dtype=torch.int64),
            "view_camera_ids": torch.tensor([[10, 11]], dtype=torch.int64),
            "reference_camera_id": torch.tensor([11], dtype=torch.int64),
            "reference_from_physical": torch.eye(3).unsqueeze(0),
        }
    )
    write_track_query_reference_contract(
        result,
        TrackQueryReferenceContract.reference_v2(ReferenceSelectorMode.REFERENCE),
    )
    return result


def test_tracking_query_runs_cpu_forward_backward_and_preserves_outputs() -> None:
    binding = compose_blcs_track_query_model_io(_config())
    call = binding.build_call(_batch())
    raw = binding.execute_call(call)
    assert set(raw) == {"position", "presence_logits"}
    assert raw["position"].shape == (1, 4, 2, 3)
    assert raw["presence_logits"].shape == (1, 4, 2)
    assert raw["position"].dtype == torch.float32
    assert torch.isfinite(raw["position"]).all()
    assert torch.isfinite(raw["presence_logits"]).all()

    (raw["position"].square().sum() + raw["presence_logits"].square().sum()).backward()
    gradients = [
        parameter.grad
        for parameter in binding.model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_reference_tracking_query_runs_six_input_cpu_forward_backward() -> None:
    binding = compose_blcs_track_query_model_io(_reference_config())
    call = binding.build_call(_reference_batch())
    assert len(call.kwargs) == 6
    raw = binding.execute_call(call)
    loss = raw["position"].square().mean() + raw["presence_logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert raw["position"].shape == (1, 4, 2, 3)
    assert all(torch.isfinite(value).all() for value in raw.values())


@pytest.mark.cuda
@pytest.mark.skipif(
    os.environ.get("TENNIS_LAB_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="CUDA integration tests require TENNIS_LAB_RUN_CUDA_TESTS=1 and CUDA",
)
def test_cuda_default_width_matches_reference_forward_backward() -> None:
    torch.manual_seed(753)
    config_kwargs = {
        "hidden_dim": 64,
        "ffn_dim": 128,
        "rope_dim": 16,
        "compression_ratio": 4,
    }
    reference_model = compose_blcs_track_query_model_io(
        _config(backend="reference", **config_kwargs)
    ).model.cuda().eval()
    cuda_model = compose_blcs_track_query_model_io(
        _config(backend="cuda", **config_kwargs)
    ).model.cuda().eval()
    cuda_model.load_state_dict(reference_model.state_dict(), strict=True)
    batch = _batch()
    reference_ball_uv = batch["ball_uv"].cuda().requires_grad_(True)
    cuda_ball_uv = reference_ball_uv.detach().clone().requires_grad_(True)
    reference_court_kp = batch["court_kp"].cuda().requires_grad_(True)
    cuda_court_kp = reference_court_kp.detach().clone().requires_grad_(True)
    ball_vis = batch["ball_vis"].cuda()
    court_vis = batch["court_vis"].cuda()
    padding_mask = batch["padding_mask"].cuda()

    expected = reference_model(
        reference_ball_uv,
        ball_vis,
        reference_court_kp,
        court_vis,
        padding_mask,
    )
    actual = cuda_model(
        cuda_ball_uv,
        ball_vis,
        cuda_court_kp,
        court_vis,
        padding_mask,
    )
    position_upstream = torch.randn_like(expected["position"])
    presence_upstream = torch.randn_like(expected["presence_logits"])
    expected_gradients = torch.autograd.grad(
        (expected["position"], expected["presence_logits"]),
        (reference_ball_uv, reference_court_kp),
        (position_upstream, presence_upstream),
    )
    actual_gradients = torch.autograd.grad(
        (actual["position"], actual["presence_logits"]),
        (cuda_ball_uv, cuda_court_kp),
        (position_upstream, presence_upstream),
    )

    for name in ("position", "presence_logits"):
        torch.testing.assert_close(actual[name], expected[name], atol=5e-4, rtol=5e-4)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=5e-4,
            rtol=5e-4,
        )


def test_old_track_query_state_dict_is_intentionally_strictly_incompatible() -> None:
    binding = compose_blcs_track_query_model_io(_config())
    model = binding.model
    assert isinstance(model, BLCSTrackQueryModel)
    old_state = {
        "slot_embeddings": model.slot_embeddings.detach().clone(),
        "spatial_blocks.0.attn_norm.weight": torch.ones(16),
    }

    with pytest.raises(RuntimeError, match="spatial_blocks"):
        model.load_state_dict(old_state, strict=True)


def test_tracking_query_state_dict_round_trip_preserves_outputs() -> None:
    source = compose_blcs_track_query_model_io(_config())
    clone = compose_blcs_track_query_model_io(_config())
    clone.model.load_state_dict(source.model.state_dict(), strict=True)
    source.model.eval()
    clone.model.eval()
    batch = _batch()

    with torch.no_grad():
        source_output = source.execute_call(source.build_call(batch))
        clone_output = clone.execute_call(clone.build_call(batch))

    assert (
        set(source_output)
        == set(clone_output)
        == {
            "position",
            "presence_logits",
        }
    )
    torch.testing.assert_close(clone_output["position"], source_output["position"])
    torch.testing.assert_close(
        clone_output["presence_logits"], source_output["presence_logits"]
    )


def test_requested_unavailable_cuda_backend_fails_during_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("requested CUDA backend is unavailable")

    monkeypatch.setattr(
        "src.utils.models.components.compressor.resolve_token_compressor_pool",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "src.utils.models.components.cswa.resolve_compressed_time_local_attention",
        unavailable,
    )

    with pytest.raises(RuntimeError, match="requested CUDA backend is unavailable"):
        compose_blcs_track_query_model_io(_config(backend="cuda"))
