"""CPU integration tests for fixed-width hybrid-CSWA BLCS track queries."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.blcs.model_io import compose_blcs_track_query_model_io
from src.tasks.blcs.models import BLCSTrackQueryModel

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _config(*, backend: str = "reference") -> object:
    overrides = [
        "model.hidden_dim=16",
        "model.num_heads=4",
        "model.ffn_dim=32",
        "model.rope_dim=4",
        "model.num_queries=2",
        "model.num_stages=4",
        "model.dropout=0.0",
        "model.mhc.coefficient_dim=8",
        "model.mhc.sinkhorn_iters=5",
        "model.cswa.compression_ratio=2",
        "model.cswa.window_radius=1",
        f"model.cswa.backend={backend}",
    ]
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train_tracking", overrides=overrides)


def _batch() -> dict[str, torch.Tensor]:
    candidate_shape = (1, 2, 4, 2)
    return {
        "ball_uv": torch.rand(*candidate_shape, 2),
        "ball_visible": torch.ones(candidate_shape, dtype=torch.bool),
        "candidate_mask": torch.ones(candidate_shape, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 4, 14, 2),
        "court_vis": torch.ones(1, 2, 4, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 4, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def test_hybrid_candidate_runs_cpu_forward_backward_and_preserves_outputs() -> None:
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


def test_hybrid_candidate_state_dict_round_trip_preserves_outputs() -> None:
    source = compose_blcs_track_query_model_io(_config())
    clone = compose_blcs_track_query_model_io(_config())
    clone.model.load_state_dict(source.model.state_dict(), strict=True)
    source.model.eval()
    clone.model.eval()
    batch = _batch()

    with torch.no_grad():
        source_output = source.execute_call(source.build_call(batch))
        clone_output = clone.execute_call(clone.build_call(batch))

    assert set(source_output) == set(clone_output) == {
        "position",
        "presence_logits",
    }
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
