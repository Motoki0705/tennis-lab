"""Small CPU forward/backward smoke tests for every BLCS ablation condition."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.configuration import (
    TrackQueryAblationModelConfig,
    parse_model_config,
)
from src.tasks.blcs.model_io import compose_blcs_track_query_model_io
from src.tasks.blcs.models.blcs_track_query_ablation_model import (
    BLCSTrackQueryAblationModel,
)
from src.utils.models.components.fixed_query_track_ablation_stage import (
    FFNMode,
    MHCWriteback,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _config(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    query_ffn_after_spatial: bool,
) -> TrackQueryAblationModelConfig:
    parsed = parse_model_config(
        {
            "model": {
                "name": "blcs_track_query_ablation",
                "hidden_dim": 16,
                "num_heads": 4,
                "num_stages": 4,
                "ffn_dim": 32,
                "num_queries": 4,
                "rope_dim": 4,
                "dropout": 0.0,
                "role_rope_enabled": True,
                "invisible_init_std": 0.02,
                "ffn_mode": ffn_mode,
                "mhc_writeback": mhc_writeback,
                "query_ffn_after_spatial": query_ffn_after_spatial,
                "mhc": {
                    "coefficient_dim": 8,
                    "sinkhorn_iters": 5,
                    "eps": 1.0e-6,
                    "residual_identity_bias": 4.0,
                    "update_scale_init": 0.0,
                },
                "cswa": {
                    "compression_ratio": 2,
                    "window_radius": 1,
                    "backend": "reference",
                },
            }
        }
    )
    assert isinstance(parsed, TrackQueryAblationModelConfig)
    return parsed


@pytest.mark.parametrize(
    ("ffn_mode", "mhc_writeback", "query_ffn_after_spatial"),
    [
        ("per_attention", "after_object_temporal", False),
        ("shared", "after_object_temporal", False),
        ("per_attention", "layer_end", False),
        ("shared", "layer_end", False),
        ("shared", "layer_end", True),
    ],
)
def test_cpu_forward_backward_has_finite_outputs_and_gradients(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    query_ffn_after_spatial: bool,
) -> None:
    torch.manual_seed(777)
    model = BLCSTrackQueryAblationModel(
        _config(ffn_mode, mhc_writeback, query_ffn_after_spatial)
    ).train()
    ball_uv = torch.rand(1, 2, 2, 4, 2, requires_grad=True)
    court_kp = torch.rand(1, 2, 2, 14, 2, requires_grad=True)
    output = cast(
        "dict[str, Tensor]",
        model(
            ball_uv,
            torch.ones(1, 2, 2, 4, dtype=torch.bool),
            court_kp,
            torch.ones(1, 2, 2, 14, dtype=torch.bool),
            torch.tensor([[[False, True], [True, False]]]),
        ),
    )

    loss = output["position"].square().mean() + output[
        "presence_logits"
    ].square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert ball_uv.grad is not None and torch.isfinite(ball_uv.grad).all()
    assert court_kp.grad is not None and torch.isfinite(court_kp.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.abs().any()) for gradient in gradients)


@pytest.mark.parametrize(
    ("profile", "selector_mode"),
    [
        (
            "track_query_ablation_d_v2_selector",
            ReferenceSelectorMode.REFERENCE,
        ),
        (
            "track_query_ablation_d_v2_selector_zero",
            ReferenceSelectorMode.SELECTOR_ZERO,
        ),
    ],
)
def test_reference_v2_d_runs_six_input_cpu_forward_backward(
    profile: str,
    selector_mode: ReferenceSelectorMode,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_queries=4",
                "model.num_stages=4",
                "model.dropout=0.0",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
                "model.cswa.backend=reference",
            ],
        )
    batch: dict[str, object] = {
        "ball_uv": torch.rand(1, 2, 2, 4, 2),
        "ball_vis": torch.zeros(1, 2, 2, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 2, 14, 2),
        "court_vis": torch.zeros(1, 2, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "reference_view_index": torch.tensor([1], dtype=torch.int64),
        "view_camera_ids": torch.tensor([[10, 11]], dtype=torch.int64),
        "reference_camera_id": torch.tensor([11], dtype=torch.int64),
        "reference_from_physical": torch.eye(3).unsqueeze(0),
    }
    write_track_query_reference_contract(
        batch,
        TrackQueryReferenceContract.reference_v2(selector_mode),
    )
    binding = compose_blcs_track_query_model_io(config)
    call = binding.build_call(batch)
    assert len(call.kwargs) == 6
    output = binding.execute_call(call)
    loss = output["position"].square().mean() + output[
        "presence_logits"
    ].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
