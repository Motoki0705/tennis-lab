"""BLCS fixed track-query architecture and public-contract tests."""

from __future__ import annotations

import hashlib
import inspect
from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.blcs.configuration import (
    TrackQueryAblationModelConfig,
    TrackQueryModelConfig,
    parse_model_config,
)
from src.tasks.blcs.models.blcs_track_query_ablation_model import (
    BLCSTrackQueryAblationModel,
)
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel
from src.utils.models.components.ffn_layers import FFNType, GPTOSSSwiGLU, SwiGLU
from src.utils.models.components.fixed_query_track_compressed_stage import (
    FixedQueryTrackCompressedStage,
)


def _raw_model() -> dict[str, object]:
    return {
        "name": "blcs_track_query_ablation",
        "hidden_dim": 16,
        "num_heads": 4,
        "num_stages": 4,
        "ffn_dim": 32,
        "ffn_type": "swiglu",
        "num_queries": 4,
        "rope_dim": 4,
        "dropout": 0.0,
        "role_rope_enabled": True,
        "invisible_init_std": 0.02,
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


def _config(*, ffn_type: FFNType = "swiglu") -> TrackQueryAblationModelConfig:
    raw = _raw_model()
    raw["ffn_type"] = ffn_type
    parsed = parse_model_config({"model": raw})
    assert isinstance(parsed, TrackQueryAblationModelConfig)
    return parsed


def _baseline_config() -> TrackQueryModelConfig:
    raw = _raw_model()
    raw["name"] = "blcs_track_query"
    parsed = parse_model_config({"model": raw})
    assert isinstance(parsed, TrackQueryModelConfig)
    return parsed


def _model(*, ffn_type: FFNType = "swiglu") -> BLCSTrackQueryAblationModel:
    model = BLCSTrackQueryAblationModel(_config(ffn_type=ffn_type))
    model.eval()
    return model


def _inputs() -> dict[str, Tensor]:
    padding_mask = torch.tensor(
        [
            [
                [False, True, True],
                [True, False, True],
                [True, True, True],
            ],
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        ]
    )
    return {
        "ball_uv": torch.rand(2, 3, 3, 4, 2),
        "ball_vis": torch.zeros(2, 3, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(2, 3, 3, 14, 2),
        "court_vis": torch.zeros(2, 3, 3, 14, dtype=torch.bool),
        "padding_mask": padding_mask,
    }


def _forward(
    model: BLCSTrackQueryAblationModel,
    inputs: dict[str, Tensor],
) -> dict[str, Tensor]:
    return cast("dict[str, Tensor]", model(**inputs))


def test_model_keeps_public_name_and_five_tensor_contract() -> None:
    assert BLCSTrackQueryAblationModel.__name__ == "BLCSTrackQueryAblationModel"
    parameters = list(inspect.signature(BLCSTrackQueryAblationModel.forward).parameters)
    assert parameters == [
        "self",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


def test_model_builds_only_fixed_compressed_stages() -> None:
    model = _model()

    assert all(
        isinstance(stage, FixedQueryTrackCompressedStage) for stage in model.stages
    )
    for module in model.stages:
        stage = cast(FixedQueryTrackCompressedStage, module)
        assert stage.object_temporal_block.ffn is None
        assert stage.spatial_block.ffn is None
        assert stage.query_temporal_block.ffn is None
        assert isinstance(stage.shared_ffn, SwiGLU)


def test_configured_ffn_type_reaches_stage_end_shared_ffn() -> None:
    model = _model(ffn_type="gpt_oss_swiglu")

    assert all(isinstance(stage.shared_ffn, GPTOSSSwiGLU) for stage in model.stages)


def test_forward_uses_q_plus_v_width_and_ignores_padded_values() -> None:
    torch.manual_seed(777)
    model = _model()
    baseline = _inputs()
    contaminated = {name: value.clone() for name, value in baseline.items()}
    padding_mask = contaminated["padding_mask"]
    contaminated["ball_uv"][padding_mask] = torch.nan
    contaminated["court_kp"][padding_mask] = torch.nan
    contaminated["ball_vis"][padding_mask] = True
    contaminated["court_vis"][padding_mask] = True
    captured: dict[str, Tensor] = {}

    def capture_stage_inputs(
        _module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        for key in (
            "object_state_valid",
            "spatial_attention_keep_mask",
            "spatial_freqs",
        ):
            value = kwargs[key]
            assert isinstance(value, Tensor)
            captured[key] = value
        object_tokens = args[0]
        assert isinstance(object_tokens, Tensor)
        captured["object_tokens"] = object_tokens

    hook = model.stages[0].register_forward_pre_hook(
        capture_stage_inputs,
        with_kwargs=True,
    )
    try:
        with torch.no_grad():
            output = _forward(model, baseline)
            contaminated_output = _forward(model, contaminated)
    finally:
        hook.remove()

    assert set(output) == {"position", "presence_logits"}
    assert output["position"].shape == (2, 3, 4, 3)
    assert output["presence_logits"].shape == (2, 3, 4)
    expected_valid = (~padding_mask).unsqueeze(-1).expand(-1, -1, -1, 4)
    assert torch.equal(captured["object_state_valid"], expected_valid)
    assert captured["spatial_attention_keep_mask"].shape == (6, 7, 7)
    assert captured["spatial_freqs"].shape == (6, 7, 1, 2)
    assert captured["object_tokens"][expected_valid].abs().any()
    for key in output:
        assert torch.isfinite(contaminated_output[key]).all()
        torch.testing.assert_close(output[key], contaminated_output[key])
    assert not output["position"][:, 2].any()
    assert not output["presence_logits"][:, 2].any()
    assert not output["position"][1].any()
    assert not output["presence_logits"][1].any()


def test_baseline_and_fixed_experiment_checkpoints_are_incompatible() -> None:
    baseline = BLCSTrackQueryModel(_baseline_config())
    experiment = _model()

    with pytest.raises(RuntimeError):
        experiment.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(experiment.state_dict(), strict=True)


def test_baseline_state_key_inventory_is_preserved() -> None:
    baseline = BLCSTrackQueryModel(_baseline_config())
    serialized_keys = "\n".join(sorted(baseline.state_dict()))

    assert len(baseline.state_dict()) == 192
    assert hashlib.sha256(serialized_keys.encode()).hexdigest() == (
        "9e531abff0eac4d1b4344a03ae347dd04eb6033c95a8f5e008d29e9d634d55eb"
    )
