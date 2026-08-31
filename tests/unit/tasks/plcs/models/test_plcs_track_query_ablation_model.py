"""PLCS track-query ablation architecture and public-contract tests."""

from __future__ import annotations

import hashlib
import inspect
from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.components.presence_competition import (
    DeepSetsPresenceResidual,
)
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel
from src.utils.models.components.ffn_layers import FFNType, GPTOSSSwiGLU, SwiGLU
from src.utils.models.components.fixed_query_track_ablation_stage import (
    FFNMode,
    FixedQueryTrackAblationStage,
    MHCWriteback,
)

_CONDITIONS: tuple[tuple[str, FFNMode, MHCWriteback, int], ...] = (
    ("A", "per_attention", "after_object_temporal", 16),
    ("B", "shared", "after_object_temporal", 16),
    ("C", "per_attention", "layer_end", 7),
    ("D", "shared", "layer_end", 7),
)


def _raw_model() -> dict[str, object]:
    return {
        "name": "plcs_track_query_ablation",
        "hidden_dim": 16,
        "num_heads": 4,
        "ffn_dim": 32,
        "num_queries": 4,
        "num_stages": 4,
        "num_joints": 17,
        "predict_canonical_pose": True,
        "rope_dim": 4,
        "rope_theta": 10_000.0,
        "ffn_type": "swiglu",
        "dropout": 0.0,
        "role_rope_enabled": True,
        "invisible_init_std": 0.02,
        "ffn_mode": "per_attention",
        "mhc_writeback": "after_object_temporal",
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


def _config(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    *,
    ffn_type: FFNType = "swiglu",
    presence_competition: str = "none",
) -> PLCSModelConfig:
    raw = _raw_model()
    raw["ffn_mode"] = ffn_mode
    raw["mhc_writeback"] = mhc_writeback
    raw["ffn_type"] = ffn_type
    raw["presence_competition"] = presence_competition
    return PLCSModelConfig.from_mapping(raw)


def _baseline_config(*, predict_canonical_pose: bool = True) -> PLCSModelConfig:
    raw = _raw_model()
    raw["name"] = "plcs_track_query"
    del raw["ffn_mode"]
    del raw["mhc_writeback"]
    if not predict_canonical_pose:
        del raw["predict_canonical_pose"]
    return PLCSModelConfig.from_mapping(raw)


def _model(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    *,
    ffn_type: FFNType = "swiglu",
    presence_competition: str = "none",
) -> PLCSTrackQueryAblationModel:
    model = PLCSTrackQueryAblationModel(
        _config(
            ffn_mode,
            mhc_writeback,
            ffn_type=ffn_type,
            presence_competition=presence_competition,
        )
    )
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
        "human_kp": torch.rand(2, 3, 3, 4, 17, 2),
        "human_vis": torch.zeros(2, 3, 3, 4, 17, dtype=torch.bool),
        "court_kp": torch.rand(2, 3, 3, 14, 2),
        "court_vis": torch.zeros(2, 3, 3, 14, dtype=torch.bool),
        "padding_mask": padding_mask,
    }


def _forward(
    model: PLCSTrackQueryAblationModel,
    inputs: dict[str, Tensor],
) -> dict[str, Tensor]:
    return cast("dict[str, Tensor]", model(**inputs))


def test_ablation_model_is_a_distinct_named_public_architecture() -> None:
    assert PLCSTrackQueryAblationModel.__name__ == "PLCSTrackQueryAblationModel"
    assert PLCSTrackQueryAblationModel.__module__.endswith(
        ".plcs_track_query_ablation_model"
    )
    parameters = list(
        inspect.signature(PLCSTrackQueryAblationModel.forward).parameters
    )
    assert parameters == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


def test_four_conditions_build_exact_stage_ffn_ownership_and_parameter_counts() -> None:
    models = {
        condition: _model(ffn_mode, mhc_writeback)
        for condition, ffn_mode, mhc_writeback, _ in _CONDITIONS
    }

    for condition, model in models.items():
        expected_shared = condition in {"B", "D"}
        assert all(
            isinstance(stage, FixedQueryTrackAblationStage)
            for stage in model.stages
        )
        for stage in model.stages:
            block_ffns = (
                stage.object_temporal_block.ffn,
                stage.spatial_block.ffn,
                stage.query_temporal_block.ffn,
            )
            if expected_shared:
                assert block_ffns == (None, None, None)
                assert isinstance(stage.shared_ffn, SwiGLU)
            else:
                assert len({id(module) for module in block_ffns}) == 3
                assert all(isinstance(module, SwiGLU) for module in block_ffns)
                assert stage.shared_ffn is None

    parameter_counts = {
        condition: sum(parameter.numel() for parameter in model.parameters())
        for condition, model in models.items()
    }
    assert parameter_counts["A"] == parameter_counts["C"]
    assert parameter_counts["B"] == parameter_counts["D"]
    assert parameter_counts["A"] > parameter_counts["B"]


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback", "spatial_width"),
    _CONDITIONS,
)
def test_all_four_ablation_variants_register_explicit_competition_only(
    condition: str,
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    spatial_width: int,
) -> None:
    del condition, spatial_width
    model = _model(
        ffn_mode,
        mhc_writeback,
        presence_competition="deepsets",
    )
    inputs = _inputs()

    with torch.no_grad():
        output = _forward(model, inputs)

    assert isinstance(model.presence_competition, DeepSetsPresenceResidual)
    assert output["presence_logits"].shape == (2, 3, 4)
    frame_valid = (~inputs["padding_mask"]).any(dim=1)
    assert not output["presence_logits"][~frame_valid].any()


def test_configured_ffn_reaches_shared_stage_ffn() -> None:
    model = _model(
        "shared",
        "layer_end",
        ffn_type="gpt_oss_swiglu",
    )

    assert all(isinstance(stage.shared_ffn, GPTOSSSwiGLU) for stage in model.stages)


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback", "spatial_width"),
    _CONDITIONS,
)
def test_four_conditions_preserve_io_padding_visibility_and_rope_contracts(
    condition: str,
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    spatial_width: int,
) -> None:
    del condition
    torch.manual_seed(777)
    model = _model(ffn_mode, mhc_writeback)
    baseline = _inputs()
    contaminated = {name: value.clone() for name, value in baseline.items()}
    padding_mask = contaminated["padding_mask"]
    contaminated["human_kp"][padding_mask] = torch.nan
    contaminated["court_kp"][padding_mask] = torch.nan
    contaminated["human_vis"][padding_mask] = True
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

    assert set(output) == {
        "position",
        "rotation",
        "presence_logits",
        "canonical_pose",
    }
    assert output["position"].shape == (2, 3, 4, 3)
    assert output["rotation"].shape == (2, 3, 4, 2)
    assert output["presence_logits"].shape == (2, 3, 4)
    assert output["canonical_pose"].shape == (2, 3, 4, 17, 3)
    expected_valid = (~padding_mask).unsqueeze(-1).expand(-1, -1, -1, 4)
    assert torch.equal(captured["object_state_valid"], expected_valid)
    assert captured["spatial_attention_keep_mask"].shape == (
        6,
        spatial_width,
        spatial_width,
    )
    assert captured["spatial_freqs"].shape == (6, spatial_width, 1, 2)
    assert captured["object_tokens"][expected_valid].abs().any()
    for key in output:
        assert torch.isfinite(contaminated_output[key]).all()
        torch.testing.assert_close(output[key], contaminated_output[key])
    frame_valid = (~padding_mask).any(dim=1)
    valid_rotation = output["rotation"][frame_valid]
    torch.testing.assert_close(
        torch.linalg.vector_norm(valid_rotation, dim=-1),
        torch.ones_like(valid_rotation[..., 0]),
    )
    assert not output["position"][:, 2].any()
    assert not output["rotation"][:, 2].any()
    assert not output["presence_logits"][:, 2].any()
    assert not output["canonical_pose"][:, 2].any()
    assert not output["position"][1].any()
    assert not output["rotation"][1].any()
    assert not output["presence_logits"][1].any()
    assert not output["canonical_pose"][1].any()


def test_baseline_and_ablation_checkpoints_are_strictly_incompatible_both_ways() -> None:
    baseline = PLCSTrackQueryModel(_baseline_config())
    ablation = _model("per_attention", "after_object_temporal")

    with pytest.raises(RuntimeError):
        ablation.load_state_dict(baseline.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(ablation.state_dict(), strict=True)


def test_legacy_baseline_state_key_inventory_is_preserved() -> None:
    baseline = PLCSTrackQueryModel(_baseline_config(predict_canonical_pose=False))
    serialized_keys = "\n".join(sorted(baseline.state_dict()))

    assert len(baseline.state_dict()) == 194
    assert hashlib.sha256(serialized_keys.encode()).hexdigest() == (
        "fd286692d51e1d5f04a38e5382fe1a5cb1980bf31b494d49490c983db84f6746"
    )
