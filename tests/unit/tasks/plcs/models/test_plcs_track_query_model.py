from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.components.presence_competition import (
    DeepSetsPresenceResidual,
)
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel
from src.utils.models.components.ffn_layers import DeepSeekV4SwiGLU, FFNType
from src.utils.models.components.fixed_query_track_stage import FixedQueryTrackStage
from src.utils.models.embeddings import CourtPlayerGroupEmbedding

MODEL_CONFIG_DIR = Path(__file__).parents[5] / "src/tasks/plcs/configs/model"


def _model(
    *,
    backend: str = "reference",
    ffn_type: FFNType = "swiglu",
    predict_canonical_pose: bool = True,
    presence_competition: str = "none",
) -> PLCSTrackQueryModel:
    with initialize_config_dir(
        config_dir=str(MODEL_CONFIG_DIR), version_base="1.3"
    ):
        overrides = [
            f"cswa.backend={backend}",
            f"ffn_type={ffn_type}",
            f"presence_competition={presence_competition}",
        ]
        if predict_canonical_pose:
            overrides.append("+predict_canonical_pose=true")
        raw = compose(config_name="track_query", overrides=overrides)
    config = PLCSModelConfig.from_mapping(raw)
    model = PLCSTrackQueryModel(config)
    model.eval()
    return model


def _inputs(model: PLCSTrackQueryModel) -> dict[str, torch.Tensor]:
    prefix = (1, 2, 3)
    return {
        "human_kp": torch.rand(*prefix, model.num_queries, 17, 2),
        "human_vis": torch.ones(
            *prefix, model.num_queries, 17, dtype=torch.bool
        ),
        "court_kp": torch.rand(*prefix, 14, 2),
        "court_vis": torch.ones(*prefix, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
    }


def _forward(
    model: PLCSTrackQueryModel, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return cast("dict[str, torch.Tensor]", model(**inputs))


def test_player_role_coordinates_share_role_within_fixed_query_axis() -> None:
    coordinates = PLCSTrackQueryModel.build_spatial_coordinates(
        batch_size=1,
        num_frames=1,
        num_views=2,
        num_detections=3,
        num_queries=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(coordinates[0, :3], torch.zeros(3, 3, dtype=torch.long))
    assert torch.equal(
        coordinates[0, 3:],
        torch.tensor(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [0, 2, 1],
                [0, 2, 1],
                [0, 2, 1],
            ]
        ),
    )


def test_forward_public_contract_has_exactly_five_tensors() -> None:
    parameters = list(inspect.signature(PLCSTrackQueryModel.forward).parameters)
    assert parameters == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


def test_model_uses_shared_court_player_group_embedding() -> None:
    model = _model()
    assert isinstance(model.group_embed, CourtPlayerGroupEmbedding)
    assert model.canonical_pose_head is not None


def test_legacy_model_omits_canonical_head_and_output() -> None:
    model = _model(predict_canonical_pose=False)

    with torch.no_grad():
        output = _forward(model, _inputs(model))

    assert model.canonical_pose_head is None
    assert set(output) == {"position", "rotation", "presence_logits"}


def test_presence_competition_is_absent_by_default_without_state_dict_changes() -> None:
    model = _model(predict_canonical_pose=False)

    assert model.presence_competition is None
    assert "presence_competition" not in dict(model.named_children())
    assert not any(
        key.startswith("presence_competition.") for key in model.state_dict()
    )


@pytest.mark.parametrize(
    "presence_competition",
    ["deepsets", "deepsets_centered"],
)
def test_enabled_zero_residual_is_bitwise_identical_to_legacy_presence_output(
    presence_competition: str,
) -> None:
    torch.manual_seed(11)
    legacy = _model(predict_canonical_pose=False)
    enabled = _model(
        predict_canonical_pose=False,
        presence_competition=presence_competition,
    )
    result = enabled.load_state_dict(legacy.state_dict(), strict=False)
    inputs = _inputs(legacy)

    with torch.no_grad():
        legacy_output = _forward(legacy, inputs)
        enabled_output = _forward(enabled, inputs)

    expected_missing_keys = {
        "presence_competition.feature_projection.weight",
        "presence_competition.feature_projection.bias",
        "presence_competition.output_projection.weight",
    }
    if presence_competition == "deepsets":
        expected_missing_keys.add("presence_competition.output_projection.bias")
    assert set(result.missing_keys) == expected_missing_keys
    assert not result.unexpected_keys
    assert isinstance(enabled.presence_competition, DeepSetsPresenceResidual)
    for key in legacy_output:
        assert torch.equal(enabled_output[key], legacy_output[key])
    assert torch.equal(
        enabled_output["presence_logits"].contiguous().view(torch.int32),
        legacy_output["presence_logits"].contiguous().view(torch.int32),
    )


def test_enabled_checkpoint_roundtrip_is_strict_and_output_preserving() -> None:
    torch.manual_seed(13)
    source = _model(presence_competition="deepsets")
    assert source.presence_competition is not None
    with torch.no_grad():
        source.presence_competition.output_projection.weight.normal_()
        source.presence_competition.output_projection.bias.normal_()
    target = _model(presence_competition="deepsets")
    target.load_state_dict(source.state_dict(), strict=True)
    inputs = _inputs(source)

    with torch.no_grad():
        source_output = _forward(source, inputs)
        target_output = _forward(target, inputs)

    for key in source_output:
        assert torch.equal(target_output[key], source_output[key])


def test_enabled_and_disabled_checkpoints_are_strictly_incompatible() -> None:
    disabled = _model(predict_canonical_pose=False)
    enabled = _model(
        predict_canonical_pose=False,
        presence_competition="deepsets",
    )

    with pytest.raises(RuntimeError, match="presence_competition"):
        enabled.load_state_dict(disabled.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="presence_competition"):
        disabled.load_state_dict(enabled.state_dict(), strict=True)


@pytest.mark.parametrize(
    "presence_competition",
    ["deepsets", "deepsets_centered"],
)
def test_enabled_competition_preserves_all_padding_zero_contract(
    presence_competition: str,
) -> None:
    model = _model(presence_competition=presence_competition)
    inputs = _inputs(model)
    inputs["padding_mask"][:] = True

    with torch.no_grad():
        output = _forward(model, inputs)

    for value in output.values():
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == 0


def test_model_uses_shared_stages_with_fixed_cswa_global_cycle() -> None:
    model = _model()

    assert all(isinstance(stage, FixedQueryTrackStage) for stage in model.stages)
    assert [stage.is_global for stage in model.stages] == [False, False, False, True]
    assert [
        stage.object_temporal_block.cfg.attention_type for stage in model.stages
    ] == [
        "cswa",
        "cswa",
        "cswa",
        "mha",
    ]
    assert [
        stage.query_temporal_block.cfg.attention_type for stage in model.stages
    ] == [
        "cswa",
        "cswa",
        "cswa",
        "mha",
    ]


def test_configured_ffn_reaches_every_track_query_block() -> None:
    model = _model(ffn_type="deepseek_v4_swiglu")
    blocks = [
        block
        for stage in model.stages
        for block in (
            stage.object_temporal_block,
            stage.spatial_block,
            stage.query_temporal_block,
        )
    ]

    assert blocks
    assert all(isinstance(block.ffn, DeepSeekV4SwiGLU) for block in blocks)


def test_invisible_joint_coordinates_do_not_affect_predictions() -> None:
    torch.manual_seed(9)
    model = _model()
    inputs = _inputs(model)
    inputs["human_vis"][:, 1, :, 1] = False
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["human_kp"][:, 1, :, 1] = torch.nan

    with torch.no_grad():
        output = _forward(model, inputs)
        changed_output = _forward(model, changed)

    for key in output:
        torch.testing.assert_close(output[key], changed_output[key])


def test_nonpadding_invisible_tokens_receive_gradient() -> None:
    torch.manual_seed(12)
    model = _model()
    model.train()
    inputs = _inputs(model)
    inputs["human_vis"][:] = False
    output = _forward(model, inputs)
    sum(value.square().sum() for value in output.values()).backward()

    gradient = model.invisible_token.token.grad
    assert gradient is not None
    assert bool(gradient.abs().sum() > 0)


def test_visibility_never_changes_object_attention_participation() -> None:
    model = _model()
    inputs = _inputs(model)
    inputs["human_vis"].zero_()
    observed: dict[str, torch.Tensor] = {}

    def capture_state(
        _module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        object_tokens = args[0]
        object_valid = kwargs["object_state_valid"]
        assert isinstance(object_tokens, torch.Tensor)
        assert isinstance(object_valid, torch.Tensor)
        observed["tokens"] = object_tokens
        observed["valid"] = object_valid

    hook = model.stages[0].register_forward_pre_hook(capture_state, with_kwargs=True)
    try:
        with torch.no_grad():
            _forward(model, inputs)
    finally:
        hook.remove()

    assert observed["tokens"].shape == (1, 2, 3, model.num_queries, model.hidden_dim)
    assert observed["valid"].all()


def test_nonrectangular_padding_is_forwarded_as_object_state_validity() -> None:
    model = _model()
    inputs = _inputs(model)
    inputs["padding_mask"] = torch.tensor(
        [[[False, True, True], [True, False, True]]]
    )
    observed: dict[str, torch.Tensor] = {}

    def capture_state(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        object_valid = kwargs["object_state_valid"]
        assert isinstance(object_valid, torch.Tensor)
        observed["valid"] = object_valid

    hook = model.stages[0].register_forward_pre_hook(capture_state, with_kwargs=True)
    try:
        with torch.no_grad():
            _forward(model, inputs)
    finally:
        hook.remove()

    expected = (~inputs["padding_mask"]).unsqueeze(-1).expand(
        -1, -1, -1, model.num_queries
    )
    torch.testing.assert_close(observed["valid"], expected)


def test_padded_values_cannot_change_valid_outputs() -> None:
    torch.manual_seed(15)
    model = _model()
    inputs = _inputs(model)
    inputs["padding_mask"][:, 1] = True
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["human_kp"][:, 1] = torch.nan
    changed["court_kp"][:, 1] = torch.nan

    with torch.no_grad():
        output = _forward(model, inputs)
        changed_output = _forward(model, changed)

    for key in output:
        torch.testing.assert_close(output[key], changed_output[key])


def test_all_padding_outputs_are_finite_and_zero() -> None:
    model = _model()
    inputs = _inputs(model)
    inputs["padding_mask"][:] = True

    with torch.no_grad():
        output = _forward(model, inputs)

    assert set(output) == {
        "position",
        "rotation",
        "presence_logits",
        "canonical_pose",
    }
    assert output["canonical_pose"].shape == (1, 3, model.num_queries, 17, 3)
    for value in output.values():
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value) == 0


def test_prediction_is_invariant_to_batch_composition() -> None:
    torch.manual_seed(18)
    model = _model()
    inputs = _inputs(model)
    companion = _inputs(model)
    batched = {
        key: torch.cat((value, companion[key]), dim=0)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        single = _forward(model, inputs)
        composed = _forward(model, batched)

    for key in single:
        torch.testing.assert_close(single[key], composed[key][:1])


def test_old_track_query_state_dict_is_intentionally_strictly_incompatible() -> None:
    model = _model()
    old_state = {
        "slot_embeddings": model.slot_embeddings.detach().clone(),
        "spatial_blocks.0.attn_norm.weight": torch.ones(model.hidden_dim),
    }

    with pytest.raises(RuntimeError, match="spatial_blocks"):
        model.load_state_dict(old_state, strict=True)


def test_requested_unavailable_cuda_backend_does_not_fall_back(
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
        _model(backend="cuda")
