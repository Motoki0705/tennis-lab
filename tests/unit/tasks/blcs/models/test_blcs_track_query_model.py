from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.blcs.configuration import TrackQueryModelConfig, parse_model_config
from src.tasks.blcs.model_io.attention_masks import prepare_tracking_attention_masks
from src.tasks.blcs.models import BLCSTrackQueryModel
from src.tasks.blcs.models.components.observation_fusion import (
    LinearTrackObservationFusion,
    PointAttentionTrackObservationFusion,
)
from src.tasks.blcs.models.components.track_query_stage import BLCSTrackQueryStage
from src.utils.configuration import ConfigurationTypeError
from src.utils.models import TransformerBlock
from src.utils.models.embeddings import CourtBallGroupEmbedding

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _composed_config(*, mask_invisible: bool = True) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking")
    config.model.mask_invisible_observations = mask_invisible
    return config


def _model_config(*, mask_invisible: bool = True) -> TrackQueryModelConfig:
    parsed = parse_model_config(_composed_config(mask_invisible=mask_invisible))
    if not isinstance(parsed, TrackQueryModelConfig):
        raise AssertionError("train_tracking must compose a track-query model.")
    return parsed


def _point_attention_model_config() -> TrackQueryModelConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                "model=track_query_large_point_attention",
                "model.hidden_dim=64",
                "model.num_heads=4",
                "model.num_stages=4",
                "model.ffn_dim=128",
                "model.rope_dim=16",
                "model.dropout=0.0",
            ],
        )
    parsed = parse_model_config(config)
    if not isinstance(parsed, TrackQueryModelConfig):
        raise AssertionError("Point-attention config must compose a track-query model.")
    return parsed


def _model(*, mask_invisible: bool = True) -> BLCSTrackQueryModel:
    model = BLCSTrackQueryModel(_model_config(mask_invisible=mask_invisible))
    model.eval()
    return model


def _forward(
    model: BLCSTrackQueryModel,
    inputs: dict[str, torch.Tensor],
    *,
    mask_invisible: bool = True,
) -> dict[str, torch.Tensor]:
    masks = prepare_tracking_attention_masks(
        ball_visible=inputs["ball_visible"],
        candidate_mask=inputs["candidate_mask"],
        court_visible=inputs["court_vis"],
        frame_mask=inputs["frame_mask"],
        view_mask=inputs["view_mask"],
        num_queries=model.num_queries,
        mask_invisible_observations=mask_invisible,
    )
    return cast(
        "dict[str, torch.Tensor]",
        model(
            ball_uv=inputs["ball_uv"],
            ball_visible=inputs["ball_visible"],
            candidate_mask=inputs["candidate_mask"],
            court_kp=inputs["court_kp"],
            court_vis=inputs["court_vis"],
            frame_mask=inputs["frame_mask"],
            camera_state_valid=masks.camera_state_valid,
            spatial_attention_mask=masks.spatial_attention_mask,
            object_temporal_state_valid=masks.object_temporal_state_valid,
            object_temporal_attention_mask=masks.object_temporal_attention_mask,
            query_temporal_state_valid=masks.query_temporal_state_valid,
            query_temporal_attention_mask=masks.query_temporal_attention_mask,
            point_attention_mask=masks.point_attention_mask,
        ),
    )


def test_spatial_coordinates_share_role_within_id_ordered_object_axis() -> None:
    coordinates = BLCSTrackQueryModel.build_spatial_coordinates(
        batch_size=1,
        num_frames=2,
        num_views=2,
        num_detections=3,
        num_queries=3,
        device=torch.device("cpu"),
    ).view(1, 2, 9, 3)
    assert torch.equal(
        coordinates[0, 1, :3],
        torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
    )
    assert torch.equal(
        coordinates[0, 0, 3:],
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


def test_model_uses_composed_ffn_and_rope_dimensions() -> None:
    config = _model_config()
    model = BLCSTrackQueryModel(config)

    assert model.rope_dim == config.rope_dim
    block = model.stages[0].spatial_block
    if not isinstance(block, TransformerBlock):
        raise AssertionError("Track-query spatial trunk must use TransformerBlock.")
    assert block.cfg.ffn_dim == config.ffn_dim


@pytest.mark.parametrize("field", ["ffn_dim", "rope_dim"])
def test_model_config_rejects_null_architecture_fields(field: str) -> None:
    config = _composed_config()
    config.model[field] = None

    with pytest.raises(ConfigurationTypeError, match=rf"model\.{field}"):
        parse_model_config(config)


def test_default_model_uses_shared_court_ball_group_embedding() -> None:
    model = _model()

    assert isinstance(model.observation_encoder, LinearTrackObservationFusion)
    assert isinstance(
        model.observation_encoder.group_embedding, CourtBallGroupEmbedding
    )


def test_point_attention_model_uses_small_fusion_before_model_projection() -> None:
    config = _point_attention_model_config()
    model = BLCSTrackQueryModel(config)

    assert isinstance(model.observation_encoder, PointAttentionTrackObservationFusion)
    fusion = model.observation_encoder.point_fusion
    assert fusion.token_dim == 32
    assert fusion.output_projection.in_features == 32
    assert fusion.output_projection.out_features == model.hidden_dim


def test_point_attention_model_forward_preserves_tracking_output_contract() -> None:
    torch.manual_seed(2)
    model = BLCSTrackQueryModel(_point_attention_model_config()).eval()
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 4, 2),
        "ball_visible": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with torch.no_grad():
        output = _forward(model, inputs)

    assert output["position"].shape == (1, 3, 4, 3)
    assert output["presence_logits"].shape == (1, 3, 4)


def test_masked_candidate_and_court_coordinates_do_not_affect_predictions() -> None:
    torch.manual_seed(7)
    model = _model()
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 4, 2),
        "ball_visible": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    inputs["ball_visible"][:, 1, :, 1] = False
    inputs["candidate_mask"][:, 1, :, 1] = False
    inputs["court_vis"][:, 1, :, 3] = False
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["ball_uv"][:, 1, :, 1] = torch.nan
    changed["court_kp"][:, 1, :, 3] = torch.nan

    with torch.no_grad():
        output = _forward(model, inputs)
        changed_output = _forward(model, changed)

    for key in output:
        torch.testing.assert_close(output[key], changed_output[key])


@pytest.mark.parametrize(
    ("mask_invisible", "expect_gradient"),
    [(True, False), (False, True)],
)
def test_invisible_token_memory_ablation_controls_gradient(
    mask_invisible: bool,
    expect_gradient: bool,
) -> None:
    torch.manual_seed(11)
    model = _model(mask_invisible=mask_invisible)
    model.train()
    visible = torch.tensor([[[[True, False, False, False]]]])
    output = _forward(
        model,
        {
            "ball_uv": torch.rand(1, 1, 1, 4, 2),
            "ball_visible": visible,
            "candidate_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
            "court_kp": torch.rand(1, 1, 1, 14, 2),
            "court_vis": torch.ones(1, 1, 1, 14, dtype=torch.bool),
            "frame_mask": torch.ones(1, 1, dtype=torch.bool),
            "view_mask": torch.ones(1, 1, dtype=torch.bool),
        },
        mask_invisible=mask_invisible,
    )
    (
        output["position"].square().sum() + output["presence_logits"].square().sum()
    ).backward()
    if not isinstance(model.observation_encoder, LinearTrackObservationFusion):
        raise AssertionError("Default model must use linear observation fusion.")
    gradient = model.observation_encoder.invisible_token.token.grad
    has_gradient = gradient is not None and bool(gradient.abs().sum() > 0)
    assert has_gradient is expect_gradient


def test_stage_schedule_is_constructor_fixed_cccg_with_matching_modes() -> None:
    model = _model()
    stages = [cast(BLCSTrackQueryStage, stage) for stage in model.stages]

    assert [stage.is_global for stage in stages] == [False, False, False, True]
    for stage in stages:
        expected = "mha" if stage.is_global else "cswa"
        assert stage.object_temporal_block.cfg.attention_type == expected
        assert stage.query_temporal_block.cfg.attention_type == expected
        assert stage.spatial_block.cfg.attention_type == "mha"
        assert not hasattr(stage.object_temporal_block.cfg, "recompute_ffn")
        assert not hasattr(stage.query_temporal_block.cfg, "recompute_ffn")
        assert not hasattr(stage.spatial_block.cfg, "recompute_ffn")


def test_object_path_uses_forward_update_and_stage_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    stage = cast(BLCSTrackQueryStage, model.stages[0])
    observed: dict[str, tuple[int, ...]] = {}
    original_update = stage.object_temporal_block.forward_update

    def record_update(
        values: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        observed["object"] = tuple(values.shape)
        return original_update(values, **kwargs)

    def prohibit_full_forward(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("object temporal path must use forward_update")

    def record_spatial(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        observed["spatial"] = tuple(args[0].shape)

    def record_query(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        observed["query"] = tuple(args[0].shape)

    monkeypatch.setattr(stage.object_temporal_block, "forward_update", record_update)
    monkeypatch.setattr(stage.object_temporal_block, "forward", prohibit_full_forward)
    hook = stage.spatial_block.register_forward_pre_hook(record_spatial)
    query_hook = stage.query_temporal_block.register_forward_pre_hook(record_query)
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 4, 2),
        "ball_visible": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    try:
        with torch.no_grad():
            output = _forward(model, inputs)
    finally:
        hook.remove()
        query_hook.remove()

    assert observed == {
        "object": (2, 3, 64),
        "spatial": (3, 12, 64),
        "query": (4, 3, 64),
    }
    assert output["position"].shape == (1, 3, 4, 3)


@pytest.mark.parametrize("mask_invisible", [True, False])
def test_all_invisible_candidates_remain_finite(mask_invisible: bool) -> None:
    model = _model(mask_invisible=mask_invisible)
    inputs = {
        "ball_uv": torch.rand(1, 1, 3, 4, 2),
        "ball_visible": torch.zeros(1, 1, 3, 4, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 1, 3, 14, 2),
        "court_vis": torch.ones(1, 1, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    with torch.no_grad():
        output = _forward(model, inputs, mask_invisible=mask_invisible)

    assert torch.isfinite(output["position"]).all()
    assert torch.isfinite(output["presence_logits"]).all()


def test_padded_frame_outputs_are_zero() -> None:
    model = _model()
    inputs = {
        "ball_uv": torch.rand(1, 1, 3, 4, 2),
        "ball_visible": torch.ones(1, 1, 3, 4, dtype=torch.bool),
        "candidate_mask": torch.ones(1, 1, 3, 4, dtype=torch.bool),
        "court_kp": torch.rand(1, 1, 3, 14, 2),
        "court_vis": torch.ones(1, 1, 3, 14, dtype=torch.bool),
        "frame_mask": torch.tensor([[True, True, False]]),
        "view_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    inputs["ball_visible"][:, :, -1] = False
    inputs["candidate_mask"][:, :, -1] = False
    inputs["court_vis"][:, :, -1] = False

    with torch.no_grad():
        output = _forward(model, inputs)

    assert not output["position"][:, -1].any()
    assert not output["presence_logits"][:, -1].any()
