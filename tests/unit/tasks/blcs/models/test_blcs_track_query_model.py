from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.blcs.configuration import TrackQueryModelConfig, parse_model_config
from src.tasks.blcs.models import BLCSTrackQueryModel
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
            overrides=["model=track_query_large_point_attention"],
        )
    parsed = parse_model_config(config)
    if not isinstance(parsed, TrackQueryModelConfig):
        raise AssertionError("Point-attention config must compose a track-query model.")
    return parsed


def _model(*, mask_invisible: bool = True) -> BLCSTrackQueryModel:
    model = BLCSTrackQueryModel(_model_config(mask_invisible=mask_invisible))
    model.eval()
    return model


def test_spatial_coordinates_share_role_within_id_ordered_object_axis() -> None:
    coordinates = BLCSTrackQueryModel.build_spatial_coordinates(
        batch_size=1,
        num_frames=2,
        num_views=2,
        num_detections=3,
        num_queries=2,
        device=torch.device("cpu"),
    ).view(1, 2, 8, 3)
    assert torch.equal(coordinates[0, 1, :2], torch.tensor([[1, 0, 0], [1, 0, 0]]))
    assert torch.equal(
        coordinates[0, 0, 2:],
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
    block = model.spatial_blocks[0]
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

    assert isinstance(model.group_embed, CourtBallGroupEmbedding)


def test_point_attention_model_uses_small_fusion_before_model_projection() -> None:
    config = _point_attention_model_config()
    model = BLCSTrackQueryModel(config)

    assert model.observation_fusion == "point_attention"
    assert model.group_embed is None
    assert model.point_fusion is not None
    assert model.point_fusion.token_dim == 32
    assert model.point_fusion.output_projection.in_features == 32
    assert model.point_fusion.output_projection.out_features == model.hidden_dim


def test_point_attention_model_forward_preserves_tracking_output_contract() -> None:
    torch.manual_seed(2)
    model = BLCSTrackQueryModel(_point_attention_model_config()).eval()
    with torch.no_grad():
        output = model(
            ball_uv=torch.rand(1, 2, 3, 4, 2),
            ball_visible=torch.ones(1, 2, 3, 4, dtype=torch.bool),
            court_kp=torch.rand(1, 2, 3, 14, 2),
            court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
            frame_mask=torch.ones(1, 3, dtype=torch.bool),
            view_mask=torch.ones(1, 2, dtype=torch.bool),
        )

    assert output["position"].shape == (1, 3, 4, 3)
    assert output["presence_logits"].shape == (1, 3, 4)


def test_masked_candidate_and_court_coordinates_do_not_affect_predictions() -> None:
    torch.manual_seed(7)
    model = _model()
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 2, 2),
        "ball_visible": torch.ones(1, 2, 3, 2, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    inputs["ball_visible"][:, 1, :, 1] = False
    inputs["court_vis"][:, 1, :, 3] = False
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["ball_uv"][:, 1, :, 1] = torch.nan
    changed["court_kp"][:, 1, :, 3] = torch.nan

    with torch.no_grad():
        output = model(**inputs)
        changed_output = model(**changed)

    for key in output:
        torch.testing.assert_close(output[key], changed_output[key])


def test_model_rejects_incomplete_court_annotation() -> None:
    model = _model()
    with pytest.raises(ValueError, match="all 14 annotated UV points"):
        model(
            ball_uv=torch.zeros(1, 1, 1, 1, 2),
            ball_visible=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            court_kp=torch.zeros(1, 1, 1, 13, 2),
            court_vis=torch.ones(1, 1, 1, 13, dtype=torch.bool),
            frame_mask=torch.ones(1, 1, dtype=torch.bool),
            view_mask=torch.ones(1, 1, dtype=torch.bool),
        )


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
    visible = torch.tensor([[[[True, False]]]])
    output = model(
        ball_uv=torch.rand(1, 1, 1, 2, 2),
        ball_visible=visible,
        court_kp=torch.rand(1, 1, 1, 14, 2),
        court_vis=torch.ones(1, 1, 1, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        view_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    (
        output["position"].square().sum() + output["presence_logits"].square().sum()
    ).backward()
    if model.invisible_token is None:
        raise AssertionError("Linear observation fusion requires an invisible token.")
    gradient = model.invisible_token.token.grad
    has_gradient = gradient is not None and bool(gradient.abs().sum() > 0)
    assert has_gradient is expect_gradient
