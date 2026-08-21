from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel
from src.utils.models.embeddings import CourtPlayerGroupEmbedding


def _model() -> PLCSTrackQueryModel:
    raw = OmegaConf.load(Path("src/tasks/plcs/configs/model/track_query.yaml"))
    config = PLCSModelConfig.from_mapping(
        cast("dict[str, object]", OmegaConf.to_container(raw, resolve=True))
    )
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
        num_detections=2,
        num_queries=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(coordinates[0, :3], torch.zeros(3, 3, dtype=torch.long))
    assert torch.equal(
        coordinates[0, 3:],
        torch.tensor([[0, 1, 1], [0, 1, 1], [0, 2, 1], [0, 2, 1]]),
    )


def test_model_uses_shared_court_player_group_embedding() -> None:
    assert isinstance(_model().group_embed, CourtPlayerGroupEmbedding)


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
