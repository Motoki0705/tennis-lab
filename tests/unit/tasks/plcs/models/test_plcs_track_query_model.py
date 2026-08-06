from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.model_io.attention_masks import prepare_tracking_attention_masks
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel
from src.utils.models.embeddings import CourtPlayerGroupEmbedding


def _model(*, mask_invisible: bool = True) -> PLCSTrackQueryModel:
    raw = OmegaConf.load(Path("src/tasks/plcs/configs/model/track_query.yaml"))
    raw.mask_invisible_observations = mask_invisible
    config = PLCSModelConfig.from_mapping(
        cast(
            "dict[str, object]",
            OmegaConf.to_container(raw, resolve=True),
        )
    )
    model = PLCSTrackQueryModel(config)
    model.eval()
    return model


def _forward(
    model: PLCSTrackQueryModel,
    inputs: dict[str, torch.Tensor],
    *,
    mask_invisible: bool = True,
) -> dict[str, torch.Tensor]:
    camera_valid, spatial_mask, temporal_mask = prepare_tracking_attention_masks(
        detection_mask=inputs["detection_mask"],
        frame_mask=inputs["frame_mask"],
        view_mask=inputs["view_mask"],
        num_queries=model.num_queries,
        mask_invisible_observations=mask_invisible,
    )
    return cast(
        "dict[str, torch.Tensor]",
        model(
            human_kp=inputs["human_kp"],
            detection_mask=inputs["detection_mask"],
            court_kp=inputs["court_kp"],
            court_vis=inputs["court_vis"],
            frame_mask=inputs["frame_mask"],
            camera_state_valid=camera_valid,
            spatial_attention_mask=spatial_mask,
            temporal_attention_mask=temporal_mask,
        ),
    )


def test_player_role_coordinates_share_role_within_id_ordered_object_axis() -> None:
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
        torch.tensor(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 2, 1],
                [0, 2, 1],
            ]
        ),
    )


def test_model_uses_shared_court_player_group_embedding() -> None:
    model = _model()

    assert isinstance(model.group_embed, CourtPlayerGroupEmbedding)


def test_masked_detection_coordinates_do_not_affect_predictions() -> None:
    torch.manual_seed(9)
    model = _model()
    inputs = {
        "human_kp": torch.rand(1, 2, 3, 2, 17, 2),
        "detection_mask": torch.ones(1, 2, 3, 2, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 14, 2),
        "court_vis": torch.ones(1, 2, 3, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    inputs["detection_mask"][:, 1, :, 1] = False
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["human_kp"][:, 1, :, 1] = torch.nan

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
    torch.manual_seed(12)
    model = _model(mask_invisible=mask_invisible)
    model.train()
    detection_mask = torch.tensor([[[[True, False]]]])
    output = _forward(
        model,
        {
            "human_kp": torch.rand(1, 1, 1, 2, 17, 2),
            "detection_mask": detection_mask,
            "court_kp": torch.rand(1, 1, 1, 14, 2),
            "court_vis": torch.ones(1, 1, 1, 14, dtype=torch.bool),
            "frame_mask": torch.ones(1, 1, dtype=torch.bool),
            "view_mask": torch.ones(1, 1, dtype=torch.bool),
        },
        mask_invisible=mask_invisible,
    )
    loss = (
        output["position"].square().sum()
        + output["rotation"].square().sum()
        + output["presence_logits"].square().sum()
    )
    loss.backward()
    gradient = model.invisible_token.token.grad
    has_gradient = gradient is not None and bool(gradient.abs().sum() > 0)
    assert has_gradient is expect_gradient
