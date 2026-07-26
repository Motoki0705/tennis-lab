from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.plcs.models import PLCSTrackQueryModel
from src.utils.models.embeddings import CourtPlayerGroupEmbedding


def _model(*, mask_invisible: bool = True) -> PLCSTrackQueryModel:
    config = OmegaConf.load(Path("src/tasks/plcs/configs/model/track_query.yaml"))
    config.mask_invisible_observations = mask_invisible
    model = PLCSTrackQueryModel(config)
    model.eval()
    return model


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
        output = model(**inputs)
        changed_output = model(**changed)

    for key in output:
        torch.testing.assert_close(output[key], changed_output[key])


def test_model_rejects_incomplete_court_annotation() -> None:
    model = _model()
    with pytest.raises(ValueError, match="all 14 annotated UV points"):
        model(
            human_kp=torch.zeros(1, 1, 1, 1, 17, 2),
            detection_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
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
    torch.manual_seed(12)
    model = _model(mask_invisible=mask_invisible)
    model.train()
    detection_mask = torch.tensor([[[[True, False]]]])
    output = model(
        human_kp=torch.rand(1, 1, 1, 2, 17, 2),
        detection_mask=detection_mask,
        court_kp=torch.rand(1, 1, 1, 14, 2),
        court_vis=torch.ones(1, 1, 1, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        view_mask=torch.ones(1, 1, dtype=torch.bool),
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
