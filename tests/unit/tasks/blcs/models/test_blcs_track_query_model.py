from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.blcs.models import BLCSTrackQueryModel


def _model(*, mask_invisible: bool = True) -> BLCSTrackQueryModel:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/model/track_query.yaml"))
    config.mask_invisible_observations = mask_invisible
    model = BLCSTrackQueryModel(config)
    model.eval()
    return model


def test_spatial_coordinates_use_camera_and_role_not_candidate_index() -> None:
    coordinates = BLCSTrackQueryModel.build_spatial_coordinates(
        batch_size=1,
        num_frames=2,
        num_views=2,
        num_detections=3,
        num_queries=2,
        device=torch.device("cpu"),
    ).view(1, 2, 10, 3)
    assert torch.equal(coordinates[0, 1, :2], torch.tensor([[1, 0, 0], [1, 0, 0]]))
    assert torch.equal(
        coordinates[0, 0, 2:],
        torch.tensor(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 1],
                [0, 2, 1],
                [0, 2, 1],
                [0, 2, 2],
            ]
        ),
    )


def test_candidate_permutation_does_not_change_slot_predictions() -> None:
    torch.manual_seed(3)
    model = _model()
    shape = (1, 2, 4, 5)
    inputs = {
        "ball_uv": torch.rand(*shape, 2),
        "ball_visible": torch.ones(*shape, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 4, 14, 2),
        "court_vis": torch.ones(1, 2, 4, 14, dtype=torch.bool),
        "frame_mask": torch.ones(1, 4, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = {
        key: value[..., permutation, :] if key == "ball_uv" else value[..., permutation]
        for key, value in inputs.items()
        if key not in {"frame_mask", "view_mask", "court_kp", "court_vis"}
    }
    permuted["frame_mask"] = inputs["frame_mask"]
    permuted["view_mask"] = inputs["view_mask"]
    permuted["court_kp"] = inputs["court_kp"]
    permuted["court_vis"] = inputs["court_vis"]
    with torch.no_grad():
        output = model(**inputs)
        permuted_output = model(**permuted)
    for key in output:
        torch.testing.assert_close(
            output[key], permuted_output[key], atol=1e-5, rtol=1e-5
        )


def test_masked_candidate_coordinates_do_not_affect_predictions() -> None:
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
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["ball_uv"][:, 1, :, 1] = torch.nan

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
    gradient = model.invisible_token.token.grad
    has_gradient = gradient is not None and bool(gradient.abs().sum() > 0)
    assert has_gradient is expect_gradient
