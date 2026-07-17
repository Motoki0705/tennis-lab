from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.blcs.models import BLCSTrackQueryModel


def _model() -> BLCSTrackQueryModel:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/model/track_query.yaml"))
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
    ).view(1, 2, 8, 3)
    assert torch.equal(coordinates[0, 1, :2], torch.tensor([[1, 0, 0], [1, 0, 0]]))
    assert torch.equal(
        coordinates[0, 0, 2:],
        torch.tensor(
            [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 2, 1], [0, 2, 1], [0, 2, 1]]
        ),
    )


def test_candidate_permutation_does_not_change_slot_predictions() -> None:
    torch.manual_seed(3)
    model = _model()
    shape = (1, 2, 4, 5)
    inputs = {
        "ball_uv": torch.rand(*shape, 2),
        "ball_score": torch.rand(*shape),
        "ball_candidate_mask": torch.ones(*shape, dtype=torch.bool),
        "ball_visible": torch.ones(*shape, dtype=torch.bool),
        "frame_mask": torch.ones(1, 4, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = {
        key: value[..., permutation, :] if key == "ball_uv" else value[..., permutation]
        for key, value in inputs.items()
        if key not in {"frame_mask", "view_mask"}
    }
    permuted["frame_mask"] = inputs["frame_mask"]
    permuted["view_mask"] = inputs["view_mask"]
    with torch.no_grad():
        output = model(**inputs)
        permuted_output = model(**permuted)
    for key in output:
        torch.testing.assert_close(
            output[key], permuted_output[key], atol=1e-5, rtol=1e-5
        )
