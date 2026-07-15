from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.player_tracking.models import PlayerTrackQueryModel


def _model() -> PlayerTrackQueryModel:
    config = OmegaConf.load(Path("src/tasks/player_tracking/configs/train.yaml")).model
    model = PlayerTrackQueryModel(config)
    model.eval()
    return model


def test_player_role_coordinates_do_not_encode_detection_index() -> None:
    coordinates = PlayerTrackQueryModel.build_spatial_coordinates(
        batch_size=1,
        num_frames=1,
        num_views=2,
        num_detections=2,
        num_queries=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(coordinates[0, :3], torch.zeros(3, 3, dtype=torch.long))
    assert torch.equal(
        coordinates[0, 3:], torch.tensor([[0, 1, 1], [0, 1, 1], [0, 2, 1], [0, 2, 1]])
    )


def test_detection_permutation_keeps_player_slot_outputs_identical() -> None:
    torch.manual_seed(5)
    model = _model()
    shape = (1, 2, 4, 3)
    inputs = {
        "human_kp": torch.rand(*shape, 17, 2),
        "human_vis": torch.ones(*shape, 17, dtype=torch.bool),
        "detection_mask": torch.ones(*shape, dtype=torch.bool),
        "detection_score": torch.rand(*shape),
        "bbox": torch.rand(*shape, 4),
        "frame_mask": torch.ones(1, 4, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(inputs)
    for key in ("human_kp", "human_vis", "detection_mask", "detection_score", "bbox"):
        permuted[key] = inputs[key][:, :, :, permutation]
    with torch.no_grad():
        output = model(**inputs)
        permuted_output = model(**permuted)
    for key in output:
        torch.testing.assert_close(
            output[key], permuted_output[key], atol=1e-5, rtol=1e-5
        )
