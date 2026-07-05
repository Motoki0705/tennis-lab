from __future__ import annotations

import torch.nn as nn
from omegaconf import OmegaConf

from src.tasks.court_detection.models import build_court_detection_model
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


def test_dinov3_dpt_can_build_for_keypoint_heatmaps(monkeypatch) -> None:
    expected = DummyModel()

    def fake_from_config(config: object) -> DummyModel:
        return expected

    monkeypatch.setattr(CourtHierarchicalModel, "from_config", fake_from_config)
    config = OmegaConf.create(
        {
            "data": {"task": "kp", "num_keypoints": 14},
            "model": {
                "name": "court_hierarchical",
                "num_classes": 14,
                "encoder": {"name": "dinov3"},
                "decoder": {"name": "dpt"},
            },
        }
    )

    model = build_court_detection_model(config)

    assert model is expected
