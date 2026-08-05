from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.models import build_court_detection_model
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


def test_dinov3_dpt_can_build_for_keypoint_heatmaps(monkeypatch) -> None:
    expected = DummyModel()

    def fake_from_config(config: object) -> DummyModel:
        return expected

    monkeypatch.setattr(CourtHierarchicalModel, "from_config", fake_from_config)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data=court_kp",
                "model/encoder=dinov3",
                "model/decoder=dpt",
                "loss=kp",
            ],
        )

    model = build_court_detection_model(config)

    assert model is expected
