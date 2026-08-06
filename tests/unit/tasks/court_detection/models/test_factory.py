from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.model_io.adapters import CourtKeypointModelIO
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def test_dinov3_dpt_can_build_for_keypoint_heatmaps(monkeypatch) -> None:
    expected = object.__new__(CourtHierarchicalModel)
    torch.nn.Module.__init__(expected)
    expected.in_channels = 3
    expected.num_classes = 14
    encoder = object.__new__(CourtDINOv3Encoder)
    torch.nn.Module.__init__(encoder)
    expected.encoder = encoder

    def fake_from_config(config: object) -> CourtHierarchicalModel:
        _ = config
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

    pair = build_court_detection_pair(config)

    assert pair.model is expected
    assert isinstance(pair.adapter, CourtKeypointModelIO)
