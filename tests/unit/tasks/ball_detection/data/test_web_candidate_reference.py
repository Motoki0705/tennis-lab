"""Web static sampling supplies raw candidate targets without repeating counts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.types import FrameLabel
from src.tasks.ball_detection.data.web_datamodule import WebBallDetectionDataset
from src.tasks.ball_detection.training.candidate_recall import ValidationCandidateRecall
from tests.support.tasks.ball_detection.store import store_config


def test_web_static_repeats_count_once_and_distractors_are_not_targets(tmp_path: Path) -> None:
    labels = [
        [FrameLabel(1, 16, 24), FrameLabel(1, 40, 40, role="distractor")],
        [FrameLabel(1, 16, 24), FrameLabel(1, 40, 40)],
        [],
    ]
    store = SimpleNamespace(
        original_size=lambda _: (64, 48),
        source_name=lambda _: "web_fixture",
        labels=lambda index: labels[index],
        decode_bgr=lambda _: np.zeros((48, 64, 3), dtype=np.uint8),
    )
    config = store_config(tmp_path)
    dataset = WebBallDetectionDataset(
        store=cast(Any, store), sample_windows=[(0, 0), (1, 1), (2, 2)], config=config,
    )
    first = dataset[0]["candidate_reference"]
    assert first["frame_id"].tolist() == [0, 0]
    assert first["observed"].tolist() == [True, True]
    assert first["source_scale"].item() == 1
    assert first["namespace"] == "web"
    assert not dataset[1]["candidate_reference"]["observed"].any()
    metric = ValidationCandidateRecall(config.training.validation_candidates)
    values = torch.zeros(3, 2, 11, 21)
    values[:, :, 5, 5] = 0.1
    metric.update(values, next(iter(DataLoader(dataset, batch_size=3))))
    report = metric.compute()[""]
    assert report["frames"] == 3
    assert report["observed"] == report["recalled_at_k"] == 1
