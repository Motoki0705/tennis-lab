"""Packed Court input remains usable after the source images are removed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.inputs.tennis_store_migration import (
    migrate_tennis_store,
)
from src.tasks.court_detection.data.target_generation.online import (
    generate_online_targets,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    SEMANTIC_LINE_TARGET_SCHEMA,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def test_migration_preserves_jpeg_points_and_splits_without_source_files(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "source", tmp_path / "packed"
    (source / "images").mkdir(parents=True)
    geometry = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    points = torch.stack(
        ((geometry[:, 0] / 12 + 0.5) * 255, (0.5 - geometry[:, 1] / 26) * 255), dim=-1
    ).tolist()
    original_jpegs = {}
    for split, name in (("train", "first"), ("val", "second")):
        path = source / "images" / f"{name}.jpg"
        Image.fromarray(np.full((256, 256, 3), 120, np.uint8)).save(path, quality=98)
        original_jpegs[name] = path.read_bytes()
        (source / f"data_{split}.json").write_text(
            json.dumps([{"id": name, "kps": points, "metric": 0.25}])
        )
    report = migrate_tennis_store(source, destination, excluded_sample_ids=())
    assert report["jpeg_bytes_verified"] is True
    shutil.rmtree(source)
    layer = TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            "tennis_court_detector",
            destination,
            {"train": "train", "val": "val", "test": None},
            (),
        )
    )
    assert layer.available_splits == ("train", "val")
    for store_split in layer.available_splits:
        record = layer.records(store_split)[0]
        raw = layer.load(record)

        assert raw.keypoint_channels is not None
        torch.testing.assert_close(
            raw.keypoint_channels.points_xy[:, 0], torch.tensor(points)
        )
        assert (
            layer.image_store.jpeg(record.payload["image_index"])
            == original_jpegs[record.sample_id]
        )
        dense = generate_online_targets(
            raw,
            {
                "seg": SEGMENTATION_TARGET_SCHEMA,
                "line": LINE_TARGET_SCHEMA,
                "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
            },
        )
        assert all(value.any() for value in dense.values())


def test_packed_input_never_falls_back_to_loose_images(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "data_train.json").write_text("[]")
    config = TennisCourtDetectorSourceConfig(
        "tennis_court_detector",
        tmp_path,
        {"train": "train", "val": "val", "test": None},
        (),
    )
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        TennisCourtDetectorInput(config)
