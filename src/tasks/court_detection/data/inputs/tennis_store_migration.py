"""One-time import of the upstream TennisCourtDetector image/KP14 release."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    LegacyTennisCourtDetectorInput,
    TennisCourtDetectorInput,
)
from src.utils.data.image_record_store import (
    ImageRecordStore,
    ImageRecordWriter,
    encode_rgb_jpeg,
)
from src.utils.io import save_json_atomic


def migrate_tennis_store(
    source: Path,
    destination: Path,
    *,
    excluded_sample_ids: tuple[str, ...] = ("QszoUKyCOHo_600",),
) -> dict[str, object]:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Tennis store destination must be new and disjoint.")
    config = TennisCourtDetectorSourceConfig(
        "tennis_court_detector",
        source,
        {"train": "train", "val": "val", "test": None},
        (),
    )
    legacy = LegacyTennisCourtDetectorInput(config)
    records = [
        record for split in legacy.available_splits for record in legacy.records(split)
    ]
    destination.mkdir(parents=True, exist_ok=False)
    originals = []
    with ImageRecordWriter(destination) as writer:
        for index, record in enumerate(records):
            if record.image_path.suffix.lower() in {".jpg", ".jpeg"}:
                jpeg = record.image_path.read_bytes()
            else:
                with Image.open(record.image_path) as handle:
                    jpeg = encode_rgb_jpeg(np.asarray(handle.convert("RGB")))
            sparse = {
                "id": record.sample_id,
                "split": record.split,
                "kps": record.payload["keypoints"],
                "metric": record.payload["annotation_metric"],
                "width": record.payload["width"],
                "height": record.payload["height"],
            }
            writer.append(jpeg, sparse)
            originals.append(json.loads(json.dumps(sparse)))
            if index % 500 == 0:
                print(f"[tennis-migration] {index}/{len(records)}", flush=True)
        storage = writer.finish(
            {
                "source_schema": "tennis_court_detector_annotations_v1",
                "jpeg_quality": 95,
                "existing_jpeg_bytes_preserved": True,
            }
        )
    save_json_atomic(
        {"schema": "tennis_court_detector_store_v1", "storage": storage},
        destination / "dataset.json",
    )
    packed = ImageRecordStore(destination, storage)
    packed.validate()
    if [packed.record(row) for row in range(len(packed))] != originals:
        raise ValueError("Tennis migration changed KP14 annotations.")
    if any(
        packed.jpeg(row) != record.image_path.read_bytes()
        for row, record in enumerate(records)
        if record.image_path.suffix.lower() in {".jpg", ".jpeg"}
    ):
        raise ValueError("Tennis migration recompressed an existing JPEG.")
    loaded = TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            "tennis_court_detector",
            destination,
            dict(config.split_mapping),
            excluded_sample_ids,
        )
    )
    return {
        "source": str(source),
        "destination": str(destination),
        "samples": len(records),
        "retained_split_counts": {
            split: len(loaded.records(split)) for split in loaded.available_splits
        },
        "jpeg_bytes_verified": True,
        "sparse_labels_equal": True,
    }
