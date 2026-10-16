"""Build published JPEG stores from the small upstream-shaped test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.data.image_record_store import ImageRecordWriter, encode_rgb_jpeg


def pack_tennis_fixture(source: Path, destination: Path) -> None:
    """Publish fixture records without depending on a production migration tool."""
    with ImageRecordWriter(destination) as writer:
        for split in ("train", "val"):
            records = json.loads((source / f"data_{split}.json").read_text())
            for record in records:
                paths = [
                    source / "images" / f"{record['id']}{suffix}"
                    for suffix in (".jpg", ".png", ".jpeg")
                ]
                image_path = next(path for path in paths if path.is_file())
                with Image.open(image_path) as image:
                    width, height = image.size
                    jpeg = (
                        image_path.read_bytes()
                        if image_path.suffix in {".jpg", ".jpeg"}
                        else encode_rgb_jpeg(np.asarray(image.convert("RGB")))
                    )
                writer.append(
                    jpeg,
                    {
                        "metric": None,
                        **record,
                        "split": split,
                        "width": width,
                        "height": height,
                    },
                )
        storage = writer.finish(
            {"source_schema": "tennis_court_detector_annotations_v1"}
        )
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "tennis_court_detector_store_v1",
                "storage": storage,
            }
        )
    )


def write_tennis_records(root: Path, records: list[dict[str, object]]) -> None:
    """Create a packed fixture, including deliberately malformed sparse records."""
    jpeg = encode_rgb_jpeg(np.full((24, 32, 3), 120, dtype=np.uint8))
    with ImageRecordWriter(root) as writer:
        for record in records:
            writer.append(jpeg, {"metric": None, "width": 32, "height": 24, **record})
        storage = writer.finish(
            {"source_schema": "tennis_court_detector_annotations_v1"}
        )
    (root / "dataset.json").write_text(
        json.dumps({"schema": "tennis_court_detector_store_v1", "storage": storage})
    )
