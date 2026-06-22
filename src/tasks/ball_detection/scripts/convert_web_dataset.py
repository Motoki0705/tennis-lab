"""Convert ``data/tennis/web`` sources into the unified web ball-detection store.

Only frames that carry a ball annotation are kept. Frames decoded from videos
are JPEG-encoded and packed into ``shards/shard-*.bin``; COCO still images are
referenced in place (no duplication). The resulting ``index.npz`` /
``index_strings.json`` are consumed by
:class:`src.tasks.ball_detection.data.web_datamodule.WebBallDataModule`.

Usage:
    python -m src.tasks.ball_detection.scripts.convert_web_dataset
    python -m src.tasks.ball_detection.scripts.convert_web_dataset \
        convert.limit_per_source=50 convert.overwrite=true

Notes:
    - Hydra config: ``src/tasks/ball_detection/configs/convert_web_dataset.yaml``.
    - See :mod:`src.tasks.ball_detection.data.web_store` for the on-disk schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import cv2
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from src.tasks.ball_detection.data.web_store import (
    INDEX_FILE,
    INSTANCE_FIELDS,
    MANIFEST_FILE,
    SAMPLE_FIELDS,
    SCHEMA_VERSION,
    SHARDS_DIR,
    SPLIT_CODES,
    STORE_FILE,
    STORE_SHARD,
    STRINGS_FILE,
    shard_name,
)

ROBOFLOW_DATASETS = (
    "roboflow_tennis_ball_tracking_detection_h9rat_v1",
    "roboflow_tennis_ball_tracking_1wnxz_v2",
    "roboflow_tennis_ball_wafqb_v2",
)
ROBOFLOW_SPLIT_DIRS = {"train": "train", "valid": "val", "test": "test"}


@dataclass
class SampleRecord:
    """One annotated frame to be added to the store."""

    instances: list[tuple[float, float, int]]  # (x, y, visibility)
    orig_w: int
    orig_h: int
    temporal: int
    source: str
    frame_index: int
    split: str
    jpeg: bytes | None = None  # packed into a shard when present
    file_path: Path | None = None  # referenced in place when present


# --------------------------------------------------------------------------- #
# Writers / accumulators
# --------------------------------------------------------------------------- #


class ShardWriter:
    """Append-only writer that packs bytes into rotating shard files."""

    def __init__(self, shards_dir: Path, shard_size_bytes: int) -> None:
        self.shards_dir = shards_dir
        self.shard_size_bytes = shard_size_bytes
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self._shard_id = 0
        self._offset = 0
        self._handle = (self.shards_dir / shard_name(0)).open("wb")
        self.shard_count = 1
        self.total_bytes = 0

    def write(self, data: bytes) -> tuple[int, int, int]:
        """Write ``data`` and return ``(shard_id, offset, length)``."""
        if self._offset > 0 and self._offset + len(data) > self.shard_size_bytes:
            self._handle.close()
            self._shard_id += 1
            self._offset = 0
            self._handle = (self.shards_dir / shard_name(self._shard_id)).open("wb")
            self.shard_count = self._shard_id + 1
        offset = self._offset
        self._handle.write(data)
        self._offset += len(data)
        self.total_bytes += len(data)
        return self._shard_id, offset, len(data)

    def close(self) -> None:
        self._handle.close()


class Interner:
    """Maps strings to stable integer ids."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self._lookup: dict[str, int] = {}

    def intern(self, value: str) -> int:
        existing = self._lookup.get(value)
        if existing is not None:
            return existing
        index = len(self.values)
        self.values.append(value)
        self._lookup[value] = index
        return index


class IndexBuilder:
    """Accumulates per-sample and per-instance columns for ``index.npz``."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {name: [] for name in SAMPLE_FIELDS}
        self.instances: dict[str, list[float]] = {name: [] for name in INSTANCE_FIELDS}
        self.sources = Interner()
        self.paths = Interner()
        self.split_counts: dict[str, int] = {name: 0 for name in SPLIT_CODES}
        self.source_counts: dict[str, int] = {}
        self.temporal_count = 0

    def add(self, record: SampleRecord, writer: ShardWriter) -> None:
        if record.jpeg is not None:
            shard_id, offset, length = writer.write(record.jpeg)
            store, path_id = STORE_SHARD, -1
        elif record.file_path is not None:
            shard_id, offset, length = -1, 0, 0
            store = STORE_FILE
            rel = os.path.relpath(record.file_path.resolve(), writer.shards_dir.parent)
            path_id = self.paths.intern(rel)
        else:  # pragma: no cover - guarded by callers
            raise ValueError("SampleRecord needs either jpeg bytes or a file_path.")

        inst_start = len(self.instances["inst_x"])
        for x, y, vis in record.instances:
            self.instances["inst_x"].append(float(x))
            self.instances["inst_y"].append(float(y))
            self.instances["inst_vis"].append(int(vis))

        sample = self.samples
        sample["store"].append(store)
        sample["shard"].append(shard_id)
        sample["offset"].append(offset)
        sample["length"].append(length)
        sample["path_id"].append(path_id)
        sample["orig_w"].append(record.orig_w)
        sample["orig_h"].append(record.orig_h)
        sample["temporal"].append(record.temporal)
        sample["split"].append(SPLIT_CODES[record.split])
        sample["source_id"].append(self.sources.intern(record.source))
        sample["frame_index"].append(record.frame_index)
        sample["inst_start"].append(inst_start)
        sample["inst_count"].append(len(record.instances))

        self.split_counts[record.split] += 1
        self.source_counts[record.source] = self.source_counts.get(record.source, 0) + 1
        self.temporal_count += int(record.temporal)

    def __len__(self) -> int:
        return len(self.samples["store"])

    def save(self, output_dir: Path) -> None:
        arrays: dict[str, np.ndarray] = {
            name: np.asarray(self.samples[name], dtype=dtype)
            for name, dtype in SAMPLE_FIELDS.items()
        }
        arrays.update(
            {
                name: np.asarray(self.instances[name], dtype=dtype)
                for name, dtype in INSTANCE_FIELDS.items()
            }
        )
        np.savez(output_dir / INDEX_FILE, **arrays)  # type: ignore[arg-type]
        (output_dir / STRINGS_FILE).write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "sources": self.sources.values,
                    "paths": self.paths.values,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# Frame streaming helpers
# --------------------------------------------------------------------------- #


def video_dims(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def stream_video_jpegs(
    parts: list[Path],
    needed: set[int],
    jpeg_quality: int,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(global_frame_index, jpeg_bytes)`` for needed frames in order.

    ``parts`` is a list of one or more video files virtually concatenated by
    frame count (used for split source videos).
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    remaining = set(needed)
    base = 0
    for part in parts:
        if not remaining:
            break
        cap = cv2.VideoCapture(str(part))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {part}")
        local = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            global_index = base + local
            if global_index in remaining:
                ok_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok_enc:
                    raise RuntimeError(f"Failed to JPEG-encode frame {global_index}.")
                yield global_index, buffer.tobytes()
                remaining.discard(global_index)
            local += 1
        cap.release()
        base += local


def make_splitter(
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Callable[[str], str]:
    def splitter(key: str) -> str:
        digest = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
        bucket = (int(digest[:8], 16) % 10_000) / 10_000.0
        if bucket < test_ratio:
            return "test"
        if bucket < test_ratio + val_ratio:
            return "val"
        return "train"

    return splitter


def _clamp(value: float, high: int) -> float:
    return float(min(max(value, 0.0), max(high - 1, 0)))


# --------------------------------------------------------------------------- #
# Source iterators
# --------------------------------------------------------------------------- #


def iter_roboflow(web_root: Path, name: str) -> Iterator[SampleRecord]:
    dataset_dir = web_root / name
    for raw_split, split in ROBOFLOW_SPLIT_DIRS.items():
        split_dir = dataset_dir / raw_split
        annotations = split_dir / "_annotations.coco.json"
        if not annotations.exists():
            continue
        coco = json.loads(annotations.read_text(encoding="utf-8"))
        ball_cats = {
            cat["id"]
            for cat in coco["categories"]
            if str(cat.get("supercategory", "none")).lower() != "none"
        }
        boxes_by_image: dict[int, list[tuple[float, float, int]]] = {}
        for ann in coco["annotations"]:
            if ann["category_id"] not in ball_cats:
                continue
            x, y, w, h = ann["bbox"]
            boxes_by_image.setdefault(ann["image_id"], []).append(
                (x + w / 2.0, y + h / 2.0, 1)
            )
        for image in coco["images"]:
            boxes = boxes_by_image.get(image["id"])
            if not boxes:
                continue
            width = int(image["width"])
            height = int(image["height"])
            instances = [
                (_clamp(cx, width), _clamp(cy, height), vis) for cx, cy, vis in boxes
            ]
            yield SampleRecord(
                instances=instances,
                orig_w=width,
                orig_h=height,
                temporal=0,
                source=name,
                frame_index=-1,
                split=split,
                file_path=split_dir / image["file_name"],
            )


def iter_racketvision(
    web_root: Path,
    jpeg_quality: int,
    splitter: Callable[[str], str],
) -> Iterator[SampleRecord]:
    root = web_root / "racketvision_tennis" / "tennis"
    videos_dir = root / "videos"
    for match_dir in sorted((root / "all").iterdir()):
        if not match_dir.is_dir():
            continue
        match_id = match_dir.name  # e.g. "match100"
        for csv_path in sorted(match_dir.glob("csv/*_ball.csv")):
            clip_id = csv_path.stem.split("_")[0]  # "000"
            video = videos_dir / f"{match_id}_{clip_id}.mp4"
            if not video.exists():
                continue
            frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
            for row in csv.DictReader(csv_path.open(encoding="utf-8")):
                if str(row.get("Visibility")) != "1":
                    continue
                frame_boxes.setdefault(int(row["Frame"]), []).append(
                    (float(row["X"]), float(row["Y"]), 1)
                )
            if not frame_boxes:
                continue
            width, height = video_dims(video)
            split = splitter(f"rv:{match_id}_{clip_id}")
            for index, jpeg in stream_video_jpegs(
                [video], set(frame_boxes), jpeg_quality
            ):
                instances = [
                    (_clamp(x, width), _clamp(y, height), vis)
                    for x, y, vis in frame_boxes[index]
                ]
                yield SampleRecord(
                    instances=instances,
                    orig_w=width,
                    orig_h=height,
                    temporal=1,
                    source="racketvision",
                    frame_index=index,
                    split=split,
                    jpeg=jpeg,
                )


def iter_kaggle(
    web_root: Path,
    jpeg_quality: int,
    splitter: Callable[[str], str],
    corner_frac: float,
) -> Iterator[SampleRecord]:
    root = web_root / "kaggle_tenis_backview"
    for ball_csv in sorted(root.glob("video*_ball.csv")):
        video_id = ball_csv.name[: -len("_ball.csv")]
        video = root / f"{video_id}.mp4"
        if not video.exists():
            continue
        width, height = video_dims(video)
        corner_x = width * (1.0 - corner_frac)
        corner_y = height * corner_frac
        frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
        for row in csv.DictReader(ball_csv.open(encoding="utf-8")):
            try:
                x = float(row["ball_x"])
                y = float(row["ball_y"])
            except (TypeError, ValueError):
                continue
            if x >= corner_x and y <= corner_y:  # top-right "absent" sentinel
                continue
            frame_index = int(str(row["frame"]).split("_")[-1])
            frame_boxes.setdefault(frame_index, []).append(
                (_clamp(x, width), _clamp(y, height), 1)
            )
        if not frame_boxes:
            continue
        split = splitter(f"kg:{video_id}")
        for index, jpeg in stream_video_jpegs([video], set(frame_boxes), jpeg_quality):
            yield SampleRecord(
                instances=frame_boxes[index],
                orig_w=width,
                orig_h=height,
                temporal=1,
                source="kaggle_backview",
                frame_index=index,
                split=split,
                jpeg=jpeg,
            )


def iter_ball_yolo(
    web_root: Path,
    jpeg_quality: int,
    splitter: Callable[[str], str],
) -> Iterator[SampleRecord]:
    labels_root = web_root / "ball_yolo_sport_ball_labels" / "tennis" / "Labels"
    videos_dir = web_root / "sport_ball_detection_videos" / "tennis" / "Videos"
    mapping_csv = web_root / "ball_yolo_tennis_video_mapping.csv"
    mapping = {
        row["label_folder"]: row
        for row in csv.DictReader(mapping_csv.open(encoding="utf-8"))
    }
    for folder in sorted(labels_root.iterdir()):
        if not folder.is_dir() or folder.name not in mapping:
            continue
        parts = [
            videos_dir / name
            for name in mapping[folder.name]["official_video_files"].split(";")
        ]
        parts = [part for part in parts if part.exists()]
        if not parts:
            continue
        frame_boxes: dict[int, list[tuple[float, float, int]]] = {}
        for label_file in folder.glob("*.txt"):
            frame_index = int(label_file.stem.rsplit("_", 1)[1])
            for line in label_file.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                cx, cy = float(fields[1]), float(fields[2])
                frame_boxes.setdefault(frame_index, []).append((cx, cy, 1))
        if not frame_boxes:
            continue
        width, height = video_dims(parts[0])
        split = splitter(f"by:{folder.name}")
        for index, jpeg in stream_video_jpegs(parts, set(frame_boxes), jpeg_quality):
            instances = [
                (_clamp(cx * width, width), _clamp(cy * height, height), vis)
                for cx, cy, vis in frame_boxes[index]
            ]
            yield SampleRecord(
                instances=instances,
                orig_w=width,
                orig_h=height,
                temporal=1,
                source="ball_yolo",
                frame_index=index,
                split=split,
                jpeg=jpeg,
            )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _hydra_main(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return cast(Callable[[Any], Any], hydra.main(*args, **kwargs))


@_hydra_main(
    config_path="../configs",
    config_name="convert_web_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    convert = cfg.convert
    web_root = Path(to_absolute_path(str(convert.web_root)))
    output_dir = Path(to_absolute_path(str(convert.output_dir)))
    index_path = output_dir / INDEX_FILE

    if index_path.exists() and not bool(convert.overwrite):
        print(f"[convert_web_dataset] index exists, skipping: {index_path}")
        print("  pass convert.overwrite=true to rebuild.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (output_dir / SHARDS_DIR).glob("shard-*.bin"):
        stale.unlink()

    splitter = make_splitter(
        float(convert.val_ratio), float(convert.test_ratio), int(convert.split_seed)
    )
    writer = ShardWriter(output_dir / SHARDS_DIR, int(convert.shard_size_bytes))
    index = IndexBuilder()
    limit = int(convert.limit_per_source)

    quality = int(convert.jpeg_quality)
    generators: list[tuple[str, Callable[[], Iterator[SampleRecord]]]] = []
    if bool(convert.sources.roboflow):
        for name in ROBOFLOW_DATASETS:
            generators.append((name, partial(iter_roboflow, web_root, name)))
    if bool(convert.sources.racketvision):
        generators.append(
            ("racketvision", partial(iter_racketvision, web_root, quality, splitter))
        )
    if bool(convert.sources.kaggle):
        generators.append(
            (
                "kaggle_backview",
                partial(
                    iter_kaggle,
                    web_root,
                    quality,
                    splitter,
                    float(convert.kaggle_corner_frac),
                ),
            )
        )
    if bool(convert.sources.ball_yolo):
        generators.append(
            ("ball_yolo", partial(iter_ball_yolo, web_root, quality, splitter))
        )

    for label, factory in generators:
        added = 0
        for record in tqdm(factory(), desc=f"convert:{label}", unit="frame"):
            index.add(record, writer)
            added += 1
            if limit and added >= limit:
                break
        print(f"[convert_web_dataset] {label}: {added} frames")

    writer.close()
    index.save(output_dir)
    _write_manifest(output_dir, index, writer)
    _write_readme(output_dir)
    print(
        f"[convert_web_dataset] wrote {len(index)} samples, "
        f"{writer.shard_count} shard(s), {writer.total_bytes / 1e9:.2f} GB packed "
        f"-> {output_dir}"
    )
    return 0


def _write_manifest(output_dir: Path, index: IndexBuilder, writer: ShardWriter) -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_samples": len(index),
        "total_instances": len(index.instances["inst_x"]),
        "temporal_samples": index.temporal_count,
        "non_temporal_samples": len(index) - index.temporal_count,
        "split_counts": index.split_counts,
        "source_counts": index.source_counts,
        "shard_count": writer.shard_count,
        "packed_bytes": writer.total_bytes,
        "referenced_files": len(index.paths.values),
    }
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "# Unified web ball-detection store\n\n"
        f"Schema: `{SCHEMA_VERSION}` (see "
        "`src/tasks/ball_detection/data/web_store.py`).\n\n"
        "Generated by `python -m "
        "src.tasks.ball_detection.scripts.convert_web_dataset` from "
        "`data/tennis/web`. Only frames with a ball annotation are kept.\n\n"
        "- `shards/shard-*.bin`: packed JPEG bytes for video-extracted frames.\n"
        "- COCO still images are referenced in place (see `index_strings.json`).\n"
        "- `index.npz` / `index_strings.json`: columnar per-sample index.\n"
        "- `manifest.json`: human-readable summary.\n\n"
        "Each sample carries a `temporal` flag (1 = from an ordered video, "
        "0 = shuffled still image) plus `frame_index`/`source` provenance so a "
        "later multi-frame phase can rebuild temporal windows.\n\n"
        "Load via `data=web_frames` (`WebBallDataModule`).\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
