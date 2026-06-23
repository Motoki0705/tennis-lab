"""Convert ``data/tennis/web`` sources into the unified web ball-detection store.

Positive frames and frames explicitly annotated as ball-absent are retained.
Frames with unknown annotation state are excluded. Frames decoded from videos
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
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
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
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
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
ROBOFLOW_SPLIT_DIRS = ("train", "valid", "test")


@dataclass
class SampleRecord:
    """One explicitly labeled frame to be added to the store."""

    instances: list[tuple[float, float, int]]  # (x, y, visibility)
    orig_w: int
    orig_h: int
    temporal: int
    source: str
    sequence: str
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
        self.sequences = Interner()
        self.paths = Interner()
        self.split_counts = {
            name: {"total": 0, "positive": 0, "negative": 0} for name in SPLIT_CODES
        }
        self.source_counts: dict[str, dict[str, int]] = {}
        self.source_sequences: dict[str, set[str]] = {}
        self.sequence_splits: dict[str, str] = {}
        self.sequence_sources: dict[str, str] = {}
        self.temporal_count = 0
        self.positive_count = 0
        self.negative_count = 0

    def add(self, record: SampleRecord, writer: ShardWriter) -> None:
        existing_split = self.sequence_splits.setdefault(record.sequence, record.split)
        if existing_split != record.split:
            raise ValueError(
                f"Sequence {record.sequence!r} spans splits "
                f"{existing_split!r} and {record.split!r}."
            )
        existing_source = self.sequence_sources.setdefault(
            record.sequence,
            record.source,
        )
        if existing_source != record.source:
            raise ValueError(
                f"Sequence {record.sequence!r} spans sources "
                f"{existing_source!r} and {record.source!r}."
            )

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
        sample["sequence_id"].append(self.sequences.intern(record.sequence))
        sample["frame_index"].append(record.frame_index)
        label_state = LABEL_POSITIVE if record.instances else LABEL_NEGATIVE
        sample["label_state"].append(label_state)
        sample["inst_start"].append(inst_start)
        sample["inst_count"].append(len(record.instances))

        label_name = "positive" if label_state == LABEL_POSITIVE else "negative"
        self.split_counts[record.split]["total"] += 1
        self.split_counts[record.split][label_name] += 1
        source_counts = self.source_counts.setdefault(
            record.source,
            {"total": 0, "positive": 0, "negative": 0, "sequences": 0},
        )
        source_counts["total"] += 1
        source_counts[label_name] += 1
        source_sequences = self.source_sequences.setdefault(record.source, set())
        source_sequences.add(record.sequence)
        source_counts["sequences"] = len(source_sequences)
        self.temporal_count += int(record.temporal)
        self.positive_count += int(label_state == LABEL_POSITIVE)
        self.negative_count += int(label_state == LABEL_NEGATIVE)

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
                    "sequences": self.sequences.values,
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


def make_group_split_map(
    group_weights: Mapping[str, int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign complete groups while balancing group and sample counts.

    Every enabled source is split independently. When at least three groups
    exist, non-zero validation/test ratios receive at least one group each.
    """
    if not group_weights:
        return {}
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Split ratios must be non-negative and sum to less than 1.")
    keys = list(group_weights)
    group_count = len(keys)

    def requested_count(ratio: float) -> int:
        if ratio == 0 or group_count < 3:
            return 0
        return max(1, int(round(group_count * ratio)))

    split_group_counts = {
        "test": requested_count(test_ratio),
        "val": requested_count(val_ratio),
    }
    while sum(split_group_counts.values()) >= group_count:
        larger = max(split_group_counts, key=split_group_counts.__getitem__)
        if split_group_counts[larger] <= 1:
            break
        split_group_counts[larger] -= 1

    hash_rank = {
        key: hashlib.sha1(f"{seed}:{key}".encode()).hexdigest() for key in keys
    }
    total_weight = sum(max(int(weight), 1) for weight in group_weights.values())
    ratios = {"test": test_ratio, "val": val_ratio}
    remaining = set(keys)
    assignments: dict[str, str] = {}
    for split in ("test", "val"):
        target_count = split_group_counts[split]
        target_weight = total_weight * ratios[split]
        selected_weight = 0
        for selected_count in range(target_count):
            step_target = target_weight * (selected_count + 1) / target_count
            key = min(
                remaining,
                key=lambda candidate: (
                    abs(
                        selected_weight
                        + max(int(group_weights[candidate]), 1)
                        - step_target
                    ),
                    hash_rank[candidate],
                ),
            )
            assignments[key] = split
            remaining.remove(key)
            selected_weight += max(int(group_weights[key]), 1)
    assignments.update({key: "train" for key in remaining})
    return assignments


def _clamp(value: float, high: int) -> float:
    return float(min(max(value, 0.0), max(high - 1, 0)))


def roboflow_source_group(file_name: str) -> str:
    """Remove the Roboflow-generated content hash from an exported file name."""
    stem = Path(file_name).stem
    return re.sub(r"\.rf\.[^.]+$", "", stem)


def roboflow_group_weights(web_root: Path, name: str) -> dict[str, int]:
    """Count exported variants per Roboflow source-name group."""
    counts: Counter[str] = Counter()
    dataset_dir = web_root / name
    for raw_split in ROBOFLOW_SPLIT_DIRS:
        annotations = dataset_dir / raw_split / "_annotations.coco.json"
        if not annotations.exists():
            continue
        coco = json.loads(annotations.read_text(encoding="utf-8"))
        counts.update(
            f"{name}:{roboflow_source_group(str(image['file_name']))}"
            for image in coco["images"]
        )
    return dict(counts)


def kaggle_group_weights(web_root: Path) -> dict[str, int]:
    """Count explicitly labeled rows per Kaggle video."""
    root = web_root / "kaggle_tenis_backview"
    weights: dict[str, int] = {}
    for ball_csv in root.glob("video*_ball.csv"):
        video_id = ball_csv.name[: -len("_ball.csv")]
        row_count = sum(1 for _ in csv.DictReader(ball_csv.open(encoding="utf-8")))
        weights[f"kaggle_backview:{video_id}"] = row_count
    return weights


def ball_yolo_group_weights(web_root: Path) -> dict[str, int]:
    """Count explicitly labeled frames per Ball-YOLO video sequence."""
    labels_root = web_root / "ball_yolo_sport_ball_labels" / "tennis" / "Labels"
    return {
        f"ball_yolo:{folder.name}": sum(1 for _ in folder.glob("*.txt"))
        for folder in labels_root.iterdir()
        if folder.is_dir()
    }


def load_racketvision_split_map(root: Path) -> dict[tuple[str, str], str]:
    """Load the official sequence-level RacketVision train/val/test split."""
    split_map: dict[tuple[str, str], str] = {}
    for split in SPLIT_CODES:
        path = root / "info" / f"{split}.json"
        entries = json.loads(path.read_text(encoding="utf-8"))
        for match_id, clip_id in entries:
            key = (str(match_id), str(clip_id))
            if key in split_map:
                raise ValueError(f"RacketVision sequence appears twice: {key}.")
            split_map[key] = split
    return split_map


# --------------------------------------------------------------------------- #
# Source iterators
# --------------------------------------------------------------------------- #


def iter_roboflow(
    web_root: Path,
    name: str,
    split_map: Mapping[str, str],
) -> Iterator[SampleRecord]:
    dataset_dir = web_root / name
    for raw_split in ROBOFLOW_SPLIT_DIRS:
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
            boxes = boxes_by_image.get(image["id"], [])
            width = int(image["width"])
            height = int(image["height"])
            instances = [
                (_clamp(cx, width), _clamp(cy, height), vis) for cx, cy, vis in boxes
            ]
            group = roboflow_source_group(str(image["file_name"]))
            sequence = f"{name}:{group}"
            yield SampleRecord(
                instances=instances,
                orig_w=width,
                orig_h=height,
                temporal=0,
                source=name,
                sequence=sequence,
                frame_index=-1,
                split=split_map[sequence],
                file_path=split_dir / image["file_name"],
            )


def iter_racketvision(
    web_root: Path,
    jpeg_quality: int,
) -> Iterator[SampleRecord]:
    root = web_root / "racketvision_tennis" / "tennis"
    videos_dir = root / "videos"
    split_map = load_racketvision_split_map(root)
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
                frame_index = int(row["Frame"])
                frame_boxes.setdefault(frame_index, [])
                if str(row.get("Visibility")) == "1":
                    frame_boxes[frame_index].append(
                        (float(row["X"]), float(row["Y"]), 1)
                    )
            if not frame_boxes:
                continue
            width, height = video_dims(video)
            split = split_map[(match_id, clip_id)]
            sequence = f"racketvision:{match_id}_{clip_id}"
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
                    sequence=sequence,
                    frame_index=index,
                    split=split,
                    jpeg=jpeg,
                )


def iter_kaggle(
    web_root: Path,
    jpeg_quality: int,
    split_map: Mapping[str, str],
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
            frame_index = int(str(row["frame"]).split("_")[-1])
            frame_boxes.setdefault(frame_index, [])
            if not (x >= corner_x and y <= corner_y):
                frame_boxes[frame_index].append(
                    (_clamp(x, width), _clamp(y, height), 1)
                )
        if not frame_boxes:
            continue
        sequence = f"kaggle_backview:{video_id}"
        split = split_map[sequence]
        for index, jpeg in stream_video_jpegs([video], set(frame_boxes), jpeg_quality):
            yield SampleRecord(
                instances=frame_boxes[index],
                orig_w=width,
                orig_h=height,
                temporal=1,
                source="kaggle_backview",
                sequence=sequence,
                frame_index=index,
                split=split,
                jpeg=jpeg,
            )


def iter_ball_yolo(
    web_root: Path,
    jpeg_quality: int,
    split_map: Mapping[str, str],
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
        sequence = f"ball_yolo:{folder.name}"
        split = split_map[sequence]
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
                sequence=sequence,
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
        strings_path = output_dir / STRINGS_FILE
        existing_schema = None
        if strings_path.exists():
            existing_schema = json.loads(strings_path.read_text(encoding="utf-8")).get(
                "schema"
            )
        if existing_schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"Existing web store schema is {existing_schema!r}, expected "
                f"{SCHEMA_VERSION!r}. Rebuild with convert.overwrite=true."
            )
        print(f"[convert_web_dataset] index exists, skipping: {index_path}")
        print("  pass convert.overwrite=true to rebuild.")
        return 0

    build_dir = output_dir.with_name(f".{output_dir.name}.building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    val_ratio = float(convert.val_ratio)
    test_ratio = float(convert.test_ratio)
    split_seed = int(convert.split_seed)
    writer = ShardWriter(build_dir / SHARDS_DIR, int(convert.shard_size_bytes))
    index = IndexBuilder()
    limit = int(convert.limit_per_source)

    quality = int(convert.jpeg_quality)
    generators: list[tuple[str, Callable[[], Iterator[SampleRecord]]]] = []
    if bool(convert.sources.roboflow):
        for name in ROBOFLOW_DATASETS:
            split_map = make_group_split_map(
                roboflow_group_weights(web_root, name),
                val_ratio,
                test_ratio,
                split_seed,
            )
            generators.append((name, partial(iter_roboflow, web_root, name, split_map)))
    if bool(convert.sources.racketvision):
        generators.append(
            ("racketvision", partial(iter_racketvision, web_root, quality))
        )
    if bool(convert.sources.kaggle):
        split_map = make_group_split_map(
            kaggle_group_weights(web_root),
            val_ratio,
            test_ratio,
            split_seed,
        )
        generators.append(
            (
                "kaggle_backview",
                partial(
                    iter_kaggle,
                    web_root,
                    quality,
                    split_map,
                    float(convert.kaggle_corner_frac),
                ),
            )
        )
    if bool(convert.sources.ball_yolo):
        split_map = make_group_split_map(
            ball_yolo_group_weights(web_root),
            val_ratio,
            test_ratio,
            split_seed,
        )
        generators.append(
            ("ball_yolo", partial(iter_ball_yolo, web_root, quality, split_map))
        )

    try:
        for label, factory in generators:
            added = 0
            for record in tqdm(factory(), desc=f"convert:{label}", unit="frame"):
                index.add(record, writer)
                added += 1
                if limit and added >= limit:
                    break
            print(f"[convert_web_dataset] {label}: {added} frames")

        writer.close()
        index.save(build_dir)
        _write_manifest(build_dir, index, writer)
        _write_readme(build_dir)
        _publish_store(build_dir, output_dir)
    except BaseException:
        writer.close()
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    print(
        f"[convert_web_dataset] wrote {len(index)} samples, "
        f"{writer.shard_count} shard(s), {writer.total_bytes / 1e9:.2f} GB packed "
        f"-> {output_dir}"
    )
    return 0


def _publish_store(build_dir: Path, output_dir: Path) -> None:
    """Replace the previous store only after a complete conversion."""
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    if backup_dir.exists():
        if output_dir.exists():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.rename(output_dir)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        build_dir.rename(output_dir)
    except BaseException:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _write_manifest(output_dir: Path, index: IndexBuilder, writer: ShardWriter) -> None:
    split_sequence_counts = {
        split: sum(
            assigned_split == split for assigned_split in index.sequence_splits.values()
        )
        for split in SPLIT_CODES
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_samples": len(index),
        "total_instances": len(index.instances["inst_x"]),
        "label_counts": {
            "positive": index.positive_count,
            "negative": index.negative_count,
            "unknown": 0,
        },
        "temporal_samples": index.temporal_count,
        "non_temporal_samples": len(index) - index.temporal_count,
        "sequence_count": len(index.sequence_splits),
        "split_sequence_counts": split_sequence_counts,
        "split_counts": index.split_counts,
        "source_counts": index.source_counts,
        "sequence_split_leaks": 0,
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
        "`data/tennis/web`. Positive frames and frames explicitly annotated "
        "as ball-absent are kept; unknown frames are excluded.\n\n"
        "- `shards/shard-*.bin`: packed JPEG bytes for video-extracted frames.\n"
        "- COCO still images are referenced in place (see `index_strings.json`).\n"
        "- `index.npz` / `index_strings.json`: columnar per-sample index.\n"
        "- `manifest.json`: human-readable summary.\n\n"
        "Every sample carries `source`, split-safe `sequence_id`, "
        "`frame_index`, `temporal`, and `label_state` provenance. Sequence ids "
        "are never shared across train/validation/test.\n\n"
        "Load via `data=web_frames` (`WebBallDataModule`).\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
