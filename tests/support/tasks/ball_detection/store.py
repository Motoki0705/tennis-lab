"""Small JPEG-sharded ball stores for CPU unit and integration tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.ball_detection.data.store import (
    SCHEMA_VERSION,
    BallFrameStore,
    Source,
    Split,
    shard_name,
)
from src.tasks.ball_detection.generate_dataset.frame_store.builder import (
    EncodedClip,
    merge_tables,
)
from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    SourceFrame,
)


def frame(index: int, *balls: BallInstance, annotated: bool = True) -> SourceFrame:
    return SourceFrame(index, annotated, True, False, "unlabeled", tuple(balls))


def ball(
    kind: str = "observed",
    xy: tuple[float, float] | None = (10.0, 20.0),
    track: str = "b001",
) -> BallInstance:
    return BallInstance(track, kind, xy, kind == "occlusion_estimated")


def write_store_clip(
    directory: Path,
    clip_id: str,
    frames: list[SourceFrame],
    *,
    source: Source = "tracknet",
    split: Split = "train",
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Add/replace a fixture clip; preserve existing clips without legacy files."""
    width, height = size
    clips: list[dict[str, Any]] = []
    tables: list[EncodedClip] = []
    payloads: list[bytes] = []
    if (directory / "metadata.json").exists():
        old = BallFrameStore(directory.resolve())
        for clip in old.clips:
            if clip.clip_id == clip_id:
                continue
            rows = old.clip_rows(clip)
            start = int(old.frames["inst_start"][rows[0]])
            stop = int(
                old.frames["inst_start"][rows[-1]] + old.frames["inst_count"][rows[-1]]
            )
            tables.append(
                EncodedClip(
                    {key: value[rows].copy() for key, value in old.frames.items()},
                    {
                        key: value[start:stop].copy()
                        for key, value in old.instances.items()
                    },
                )
            )
            clips.append(asdict(clip))
            payloads.append(b"".join(old.read_jpeg(int(row)).tobytes() for row in rows))
    labels = ClipLabels.from_frames(frames, width=width, height=height)
    encoded = []
    for index in range(len(frames)):
        pixels: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 2] = (index * 7) % 255
        pixels[..., 1] = (index * 13) % 255
        ok, jpeg = cv2.imencode(".jpg", pixels)
        assert ok
        encoded.append(jpeg.tobytes())
    lengths = np.array([len(jpeg) for jpeg in encoded], dtype=np.int64)
    columns = dict(labels.frames)
    columns.update(
        frame_index=np.arange(len(frames), dtype=np.int32),
        offset=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
        length=lengths,
    )
    instances = dict(labels.instances, xy=labels.xy.astype(np.float32))
    tables.append(EncodedClip(columns, instances))
    payloads.append(b"".join(encoded))
    clips.append(
        dict(
            index=len(clips),
            clip_id=clip_id,
            source=source,
            group_id=clip_id.rsplit("/", 1)[0],
            camera_id=None,
            split=split,
            width=width,
            height=height,
            source_width=width,
            source_height=height,
            frame_count=len(frames),
            time_base="1/30",
            fps="30",
            has_events=False,
            track_ids=list(labels.track_ids),
            annotation_sha256="fixture",
            media_sha256="fixture",
        )
    )
    (directory / "shards").mkdir(parents=True, exist_ok=True)
    for index, (clip_record, payload) in enumerate(zip(clips, payloads, strict=True)):
        clip_record["index"] = index
        (directory / "shards" / shard_name(index)).write_bytes(payload)
    savez: Any = np.savez
    savez(directory / "index.npz", **merge_tables(tables))
    (directory / "metadata.json").write_text(
        json.dumps(dict(schema_version=SCHEMA_VERSION, clips=clips))
    )
    BallFrameStore(directory.resolve())
    return directory


CONFIG = Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs"


def store_config(tmp_path: Path, *, frames: int = 2) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["model=conv_next_unet"])
    cfg.paths.data_root = str(tmp_path)
    cfg.data.data_dir = "ball_detection/test-v1"
    cfg.data.sources = ["tracknet", "meiji"]
    cfg.data.train_sampling.source_weights = {"tracknet": 1.0, "meiji": 1.0}
    cfg.data.train_sampling.windows_per_epoch = 9
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.image_size = [48, 64]
    cfg.data.heatmap_size = [24, 32]
    cfg.model.num_frames = frames
    for entry in cfg.data.augmentation.values():
        entry.enabled = False
    return cfg
