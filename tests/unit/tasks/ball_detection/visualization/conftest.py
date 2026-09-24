"""Fixtures for the ball-detection review/inference backend unit tests.

Everything here is synthetic and CPU-only: a minimal unified web store, a
minimal TrackNet-style clip tree, and a tiny real convolution checkpoint.  The
real curated checkpoint and real clips are exercised separately by
``local_data`` tests so the default suite stays fast.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair

WEB_SCHEMA = "web_ball_frames_v2"
# A tiny spatial size keeps the real convolution model fast on CPU.
TINY_IMAGE_SIZE = (64, 128)

_ROW_FIELDS = (
    "file name",
    "instance id",
    "visibility",
    "x-coordinate",
    "y-coordinate",
    "ball state",
    "role",
)


def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return bytes(buffer.tobytes())


def write_clip(
    clip_dir: Path,
    *,
    frames: int,
    rows: Sequence[dict[str, Any]],
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Write one ``Label.csv`` clip directory with numbered JPEG frames."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    width, height = size
    for offset in range(frames):
        rgb: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (offset * 7) % 255
        rgb[..., 1] = (offset * 13) % 255
        (clip_dir / f"{offset:04d}.jpg").write_bytes(_jpeg(rgb))
    lines = [",".join(_ROW_FIELDS)]
    lines.extend(
        ",".join(str(row.get(name, "")) for name in _ROW_FIELDS) for row in rows
    )
    (clip_dir / "Label.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return clip_dir


@pytest.fixture
def make_clip_dataset() -> Callable[..., Path]:
    """Return a factory writing one synthetic TrackNet-style source tree."""

    def _factory(
        root: Path,
        *,
        clips: int = 1,
        frames: int = 6,
        extra_rows: Sequence[dict[str, Any]] = (),
    ) -> Path:
        for clip_index in range(clips):
            clip_dir = root / "game1" / f"Clip{clip_index + 1}"
            rows: list[dict[str, Any]] = [
                {
                    "file name": f"{offset:04d}.jpg",
                    "instance id": "b001",
                    "visibility": 1,
                    "x-coordinate": 10.0 + offset,
                    "y-coordinate": 20.0 + offset,
                    "ball state": "visible",
                    "role": "target",
                }
                for offset in range(frames)
            ]
            rows.extend(extra_rows)
            write_clip(clip_dir, frames=frames, rows=rows)
        return root

    return _factory


@pytest.fixture
def make_web_store() -> Callable[[Path], Path]:
    """Return the :func:`write_unified_store` builder as a fixture."""
    return write_unified_store


def write_unified_store(data_root: Path) -> Path:
    """Write a minimal unified web store and return its directory.

    The store carries one referenced positive still, one explicitly-negative
    still, and a two-frame temporal sequence: the smallest layout that
    exercises every catalog branch (static/temporal, positive/negative).
    """
    store = data_root / "tennis" / "web" / "unified"
    (store / "shards").mkdir(parents=True, exist_ok=True)
    (store / "stills").mkdir(parents=True, exist_ok=True)

    (store / "stills" / "positive.jpg").write_bytes(
        _jpeg(np.full((48, 64, 3), 40, dtype=np.uint8))
    )
    (store / "stills" / "negative.jpg").write_bytes(
        _jpeg(np.full((48, 64, 3), 90, dtype=np.uint8))
    )

    temporal_bytes = [
        _jpeg(np.full((48, 64, 3), value, dtype=np.uint8)) for value in (10, 200)
    ]
    shard = store / "shards" / "shard-00000.bin"
    offsets: list[tuple[int, int]] = []
    with shard.open("wb") as handle:
        for payload in temporal_bytes:
            offset = handle.tell()
            handle.write(payload)
            offsets.append((offset, len(payload)))

    np.savez(
        store / "index.npz",
        # 0 = shard-backed, 1 = referenced file
        store=np.asarray([1, 1, 0, 0], dtype=np.uint8),
        shard=np.zeros(4, dtype=np.int32),
        offset=np.asarray([0, 0, offsets[0][0], offsets[1][0]], dtype=np.int64),
        length=np.asarray([0, 0, offsets[0][1], offsets[1][1]], dtype=np.int64),
        path_id=np.asarray([0, 1, -1, -1], dtype=np.int32),
        orig_w=np.full(4, 64, dtype=np.int32),
        orig_h=np.full(4, 48, dtype=np.int32),
        temporal=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        # train, val, train, train
        split=np.asarray([0, 1, 0, 0], dtype=np.uint8),
        source_id=np.asarray([0, 0, 1, 1], dtype=np.int32),
        sequence_id=np.asarray([0, 1, 2, 2], dtype=np.int32),
        frame_index=np.asarray([-1, -1, 0, 1], dtype=np.int32),
        # positive, negative, positive, positive
        label_state=np.asarray([1, 0, 1, 1], dtype=np.uint8),
        inst_start=np.asarray([0, 1, 1, 2], dtype=np.int64),
        inst_count=np.asarray([1, 0, 1, 1], dtype=np.int32),
        inst_x=np.asarray([12.5, 7.0, 30.0], dtype=np.float32),
        inst_y=np.asarray([20.5, 9.0, 11.0], dtype=np.float32),
        inst_vis=np.asarray([1, 1, 1], dtype=np.uint8),
    )
    (store / "index_strings.json").write_text(
        json.dumps(
            {
                "schema": WEB_SCHEMA,
                "sources": ["roboflow", "racketvision"],
                "sequences": ["still-0001", "still-0002", "video-0001"],
                "paths": ["stills/positive.jpg", "stills/negative.jpg"],
            }
        ),
        encoding="utf-8",
    )
    return store


def tiny_model_config(
    *,
    model_name: str = "conv_next_unet",
    num_frames: int = 2,
    input_mode: str = "mdd",
    image_size: tuple[int, int] = TINY_IMAGE_SIZE,
) -> DictConfig:
    """Build a minimal ball model config with a CPU-cheap architecture."""
    model: dict[str, Any] = {
        "name": model_name,
        "input_mode": input_mode,
        "in_channels": 3 if input_mode == "rgb" else 2,
        "num_classes": 1,
        "num_frames": num_frames,
        "input_layout": "bcthw",
        "mdd_a": 0.2,
        "mdd_b": 0.15,
    }
    if model_name == "conv_next_unet":
        model.update(
            {"dims": [4, 8, 16, 32], "depth": 1, "drop_path_prob": 0.0}
        )
    elif model_name != "stunet":
        # dinov3_rope needs its backbone/decoder bundle and a real pretrained
        # backbone, so it is deliberately outside this cheap fixture.
        raise ValueError(f"tiny_model_config does not support {model_name!r}.")
    return OmegaConf.create(
        {"model": model, "data": {"image_size": list(image_size), "augmentation": {"normalize_imagenet": {"enabled": False}}}}
    )


@pytest.fixture
def make_tiny_checkpoint() -> Callable[..., Path]:
    """Return the :func:`tiny_checkpoint_from_config` builder as a fixture."""
    return tiny_checkpoint_from_config


def tiny_checkpoint_from_config(
    path: Path,
    *,
    model_name: str = "conv_next_unet",
    num_frames: int = 2,
    input_mode: str = "mdd",
    state_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Path:
    """Write a tiny but real convolution checkpoint and return its path."""
    config = tiny_model_config(
        model_name=model_name,
        num_frames=num_frames,
        input_mode=input_mode,
    )
    bound = build_ball_detection_pair(config)
    state = {f"model.{key}": value for key, value in bound.model.state_dict().items()}
    if state_transform is not None:
        state = state_transform(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": {"config": OmegaConf.to_container(config)},
        },
        path,
    )
    return path


__all__ = [
    "TINY_IMAGE_SIZE",
    "WEB_SCHEMA",
    "tiny_model_config",
    "tiny_checkpoint_from_config",
    "write_clip",
    "write_unified_store",
]
