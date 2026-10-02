"""Fixtures for the ball-detection review/inference backend unit tests.

Everything here is synthetic and CPU-only: a minimal ball frame store and
a tiny real convolution checkpoint.  The
real curated checkpoint and real clips are exercised separately by
``local_data`` tests so the default suite stays fast.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

# A tiny spatial size keeps the real convolution model fast on CPU.
TINY_IMAGE_SIZE = (64, 128)


def write_clip(
    clip_dir: Path,
    *,
    frames: int,
    rows: Sequence[dict[str, Any]],
    size: tuple[int, int] = (64, 48),
) -> Path:
    """Write a clip into a ball store; rows are synthetic fixture annotations."""
    directory = clip_dir.parents[2]
    clip_id = '/'.join(clip_dir.parts[-3:])
    labels = []
    for index in range(frames):
        frame_rows = [row for row in rows if row['file name'] == f'{index:04d}.jpg']
        instances = []
        for row in frame_rows:
            if float(row['visibility'] or 0) <= 0:
                continue
            instances.append(ball(
                str(row.get('point_kind', 'observed')),
                (float(row['x-coordinate']), float(row['y-coordinate'])),
                str(row.get('instance id') or 'b001'),
            ))
        labels.append(frame(index, *instances, annotated=bool(frame_rows)))
    return cast(Path, write_store_clip(directory, clip_id, labels, size=size))


@pytest.fixture
def make_clip_dataset() -> Callable[..., Path]:
    """Return a factory writing one synthetic ball frame store."""

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
    "tiny_model_config",
    "tiny_checkpoint_from_config",
    "write_clip",
]
