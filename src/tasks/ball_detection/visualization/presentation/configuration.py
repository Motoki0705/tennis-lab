"""Strict Hydra configuration for CPU-only learning-stage presentations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig

from src.tasks.ball_detection.configuration import (
    BallRuntimePaths,
    exact_mapping,
    typed,
)
from src.tasks.base.visualization.presentation import Segment, StepClock
from src.utils.hydra import register_boundary_validator

SEGMENT_NAMES = (
    "intro",
    "phase1",
    "handoff12",
    "phase2",
    "handoff23",
    "phase3",
    "outro",
)


@dataclass(frozen=True)
class InputClip:
    store_dir: Path
    clip_id: str
    frame: int


@dataclass(frozen=True)
class LearningStagesConfig:
    pretraining: Path
    posttraining: Path
    input_panel: Literal["real_clip", "schematic"]
    input_clip: InputClip | None
    durations: Mapping[str, float]
    fps: float
    width: int
    height: int
    gif_width: int
    gif_fps: float
    output_dir: Path


def build_learning_stages_config(config: DictConfig) -> LearningStagesConfig:
    """Reject missing/unknown options and resolve paths through their role roots."""
    raw = exact_mapping(
        config,
        path="configuration",
        required={
            "paths",
            "runs",
            "input_panel",
            "input_clip",
            "durations",
            "fps",
            "width",
            "height",
            "gif_width",
            "gif_fps",
            "run_id",
        },
    )
    paths = BallRuntimePaths.from_config(config)
    runs = exact_mapping(
        raw["runs"], path="runs", required={"pretraining", "posttraining"}
    )
    pretraining = paths.project(cast(str, typed(runs, "pretraining", str, path="runs")))
    posttraining = paths.project(
        cast(str, typed(runs, "posttraining", str, path="runs"))
    )
    panel = typed(raw, "input_panel", str, path="configuration")
    if panel not in ("real_clip", "schematic"):
        raise ValueError("input_panel must explicitly be real_clip or schematic")
    clip = None
    if panel == "real_clip":
        clip_raw = exact_mapping(
            raw["input_clip"],
            path="input_clip",
            required={"store_dir", "clip_id", "frame"},
        )
        frame = cast(int, typed(clip_raw, "frame", int, path="input_clip"))
        clip_id = cast(str, typed(clip_raw, "clip_id", str, path="input_clip"))
        if frame < 0 or not clip_id.strip():
            raise ValueError(
                "input_clip requires a nonnegative frame and nonempty clip_id"
            )
        store_dir = cast(str, typed(clip_raw, "store_dir", str, path="input_clip"))
        # The data root is explicit; do not search another tree/store on failure.
        clip = InputClip(paths.data(store_dir), clip_id, frame)
    elif raw["input_clip"] is not None:
        raise ValueError("input_clip must be null when input_panel=schematic")
    durations_raw = exact_mapping(
        raw["durations"], path="durations", required=set(SEGMENT_NAMES)
    )
    durations = {
        name: float(
            cast(
                float | int, typed(durations_raw, name, (int, float), path="durations")
            )
        )
        for name in SEGMENT_NAMES
    }
    fps = float(
        cast(float | int, typed(raw, "fps", (int, float), path="configuration"))
    )
    # Validate timing even before logs or images are read.
    StepClock([Segment(name, durations[name], 0, 0) for name in SEGMENT_NAMES], fps)
    sizes = {
        name: cast(int, typed(raw, name, int, path="configuration"))
        for name in ("width", "height", "gif_width")
    }
    if any(value <= 0 for value in sizes.values()) or any(
        sizes[name] % 2 for name in ("width", "height")
    ):
        raise ValueError(
            "Resolution must be positive and H.264 width/height must be even"
        )
    gif_fps = float(
        cast(float | int, typed(raw, "gif_fps", (int, float), path="configuration"))
    )
    if not math.isfinite(gif_fps) or gif_fps <= 0:
        raise ValueError("gif_fps must be finite and positive")
    run_id = cast(str, typed(raw, "run_id", str, path="configuration"))
    if not run_id.strip() or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a nonempty filename component")
    return LearningStagesConfig(
        pretraining,
        posttraining,
        cast(Literal["real_clip", "schematic"], panel),
        clip,
        durations,
        fps,
        sizes["width"],
        sizes["height"],
        sizes["gif_width"],
        gif_fps,
        paths.output(f"ball_detection/visualize/presentation/learning_stages/{run_id}"),
    )


def validate_learning_stages_boundary(config: DictConfig) -> None:
    build_learning_stages_config(config)


register_boundary_validator("ball.learning_stages", validate_learning_stages_boundary)
