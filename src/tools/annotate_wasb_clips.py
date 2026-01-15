"""Sample clips from a video and annotate tennis ball positions for WASB.

This tool supports a two-phase workflow:
1) Sample clip segments from a video and export them in WASB's dataset structure.
2) Manually annotate ball positions per frame for the sampled clips.

Example commands:
    # Sample clips and then annotate them
    `uv run python -m src.tools.annotate_wasb_clips mode=all video_path=data/raw/match.mp4 \
        output.output_dir=data/tennis output.game_name=game_manual`

    # Only sample clips
    `uv run python -m src.tools.annotate_wasb_clips mode=sample video_path=data/raw/match.mp4 \
        sampling.num_clips=8 sampling.clip_length=90`

    # Only annotate existing clips
    `uv run python -m src.tools.annotate_wasb_clips mode=annotate output.output_dir=data/tennis \
        output.game_name=game_manual annotate.clip_indices=[1,3]`

Config entry point: `src/tools/configs/annotate_wasb_clips.yaml`
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import hydra
from omegaconf import DictConfig

from src.wasb.tennis_format import (
    TennisLabelRow,
    load_label_csv,
    row_from_visibility,
    save_label_csv,
)
from src.wasb.utils.video_extractor import VideoExtractor

LOGGER = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for clip sampling from a video."""

    method: Literal["uniform", "random"] = "uniform"
    clip_length: int = 90
    num_clips: int = 10
    min_gap: int = 30
    start_frame: int | None = None
    end_frame: int | None = None
    explicit_starts: list[int] = field(default_factory=list)
    random_seed: int = 7


@dataclass
class OutputConfig:
    """Configuration for output dataset structure."""

    output_dir: str = "data/tennis"
    game_name: str = "game_manual"
    frame_format: str = "frame_{:04d}.jpg"
    label_format: str = "{:04d}.jpg"
    jpeg_quality: int = 95
    overwrite: bool = False
    skip_existing: bool = True


@dataclass
class AnnotationConfig:
    """Configuration for manual annotation UI."""

    clip_indices: list[int] = field(default_factory=list)
    window_name: str = "WASB Annotator"
    circle_radius: int = 6
    circle_color_bgr: tuple[int, int, int] = (0, 0, 255)
    text_color_bgr: tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.6
    line_thickness: int = 2


@dataclass
class ToolConfig:
    """Top-level configuration for the annotation tool."""

    mode: Literal["sample", "annotate", "all"] = "all"
    video_path: str | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    annotate: AnnotationConfig = field(default_factory=AnnotationConfig)


@dataclass(frozen=True)
class ClipPlan:
    """Plan for a single clip segment."""

    index: int
    start_frame: int
    end_frame: int


def _resolve_game_dir(output: OutputConfig) -> Path:
    return Path(output.output_dir) / output.game_name


def _resolve_frame_bounds(
    extractor: VideoExtractor, sampling: SamplingConfig
) -> tuple[int, int]:
    start = sampling.start_frame or 0
    end = sampling.end_frame or extractor.frame_count

    if start < 0 or end <= start:
        raise ValueError("Invalid start/end frame range")

    end = min(end, extractor.frame_count)

    return start, end


def _build_uniform_starts(
    start: int, end: int, clip_length: int, num_clips: int
) -> list[int]:
    max_start = end - clip_length
    if max_start < start:
        raise ValueError("Clip length is longer than the available frame range")

    if num_clips <= 0:
        raise ValueError("num_clips must be >= 1")

    if num_clips == 1:
        return [int(round((start + max_start) / 2))]

    step = (max_start - start) / (num_clips - 1)
    starts = [int(round(start + step * idx)) for idx in range(num_clips)]

    deduped: list[int] = []
    for value in starts:
        value = min(max(value, start), max_start)
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return deduped


def _build_random_starts(
    start: int,
    end: int,
    clip_length: int,
    num_clips: int,
    min_gap: int,
    seed: int,
) -> list[int]:
    max_start = end - clip_length
    if max_start < start:
        raise ValueError("Clip length is longer than the available frame range")

    candidates = list(range(start, max_start + 1))
    rng = random.Random(seed)
    rng.shuffle(candidates)

    starts: list[int] = []
    for candidate in candidates:
        if all(abs(candidate - chosen) >= min_gap for chosen in starts):
            starts.append(candidate)
            if len(starts) >= num_clips:
                break

    if len(starts) < num_clips:
        raise ValueError("Unable to sample enough clips with the requested gap")

    return sorted(starts)


def build_clip_plan(extractor: VideoExtractor, sampling: SamplingConfig) -> list[ClipPlan]:
    """Build a list of clips to sample from the video."""
    start_frame, end_frame = _resolve_frame_bounds(extractor, sampling)

    if sampling.explicit_starts:
        starts = []
        max_start = end_frame - sampling.clip_length
        for value in sampling.explicit_starts:
            if start_frame <= value <= max_start:
                starts.append(value)
        if not starts:
            raise ValueError("No valid explicit start frames within range")
    elif sampling.method == "uniform":
        starts = _build_uniform_starts(
            start_frame, end_frame, sampling.clip_length, sampling.num_clips
        )
    elif sampling.method == "random":
        starts = _build_random_starts(
            start_frame,
            end_frame,
            sampling.clip_length,
            sampling.num_clips,
            sampling.min_gap,
            sampling.random_seed,
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling.method}")

    plans: list[ClipPlan] = []
    for idx, clip_start in enumerate(starts, 1):
        plans.append(
            ClipPlan(
                index=idx,
                start_frame=clip_start,
                end_frame=clip_start + sampling.clip_length,
            )
        )
    return plans


def save_manifest(
    output_dir: Path,
    video_path: Path,
    extractor: VideoExtractor,
    output: OutputConfig,
    plans: list[ClipPlan],
) -> None:
    manifest = {
        "video_path": str(video_path),
        "fps": extractor.fps,
        "frame_count": extractor.frame_count,
        "frame_format": output.frame_format,
        "label_format": output.label_format,
        "clips": [
            {
                "index": plan.index,
                "start_frame": plan.start_frame,
                "end_frame": plan.end_frame,
            }
            for plan in plans
        ],
    }
    manifest_path = output_dir / "clip_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def sample_clips(cfg: ToolConfig) -> list[ClipPlan]:
    """Sample clips from a video and export frames to WASB dataset structure."""
    if cfg.video_path is None:
        raise ValueError("video_path is required for sampling")

    video_path = Path(cfg.video_path)
    extractor = VideoExtractor(video_path)
    output_dir = _resolve_game_dir(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = build_clip_plan(extractor, cfg.sampling)
    if not plans:
        raise ValueError("No clips selected for sampling")

    for plan in plans:
        clip_dir = output_dir / f"Clip{plan.index}"
        if clip_dir.exists():
            if cfg.output.skip_existing:
                LOGGER.info("Skipping existing clip: %s", clip_dir)
                continue
            if cfg.output.overwrite:
                for path in clip_dir.iterdir():
                    if path.is_file():
                        path.unlink()
                LOGGER.info("Overwriting clip contents: %s", clip_dir)
            else:
                raise FileExistsError(f"Clip directory exists: {clip_dir}")

        clip_dir.mkdir(parents=True, exist_ok=True)
        extractor.extract_segment(
            start_frame=plan.start_frame,
            end_frame=plan.end_frame,
            output_dir=clip_dir,
            frame_format=cfg.output.frame_format,
            jpeg_quality=cfg.output.jpeg_quality,
        )

    save_manifest(output_dir, video_path, extractor, cfg.output, plans)
    return plans


def _list_frame_files(clip_dir: Path) -> list[Path]:
    return sorted(
        [path for path in clip_dir.iterdir() if path.suffix.lower() in {".jpg", ".png"}]
    )


def _build_label_rows(
    frame_files: list[Path],
    label_format: str,
    existing: dict[str, TennisLabelRow] | None,
) -> list[TennisLabelRow]:
    rows: list[TennisLabelRow] = []
    for idx, _ in enumerate(frame_files):
        label_name = label_format.format(idx)
        if existing and label_name in existing:
            rows.append(existing[label_name])
        else:
            rows.append(
                row_from_visibility(
                    file_name=label_name,
                    x=0.0,
                    y=0.0,
                    visibility=0,
                    score=0.0,
                )
            )
    return rows


def _draw_overlay(
    frame: cv2.Mat,
    row: TennisLabelRow,
    frame_idx: int,
    frame_count: int,
    config: AnnotationConfig,
) -> cv2.Mat:
    annotated = frame.copy()

    if row.visibility > 0:
        x = int(round(row.x))
        y = int(round(row.y))
        cv2.circle(
            annotated,
            (x, y),
            config.circle_radius,
            config.circle_color_bgr,
            config.line_thickness,
            cv2.LINE_AA,
        )

    text_lines = [
        f"Frame {frame_idx + 1}/{frame_count}",
        f"Visibility: {row.visibility}",
        "LMB: set ball | RMB/C: clear | N/P: next/prev | R: copy prev | Q: save+quit",
    ]
    for idx, text in enumerate(text_lines):
        cv2.putText(
            annotated,
            text,
            (12, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.text_color_bgr,
            1,
            cv2.LINE_AA,
        )

    return annotated


def annotate_clip(clip_dir: Path, output: OutputConfig, config: AnnotationConfig) -> None:
    """Annotate a single clip directory in-place."""
    frame_files = _list_frame_files(clip_dir)
    if not frame_files:
        LOGGER.warning("No frames found in %s", clip_dir)
        return

    label_path = clip_dir / "Label.csv"
    existing_rows: dict[str, TennisLabelRow] | None = None
    if label_path.exists():
        existing_rows = {row.file_name: row for row in load_label_csv(label_path)}

    label_rows = _build_label_rows(frame_files, output.label_format, existing_rows)

    state = {"index": 0}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            label_name = output.label_format.format(state["index"])
            label_rows[state["index"]] = row_from_visibility(
                file_name=label_name,
                x=float(x),
                y=float(y),
                visibility=1,
                score=1.0,
            )
        elif event == cv2.EVENT_RBUTTONDOWN:
            label_name = output.label_format.format(state["index"])
            label_rows[state["index"]] = row_from_visibility(
                file_name=label_name,
                x=0.0,
                y=0.0,
                visibility=0,
                score=0.0,
            )

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(config.window_name, on_mouse)

    try:
        while True:
            idx = state["index"]
            frame = cv2.imread(str(frame_files[idx]))
            if frame is None:
                LOGGER.warning("Failed to read frame: %s", frame_files[idx])
                break

            overlay = _draw_overlay(
                frame,
                label_rows[idx],
                idx,
                len(frame_files),
                config,
            )
            cv2.imshow(config.window_name, overlay)
            key = cv2.waitKey(0) & 0xFF

            if key in {ord("n"), ord(" ")}:
                state["index"] = min(idx + 1, len(frame_files) - 1)
            elif key == ord("p"):
                state["index"] = max(idx - 1, 0)
            elif key == ord("c"):
                label_name = output.label_format.format(idx)
                label_rows[idx] = row_from_visibility(
                    file_name=label_name,
                    x=0.0,
                    y=0.0,
                    visibility=0,
                    score=0.0,
                )
            elif key == ord("r"):
                if idx > 0:
                    label_rows[idx] = label_rows[idx - 1]
            elif key in {ord("q"), 27}:
                break

        save_label_csv(label_path, label_rows)
    finally:
        cv2.destroyWindow(config.window_name)


def annotate_clips(cfg: ToolConfig) -> None:
    """Annotate all requested clips under the game directory."""
    output_dir = _resolve_game_dir(cfg.output)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if cfg.annotate.clip_indices:
        clip_dirs = [
            output_dir / f"Clip{idx}"
            for idx in cfg.annotate.clip_indices
            if (output_dir / f"Clip{idx}").exists()
        ]
    else:
        clip_dirs = sorted(
            [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("Clip")]
        )

    if not clip_dirs:
        raise FileNotFoundError("No clip directories found for annotation")

    for clip_dir in clip_dirs:
        print(f"Annotating {clip_dir}...")
        annotate_clip(clip_dir, cfg.output, cfg.annotate)


@hydra.main(version_base=None, config_path="configs", config_name="annotate_wasb_clips")
def main(cfg: DictConfig) -> None:
    """Run the annotation tool based on Hydra config."""
    config = ToolConfig(
        mode=cfg.get("mode", "all"),
        video_path=cfg.get("video_path", None),
        output=OutputConfig(
            output_dir=cfg.output.get("output_dir", "data/tennis"),
            game_name=cfg.output.get("game_name", "game_manual"),
            frame_format=cfg.output.get("frame_format", "frame_{:04d}.jpg"),
            label_format=cfg.output.get("label_format", "{:04d}.jpg"),
            jpeg_quality=int(cfg.output.get("jpeg_quality", 95)),
            overwrite=bool(cfg.output.get("overwrite", False)),
            skip_existing=bool(cfg.output.get("skip_existing", True)),
        ),
        sampling=SamplingConfig(
            method=cfg.sampling.get("method", "uniform"),
            clip_length=int(cfg.sampling.get("clip_length", 90)),
            num_clips=int(cfg.sampling.get("num_clips", 10)),
            min_gap=int(cfg.sampling.get("min_gap", 30)),
            start_frame=cfg.sampling.get("start_frame", None),
            end_frame=cfg.sampling.get("end_frame", None),
            explicit_starts=list(cfg.sampling.get("explicit_starts", [])),
            random_seed=int(cfg.sampling.get("random_seed", 7)),
        ),
        annotate=AnnotationConfig(
            clip_indices=list(cfg.annotate.get("clip_indices", [])),
            window_name=cfg.annotate.get("window_name", "WASB Annotator"),
            circle_radius=int(cfg.annotate.get("circle_radius", 6)),
            circle_color_bgr=tuple(cfg.annotate.get("circle_color_bgr", (0, 0, 255))),
            text_color_bgr=tuple(cfg.annotate.get("text_color_bgr", (255, 255, 255))),
            font_scale=float(cfg.annotate.get("font_scale", 0.6)),
            line_thickness=int(cfg.annotate.get("line_thickness", 2)),
        ),
    )

    if config.mode in {"sample", "all"}:
        sample_clips(config)

    if config.mode in {"annotate", "all"}:
        annotate_clips(config)


if __name__ == "__main__":
    main()
