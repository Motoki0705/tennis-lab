"""Manually create clips from a video and annotate tennis ball positions for WASB.

This tool supports a two-phase workflow:
1) Manually create clip segments from a video and export them in WASB's dataset structure.
2) Manually annotate ball positions per frame for the created clips.

Example commands:
    # Manually create clips
    `uv run python -m src.tools.annotate_wasb_clips mode=clip video_path=data/raw/match.mp4 \
        output.output_dir=data/tennis output.game_name=game_manual`

    # Only annotate existing clips
    `uv run python -m src.tools.annotate_wasb_clips mode=annotation output.output_dir=data/tennis \
        output.game_name=game_manual annotate.clip_indices=[1,3]`

Config entry point: `src/tools/configs/annotate_wasb_clips.yaml`
"""

from __future__ import annotations

import json
import logging
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
class ClipConfig:
    """Configuration for manual clip selection UI."""

    window_name: str = "WASB Clip Selector"
    step_small: int = 1
    step_large: int = 10
    skip_multipliers: tuple[int, ...] = (1, 5, 10, 20)
    display_scale: float = 1.0


@dataclass
class OutputConfig:
    """Configuration for output dataset structure."""

    output_dir: str = "data/tennis"
    game_name: str = "game_manual"
    frame_format: str = "frame_{:04d}.jpg"
    label_format: str = "{:04d}.jpg"
    jpeg_quality: int = 95
    precise_seek: bool = True
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

    mode: Literal["clip", "annotation"] = "clip"
    video_path: str | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    annotate: AnnotationConfig = field(default_factory=AnnotationConfig)


@dataclass(frozen=True)
class ClipPlan:
    """Plan for a single clip segment."""

    index: int
    start_frame: int
    end_frame: int


def _resolve_game_dir(output: OutputConfig) -> Path:
    return Path(output.output_dir) / output.game_name


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


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> cv2.Mat | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _draw_clip_overlay(
    frame: cv2.Mat,
    frame_idx: int,
    frame_count: int,
    in_mark: int | None,
    out_mark: int | None,
    clip_count: int,
    skip_multiplier: int,
    config: ClipConfig,
) -> cv2.Mat:
    annotated = frame.copy()
    text_lines = [
        f"Frame {frame_idx + 1}/{frame_count}",
        f"In: {in_mark + 1 if in_mark is not None else '-'}",
        f"Out: {out_mark + 1 if out_mark is not None else '-'}",
        f"Clips: {clip_count}",
        f"Skip: x{skip_multiplier} (1-{len(config.skip_multipliers)} set, [/] cycle)",
        "I: set in | O: set out | S: add clip | D: delete last | C: clear",
        "N/P: next/prev | F/B: jump | Q: save+quit",
    ]
    for idx, text in enumerate(text_lines):
        cv2.putText(
            annotated,
            text,
            (12, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if config.display_scale != 1.0:
        annotated = cv2.resize(
            annotated,
            None,
            fx=config.display_scale,
            fy=config.display_scale,
            interpolation=cv2.INTER_AREA,
        )
    return annotated


def manual_clip_selection(video_path: Path, config: ClipConfig) -> list[tuple[int, int]]:
    """Launch a UI for manually selecting clip ranges."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    in_mark: int | None = None
    out_mark: int | None = None
    clips: list[tuple[int, int]] = []
    skip_multipliers = list(config.skip_multipliers) or [1]
    skip_index = 0

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = _read_frame(cap, frame_idx)
            if frame is None:
                LOGGER.warning("Failed to read frame: %s", frame_idx)
                break

            overlay = _draw_clip_overlay(
                frame,
                frame_idx,
                frame_count,
                in_mark,
                out_mark,
                len(clips),
                skip_multipliers[skip_index],
                config,
            )
            cv2.imshow(config.window_name, overlay)
            key = cv2.waitKey(0) & 0xFF

            if key in {ord("n"), ord(" ")}:
                step = config.step_small * skip_multipliers[skip_index]
                frame_idx = min(frame_idx + step, frame_count - 1)
            elif key == ord("p"):
                step = config.step_small * skip_multipliers[skip_index]
                frame_idx = max(frame_idx - step, 0)
            elif key == ord("f"):
                frame_idx = min(frame_idx + config.step_large, frame_count - 1)
            elif key == ord("b"):
                frame_idx = max(frame_idx - config.step_large, 0)
            elif key == ord("["):
                skip_index = (skip_index - 1) % len(skip_multipliers)
            elif key == ord("]"):
                skip_index = (skip_index + 1) % len(skip_multipliers)
            elif ord("1") <= key <= ord("9"):
                candidate = key - ord("1")
                if candidate < len(skip_multipliers):
                    skip_index = candidate
            elif key == ord("i"):
                in_mark = frame_idx
            elif key == ord("o"):
                out_mark = frame_idx
            elif key == ord("c"):
                in_mark = None
                out_mark = None
            elif key == ord("d"):
                if clips:
                    clips.pop()
            elif key == ord("s"):
                if in_mark is None or out_mark is None:
                    LOGGER.info("Set both in/out before adding a clip.")
                else:
                    start = min(in_mark, out_mark)
                    end = max(in_mark, out_mark) + 1
                    if end > start:
                        clips.append((start, end))
                    in_mark = None
                    out_mark = None
            elif key in {ord("q"), 27}:
                break
    finally:
        cap.release()
        cv2.destroyWindow(config.window_name)

    if not clips:
        raise ValueError("No clips selected in manual clip UI")

    return clips


def export_clips(cfg: ToolConfig, clip_ranges: list[tuple[int, int]]) -> list[ClipPlan]:
    """Export manually selected clips to WASB dataset structure."""
    if cfg.video_path is None:
        raise ValueError("video_path is required for clip export")

    video_path = Path(cfg.video_path)
    extractor = VideoExtractor(video_path)
    output_dir = _resolve_game_dir(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans: list[ClipPlan] = []
    for idx, (start_frame, end_frame) in enumerate(clip_ranges, 1):
        if end_frame <= start_frame:
            raise ValueError("Clip end must be greater than start")
        if start_frame < 0 or end_frame > extractor.frame_count:
            raise ValueError("Clip range is out of bounds")
        plans.append(ClipPlan(index=idx, start_frame=start_frame, end_frame=end_frame))

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
            seek_every_frame=cfg.output.precise_seek,
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
        mode=cfg.get("mode", "clip"),
        video_path=cfg.get("video_path", None),
        output=OutputConfig(
            output_dir=cfg.output.get("output_dir", "data/tennis"),
            game_name=cfg.output.get("game_name", "game_manual"),
            frame_format=cfg.output.get("frame_format", "frame_{:04d}.jpg"),
            label_format=cfg.output.get("label_format", "{:04d}.jpg"),
            jpeg_quality=int(cfg.output.get("jpeg_quality", 95)),
            precise_seek=bool(cfg.output.get("precise_seek", True)),
            overwrite=bool(cfg.output.get("overwrite", False)),
            skip_existing=bool(cfg.output.get("skip_existing", True)),
        ),
        clip=ClipConfig(
            window_name=cfg.clip.get("window_name", "WASB Clip Selector"),
            step_small=int(cfg.clip.get("step_small", 1)),
            step_large=int(cfg.clip.get("step_large", 10)),
            skip_multipliers=tuple(cfg.clip.get("skip_multipliers", (1, 5, 10, 20))),
            display_scale=float(cfg.clip.get("display_scale", 1.0)),
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

    if config.mode == "clip":
        if config.video_path is None:
            raise ValueError("video_path is required for clip mode")
        clip_ranges = manual_clip_selection(Path(config.video_path), config.clip)
        export_clips(config, clip_ranges)

    if config.mode == "annotation":
        annotate_clips(config)


if __name__ == "__main__":
    main()
