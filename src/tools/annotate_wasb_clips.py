"""Annotate WASB clip ranges with OpenCV UI using Hydra-managed configuration.

Example commands:
    `uv run python -m src.tools.annotate_wasb_clips video_path=data/sample.mp4`
    `uv run python -m src.tools.annotate_wasb_clips sampling.method=manual sampling.preview_stride=3`

Config entry point: `src/tools/configs/annotate_wasb_clips.yaml`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig


@dataclass
class ClipPlan:
    """Clip span in frame indices."""

    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        """Return number of frames in the clip (inclusive)."""
        return self.end_frame - self.start_frame + 1


@dataclass
class SamplingConfig:
    """Sampling configuration for clip selection."""

    method: str = "manual"
    preview_stride: int = 3
    preview_scale: float = 0.5
    min_clip_length: int = 15


@dataclass
class AnnotationConfig:
    """Top-level configuration for WASB clip annotation."""

    video_path: str
    output_dir: str = "outputs/wasb/clip_annotations"
    manifest_name: str = "clip_manifest.json"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser()


def _resize_frame(frame: cv2.Mat, scale: float) -> cv2.Mat:
    if scale == 1.0:
        return frame
    height, width = frame.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _overlay_status(
    frame: cv2.Mat,
    frame_index: int,
    start_mark: int | None,
    end_mark: int | None,
    clips: list[ClipPlan],
) -> None:
    status_lines = [
        f"Frame: {frame_index}",
        f"Start: {start_mark if start_mark is not None else '-'}",
        f"End: {end_mark if end_mark is not None else '-'}",
        f"Clips: {len(clips)}",
        "Keys: S=start, E=end, A=add, D=delete, Q=quit",
    ]
    y = 24
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


def select_clips_manual(video_path: Path, sampling: SamplingConfig) -> list[ClipPlan]:
    """Interactively select clip ranges using OpenCV UI."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        stride = max(1, int(sampling.preview_stride))
        max_preview_index = max(0, (total_frames - 1) // stride)
        window_name = "WASB Clip Sampler"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        state = {
            "preview_index": 0,
        }

        def on_trackbar(val: int) -> None:
            state["preview_index"] = val

        cv2.createTrackbar("Frame", window_name, 0, max_preview_index, on_trackbar)

        start_mark: int | None = None
        end_mark: int | None = None
        clips: list[ClipPlan] = []

        while True:
            preview_index = state["preview_index"]
            frame_index = min(total_frames - 1, preview_index * stride)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            display = _resize_frame(frame, sampling.preview_scale)
            _overlay_status(display, frame_index, start_mark, end_mark, clips)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                start_mark = frame_index
            if key in (ord("e"), ord("E")):
                end_mark = frame_index
            if key in (ord("d"), ord("D")) and clips:
                clips.pop()
            if key in (ord("a"), ord("A")):
                if start_mark is None or end_mark is None:
                    print("Set both start (S) and end (E) before adding.")
                    continue
                start = min(start_mark, end_mark)
                end = max(start_mark, end_mark)
                clip = ClipPlan(start_frame=start, end_frame=end)
                if clip.length < sampling.min_clip_length:
                    print(
                        f"Clip length {clip.length} < min_clip_length={sampling.min_clip_length}; skipped."
                    )
                    continue
                clips.append(clip)
                start_mark = None
                end_mark = None

        cv2.destroyWindow(window_name)
        return clips
    finally:
        cap.release()


def build_clip_plan(video_path: Path, sampling: SamplingConfig) -> list[ClipPlan]:
    """Build clip plan based on sampling method."""
    method = sampling.method.lower()
    if method == "manual":
        return select_clips_manual(video_path, sampling)
    raise ValueError(f"Unsupported sampling method: {sampling.method}")


def save_clip_manifest(
    output_dir: Path,
    video_path: Path,
    sampling: SamplingConfig,
    clips: list[ClipPlan],
    manifest_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    payload = {
        "video_path": str(video_path),
        "method": sampling.method,
        "preview_stride": sampling.preview_stride,
        "preview_scale": sampling.preview_scale,
        "min_clip_length": sampling.min_clip_length,
        "clips": [
            {"start_frame": clip.start_frame, "end_frame": clip.end_frame}
            for clip in clips
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="annotate_wasb_clips",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for WASB clip annotation."""
    sampling_cfg = SamplingConfig(
        method=str(cfg.sampling.get("method", "manual")),
        preview_stride=int(cfg.sampling.get("preview_stride", 3)),
        preview_scale=float(cfg.sampling.get("preview_scale", 0.5)),
        min_clip_length=int(cfg.sampling.get("min_clip_length", 15)),
    )
    config = AnnotationConfig(
        video_path=str(cfg.video_path),
        output_dir=str(cfg.get("output_dir", "outputs/wasb/clip_annotations")),
        manifest_name=str(cfg.get("manifest_name", "clip_manifest.json")),
        sampling=sampling_cfg,
    )

    video_path = _resolve_path(config.video_path)
    clips = build_clip_plan(video_path, config.sampling)
    manifest_path = save_clip_manifest(
        _resolve_path(config.output_dir),
        video_path,
        config.sampling,
        clips,
        config.manifest_name,
    )
    print(f"Saved clip manifest: {manifest_path}")


if __name__ == "__main__":
    main()
