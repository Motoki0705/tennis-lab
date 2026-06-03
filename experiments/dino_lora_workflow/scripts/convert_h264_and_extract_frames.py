"""Overview:
Convert AV1 source videos to H.264 and extract review / pseudo-label frames.
Uses seek-based extraction: jumps to each target frame index instead of
sequentially decoding every frame. This avoids processing frames that will
be discarded.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py dry_run=true frame_stride=30
    .venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py h264.enabled=false

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/convert_h264_and_extract_frames.yaml`.
    - AV1 source videos are read from `data/youtube/videos/av1/` via manifest.json or glob.
    - H.264 conversions are cached under `data/youtube/videos/h264/`.
    - Frames are stored under `data/dino_workflow/sources/youtube/frames/<video_id>/`.
    - Set `h264.enabled=false` to skip conversion and extract directly from AV1 sources.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class SourceVideo:
    """One source video resolved from the download manifest or a glob fallback."""

    video_id: str
    video_index: int
    source_path: Path
    source_url: str
    source_title: str
    h264_path: Path | None = None


@dataclass(slots=True)
class VideoProbe:
    """OpenCV metadata for one video file."""

    fps: float
    width: int
    height: int
    total_frames: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return int(value)


def parse_video_index(video_id: str, fallback: int) -> int:
    prefix = "video_"
    if video_id.startswith(prefix):
        try:
            return int(video_id[len(prefix) :])
        except ValueError:
            return fallback
    return fallback


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def resolve_ffmpeg_binary(cfg: DictConfig) -> str:
    configured = str(cfg.h264.get("ffmpeg_binary", "auto") or "auto")
    if configured != "auto":
        return configured
    return shutil.which("ffmpeg") or "ffmpeg"


def read_h264_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"entries": [], "updated_at": now_iso()}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"entries": [], "updated_at": now_iso()}


def write_h264_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = now_iso()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def find_h264_entry(manifest: dict[str, Any], video_id: str) -> dict[str, Any] | None:
    for entry in manifest.get("entries", []):
        if entry.get("video_id") == video_id:
            return entry
    return None


def ensure_h264(video: SourceVideo, cfg: DictConfig) -> Path:
    """Convert AV1 source to H.264 if not already cached. Returns H.264 path."""
    if not bool(cfg.h264.enabled):
        return video.source_path

    h264_dir = Path(to_absolute_path(str(cfg.h264_output_dir))).resolve()
    h264_dir.mkdir(parents=True, exist_ok=True)
    h264_path = (h264_dir / video.source_path.name).resolve()

    if h264_path.exists():
        return h264_path

    ffmpeg = resolve_ffmpeg_binary(cfg)
    codec = str(cfg.h264.video_codec)
    preset = str(cfg.h264.preset)
    cq = str(int(cfg.h264.cq))
    pix_fmt = str(cfg.h264.pixel_format)

    command = [
        ffmpeg,
        "-y",
        "-i", str(video.source_path),
        "-c:v", codec,
        "-preset", preset,
        "-cq", cq,
        "-pix_fmt", pix_fmt,
        "-movflags", "+faststart",
    ]
    if bool(cfg.h264.drop_audio):
        command.append("-an")

    command.append(str(h264_path))

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(cfg.h264.timeout_seconds),
    )

    if not h264_path.exists():
        raise FileNotFoundError(f"H.264 output was not created: {h264_path}")

    # Update conversion manifest
    manifest_path = h264_dir / str(cfg.h264_manifest_file)
    h264_manifest = read_h264_manifest(manifest_path)

    existing = find_h264_entry(h264_manifest, video.video_id)
    entry = {
        "video_id": video.video_id,
        "video_index": video.video_index,
        "av1_path": str(video.source_path),
        "h264_path": str(h264_path),
        "h264_size_bytes": h264_path.stat().st_size,
        "ffmpeg_codec": codec,
        "converted_at": now_iso(),
    }
    if existing is not None:
        existing.update(entry)
    else:
        h264_manifest.setdefault("entries", []).append(entry)

    write_h264_manifest(manifest_path, h264_manifest)
    return h264_path


def resolve_source_path(entry: dict[str, Any], video_dir: Path) -> Path:
    absolute_path = entry.get("absolute_path")
    if absolute_path:
        candidate = Path(str(absolute_path)).expanduser()
        if candidate.exists():
            return candidate.resolve()

    relative_path = entry.get("relative_path")
    if relative_path:
        candidate = video_dir / str(relative_path)
        if candidate.exists():
            return candidate.resolve()

    file_name = entry.get("file_name") or entry.get("source_filename")
    if file_name:
        return (video_dir / str(file_name)).resolve()

    raise ValueError(f"Manifest entry does not include a usable video path: {entry}")


def source_from_manifest_entry(
    entry: dict[str, Any],
    *,
    video_dir: Path,
    fallback_index: int,
) -> SourceVideo:
    source_path = resolve_source_path(entry, video_dir)
    video_id = source_path.stem
    video_index = int(entry.get("index") or parse_video_index(video_id, fallback_index))
    return SourceVideo(
        video_id=video_id,
        video_index=video_index,
        source_path=source_path,
        source_url=str(entry.get("url") or ""),
        source_title=str(entry.get("title") or video_id),
    )


def collect_source_videos(video_dir: Path, manifest_path: Path) -> list[SourceVideo]:
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        entries = manifest.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError(f"Manifest entries must be a list: {manifest_path}")
        videos = [
            source_from_manifest_entry(entry, video_dir=video_dir, fallback_index=index)
            for index, entry in enumerate(entries, start=1)
        ]
    else:
        videos = [
            SourceVideo(
                video_id=path.stem,
                video_index=parse_video_index(path.stem, index),
                source_path=path.resolve(),
                source_url="",
                source_title=path.stem,
            )
            for index, path in enumerate(sorted(video_dir.glob("video_*.mp4")), start=1)
        ]

    if not videos:
        source = f"manifest {manifest_path}" if manifest_path.exists() else f"glob {video_dir}/video_*.mp4"
        raise FileNotFoundError(f"No source videos found from {source}")
    return sorted(videos, key=lambda item: (item.video_index, item.video_id))


def probe_video_path(video_path: Path, video: SourceVideo) -> VideoProbe:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video: {video.source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return VideoProbe(fps=fps, width=width, height=height, total_frames=total_frames)


def plan_known_frame_indices(
    *,
    total_frames: int,
    frame_stride: int,
    start_frame: int,
    end_frame: int | None,
    max_frames_per_video: int | None,
) -> list[int] | None:
    if total_frames <= 0 and end_frame is None:
        return None

    if end_frame is None:
        resolved_end = total_frames - 1
    elif total_frames > 0:
        resolved_end = min(end_frame, total_frames - 1)
    else:
        resolved_end = end_frame

    if resolved_end < start_frame:
        return []

    frame_indices = list(range(start_frame, resolved_end + 1, frame_stride))
    if max_frames_per_video is not None and max_frames_per_video > 0:
        frame_indices = frame_indices[:max_frames_per_video]
    return frame_indices


def iter_target_frame_indices(
    *,
    total_frames: int,
    frame_stride: int,
    start_frame: int,
    end_frame: int | None,
    max_frames_per_video: int | None,
):
    emitted = 0
    current_frame = start_frame

    while True:
        if end_frame is not None and current_frame > end_frame:
            break
        if total_frames > 0 and current_frame >= total_frames:
            break
        if max_frames_per_video is not None and max_frames_per_video > 0:
            if emitted >= max_frames_per_video:
                break

        yield current_frame

        emitted += 1
        current_frame += frame_stride


def build_frame_entry(
    *,
    video: SourceVideo,
    probe: VideoProbe,
    source_frame_index: int,
    output_file: Path,
    output_dir: Path,
    status: str,
    video_path: Path | None = None,
) -> dict[str, Any]:
    resolved_path = video_path if video_path is not None else video.source_path
    return {
        "video_id": video.video_id,
        "video_index": video.video_index,
        "source_video": str(resolved_path),
        "source_url": video.source_url,
        "source_title": video.source_title,
        "fps": probe.fps,
        "width": probe.width,
        "height": probe.height,
        "source_frame_index": source_frame_index,
        "output_file": str(output_file.relative_to(output_dir)),
        "absolute_output_file": str(output_file.resolve()),
        "status": status,
        "processed_at": now_iso(),
    }


def build_video_summary(
    *,
    video: SourceVideo,
    probe: VideoProbe,
    planned_frame_count: int | str,
    written_frame_count: int,
    skipped_existing_count: int,
    failed_frame_count: int,
    output_dir: Path,
    video_output_dir: Path,
    frames_jsonl: Path,
    status: str,
    error: str | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    resolved_path = video_path if video_path is not None else video.source_path
    summary: dict[str, Any] = {
        "video_id": video.video_id,
        "video_index": video.video_index,
        "source_video": str(resolved_path),
        "source_url": video.source_url,
        "source_title": video.source_title,
        "fps": probe.fps,
        "width": probe.width,
        "height": probe.height,
        "total_frames": probe.total_frames,
        "planned_frame_count": planned_frame_count,
        "written_frame_count": written_frame_count,
        "skipped_existing_count": skipped_existing_count,
        "failed_frame_count": failed_frame_count,
        "status": status,
        "output_dir": str(video_output_dir.relative_to(output_dir)),
        "frames_jsonl": str(frames_jsonl.relative_to(output_dir)),
        "frames_manifest": str((video_output_dir / "frames_manifest.json").relative_to(output_dir)),
    }
    if error is not None:
        summary["error"] = error
    return summary


def write_video_manifests(
    *,
    video: SourceVideo,
    probe: VideoProbe,
    video_output_dir: Path,
    output_dir: Path,
    frame_entries: list[dict[str, Any]],
    summary: dict[str, Any],
    video_path: Path | None = None,
) -> None:
    resolved_path = video_path if video_path is not None else video.source_path
    frames_jsonl = video_output_dir / "frames.jsonl"
    write_jsonl(frames_jsonl, frame_entries)
    write_json(
        video_output_dir / "frames_manifest.json",
        {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "video_id": video.video_id,
            "video_index": video.video_index,
            "source_video": str(resolved_path),
            "source_url": video.source_url,
            "source_title": video.source_title,
            "fps": probe.fps,
            "width": probe.width,
            "height": probe.height,
            "total_frames": probe.total_frames,
            "frame_count": len(frame_entries),
            "frames_jsonl": str(frames_jsonl.relative_to(output_dir)),
            "summary": summary,
        },
    )


def dry_run_video(
    *,
    video: SourceVideo,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probe = probe_video_path(video.source_path, video)
    frame_indices = plan_known_frame_indices(
        total_frames=probe.total_frames,
        frame_stride=int(cfg.frame_stride),
        start_frame=int(cfg.start_frame),
        end_frame=optional_int(cfg.end_frame),
        max_frames_per_video=optional_int(cfg.max_frames_per_video),
    )
    planned_count: int | str = len(frame_indices) if frame_indices is not None else "unknown"
    video_output_dir = output_dir / video.video_id
    frames_jsonl = video_output_dir / "frames.jsonl"
    summary = build_video_summary(
        video=video,
        probe=probe,
        planned_frame_count=planned_count,
        written_frame_count=0,
        skipped_existing_count=0,
        failed_frame_count=0,
        output_dir=output_dir,
        video_output_dir=video_output_dir,
        frames_jsonl=frames_jsonl,
        status="planned",
    )
    frame_entries: list[dict[str, Any]] = []
    if frame_indices is not None:
        for frame_index in frame_indices:
            output_file = video_output_dir / f"frame_{frame_index:06d}.jpg"
            frame_entries.append(
                build_frame_entry(
                    video=video,
                    probe=probe,
                    source_frame_index=frame_index,
                    output_file=output_file,
                    output_dir=output_dir,
                    status="planned",
                )
            )
    return summary, frame_entries


def extract_video(
    *,
    video: SourceVideo,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    video_path = ensure_h264(video, cfg)
    probe = probe_video_path(video_path, video)
    frame_stride = int(cfg.frame_stride)
    start_frame = int(cfg.start_frame)
    end_frame = optional_int(cfg.end_frame)
    max_frames_per_video = optional_int(cfg.max_frames_per_video)
    jpeg_quality = int(cfg.jpeg_quality)
    video_output_dir = output_dir / video.video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    frames_jsonl = video_output_dir / "frames.jsonl"
    frame_entries: list[dict[str, Any]] = []
    written = 0
    skipped_existing = 0
    failed_frames = 0

    planned_frame_indices = plan_known_frame_indices(
        total_frames=probe.total_frames,
        frame_stride=frame_stride,
        start_frame=start_frame,
        end_frame=end_frame,
        max_frames_per_video=max_frames_per_video,
    )
    planned_count: int | str = (
        len(planned_frame_indices) if planned_frame_indices is not None else "unknown"
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video: {video.source_path}")

    try:
        for current_frame in iter_target_frame_indices(
            total_frames=probe.total_frames,
            frame_stride=frame_stride,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames_per_video=max_frames_per_video,
        ):
            output_file = video_output_dir / f"frame_{current_frame:06d}.jpg"

            if output_file.exists() and bool(cfg.skip_existing_frames):
                skipped_existing += 1
                status = "skipped_existing"

                frame_entries.append(
                    build_frame_entry(
                        video=video,
                        probe=probe,
                        source_frame_index=current_frame,
                        output_file=output_file,
                        output_dir=output_dir,
                        status=status,
                        video_path=video_path,
                    )
                )
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            ok, frame = capture.read()
            if not ok:
                failed_frames += 1
                status = "failed_read"

                frame_entries.append(
                    build_frame_entry(
                        video=video,
                        probe=probe,
                        source_frame_index=current_frame,
                        output_file=output_file,
                        output_dir=output_dir,
                        status=status,
                        video_path=video_path,
                    )
                )
                break

            saved = cv2.imwrite(
                str(output_file),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not saved:
                failed_frames += 1
                status = "failed_write"
            else:
                written += 1
                status = "written"

            frame_entries.append(
                build_frame_entry(
                    video=video,
                    probe=probe,
                    source_frame_index=current_frame,
                    output_file=output_file,
                    output_dir=output_dir,
                    status=status,
                    video_path=video_path,
                )
            )
    finally:
        capture.release()

    status = "ok" if failed_frames == 0 else "partial"
    summary = build_video_summary(
        video=video,
        probe=probe,
        planned_frame_count=planned_count,
        written_frame_count=written,
        skipped_existing_count=skipped_existing,
        failed_frame_count=failed_frames,
        output_dir=output_dir,
        video_output_dir=video_output_dir,
        frames_jsonl=frames_jsonl,
        status=status,
        video_path=video_path,
    )
    write_video_manifests(
        video=video,
        probe=probe,
        video_output_dir=video_output_dir,
        output_dir=output_dir,
        frame_entries=frame_entries,
        summary=summary,
        video_path=video_path,
    )
    return summary, frame_entries


def build_global_manifest(
    *,
    cfg: DictConfig,
    video_dir: Path,
    video_manifest: Path,
    output_dir: Path,
    videos: list[dict[str, Any]],
    frame_entries: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> dict[str, Any]:
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    manifest: dict[str, Any] = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "video_dir": str(video_dir),
        "video_manifest": str(video_manifest),
        "output_dir": str(output_dir),
        "dry_run": bool(cfg.dry_run),
        "config": config_payload,
        "video_count": len(videos),
        "frame_count": len(frame_entries),
        "failed_count": len(failed),
        "videos": videos,
        "frames": frame_entries if bool(cfg.write_frame_entries_to_global_manifest) else [],
        "failed": failed,
    }
    if bool(cfg.h264.enabled):
        h264_dir = Path(to_absolute_path(str(cfg.h264_output_dir))).resolve()
        manifest["h264_output_dir"] = str(h264_dir)
        manifest["h264_manifest_file"] = str(h264_dir / str(cfg.h264_manifest_file))
    return manifest


def convert_h264_and_extract_frames(cfg: DictConfig) -> dict[str, Any]:
    if int(cfg.frame_stride) <= 0:
        raise ValueError("frame_stride must be positive")
    if int(cfg.start_frame) < 0:
        raise ValueError("start_frame must be non-negative")
    jpeg_quality = int(cfg.jpeg_quality)
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("jpeg_quality must be in [1, 100]")

    video_dir = Path(to_absolute_path(str(cfg.video_dir))).resolve()
    video_manifest = Path(to_absolute_path(str(cfg.video_manifest))).resolve()
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    videos = collect_source_videos(video_dir, video_manifest)

    video_summaries: list[dict[str, Any]] = []
    all_frame_entries: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if not bool(cfg.dry_run):
        output_dir.mkdir(parents=True, exist_ok=True)

    for video in videos:
        try:
            if bool(cfg.dry_run):
                summary, frame_entries = dry_run_video(video=video, cfg=cfg, output_dir=output_dir)
            else:
                summary, frame_entries = extract_video(video=video, cfg=cfg, output_dir=output_dir)
            video_summaries.append(summary)
            all_frame_entries.extend(frame_entries)
        except Exception as exc:  # noqa: BLE001
            error = {"video_id": video.video_id, "source_video": str(video.source_path), "error": repr(exc)}
            failed.append(error)
            if bool(cfg.fail_fast):
                raise

    manifest = build_global_manifest(
        cfg=cfg,
        video_dir=video_dir,
        video_manifest=video_manifest,
        output_dir=output_dir,
        videos=video_summaries,
        frame_entries=all_frame_entries,
        failed=failed,
    )
    if not bool(cfg.dry_run):
        write_json(output_dir / "manifest.json", manifest)

    return {
        "dry_run": bool(cfg.dry_run),
        "video_count": len(video_summaries),
        "frame_count": len(all_frame_entries),
        "failed_count": len(failed),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "videos": video_summaries,
        "failed": failed,
    }


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="convert_h264_and_extract_frames",
)
def main(cfg: DictConfig) -> None:
    summary = convert_h264_and_extract_frames(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()