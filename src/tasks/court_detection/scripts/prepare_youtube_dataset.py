"""Prepare a YouTube-sourced court keypoint annotation dataset.

Usage:
    python -m src.tasks.court_detection.scripts.prepare_youtube_dataset
    python -m src.tasks.court_detection.scripts.prepare_youtube_dataset workflow.sources.0.url=https://www.youtube.com/watch?v=...
    python -m src.tasks.court_detection.scripts.prepare_youtube_dataset workflow.download.enabled=false

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/prepare_youtube_dataset.yaml`.
    - Videos are downloaded as AV1 first, transcoded to H.264, then sampled into frames.
    - Annotation JSON files are initialized under `data/court/youtube/annotations/{train,val}.json`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils.schema.court import COURT_KP_NAMES

F = TypeVar("F", bound=Callable[..., Any])
JSONDict = dict[str, Any]


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(
    config_path="../configs",
    config_name="prepare_youtube_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    workflow_cfg = cfg.workflow
    root = Path(to_absolute_path(str(workflow_cfg.root))).resolve()
    paths_cfg = workflow_cfg.paths
    av1_dir = root / str(paths_cfg.videos_dir) / str(paths_cfg.av1_dir)
    h264_dir = root / str(paths_cfg.videos_dir) / str(paths_cfg.h264_dir)
    frames_root = root / str(paths_cfg.frames_dir)
    annotations_dir = root / str(paths_cfg.annotations_dir)
    manifests_dir = root / str(paths_cfg.manifests_dir)
    _ensure_dirs([av1_dir, h264_dir, frames_root, annotations_dir, manifests_dir])

    sources = _source_dicts(workflow_cfg.sources)
    frame_records_by_split: dict[str, list[JSONDict]] = {"train": [], "val": []}
    source_records: list[JSONDict] = []
    download_records: list[JSONDict] = []

    for index, source in enumerate(sources, start=1):
        video_id = str(source.get("source_id") or f"video_{index:06d}")
        source = {**source, "source_id": video_id}
        split = str(source.get("split") or workflow_cfg.split.default)
        source_records.append(source)

        print(f"[prepare_youtube_dataset] source={video_id} split={split}")
        av1_video = _download_av1(source, video_id, av1_dir, workflow_cfg.download)
        h264_video = _transcode_h264(av1_video, video_id, h264_dir, workflow_cfg.transcode)
        info = _read_info_json(av1_dir / f"{video_id}.info.json")
        download_records.append({
            "video_id": video_id,
            "source_url": source["url"],
            "source_title": info.get("title"),
            "av1_video": _relative_path(av1_video, root),
            "h264_video": _relative_path(h264_video, root),
            "processed_at": _utc_now(),
        })

        frame_records = _extract_frames(
            source,
            split,
            h264_video,
            frames_root / video_id,
            root,
            workflow_cfg.frames,
            info,
        )
        frame_records_by_split.setdefault(split, []).extend(frame_records)

    _write_jsonl(manifests_dir / "sources.jsonl", source_records)
    _write_jsonl(manifests_dir / "download_manifest.jsonl", download_records)
    _write_annotations(annotations_dir, frame_records_by_split, workflow_cfg.annotation)
    _write_json_atomic(
        manifests_dir / "split_manifest.json",
        {
            "schema_name": "court_youtube_split_manifest_v1",
            "counts": {split: len(records) for split, records in frame_records_by_split.items()},
            "written_at": _utc_now(),
        },
    )
    return 0


def _source_dicts(raw_sources: Iterable[Any]) -> list[JSONDict]:
    sources: list[JSONDict] = []
    for source in raw_sources:
        source_dict = cast(JSONDict, OmegaConf.to_container(source, resolve=True))
        if not source_dict.get("url"):
            raise ValueError("Each workflow.sources entry must define a non-empty url.")
        sources.append(source_dict)
    return sources


def _download_av1(source: JSONDict, video_id: str, av1_dir: Path, cfg: DictConfig) -> Path:
    existing = _find_video(av1_dir, video_id)
    if existing is not None and (not bool(cfg.enabled) or not bool(cfg.overwrite)):
        print(f"  AV1 exists: {existing}")
        return existing
    if not bool(cfg.enabled):
        raise FileNotFoundError(f"AV1 download disabled and no existing video found for {video_id}.")

    output_template = av1_dir / f"{video_id}.%(ext)s"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        str(source["url"]),
        "-f",
        str(cfg.strict_format if bool(cfg.require_av1) else cfg.format),
        "-o",
        str(output_template),
        "--merge-output-format",
        str(cfg.merge_output_format),
        "--write-info-json",
        "--no-playlist",
    ]
    js_runtimes = cfg.get("js_runtimes")
    if js_runtimes is not None:
        cmd.extend(["--js-runtimes", str(js_runtimes)])
    remote_components = cfg.get("remote_components")
    if remote_components is not None:
        cmd.extend(["--remote-components", str(remote_components)])
    archive_path = cfg.get("download_archive")
    if archive_path is not None:
        cmd.extend(["--download-archive", str(Path(to_absolute_path(str(archive_path))).resolve())])
    cmd.append("--force-overwrites" if bool(cfg.overwrite) else "--no-overwrites")
    cmd.extend(str(value) for value in cfg.get("extra_args", []))
    _run(cmd)

    downloaded = _find_video(av1_dir, video_id)
    if downloaded is None:
        raise FileNotFoundError(f"yt-dlp finished but no AV1 video was found for {video_id}.")
    return downloaded


def _transcode_h264(av1_video: Path, video_id: str, h264_dir: Path, cfg: DictConfig) -> Path:
    output_path = h264_dir / f"{video_id}.mp4"
    if output_path.exists() and not bool(cfg.overwrite):
        print(f"  H.264 exists: {output_path}")
        return output_path
    if not bool(cfg.enabled):
        raise FileNotFoundError(f"H.264 transcode disabled and output missing: {output_path}")

    cmd = [
        str(cfg.ffmpeg_binary),
        "-y" if bool(cfg.overwrite) else "-n",
    ]
    hwaccel = cfg.get("hwaccel")
    if hwaccel is not None:
        cmd.extend(["-hwaccel", str(hwaccel)])
    hwaccel_output_format = cfg.get("hwaccel_output_format")
    if hwaccel_output_format is not None:
        cmd.extend(["-hwaccel_output_format", str(hwaccel_output_format)])

    cmd.extend([
        "-i",
        str(av1_video),
        "-map",
        "0:v:0",
    ])
    cmd.extend(_h264_encoder_args(cfg))
    cmd.extend(["-movflags", "+faststart", "-an", str(output_path)])
    _run(cmd)
    return output_path


def _h264_encoder_args(cfg: DictConfig) -> list[str]:
    """Return FFmpeg arguments for H.264 encoding."""
    encoder = str(cfg.encoder)
    if encoder == "libx264":
        return [
            "-c:v",
            encoder,
            "-preset",
            str(cfg.preset),
            "-crf",
            str(cfg.crf),
            "-pix_fmt",
            str(cfg.pix_fmt),
        ]

    if encoder in {"h264_nvenc", "avc_nvenc"}:
        args = [
            "-c:v",
            encoder,
            "-preset",
            str(cfg.preset),
            "-tune",
            str(cfg.tune),
            "-rc",
            str(cfg.rate_control),
            "-cq",
            str(cfg.cq),
            "-pix_fmt",
            str(cfg.pix_fmt),
        ]
        bitrate = cfg.get("bitrate")
        if bitrate is None:
            args.extend(["-b:v", "0"])
        else:
            args.extend(["-b:v", str(bitrate)])
        maxrate = cfg.get("maxrate")
        if maxrate is not None:
            args.extend(["-maxrate", str(maxrate)])
        bufsize = cfg.get("bufsize")
        if bufsize is not None:
            args.extend(["-bufsize", str(bufsize)])
        profile = cfg.get("profile")
        if profile is not None:
            args.extend(["-profile:v", str(profile)])
        return args

    raise ValueError(f"Unsupported H.264 encoder: {encoder!r}")


def _extract_frames(
    source: JSONDict,
    split: str,
    video_path: Path,
    output_dir: Path,
    root: Path,
    cfg: DictConfig,
    info: JSONDict,
) -> list[JSONDict]:
    if not bool(cfg.enabled):
        return _read_jsonl(output_dir / "frames.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open H.264 video for frame extraction: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by OpenCV for {video_path}: {fps}")

    duration = frame_count / fps if frame_count > 0 else float(info.get("duration") or 0.0)
    frame_indices = _sample_frame_indices(source.get("time_ranges", []), duration, fps, cfg)
    if frame_count > 0:
        frame_indices = [frame_index for frame_index in frame_indices if frame_index < frame_count]
    max_frames = cfg.get("max_frames_per_video")
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]

    records: list[JSONDict] = []
    for frame_index in frame_indices:
        image_id = f"{source['source_id'].replace('video_', 'yt_')}_f{frame_index:08d}"
        output_path = output_dir / f"{image_id}.{cfg.output_ext}"
        if not output_path.exists() or bool(cfg.overwrite):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                print(f"  SKIP frame read failed: {video_path} frame={frame_index}")
                continue
            write_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)]
            if not cv2.imwrite(str(output_path), frame, write_params):
                raise RuntimeError(f"Failed to write frame: {output_path}")

        records.append({
            "id": image_id,
            "image_path": _relative_path(output_path, root),
            "video_id": source["source_id"],
            "source_url": source["url"],
            "source_title": info.get("title"),
            "source_frame_index": frame_index,
            "timestamp_sec": frame_index / fps,
            "fps": fps,
            "width": width,
            "height": height,
            "split": split,
            "status": "pending_annotation",
            "processed_at": _utc_now(),
        })

    capture.release()
    _write_jsonl(output_dir / "frames.jsonl", records)
    print(f"  frames: {len(records)} -> {output_dir}")
    return records


def _sample_frame_indices(raw_ranges: Any, duration: float, fps: float, cfg: DictConfig) -> list[int]:
    ranges = list(raw_ranges or [])
    if not ranges:
        ranges = [{"start": 0.0, "end": duration}]
    step_sec = _sample_step_seconds(cfg, fps)
    frame_indices: list[int] = []
    seen: set[int] = set()
    for time_range in ranges:
        start_sec = _parse_time_seconds(time_range.get("start", 0.0))
        end_value = time_range.get("end")
        end_sec = duration if end_value is None else _parse_time_seconds(end_value)
        timestamp = max(start_sec, 0.0)
        while timestamp <= max(end_sec, start_sec):
            frame_index = int(round(timestamp * fps))
            if frame_index not in seen:
                seen.add(frame_index)
                frame_indices.append(frame_index)
            timestamp += step_sec
    return frame_indices


def _sample_step_seconds(cfg: DictConfig, fps: float) -> float:
    mode = str(cfg.sample_mode)
    if mode == "interval_seconds":
        return float(cfg.interval_seconds)
    if mode == "fps":
        return 1.0 / float(cfg.fps)
    if mode == "every_n_frames":
        return float(cfg.every_n_frames) / fps
    raise ValueError(f"Unsupported frames.sample_mode={mode!r}.")


def _parse_time_seconds(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value)
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def _write_annotations(
    annotations_dir: Path,
    frame_records_by_split: dict[str, list[JSONDict]],
    cfg: DictConfig,
) -> None:
    for split in ("train", "val"):
        path = annotations_dir / f"{split}.json"
        existing_by_id = _existing_annotation_items(path) if bool(cfg.merge_existing) else {}
        items: list[JSONDict] = []
        for frame in frame_records_by_split.get(split, []):
            image_id = str(frame["id"])
            items.append(existing_by_id.get(image_id) or _initial_annotation_item(frame, split, cfg))
        payload = {
            "schema_name": str(cfg.schema_name),
            "keypoint_schema": "COURT_KP_NAMES",
            "keypoint_names": list(COURT_KP_NAMES),
            "keypoint_format_schema": {
                "kp15": _labeled_keypoint_indices("kp15"),
                "kp20": _labeled_keypoint_indices("kp20"),
            },
            "visibility_schema": {
                "0": "not_labeled",
                "1": "visible",
                "2": "occluded",
                "3": "out_of_frame",
            },
            "items": items,
        }
        if path.exists() and not bool(cfg.overwrite) and not bool(cfg.merge_existing):
            raise FileExistsError(f"Annotation file exists. Set overwrite=true or merge_existing=true: {path}")
        _write_json_atomic(path, payload)
        print(f"[prepare_youtube_dataset] annotations {split}: {len(items)} -> {path}")


def _initial_annotation_item(frame: JSONDict, split: str, cfg: DictConfig) -> JSONDict:
    keypoint_format = str(cfg.keypoint_format)
    labeled_indices = _labeled_keypoint_indices(keypoint_format)
    source_dataset = cfg.get("source_dataset")
    source: JSONDict = {
        "keypoint_format": keypoint_format,
        "labeled_keypoint_indices": labeled_indices,
        "video_id": frame["video_id"],
        "source_url": frame["source_url"],
        "source_title": frame.get("source_title"),
        "source_frame_index": frame["source_frame_index"],
        "timestamp_sec": frame["timestamp_sec"],
    }
    if source_dataset is not None:
        source["dataset"] = str(source_dataset)
    return {
        "id": frame["id"],
        "image_path": frame["image_path"],
        "width": frame["width"],
        "height": frame["height"],
        "split": split,
        "annotation_status": "pending",
        "keypoint_format": keypoint_format,
        "labeled_keypoint_indices": labeled_indices,
        "keypoints": [
            {
                "index": idx,
                "name": name,
                "x": None,
                "y": None,
                "visibility": 0,
            }
            for idx, name in enumerate(COURT_KP_NAMES)
        ],
        "source": source,
        "annotation": {
            "annotator": None,
            "created_at": None,
            "updated_at": None,
            "notes": "",
        },
    }


def _labeled_keypoint_indices(keypoint_format: str) -> list[int]:
    if keypoint_format == "kp15":
        return list(range(15))
    if keypoint_format == "kp20":
        return list(range(20))
    raise ValueError(f"Unsupported keypoint_format={keypoint_format!r}.")


def _existing_annotation_items(path: Path) -> dict[str, JSONDict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    return {str(item["id"]): item for item in items}


def _find_video(directory: Path, video_id: str) -> Path | None:
    candidates = [
        path
        for path in sorted(directory.glob(f"{video_id}.*"))
        if path.suffix not in {".json", ".part", ".ytdl"} and not path.name.endswith(".info.json")
    ]
    return candidates[0] if candidates else None


def _read_info_json(path: Path) -> JSONDict:
    if not path.exists():
        return {}
    return cast(JSONDict, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JSONDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
