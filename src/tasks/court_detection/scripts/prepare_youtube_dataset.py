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

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import require_config_mapping
from src.tasks.court_detection.configuration import validate_paths_boundary
from src.utils.configuration import PathResolver, PathRole
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import (
    ensure_dirs,
    load_json,
    load_json_if_exists,
    read_jsonl,
    relative_path,
    save_json_atomic,
    utc_now_iso,
    write_jsonl,
)
from src.utils.schema.court import COURT_KP_NAMES
from src.utils.video.sampling import (
    parse_time_seconds,
    sample_frame_indices_by_time_ranges,
    sample_step_seconds,
)
from src.utils.video.youtube import (
    download_youtube_video,
    h264_encoder_args,
    transcode_h264_video,
)

JSONDict = dict[str, Any]
_BOUNDARY = "court_detection.prepare_youtube_dataset"


@dataclass(frozen=True, slots=True)
class YoutubeDatasetPaths:
    """Fully resolved DATA-role paths consumed by the preparation workflow."""

    resolver: PathResolver
    root: Path
    av1_dir: Path
    h264_dir: Path
    frames_root: Path
    annotations_dir: Path
    manifests_dir: Path
    download_archive: Path | None


def _validate_exact(value: Any, expected: set[str], *, path: str) -> None:
    raw = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, DictConfig)
        else value
    )
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"{path} requires exactly {sorted(expected)}.")


def _typed(
    mapping: Mapping[str, object],
    key: str,
    expected: type[object] | tuple[type[object], ...],
    *,
    path: str,
) -> None:
    accepted = expected if isinstance(expected, tuple) else (expected,)
    if type(mapping[key]) not in accepted:
        names = " | ".join(candidate.__name__ for candidate in accepted)
        raise TypeError(
            f"{path}.{key}: expected {names}, got {type(mapping[key]).__name__}."
        )


def _runtime(cfg: DictConfig) -> YoutubeDatasetPaths:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"workflow"})
    workflow = require_config_mapping(root, "workflow", path="configuration")
    _validate_exact(
        workflow,
        {
            "root",
            "sources",
            "paths",
            "split",
            "download",
            "transcode",
            "frames",
            "annotation",
        },
        path="workflow",
    )
    _validate_exact(
        workflow["paths"],
        {
            "videos_dir",
            "av1_dir",
            "h264_dir",
            "frames_dir",
            "annotations_dir",
            "manifests_dir",
        },
        path="workflow.paths",
    )
    _validate_exact(workflow["split"], {"default"}, path="workflow.split")
    _validate_exact(
        workflow["download"],
        {
            "enabled",
            "require_av1",
            "strict_format",
            "format",
            "merge_output_format",
            "js_runtimes",
            "remote_components",
            "download_archive",
            "overwrite",
            "extra_args",
        },
        path="workflow.download",
    )
    _validate_exact(
        workflow["transcode"],
        {
            "enabled",
            "ffmpeg_binary",
            "encoder",
            "hwaccel",
            "hwaccel_output_format",
            "preset",
            "tune",
            "rate_control",
            "cq",
            "bitrate",
            "maxrate",
            "bufsize",
            "profile",
            "pix_fmt",
            "crf",
            "overwrite",
        },
        path="workflow.transcode",
    )
    _validate_exact(
        workflow["frames"],
        {
            "enabled",
            "sample_mode",
            "interval_seconds",
            "fps",
            "every_n_frames",
            "output_ext",
            "jpeg_quality",
            "max_frames_per_video",
            "overwrite",
        },
        path="workflow.frames",
    )
    _validate_exact(
        workflow["annotation"],
        {"schema_name", "keypoint_format", "merge_existing", "overwrite"},
        path="workflow.annotation",
    )
    _typed(workflow, "root", str, path="workflow")
    split_config = cast("Mapping[str, object]", workflow["split"])
    _typed(split_config, "default", str, path="workflow.split")
    sources = workflow["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("workflow.sources must be a non-empty list.")
    source_ids: set[str] = set()
    root_raw = cast("str", workflow["root"])
    if not root_raw:
        raise ValueError("workflow.root must not be empty.")
    for index, source in enumerate(sources):
        _validate_exact(
            source,
            {"source_id", "url", "split", "time_ranges"},
            path=f"workflow.sources[{index}]",
        )
        source_mapping = cast("Mapping[str, object]", source)
        for key in ("source_id", "url", "split"):
            _typed(source_mapping, key, str, path=f"workflow.sources[{index}]")
        _typed(source_mapping, "time_ranges", list, path=f"workflow.sources[{index}]")
        source_id = cast("str", source_mapping["source_id"])
        if not source_id or source_id in source_ids:
            raise ValueError("workflow source IDs must be non-empty and unique.")
        source_ids.add(source_id)
        resolver.resolve(PathRole.DATA, root_raw, source_id)
        if not cast("str", source_mapping["url"]):
            raise ValueError(f"workflow.sources[{index}].url must not be empty.")
        if source_mapping["split"] not in {"train", "val"}:
            raise ValueError(f"workflow.sources[{index}].split must be train or val.")
        for range_index, time_range in enumerate(
            cast("list[object]", source_mapping["time_ranges"])
        ):
            range_path = f"workflow.sources[{index}].time_ranges[{range_index}]"
            _validate_exact(time_range, {"start", "end"}, path=range_path)
            range_mapping = cast("Mapping[str, object]", time_range)
            for key in ("start", "end"):
                _typed(range_mapping, key, (str, int, float), path=range_path)
                if isinstance(range_mapping[key], str) and not range_mapping[key]:
                    raise ValueError(f"{range_path}.{key} must not be empty.")
    paths = cast("Mapping[str, object]", workflow["paths"])
    for key in paths:
        _typed(paths, key, str, path="workflow.paths")
        if not cast("str", paths[key]):
            raise ValueError(f"workflow.paths.{key} must not be empty.")
    av1_dir = resolver.resolve(
        PathRole.DATA,
        root_raw,
        cast("str", paths["videos_dir"]),
        cast("str", paths["av1_dir"]),
    )
    h264_dir = resolver.resolve(
        PathRole.DATA,
        root_raw,
        cast("str", paths["videos_dir"]),
        cast("str", paths["h264_dir"]),
    )
    frames_root = resolver.resolve(
        PathRole.DATA, root_raw, cast("str", paths["frames_dir"])
    )
    annotations_dir = resolver.resolve(
        PathRole.DATA, root_raw, cast("str", paths["annotations_dir"])
    )
    manifests_dir = resolver.resolve(
        PathRole.DATA, root_raw, cast("str", paths["manifests_dir"])
    )
    download = cast("Mapping[str, object]", workflow["download"])
    for key in ("enabled", "require_av1", "overwrite"):
        _typed(download, key, bool, path="workflow.download")
    for key in ("strict_format", "format", "merge_output_format"):
        _typed(download, key, str, path="workflow.download")
    for key in ("js_runtimes", "remote_components", "download_archive"):
        _typed(download, key, (str, type(None)), path="workflow.download")
    _typed(download, "extra_args", list, path="workflow.download")
    if any(
        type(value) is not str for value in cast("list[object]", download["extra_args"])
    ):
        raise TypeError("workflow.download.extra_args must contain only strings.")
    archive_fragment = cast("str | None", download["download_archive"])
    archive: Path | None = None
    if archive_fragment is not None:
        if not archive_fragment:
            raise ValueError(
                "workflow.download.download_archive must be null or non-empty."
            )
        archive = resolver.resolve(PathRole.DATA, root_raw, archive_fragment)
    transcode = cast("Mapping[str, object]", workflow["transcode"])
    for key in ("enabled", "overwrite"):
        _typed(transcode, key, bool, path="workflow.transcode")
    for key in ("ffmpeg_binary", "encoder", "preset", "pix_fmt"):
        _typed(transcode, key, str, path="workflow.transcode")
    for key in (
        "hwaccel",
        "hwaccel_output_format",
        "tune",
        "rate_control",
        "bitrate",
        "maxrate",
        "bufsize",
        "profile",
    ):
        _typed(transcode, key, (str, type(None)), path="workflow.transcode")
    for key in ("cq", "crf"):
        _typed(transcode, key, (int, type(None)), path="workflow.transcode")
    for key in ("ffmpeg_binary", "encoder", "preset", "pix_fmt"):
        if not cast("str", transcode[key]):
            raise ValueError(f"workflow.transcode.{key} must not be empty.")
    for key in (
        "hwaccel",
        "hwaccel_output_format",
        "tune",
        "rate_control",
        "bitrate",
        "maxrate",
        "bufsize",
        "profile",
    ):
        if transcode[key] == "":
            raise ValueError(
                f"workflow.transcode.{key} must be null or a non-empty string."
            )
    for key in ("cq", "crf"):
        value = cast("int | None", transcode[key])
        if value is not None and value < 0:
            raise ValueError(f"workflow.transcode.{key} must be non-negative.")
    if transcode["encoder"] == "libx264" and transcode["crf"] is None:
        raise ValueError(
            "workflow.transcode.crf must be explicit when encoder='libx264'."
        )
    try:
        _h264_encoder_args(transcode)
    except ValueError as error:
        raise ValueError(f"Invalid workflow.transcode: {error}") from error
    frames = cast("Mapping[str, object]", workflow["frames"])
    for key in ("enabled", "overwrite"):
        _typed(frames, key, bool, path="workflow.frames")
    for key in ("sample_mode", "output_ext"):
        _typed(frames, key, str, path="workflow.frames")
    for key in ("interval_seconds", "fps"):
        _typed(frames, key, (float, int), path="workflow.frames")
    for key in ("every_n_frames", "jpeg_quality"):
        _typed(frames, key, int, path="workflow.frames")
    _typed(frames, "max_frames_per_video", (int, type(None)), path="workflow.frames")
    if frames["sample_mode"] not in {"interval_seconds", "fps", "every_n_frames"}:
        raise ValueError("workflow.frames.sample_mode is invalid.")
    if float(cast("float | int", frames["interval_seconds"])) <= 0:
        raise ValueError("workflow.frames.interval_seconds must be positive.")
    if float(cast("float | int", frames["fps"])) <= 0:
        raise ValueError("workflow.frames.fps must be positive.")
    if cast("int", frames["every_n_frames"]) <= 0:
        raise ValueError("workflow.frames.every_n_frames must be positive.")
    if not cast("str", frames["output_ext"]) or "/" in cast(
        "str", frames["output_ext"]
    ):
        raise ValueError("workflow.frames.output_ext must be a file extension.")
    if not 1 <= cast("int", frames["jpeg_quality"]) <= 100:
        raise ValueError("workflow.frames.jpeg_quality must be in [1, 100].")
    max_frames = cast("int | None", frames["max_frames_per_video"])
    if max_frames is not None and max_frames <= 0:
        raise ValueError("workflow.frames.max_frames_per_video must be positive.")
    annotation = cast("Mapping[str, object]", workflow["annotation"])
    for key in ("schema_name", "keypoint_format"):
        _typed(annotation, key, str, path="workflow.annotation")
    for key in ("merge_existing", "overwrite"):
        _typed(annotation, key, bool, path="workflow.annotation")
    if annotation["schema_name"] != "court_youtube_keypoints_v2":
        raise ValueError("workflow.annotation.schema_name is invalid.")
    if annotation["keypoint_format"] not in {"kp15", "kp20"}:
        raise ValueError("workflow.annotation.keypoint_format must be kp15 or kp20.")
    return YoutubeDatasetPaths(
        resolver=resolver,
        root=resolver.resolve(PathRole.DATA, root_raw),
        av1_dir=av1_dir,
        h264_dir=h264_dir,
        frames_root=frames_root,
        annotations_dir=annotations_dir,
        manifests_dir=manifests_dir,
        download_archive=archive,
    )


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="prepare_youtube_dataset",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    workflow_cfg = cfg.workflow
    runtime_paths = _runtime(cfg)
    root: Path = runtime_paths.root
    av1_dir: Path = runtime_paths.av1_dir
    h264_dir: Path = runtime_paths.h264_dir
    frames_root: Path = runtime_paths.frames_root
    annotations_dir: Path = runtime_paths.annotations_dir
    manifests_dir: Path = runtime_paths.manifests_dir
    ensure_dirs([av1_dir, h264_dir, frames_root, annotations_dir, manifests_dir])

    sources = _source_dicts(workflow_cfg.sources)
    frame_records_by_split: dict[str, list[JSONDict]] = {"train": [], "val": []}
    source_records: list[JSONDict] = []
    download_records: list[JSONDict] = []

    for source in sources:
        video_id = str(source["source_id"])
        split = str(source["split"])
        source_records.append(source)

        print(f"[prepare_youtube_dataset] source={video_id} split={split}")
        av1_video = _download_av1(
            source,
            video_id,
            av1_dir,
            workflow_cfg.download,
            download_archive=runtime_paths.download_archive,
        )
        h264_video = _transcode_h264(
            av1_video,
            video_id,
            h264_dir,
            workflow_cfg.transcode,
            resolver=runtime_paths.resolver,
        )
        info = _read_info_json(
            runtime_paths.resolver.resolve_beneath(
                PathRole.DATA, av1_dir, f"{video_id}.info.json"
            )
        )
        download_records.append(
            {
                "video_id": video_id,
                "source_url": source["url"],
                "source_title": info.get("title"),
                "av1_video": relative_path(av1_video, root),
                "h264_video": relative_path(h264_video, root),
                "processed_at": utc_now_iso(),
            }
        )

        frame_records = _extract_frames(
            source,
            split,
            h264_video,
            runtime_paths.resolver.resolve_beneath(
                PathRole.DATA, frames_root, video_id
            ),
            root,
            workflow_cfg.frames,
            info,
            resolver=runtime_paths.resolver,
        )
        frame_records_by_split[split].extend(frame_records)

    write_jsonl(
        runtime_paths.resolver.resolve_beneath(
            PathRole.DATA, manifests_dir, "sources.jsonl"
        ),
        source_records,
    )
    write_jsonl(
        runtime_paths.resolver.resolve_beneath(
            PathRole.DATA, manifests_dir, "download_manifest.jsonl"
        ),
        download_records,
    )
    _write_annotations(
        annotations_dir,
        frame_records_by_split,
        workflow_cfg.annotation,
        resolver=runtime_paths.resolver,
    )
    save_json_atomic(
        {
            "schema_name": "court_youtube_split_manifest_v1",
            "counts": {
                split: len(records) for split, records in frame_records_by_split.items()
            },
            "written_at": utc_now_iso(),
        },
        runtime_paths.resolver.resolve_beneath(
            PathRole.DATA, manifests_dir, "split_manifest.json"
        ),
    )
    return 0


def _source_dicts(raw_sources: Iterable[Any]) -> list[JSONDict]:
    sources: list[JSONDict] = []
    for source in raw_sources:
        source_dict = cast(JSONDict, OmegaConf.to_container(source, resolve=True))
        if not source_dict["url"]:
            raise ValueError("Each workflow.sources entry must define a non-empty url.")
        sources.append(source_dict)
    return sources


def _download_av1(
    source: JSONDict,
    video_id: str,
    av1_dir: Path,
    cfg: DictConfig,
    *,
    download_archive: Path | None,
) -> Path:
    downloaded: Path = download_youtube_video(
        url=str(source["url"]),
        video_id=video_id,
        output_dir=av1_dir,
        format_selector=str(cfg.strict_format if bool(cfg.require_av1) else cfg.format),
        merge_output_format=str(cfg.merge_output_format),
        enabled=bool(cfg.enabled),
        overwrite=bool(cfg.overwrite),
        js_runtimes=None if cfg.js_runtimes is None else str(cfg.js_runtimes),
        remote_components=(
            None if cfg.remote_components is None else str(cfg.remote_components)
        ),
        download_archive=download_archive,
        extra_args=[str(value) for value in cfg.extra_args],
    )
    return downloaded


def _transcode_h264(
    av1_video: Path,
    video_id: str,
    h264_dir: Path,
    cfg: DictConfig,
    *,
    resolver: PathResolver,
) -> Path:
    transcoded: Path = transcode_h264_video(
        source_video=resolver.validate(PathRole.DATA, av1_video),
        output_path=resolver.resolve_beneath(
            PathRole.DATA, h264_dir, f"{video_id}.mp4"
        ),
        enabled=bool(cfg.enabled),
        overwrite=bool(cfg.overwrite),
        ffmpeg_binary=str(cfg.ffmpeg_binary),
        encoder=str(cfg.encoder),
        hwaccel=None if cfg.hwaccel is None else str(cfg.hwaccel),
        hwaccel_output_format=(
            None
            if cfg.hwaccel_output_format is None
            else str(cfg.hwaccel_output_format)
        ),
        preset=str(cfg.preset),
        tune=None if cfg.tune is None else str(cfg.tune),
        rate_control=None if cfg.rate_control is None else str(cfg.rate_control),
        cq=None if cfg.cq is None else cfg.cq,
        bitrate=None if cfg.bitrate is None else str(cfg.bitrate),
        maxrate=None if cfg.maxrate is None else str(cfg.maxrate),
        bufsize=None if cfg.bufsize is None else str(cfg.bufsize),
        profile=None if cfg.profile is None else str(cfg.profile),
        pix_fmt=str(cfg.pix_fmt),
        crf=cfg.crf,
    )
    return transcoded


def _h264_encoder_args(cfg: Mapping[str, object]) -> list[str]:
    """Return FFmpeg arguments for H.264 encoding."""
    arguments: list[str] = h264_encoder_args(
        encoder=str(cfg["encoder"]),
        preset=str(cfg["preset"]),
        tune=None if cfg["tune"] is None else str(cfg["tune"]),
        rate_control=(
            None if cfg["rate_control"] is None else str(cfg["rate_control"])
        ),
        cq=cast("int | float | None", cfg["cq"]),
        bitrate=None if cfg["bitrate"] is None else str(cfg["bitrate"]),
        maxrate=None if cfg["maxrate"] is None else str(cfg["maxrate"]),
        bufsize=None if cfg["bufsize"] is None else str(cfg["bufsize"]),
        profile=None if cfg["profile"] is None else str(cfg["profile"]),
        pix_fmt=str(cfg["pix_fmt"]),
        crf=cast("int | float", cfg["crf"]),
    )
    return arguments


def _extract_frames(
    source: JSONDict,
    split: str,
    video_path: Path,
    output_dir: Path,
    root: Path,
    cfg: DictConfig,
    info: JSONDict,
    *,
    resolver: PathResolver,
) -> list[JSONDict]:
    video_path = resolver.validate(PathRole.DATA, video_path)
    output_dir = resolver.validate(PathRole.DATA, output_dir)
    frame_manifest = resolver.resolve_beneath(
        PathRole.DATA, output_dir, "frames.jsonl"
    )
    if not bool(cfg.enabled):
        cached_records: list[JSONDict] = read_jsonl(frame_manifest)
        return cached_records

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open H.264 video for frame extraction: {video_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by OpenCV for {video_path}: {fps}")

    duration = (
        frame_count / fps if frame_count > 0 else float(info.get("duration") or 0.0)
    )
    frame_indices = _sample_frame_indices(source["time_ranges"], duration, fps, cfg)
    if frame_count > 0:
        frame_indices = [
            frame_index for frame_index in frame_indices if frame_index < frame_count
        ]
    max_frames = cfg.max_frames_per_video
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]

    records: list[JSONDict] = []
    for frame_index in frame_indices:
        image_id = f"{source['source_id'].replace('video_', 'yt_')}_f{frame_index:08d}"
        output_path = resolver.resolve_beneath(
            PathRole.DATA, output_dir, f"{image_id}.{cfg.output_ext}"
        )
        if not output_path.exists() or bool(cfg.overwrite):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                print(f"  SKIP frame read failed: {video_path} frame={frame_index}")
                continue
            write_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)]
            if not cv2.imwrite(str(output_path), frame, write_params):
                raise RuntimeError(f"Failed to write frame: {output_path}")

        records.append(
            {
                "id": image_id,
                "image_path": relative_path(output_path, root),
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
                "processed_at": utc_now_iso(),
            }
        )

    capture.release()
    write_jsonl(frame_manifest, records)
    print(f"  frames: {len(records)} -> {output_dir}")
    return records


def _sample_frame_indices(
    raw_ranges: Any, duration: float, fps: float, cfg: DictConfig
) -> list[int]:
    indices: list[int] = sample_frame_indices_by_time_ranges(
        raw_ranges,
        duration=duration,
        fps=fps,
        sample_mode=str(cfg.sample_mode),
        interval_seconds=float(cfg.interval_seconds),
        target_fps=float(cfg.fps),
        every_n_frames=int(cfg.every_n_frames),
    )
    return indices


def _sample_step_seconds(cfg: DictConfig, fps: float) -> float:
    return float(
        sample_step_seconds(
            sample_mode=str(cfg.sample_mode),
            fps=fps,
            interval_seconds=float(cfg.interval_seconds),
            target_fps=float(cfg.fps),
            every_n_frames=int(cfg.every_n_frames),
        )
    )


def _parse_time_seconds(value: Any) -> float:
    return float(parse_time_seconds(value))


def _write_annotations(
    annotations_dir: Path,
    frame_records_by_split: dict[str, list[JSONDict]],
    cfg: DictConfig,
    *,
    resolver: PathResolver,
) -> None:
    annotations_dir = resolver.validate(PathRole.DATA, annotations_dir)
    for split in ("train", "val"):
        path = resolver.resolve_beneath(
            PathRole.DATA, annotations_dir, f"{split}.json"
        )
        existing_by_id = (
            _existing_annotation_items(path) if bool(cfg.merge_existing) else {}
        )
        items: list[JSONDict] = []
        generated_ids: set[str] = set()
        for frame in frame_records_by_split.get(split, []):
            image_id = str(frame["id"])
            generated_ids.add(image_id)
            existing = existing_by_id.get(image_id)
            if existing is None:
                items.append(_initial_annotation_item(frame, split, cfg))
            else:
                items.append(
                    _normalize_youtube_annotation_item(existing, frame, split, cfg)
                )
        if bool(cfg.merge_existing):
            items.extend(
                item
                for image_id, item in existing_by_id.items()
                if image_id not in generated_ids
            )
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
            raise FileExistsError(
                f"Annotation file exists. Set overwrite=true or merge_existing=true: {path}"
            )
        save_json_atomic(payload, path)
        print(f"[prepare_youtube_dataset] annotations {split}: {len(items)} -> {path}")


def _initial_annotation_item(frame: JSONDict, split: str, cfg: DictConfig) -> JSONDict:
    keypoint_format = str(cfg.keypoint_format)
    labeled_indices = _labeled_keypoint_indices(keypoint_format)
    source: JSONDict = {
        "type": "youtube",
        "video_id": frame["video_id"],
        "source_url": frame["source_url"],
        "source_title": frame.get("source_title"),
        "source_frame_index": frame["source_frame_index"],
        "timestamp_sec": frame["timestamp_sec"],
    }
    return {
        "id": frame["id"],
        "image_path": frame["image_path"],
        "width": frame["width"],
        "height": frame["height"],
        "split": split,
        "annotation_status": "pending",
        "keypoint_format": keypoint_format,
        "labeled_keypoint_indices": labeled_indices,
        "is_yastrebksv_kp15": False,
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


def _normalize_youtube_annotation_item(
    existing: JSONDict,
    frame: JSONDict,
    split: str,
    cfg: DictConfig,
) -> JSONDict:
    """Normalize an existing YouTube item without discarding annotations."""
    output = dict(existing)
    keypoint_format = str(cfg.keypoint_format)
    output.update(
        {
            "id": frame["id"],
            "image_path": frame["image_path"],
            "width": frame["width"],
            "height": frame["height"],
            "split": split,
            "keypoint_format": keypoint_format,
            "labeled_keypoint_indices": _labeled_keypoint_indices(keypoint_format),
            "is_yastrebksv_kp15": False,
        }
    )
    source = dict(output.get("source", {}))
    for key in ("dataset", "keypoint_format", "labeled_keypoint_indices"):
        source.pop(key, None)
    source.update(
        {
            "type": "youtube",
            "video_id": frame["video_id"],
            "source_url": frame["source_url"],
            "source_title": frame.get("source_title"),
            "source_frame_index": frame["source_frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
        }
    )
    output["source"] = source
    if output.get("annotation_status") == "completed" and not _named_keypoints_complete(
        output.get("keypoints"),
        _labeled_keypoint_indices(keypoint_format),
    ):
        output["annotation_status"] = "pending"
    return output


def _named_keypoints_complete(keypoints: Any, required_indices: list[int]) -> bool:
    """Return whether all required named keypoints have valid visibility data."""
    if not isinstance(keypoints, list) or len(keypoints) < len(COURT_KP_NAMES):
        return False
    points_by_index = {
        int(point["index"]): point
        for point in keypoints
        if isinstance(point, dict) and "index" in point
    }
    for index in required_indices:
        point = points_by_index.get(index)
        if point is None:
            return False
        visibility = int(point.get("visibility", 0))
        if visibility == 3:
            continue
        if visibility not in (1, 2):
            return False
        x = point.get("x")
        y = point.get("y")
        if (
            x is None
            or y is None
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            return False
    return True


def _labeled_keypoint_indices(keypoint_format: str) -> list[int]:
    if keypoint_format == "kp15":
        return list(range(15))
    if keypoint_format == "kp20":
        return list(range(20))
    raise ValueError(f"Unsupported keypoint_format={keypoint_format!r}.")


def _existing_annotation_items(path: Path) -> dict[str, JSONDict]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(
            f"Existing annotation must use the metadata-wrapped format: {path}"
        )
    items = payload["items"]
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"Existing annotation items must be objects: {path}")
    return {str(item["id"]): item for item in items}


def _read_info_json(path: Path) -> JSONDict:
    return cast(JSONDict, load_json_if_exists(path, {}))


if __name__ == "__main__":
    raise SystemExit(main())
