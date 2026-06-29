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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import cv2
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra import hydra_main
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
    ensure_dirs([av1_dir, h264_dir, frames_root, annotations_dir, manifests_dir])

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
        h264_video = _transcode_h264(
            av1_video, video_id, h264_dir, workflow_cfg.transcode
        )
        info = _read_info_json(av1_dir / f"{video_id}.info.json")
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
            frames_root / video_id,
            root,
            workflow_cfg.frames,
            info,
        )
        frame_records_by_split.setdefault(split, []).extend(frame_records)

    write_jsonl(manifests_dir / "sources.jsonl", source_records)
    write_jsonl(manifests_dir / "download_manifest.jsonl", download_records)
    _write_annotations(annotations_dir, frame_records_by_split, workflow_cfg.annotation)
    save_json_atomic(
        {
            "schema_name": "court_youtube_split_manifest_v1",
            "counts": {
                split: len(records) for split, records in frame_records_by_split.items()
            },
            "written_at": utc_now_iso(),
        },
        manifests_dir / "split_manifest.json",
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


def _download_av1(
    source: JSONDict, video_id: str, av1_dir: Path, cfg: DictConfig
) -> Path:
    archive = None
    if cfg.get("download_archive") is not None:
        archive = Path(to_absolute_path(str(cfg.download_archive))).resolve()
    return download_youtube_video(
        url=str(source["url"]),
        video_id=video_id,
        output_dir=av1_dir,
        format_selector=str(cfg.strict_format if bool(cfg.require_av1) else cfg.format),
        merge_output_format=str(cfg.merge_output_format),
        enabled=bool(cfg.enabled),
        overwrite=bool(cfg.overwrite),
        js_runtimes=None if cfg.get("js_runtimes") is None else str(cfg.js_runtimes),
        remote_components=(
            None if cfg.get("remote_components") is None else str(cfg.remote_components)
        ),
        download_archive=archive,
        extra_args=[str(value) for value in cfg.get("extra_args", [])],
    )


def _transcode_h264(
    av1_video: Path, video_id: str, h264_dir: Path, cfg: DictConfig
) -> Path:
    return transcode_h264_video(
        source_video=av1_video,
        output_path=h264_dir / f"{video_id}.mp4",
        enabled=bool(cfg.enabled),
        overwrite=bool(cfg.overwrite),
        ffmpeg_binary=str(cfg.ffmpeg_binary),
        encoder=str(cfg.encoder),
        hwaccel=None if cfg.get("hwaccel") is None else str(cfg.hwaccel),
        hwaccel_output_format=(
            None
            if cfg.get("hwaccel_output_format") is None
            else str(cfg.hwaccel_output_format)
        ),
        preset=str(cfg.preset),
        tune=None if cfg.get("tune") is None else str(cfg.tune),
        rate_control=None if cfg.get("rate_control") is None else str(cfg.rate_control),
        cq=None if cfg.get("cq") is None else cfg.cq,
        bitrate=None if cfg.get("bitrate") is None else str(cfg.bitrate),
        maxrate=None if cfg.get("maxrate") is None else str(cfg.maxrate),
        bufsize=None if cfg.get("bufsize") is None else str(cfg.bufsize),
        profile=None if cfg.get("profile") is None else str(cfg.profile),
        pix_fmt=str(cfg.pix_fmt),
        crf=cfg.crf,
    )


def _h264_encoder_args(cfg: DictConfig) -> list[str]:
    """Return FFmpeg arguments for H.264 encoding."""
    return h264_encoder_args(
        encoder=str(cfg.encoder),
        preset=str(cfg.preset),
        tune=None if cfg.get("tune") is None else str(cfg.tune),
        rate_control=None if cfg.get("rate_control") is None else str(cfg.rate_control),
        cq=None if cfg.get("cq") is None else cfg.cq,
        bitrate=None if cfg.get("bitrate") is None else str(cfg.bitrate),
        maxrate=None if cfg.get("maxrate") is None else str(cfg.maxrate),
        bufsize=None if cfg.get("bufsize") is None else str(cfg.bufsize),
        profile=None if cfg.get("profile") is None else str(cfg.profile),
        pix_fmt=str(cfg.pix_fmt),
        crf=cfg.crf,
    )


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
        return read_jsonl(output_dir / "frames.jsonl")

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
    frame_indices = _sample_frame_indices(
        source.get("time_ranges", []), duration, fps, cfg
    )
    if frame_count > 0:
        frame_indices = [
            frame_index for frame_index in frame_indices if frame_index < frame_count
        ]
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
    write_jsonl(output_dir / "frames.jsonl", records)
    print(f"  frames: {len(records)} -> {output_dir}")
    return records


def _sample_frame_indices(
    raw_ranges: Any, duration: float, fps: float, cfg: DictConfig
) -> list[int]:
    return sample_frame_indices_by_time_ranges(
        raw_ranges,
        duration=duration,
        fps=fps,
        sample_mode=str(cfg.sample_mode),
        interval_seconds=float(cfg.interval_seconds),
        target_fps=float(cfg.fps),
        every_n_frames=int(cfg.every_n_frames),
    )


def _sample_step_seconds(cfg: DictConfig, fps: float) -> float:
    return sample_step_seconds(
        sample_mode=str(cfg.sample_mode),
        fps=fps,
        interval_seconds=float(cfg.interval_seconds),
        target_fps=float(cfg.fps),
        every_n_frames=int(cfg.every_n_frames),
    )


def _parse_time_seconds(value: Any) -> float:
    return parse_time_seconds(value)


def _write_annotations(
    annotations_dir: Path,
    frame_records_by_split: dict[str, list[JSONDict]],
    cfg: DictConfig,
) -> None:
    for split in ("train", "val"):
        path = annotations_dir / f"{split}.json"
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
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    return {str(item["id"]): item for item in items}


def _read_info_json(path: Path) -> JSONDict:
    return cast(JSONDict, load_json_if_exists(path, {}))


if __name__ == "__main__":
    raise SystemExit(main())  # type: ignore[call-arg]
