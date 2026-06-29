"""Download YouTube videos and extract continuous raw frames.

Usage:
    python -m src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset
    python -m src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset workflow.sources.0.url=https://www.youtube.com/watch?v=...
    python -m src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset workflow.download.enabled=false

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/prepare_youtube_dataset.yaml`.
    - Videos are downloaded, transcoded to H.264, and extracted under `frames/video_*/raw`.
    - Candidate selection and prediction are handled by
      `scripts.youtube.clip_and_predict_youtube_dataset`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import cv2
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra import hydra_main
from src.utils.io import (
    ensure_dirs,
    load_json_if_exists,
    read_jsonl,
    relative_path,
    save_json_atomic,
    utc_now_iso,
    write_jsonl,
)
from src.utils.video.youtube import (
    download_youtube_video,
    h264_encoder_args,
    transcode_h264_video,
)

JSONDict = dict[str, Any]


@hydra_main(
    config_path="../../configs",
    config_name="prepare_youtube_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    workflow = cfg.workflow
    root = Path(to_absolute_path(str(workflow.root))).resolve()
    paths = workflow.paths
    av1_dir = root / str(paths.videos_dir) / str(paths.av1_dir)
    h264_dir = root / str(paths.videos_dir) / str(paths.h264_dir)
    frames_root = root / str(paths.frames_dir)
    manifests_dir = root / str(paths.manifests_dir)
    ensure_dirs([av1_dir, h264_dir, frames_root, manifests_dir])

    source_records: list[JSONDict] = []
    download_records: list[JSONDict] = []
    frame_counts: dict[str, int] = {"train": 0, "val": 0}
    video_counts: dict[str, int] = {"train": 0, "val": 0}

    for index, raw_source in enumerate(_source_dicts(workflow.sources), start=1):
        video_id = str(raw_source.get("source_id") or f"video_{index:06d}")
        source = {**raw_source, "source_id": video_id}
        split = str(source.get("split") or "train")
        if split not in frame_counts:
            raise ValueError(
                f"Unsupported source split={split!r}; expected train or val."
            )
        source_records.append(source)
        video_counts[split] += 1

        print(f"[prepare_ball_youtube_dataset] source={video_id} split={split}")
        source_video = _download_video(source, video_id, av1_dir, workflow.download)
        h264_video = _transcode_h264(
            source_video, video_id, h264_dir, workflow.transcode
        )
        info = _read_info_json(av1_dir / f"{video_id}.info.json")
        download_records.append(
            {
                "video_id": video_id,
                "source_url": source["url"],
                "source_title": info.get("title"),
                "source_video": relative_path(source_video, root),
                "h264_video": relative_path(h264_video, root),
                "processed_at": utc_now_iso(),
            }
        )

        raw_dir = frames_root / video_id / str(paths.raw_dir)
        records = _extract_continuous_frames(
            source=source,
            split=split,
            video_path=h264_video,
            raw_dir=raw_dir,
            root=root,
            cfg=workflow.frames,
            info=info,
        )
        frame_counts[split] += len(records)

    write_jsonl(manifests_dir / "sources.jsonl", source_records)
    write_jsonl(manifests_dir / "download_manifest.jsonl", download_records)
    save_json_atomic(
        {
            "schema_name": "ball_youtube_split_manifest_v1",
            "video_counts": video_counts,
            "raw_frame_counts": frame_counts,
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


def _download_video(
    source: JSONDict, video_id: str, output_dir: Path, cfg: DictConfig
) -> Path:
    archive = None
    if cfg.get("download_archive") is not None:
        archive = Path(to_absolute_path(str(cfg.download_archive))).resolve()
    return download_youtube_video(
        url=str(source["url"]),
        video_id=video_id,
        output_dir=output_dir,
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
    source_video: Path, video_id: str, output_dir: Path, cfg: DictConfig
) -> Path:
    return transcode_h264_video(
        source_video=source_video,
        output_path=output_dir / f"{video_id}.mp4",
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


def _extract_continuous_frames(
    *,
    source: JSONDict,
    split: str,
    video_path: Path,
    raw_dir: Path,
    root: Path,
    cfg: DictConfig,
    info: JSONDict,
) -> list[JSONDict]:
    manifest_path = raw_dir / "frames.jsonl"
    if not bool(cfg.enabled):
        return read_jsonl(manifest_path)
    if manifest_path.exists() and not bool(cfg.overwrite):
        existing_records = read_jsonl(manifest_path)
        if existing_records:
            print(f"  raw frames exist: {len(existing_records)} -> {raw_dir}")
            return existing_records

    raw_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open H.264 video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported for {video_path}: {fps}")

    max_frames_raw = cfg.get("max_frames_per_video")
    max_frames = None if max_frames_raw is None else int(max_frames_raw)
    records: list[JSONDict] = []
    frame_index = 0
    while max_frames is None or frame_index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id = f"{source['source_id']}_f{frame_index:08d}"
        output_path = raw_dir / f"frame_{frame_index:08d}.{cfg.output_ext}"
        if not output_path.exists() or bool(cfg.overwrite):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)]
            if not cv2.imwrite(str(output_path), frame, params):
                raise RuntimeError(f"Failed to write raw frame: {output_path}")
        records.append(
            {
                "frame_id": frame_id,
                "image_path": relative_path(output_path, root),
                "video_id": source["source_id"],
                "source_frame_index": frame_index,
                "timestamp_sec": frame_index / fps,
                "fps": fps,
                "width": width,
                "height": height,
                "split": split,
                "source_url": source["url"],
                "source_title": info.get("title"),
            }
        )
        frame_index += 1
    capture.release()
    write_jsonl(manifest_path, records)
    print(f"  raw frames: {len(records)} -> {raw_dir}")
    return records


def _read_info_json(path: Path) -> JSONDict:
    return cast(JSONDict, load_json_if_exists(path, {}))


if __name__ == "__main__":
    raise SystemExit(main())  # type: ignore[call-arg]
