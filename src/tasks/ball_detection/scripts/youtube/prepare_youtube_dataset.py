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

F = TypeVar("F", bound=Callable[..., Any])
JSONDict = dict[str, Any]


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


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
    _ensure_dirs([av1_dir, h264_dir, frames_root, manifests_dir])

    source_records: list[JSONDict] = []
    download_records: list[JSONDict] = []
    frame_counts: dict[str, int] = {"train": 0, "val": 0}
    video_counts: dict[str, int] = {"train": 0, "val": 0}

    for index, raw_source in enumerate(_source_dicts(workflow.sources), start=1):
        video_id = str(raw_source.get("source_id") or f"video_{index:06d}")
        source = {**raw_source, "source_id": video_id}
        split = str(source.get("split") or "train")
        if split not in frame_counts:
            raise ValueError(f"Unsupported source split={split!r}; expected train or val.")
        source_records.append(source)
        video_counts[split] += 1

        print(f"[prepare_ball_youtube_dataset] source={video_id} split={split}")
        source_video = _download_video(source, video_id, av1_dir, workflow.download)
        h264_video = _transcode_h264(source_video, video_id, h264_dir, workflow.transcode)
        info = _read_info_json(av1_dir / f"{video_id}.info.json")
        download_records.append({
            "video_id": video_id,
            "source_url": source["url"],
            "source_title": info.get("title"),
            "source_video": _relative_path(source_video, root),
            "h264_video": _relative_path(h264_video, root),
            "processed_at": _utc_now(),
        })

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

    _write_jsonl(manifests_dir / "sources.jsonl", source_records)
    _write_jsonl(manifests_dir / "download_manifest.jsonl", download_records)
    _write_json_atomic(
        manifests_dir / "split_manifest.json",
        {
            "schema_name": "ball_youtube_split_manifest_v1",
            "video_counts": video_counts,
            "raw_frame_counts": frame_counts,
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


def _download_video(source: JSONDict, video_id: str, output_dir: Path, cfg: DictConfig) -> Path:
    existing = _find_video(output_dir, video_id)
    if existing is not None and (not bool(cfg.enabled) or not bool(cfg.overwrite)):
        print(f"  source video exists: {existing}")
        return existing
    if not bool(cfg.enabled):
        raise FileNotFoundError(f"Download disabled and no source video found for {video_id}.")

    output_template = output_dir / f"{video_id}.%(ext)s"
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
    if cfg.get("js_runtimes") is not None:
        cmd.extend(["--js-runtimes", str(cfg.js_runtimes)])
    if cfg.get("remote_components") is not None:
        cmd.extend(["--remote-components", str(cfg.remote_components)])
    if cfg.get("download_archive") is not None:
        archive = Path(to_absolute_path(str(cfg.download_archive))).resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--download-archive", str(archive)])
    cmd.append("--force-overwrites" if bool(cfg.overwrite) else "--no-overwrites")
    cmd.extend(str(value) for value in cfg.get("extra_args", []))
    _run(cmd)

    downloaded = _find_video(output_dir, video_id)
    if downloaded is None:
        raise FileNotFoundError(f"yt-dlp finished but no video was found for {video_id}.")
    return downloaded


def _transcode_h264(source_video: Path, video_id: str, output_dir: Path, cfg: DictConfig) -> Path:
    output_path = output_dir / f"{video_id}.mp4"
    if output_path.exists() and not bool(cfg.overwrite):
        print(f"  H.264 exists: {output_path}")
        return output_path
    if not bool(cfg.enabled):
        raise FileNotFoundError(f"H.264 transcode disabled and output missing: {output_path}")

    cmd = [str(cfg.ffmpeg_binary), "-y" if bool(cfg.overwrite) else "-n"]
    if cfg.get("hwaccel") is not None:
        cmd.extend(["-hwaccel", str(cfg.hwaccel)])
    if cfg.get("hwaccel_output_format") is not None:
        cmd.extend(["-hwaccel_output_format", str(cfg.hwaccel_output_format)])
    cmd.extend(["-i", str(source_video), "-map", "0:v:0"])
    cmd.extend(_h264_encoder_args(cfg))
    cmd.extend(["-movflags", "+faststart", "-an", str(output_path)])
    _run(cmd)
    return output_path


def _h264_encoder_args(cfg: DictConfig) -> list[str]:
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
    if encoder not in {"h264_nvenc", "avc_nvenc"}:
        raise ValueError(f"Unsupported H.264 encoder: {encoder!r}")
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
        "-b:v",
        "0" if cfg.get("bitrate") is None else str(cfg.bitrate),
    ]
    for option in ("maxrate", "bufsize", "profile"):
        value = cfg.get(option)
        if value is not None:
            flag = "-profile:v" if option == "profile" else f"-{option}"
            args.extend([flag, str(value)])
    return args


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
        return _read_jsonl(manifest_path)
    if manifest_path.exists() and not bool(cfg.overwrite):
        records = _read_jsonl(manifest_path)
        if records:
            print(f"  raw frames exist: {len(records)} -> {raw_dir}")
            return records

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
        records.append({
            "frame_id": frame_id,
            "image_path": _relative_path(output_path, root),
            "video_id": source["source_id"],
            "source_frame_index": frame_index,
            "timestamp_sec": frame_index / fps,
            "fps": fps,
            "width": width,
            "height": height,
            "split": split,
            "source_url": source["url"],
            "source_title": info.get("title"),
        })
        frame_index += 1
    capture.release()
    _write_jsonl(manifest_path, records)
    print(f"  raw frames: {len(records)} -> {raw_dir}")
    return records


def _find_video(directory: Path, video_id: str) -> Path | None:
    candidates = [
        path
        for path in sorted(directory.glob(f"{video_id}.*"))
        if path.suffix not in {".json", ".part", ".ytdl"} and not path.name.endswith(".info.json")
    ]
    return candidates[0] if candidates else None


def _read_info_json(path: Path) -> JSONDict:
    return {} if not path.exists() else cast(JSONDict, json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JSONDict]:
    if not path.exists():
        return []
    return [
        cast(JSONDict, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
