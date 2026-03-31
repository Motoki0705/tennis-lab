"""Download raw tennis videos from configured URLs using ``yt-dlp``.

Usage:
    python -m src.tasks.ball_detection.scripts.download_videos
    python -m src.tasks.ball_detection.scripts.download_videos urls=[https://example.com/video]

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/downlaod.yaml`.
    - Downloads are renamed into `video_<n>.mp4` files and summarized in JSON.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(slots=True)
class SummaryEntry:
    """Track one downloaded URL and its renamed local file."""

    index: int
    url: str
    title: str
    source_filename: str
    file_name: str
    relative_path: str
    absolute_path: str
    file_size_bytes: int
    downloaded_at: str


@dataclass(slots=True)
class DownloadResult:
    """Metadata emitted by one yt-dlp download."""

    title: str
    source_path: Path


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_yt_dlp_command() -> list[str]:
    yt_dlp_binary = shutil.which("yt-dlp")
    if yt_dlp_binary is not None:
        return [yt_dlp_binary]
    return [sys.executable, "-m", "yt_dlp"]


def _read_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            "updated_at": _timestamp(),
            "entries": [],
            "url_to_file": {},
        }

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in summary file: {summary_path}") from exc
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid summary format: {summary_path}")
    data["entries"] = entries
    data["url_to_file"] = data.get("url_to_file", {})
    return data


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary["updated_at"] = _timestamp()
    summary["entries"] = sorted(summary["entries"], key=lambda item: int(item["index"]))
    summary["url_to_file"] = {
        entry["url"]: entry["file_name"] for entry in summary["entries"]
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _find_entry(summary: dict[str, Any], url: str) -> dict[str, Any] | None:
    for entry in summary["entries"]:
        if entry["url"] == url:
            return entry
    return None


def _extract_index(file_name: str) -> int:
    stem = Path(file_name).stem
    prefix = "video_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected video file name: {file_name}")
    return int(stem[len(prefix) :])


def _collect_reserved_indices(output_dir: Path, summary: dict[str, Any]) -> set[int]:
    reserved_indices = set()
    for entry in summary["entries"]:
        reserved_indices.add(int(entry["index"]))
        reserved_indices.add(_extract_index(str(entry["file_name"])))

    for video_path in output_dir.glob("video_*.mp4"):
        reserved_indices.add(_extract_index(video_path.name))
    return reserved_indices


def _allocate_index(output_dir: Path, summary: dict[str, Any]) -> int:
    reserved_indices = _collect_reserved_indices(output_dir, summary)
    next_index = 1
    while next_index in reserved_indices:
        next_index += 1
    return next_index


def _fetch_video_metadata(url: str, timeout_seconds: int) -> dict[str, Any]:
    command = [
        *_resolve_yt_dlp_command(),
        "--dump-single-json",
        "--no-playlist",
        "--skip-download",
        url,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return json.loads(result.stdout)


def _download_with_yt_dlp(
    *,
    url: str,
    temp_dir: Path,
    cfg: DictConfig,
) -> DownloadResult:
    metadata = _fetch_video_metadata(
        url, timeout_seconds=int(cfg.download.timeout_seconds)
    )
    title = str(metadata.get("title") or metadata.get("fulltitle") or url)
    output_template = str(temp_dir / "%(title)s.%(ext)s")
    command = [
        *_resolve_yt_dlp_command(),
        "--no-playlist",
        "--format",
        str(cfg.download.format),
        "--merge-output-format",
        str(cfg.download.merge_output_format),
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
    ]
    remux_video = str(cfg.download.get("remux_video", "") or "").strip()
    if remux_video:
        command.extend(["--remux-video", remux_video])
    command.append(url)

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(cfg.download.timeout_seconds),
    )
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError(f"yt-dlp did not report an output path for URL: {url}")

    source_path = Path(output_lines[-1])
    if not source_path.exists():
        raise FileNotFoundError(f"Downloaded file not found: {source_path}")
    return DownloadResult(title=title, source_path=source_path)


def _upsert_entry(summary: dict[str, Any], entry: SummaryEntry) -> None:
    payload = asdict(entry)
    existing_entry = _find_entry(summary, entry.url)
    if existing_entry is None:
        summary["entries"].append(payload)
        return

    existing_entry.clear()
    existing_entry.update(payload)


def _build_entry(
    *, url: str, target_path: Path, result: DownloadResult, index: int
) -> SummaryEntry:
    return SummaryEntry(
        index=index,
        url=url,
        title=result.title,
        source_filename=result.source_path.name,
        file_name=target_path.name,
        relative_path=target_path.name,
        absolute_path=str(target_path.resolve()),
        file_size_bytes=target_path.stat().st_size,
        downloaded_at=_timestamp(),
    )


def _rename_download(
    *,
    result: DownloadResult,
    output_dir: Path,
    index: int,
    allow_overwrite: bool,
) -> Path:
    target_path = output_dir / f"video_{index}.mp4"
    if target_path.exists():
        if not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target_path}")
        target_path.unlink()

    shutil.move(str(result.source_path), str(target_path))
    return target_path


def _transcode_to_h264(
    *,
    input_path: Path,
    cfg: DictConfig,
) -> Path:
    """Transcode one downloaded video to an H.264 MP4 file."""
    output_path = input_path.with_name(f"{input_path.stem}.h264.mp4")
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        str(cfg.download.get("video_codec", "libx264")),
        "-pix_fmt",
        str(cfg.download.get("pixel_format", "yuv420p")),
        "-preset",
        str(cfg.download.get("preset", "medium")),
        "-crf",
        str(int(cfg.download.get("crf", 18))),
    ]
    if bool(cfg.download.get("drop_audio", True)):
        command.append("-an")
    else:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.append(str(output_path))

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(cfg.download.timeout_seconds),
    )
    if not output_path.exists():
        raise FileNotFoundError(f"Transcoded H.264 file not found: {output_path}")
    input_path.unlink(missing_ok=True)
    return output_path


def download_videos(cfg: DictConfig) -> dict[str, Any]:
    """Download configured URLs into `video_<n>.mp4` files and update the summary."""
    urls = [str(url).strip() for url in cfg.get("urls", []) if str(url).strip()]
    if not urls:
        raise ValueError("No URLs configured. Set `urls` in configs/downlaod.yaml.")

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(to_absolute_path(str(cfg.summary_path)))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _read_summary(summary_path)
    summary["output_dir"] = str(output_dir)
    summary["summary_path"] = str(summary_path)

    downloaded = 0
    skipped = 0

    with tempfile.TemporaryDirectory(
        prefix=".yt-dlp-", dir=output_dir
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for url in urls:
            existing_entry = _find_entry(summary, url)
            if existing_entry is not None:
                expected_path = output_dir / str(existing_entry["file_name"])
                if bool(cfg.download.skip_existing_urls) and expected_path.exists():
                    skipped += 1
                    continue
                index = int(existing_entry["index"])
            else:
                index = _allocate_index(output_dir, summary)

            result = _download_with_yt_dlp(url=url, temp_dir=temp_dir, cfg=cfg)
            result = DownloadResult(
                title=result.title,
                source_path=_transcode_to_h264(input_path=result.source_path, cfg=cfg),
            )
            target_path = _rename_download(
                result=result,
                output_dir=output_dir,
                index=index,
                allow_overwrite=existing_entry is not None,
            )
            entry = _build_entry(
                url=url, target_path=target_path, result=result, index=index
            )
            _upsert_entry(summary, entry)
            _write_summary(summary_path, summary)
            downloaded += 1

    _write_summary(summary_path, summary)
    return {
        "configured_urls": len(urls),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
    }


@hydra.main(config_path="../configs", config_name="downlaod", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for downloading raw videos into the local dataset area."""
    summary = download_videos(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
