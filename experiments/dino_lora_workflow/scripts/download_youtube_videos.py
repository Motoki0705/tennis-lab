"""Overview:
Download YouTube videos listed in a plain text URL file for the DINO LoRA workflow.

Usage:
    .venv/bin/python experiments/dino_lora_workflow/scripts/download_youtube_videos.py
    .venv/bin/python experiments/dino_lora_workflow/scripts/download_youtube_videos.py dry_run=true urls_file=data/dino_workflow/sources/youtube/urls.txt

Notes:
    - Hydra loads configuration from `experiments/dino_lora_workflow/configs/download_youtube_videos.yaml`.
    - The URL file is plain text with one URL per line; blank lines and lines beginning with `#` are ignored.
    - Videos are stored as `video_000001.mp4`, `video_000002.mp4`, and so on under `data/youtube/videos/av1/`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class DownloadResult:
    """Metadata returned from one yt-dlp download."""

    title: str
    source_path: Path
    source_filename: str


@dataclass(slots=True)
class ManifestEntry:
    """One downloaded video entry stored in `manifest.json`."""

    index: int
    url: str
    title: str
    source_filename: str
    file_name: str
    relative_path: str
    absolute_path: str
    file_size_bytes: int
    downloaded_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_urls(urls_file: Path) -> list[str]:
    if not urls_file.exists():
        raise FileNotFoundError(f"URL file not found: {urls_file}")

    urls: list[str] = []
    with urls_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            urls.append(stripped)

    if not urls:
        raise ValueError(f"No URLs found in {urls_file}")
    return urls


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "output_dir": str(manifest_path.parent),
            "urls_file": "",
            "entries": [],
            "url_to_file": {},
            "updated_at": now_iso(),
        }

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest JSON: {manifest_path}") from exc

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid manifest format, entries must be a list: {manifest_path}")
    data["entries"] = entries
    data["url_to_file"] = data.get("url_to_file", {})
    return data


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = now_iso()
    manifest["entries"] = sorted(manifest["entries"], key=lambda item: int(item["index"]))
    manifest["url_to_file"] = {
        str(entry["url"]): str(entry["file_name"]) for entry in manifest["entries"]
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def find_entry(manifest: dict[str, Any], url: str) -> dict[str, Any] | None:
    for entry in manifest.get("entries", []):
        if str(entry.get("url")) == url:
            return entry
    return None


def extract_video_index(file_name: str) -> int:
    stem = Path(file_name).stem
    prefix = "video_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected video file name: {file_name}")
    return int(stem[len(prefix) :])


def collect_reserved_indices(output_dir: Path, manifest: dict[str, Any]) -> set[int]:
    reserved_indices: set[int] = set()
    for entry in manifest.get("entries", []):
        if "index" in entry:
            reserved_indices.add(int(entry["index"]))
        if "file_name" in entry:
            reserved_indices.add(extract_video_index(str(entry["file_name"])))
    for video_path in output_dir.glob("video_*.mp4"):
        reserved_indices.add(extract_video_index(video_path.name))
    return reserved_indices


def allocate_index(reserved_indices: set[int]) -> int:
    index = 1
    while index in reserved_indices:
        index += 1
    reserved_indices.add(index)
    return index


def resolve_yt_dlp_command(cfg: DictConfig) -> list[str]:
    configured_binary = str(cfg.download.get("yt_dlp_binary", "auto") or "auto")
    if configured_binary != "auto":
        return [configured_binary]

    yt_dlp_binary = shutil.which("yt-dlp")
    if yt_dlp_binary is not None:
        return [yt_dlp_binary]

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python), "-m", "yt_dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def resolve_ffmpeg_command(cfg: DictConfig) -> str:
    configured_binary = str(cfg.h264.get("ffmpeg_binary", "ffmpeg") or "ffmpeg")
    if configured_binary != "ffmpeg":
        return configured_binary
    return shutil.which("ffmpeg") or "ffmpeg"


def fetch_metadata(url: str, cfg: DictConfig) -> dict[str, Any]:
    command = [
        *resolve_yt_dlp_command(cfg),
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
        timeout=int(cfg.download.timeout_seconds),
    )
    return json.loads(result.stdout)


def download_with_yt_dlp(url: str, temp_dir: Path, cfg: DictConfig) -> DownloadResult:
    metadata = fetch_metadata(url, cfg)
    title = str(metadata.get("title") or metadata.get("fulltitle") or url)
    output_template = str(temp_dir / "%(title).200B.%(ext)s")
    command = [
        *resolve_yt_dlp_command(cfg),
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

    extra_args = cfg.download.get("extra_args", [])
    if extra_args:
        command.extend([str(item) for item in extra_args])
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

    source_path = Path(output_lines[-1]).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Downloaded file not found: {source_path}")
    return DownloadResult(
        title=title,
        source_path=source_path,
        source_filename=source_path.name,
    )


def transcode_to_h264(result: DownloadResult, cfg: DictConfig) -> DownloadResult:
    input_path = result.source_path
    output_path = input_path.with_name(f"{input_path.stem}.h264.mp4")
    command = [
        resolve_ffmpeg_command(cfg),
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        str(cfg.h264.video_codec),
        "-preset",
        str(cfg.h264.preset),
        "-crf",
        str(int(cfg.h264.crf)),
        "-pix_fmt",
        str(cfg.h264.pixel_format),
        "-movflags",
        "+faststart",
    ]
    if bool(cfg.h264.drop_audio):
        command.append("-an")
    else:
        command.extend(
            [
                "-c:a",
                str(cfg.h264.audio_codec),
                "-b:a",
                str(cfg.h264.audio_bitrate),
            ]
        )
    command.append(str(output_path))

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(cfg.h264.timeout_seconds),
    )
    if not output_path.exists():
        raise FileNotFoundError(f"H.264 output was not created: {output_path}")
    input_path.unlink(missing_ok=True)
    return DownloadResult(
        title=result.title,
        source_path=output_path,
        source_filename=result.source_filename,
    )


def move_to_canonical_file(
    result: DownloadResult,
    target_path: Path,
    allow_overwrite: bool,
) -> Path:
    if target_path.exists():
        if not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target_path}")
        target_path.unlink()
    shutil.move(str(result.source_path), str(target_path))
    return target_path


def build_manifest_entry(
    *,
    index: int,
    url: str,
    result: DownloadResult,
    target_path: Path,
    output_dir: Path,
) -> ManifestEntry:
    return ManifestEntry(
        index=index,
        url=url,
        title=result.title,
        source_filename=result.source_filename,
        file_name=target_path.name,
        relative_path=str(target_path.relative_to(output_dir)),
        absolute_path=str(target_path.resolve()),
        file_size_bytes=target_path.stat().st_size,
        downloaded_at=now_iso(),
    )


def upsert_manifest_entry(manifest: dict[str, Any], entry: ManifestEntry) -> None:
    payload = asdict(entry)
    existing_entry = find_entry(manifest, entry.url)
    if existing_entry is None:
        manifest["entries"].append(payload)
        return
    existing_entry.clear()
    existing_entry.update(payload)


def build_plan(
    *,
    urls: list[str],
    output_dir: Path,
    manifest: dict[str, Any],
    skip_existing_urls: bool,
) -> list[dict[str, Any]]:
    reserved_indices = collect_reserved_indices(output_dir, manifest)
    plan: list[dict[str, Any]] = []
    for url in urls:
        existing_entry = find_entry(manifest, url)
        if existing_entry is not None:
            index = int(existing_entry["index"])
            file_name = str(existing_entry["file_name"])
            target_path = output_dir / file_name
            if skip_existing_urls and target_path.exists():
                action = "skip_existing"
            else:
                action = "download"
        else:
            index = allocate_index(reserved_indices)
            file_name = f"video_{index:06d}.mp4"
            target_path = output_dir / file_name
            action = "download"
        plan.append(
            {
                "index": index,
                "url": url,
                "action": action,
                "file_name": file_name,
                "absolute_path": str(target_path.resolve()),
            }
        )
    return plan


def run_dry_run(
    *,
    urls: list[str],
    urls_file: Path,
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    plan = build_plan(
        urls=urls,
        output_dir=output_dir,
        manifest=manifest,
        skip_existing_urls=bool(cfg.skip_existing_urls),
    )
    would_download = sum(1 for item in plan if item["action"] == "download")
    would_skip = sum(1 for item in plan if item["action"] == "skip_existing")
    return {
        "dry_run": True,
        "urls_file": str(urls_file),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "configured_urls": len(urls),
        "existing_manifest_entries": len(manifest.get("entries", [])),
        "would_download": would_download,
        "would_skip_existing": would_skip,
        "plan": plan,
    }


def download_youtube_videos(cfg: DictConfig) -> dict[str, Any]:
    urls_file = Path(to_absolute_path(str(cfg.urls_file))).resolve()
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    manifest_path = output_dir / str(cfg.manifest_file)
    urls = read_urls(urls_file)
    manifest = read_manifest(manifest_path)
    manifest["output_dir"] = str(output_dir)
    manifest["urls_file"] = str(urls_file)

    if bool(cfg.dry_run):
        return run_dry_run(
            urls=urls,
            urls_file=urls_file,
            output_dir=output_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            cfg=cfg,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    plan = build_plan(
        urls=urls,
        output_dir=output_dir,
        manifest=manifest,
        skip_existing_urls=bool(cfg.skip_existing_urls),
    )

    with tempfile.TemporaryDirectory(prefix=".yt-dlp-", dir=output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for item in plan:
            if item["action"] == "skip_existing":
                skipped += 1
                continue

            url = str(item["url"])
            index = int(item["index"])
            target_path = output_dir / str(item["file_name"])
            try:
                result = download_with_yt_dlp(url, temp_dir, cfg)
                if bool(cfg.h264.enabled):
                    result = transcode_to_h264(result, cfg)
                move_to_canonical_file(
                    result,
                    target_path,
                    allow_overwrite=bool(cfg.allow_overwrite),
                )
                entry = build_manifest_entry(
                    index=index,
                    url=url,
                    result=result,
                    target_path=target_path,
                    output_dir=output_dir,
                )
                upsert_manifest_entry(manifest, entry)
                write_manifest(manifest_path, manifest)
                downloaded += 1
            except Exception as exc:  # noqa: BLE001
                if not bool(cfg.continue_on_error):
                    raise
                failed.append({"url": url, "error": repr(exc)})

    write_manifest(manifest_path, manifest)
    return {
        "dry_run": False,
        "configured_urls": len(urls),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
    }


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="download_youtube_videos",
)
def main(cfg: DictConfig) -> None:
    summary = download_youtube_videos(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
