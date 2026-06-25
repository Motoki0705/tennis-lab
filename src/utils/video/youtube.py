"""YouTube download and FFmpeg transcode helpers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from src.utils.commands import run_command

VIDEO_EXCLUDED_SUFFIXES = {".json", ".part", ".ytdl"}


def find_downloaded_video(directory: str | Path, video_id: str) -> Path | None:
    """Return the first downloaded video matching ``video_id`` in ``directory``."""
    root = Path(directory)
    candidates = [
        path
        for path in sorted(root.glob(f"{video_id}.*"))
        if path.suffix not in VIDEO_EXCLUDED_SUFFIXES
        and not path.name.endswith(".info.json")
    ]
    return candidates[0] if candidates else None


def download_youtube_video(
    *,
    url: str,
    video_id: str,
    output_dir: str | Path,
    format_selector: str,
    merge_output_format: str = "mkv",
    enabled: bool = True,
    overwrite: bool = False,
    write_info_json: bool = True,
    no_playlist: bool = True,
    js_runtimes: str | None = None,
    remote_components: str | None = None,
    download_archive: str | Path | None = None,
    extra_args: Sequence[str] = (),
    python_executable: str = sys.executable,
) -> Path:
    """Download one YouTube video with ``yt-dlp`` and return its local path."""
    destination = Path(output_dir)
    existing = find_downloaded_video(destination, video_id)
    if existing is not None and (not enabled or not overwrite):
        print(f"  source video exists: {existing}")
        return existing
    if not enabled:
        raise FileNotFoundError(f"Download disabled and no video found for {video_id}.")

    destination.mkdir(parents=True, exist_ok=True)
    output_template = destination / f"{video_id}.%(ext)s"
    cmd = [
        python_executable,
        "-m",
        "yt_dlp",
        url,
        "-f",
        format_selector,
        "-o",
        str(output_template),
        "--merge-output-format",
        merge_output_format,
    ]
    if write_info_json:
        cmd.append("--write-info-json")
    if no_playlist:
        cmd.append("--no-playlist")
    if js_runtimes is not None:
        cmd.extend(["--js-runtimes", js_runtimes])
    if remote_components is not None:
        cmd.extend(["--remote-components", remote_components])
    if download_archive is not None:
        archive = Path(download_archive).resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--download-archive", str(archive)])
    cmd.append("--force-overwrites" if overwrite else "--no-overwrites")
    cmd.extend(str(value) for value in extra_args)
    run_command(cmd)

    downloaded = find_downloaded_video(destination, video_id)
    if downloaded is None:
        raise FileNotFoundError(
            f"yt-dlp finished but no video was found for {video_id}."
        )
    return downloaded


def h264_encoder_args(
    *,
    encoder: str,
    preset: str,
    pix_fmt: str,
    crf: int | float,
    tune: str | None = None,
    rate_control: str | None = None,
    cq: int | float | None = None,
    bitrate: str | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
    profile: str | None = None,
) -> list[str]:
    """Return FFmpeg arguments for H.264 encoding."""
    if encoder == "libx264":
        return [
            "-c:v",
            encoder,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pix_fmt,
        ]

    if encoder not in {"h264_nvenc", "avc_nvenc"}:
        raise ValueError(f"Unsupported H.264 encoder: {encoder!r}")

    args = [
        "-c:v",
        encoder,
        "-preset",
        preset,
        "-tune",
        "hq" if tune is None else tune,
        "-rc",
        "vbr" if rate_control is None else rate_control,
        "-cq",
        "20" if cq is None else str(cq),
        "-pix_fmt",
        pix_fmt,
        "-b:v",
        "0" if bitrate is None else bitrate,
    ]
    if maxrate is not None:
        args.extend(["-maxrate", maxrate])
    if bufsize is not None:
        args.extend(["-bufsize", bufsize])
    if profile is not None:
        args.extend(["-profile:v", profile])
    return args


def transcode_h264_video(
    *,
    source_video: str | Path,
    output_path: str | Path,
    enabled: bool = True,
    overwrite: bool = False,
    ffmpeg_binary: str = "ffmpeg",
    encoder: str = "libx264",
    hwaccel: str | None = None,
    hwaccel_output_format: str | None = None,
    preset: str = "medium",
    tune: str | None = None,
    rate_control: str | None = None,
    cq: int | float | None = None,
    bitrate: str | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
    profile: str | None = None,
    pix_fmt: str = "yuv420p",
    crf: int | float = 20,
) -> Path:
    """Transcode a video to H.264 MP4 and return the output path."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        print(f"  H.264 exists: {destination}")
        return destination
    if not enabled:
        raise FileNotFoundError(
            f"H.264 transcode disabled and output missing: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_binary, "-y" if overwrite else "-n"]
    if hwaccel is not None:
        cmd.extend(["-hwaccel", hwaccel])
    if hwaccel_output_format is not None:
        cmd.extend(["-hwaccel_output_format", hwaccel_output_format])
    cmd.extend(["-i", str(source_video), "-map", "0:v:0"])
    cmd.extend(
        h264_encoder_args(
            encoder=encoder,
            preset=preset,
            tune=tune,
            rate_control=rate_control,
            cq=cq,
            bitrate=bitrate,
            maxrate=maxrate,
            bufsize=bufsize,
            profile=profile,
            pix_fmt=pix_fmt,
            crf=crf,
        )
    )
    cmd.extend(["-movflags", "+faststart", "-an", str(destination)])
    run_command(cmd)
    return destination


__all__ = [
    "download_youtube_video",
    "find_downloaded_video",
    "h264_encoder_args",
    "transcode_h264_video",
]
