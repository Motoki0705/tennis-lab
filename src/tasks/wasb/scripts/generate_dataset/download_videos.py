#!/usr/bin/env python
"""Download tennis match videos from YouTube using yt-dlp.

This script reads video URLs from urls.yaml and downloads them to
data/tennis/raw/ with progress tracking via meta.json.

Features:
- Resume capability (tracks which URLs have been downloaded)
- Automatic detection of new URLs in urls.yaml
- Custom filename support
- Time range extraction (start/end times)

Requirements:
    pip install yt-dlp pyyaml

Usage:
    # Download all videos in urls.yaml
    uv run python -m src.wasb.scripts.generate_dataset.download_videos

    # Download with custom urls.yaml location
    uv run python -m src.wasb.scripts.generate_dataset.download_videos urls_path=path/to/urls.yaml

    # Check status
    uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=status

    # Reset failed downloads
    uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_failed

    # Force re-download specific URL
    uv run python -m src.wasb.scripts.generate_dataset.download_videos mode=reset_url reset_url="https://..."

"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import hydra
import yaml
from omegaconf import DictConfig

# Default paths
DEFAULT_URLS_PATH = Path("data/tennis/raw/urls.yaml")
DEFAULT_OUTPUT_DIR = Path("data/tennis/raw/videos")
DEFAULT_META_PATH = Path("data/tennis/raw/meta.json")
META_VERSION = "1.0"


@dataclass
class VideoEntry:
    """A video entry from urls.yaml.

    Attributes:
        url: YouTube URL.
        name: Custom filename (without extension).
        start: Start time for extraction.
        end: End time for extraction.

    """

    url: str
    name: str | None = None
    start: str | None = None
    end: str | None = None

    @property
    def url_hash(self) -> str:
        """Short hash of URL for identification."""
        return hashlib.md5(self.url.encode()).hexdigest()[:12]


@dataclass
class DownloadStatus:
    """Status of a single video download.

    Attributes:
        status: Download status.
        url: Original URL.
        filename: Downloaded filename (if completed).
        downloaded_at: ISO timestamp of completion.
        error_message: Error message if failed.
        file_size: File size in bytes (if completed).

    """

    status: Literal["pending", "in_progress", "completed", "failed"]
    url: str
    filename: str | None = None
    downloaded_at: str | None = None
    error_message: str | None = None
    file_size: int | None = None


@dataclass
class DownloadMeta:
    """Metadata for download state.

    Attributes:
        version: Meta file format version.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        urls_hash: Hash of urls.yaml for change detection.
        downloads: Dictionary mapping URL hash to DownloadStatus.

    """

    version: str = META_VERSION
    created_at: str = ""
    updated_at: str = ""
    urls_hash: str = ""
    downloads: dict[str, DownloadStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "urls_hash": self.urls_hash,
            "downloads": {
                key: asdict(status) for key, status in self.downloads.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> DownloadMeta:
        """Create from dictionary."""
        downloads = {}
        for key, status_dict in data.get("downloads", {}).items():
            downloads[key] = DownloadStatus(**status_dict)

        return cls(
            version=data.get("version", META_VERSION),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            urls_hash=data.get("urls_hash", ""),
            downloads=downloads,
        )


class VideoDownloader:
    """YouTube video downloader with progress tracking.

    Example:
        >>> downloader = VideoDownloader(output_dir="data/tennis/raw")
        >>> result = downloader.download_all("urls.yaml")

    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        meta_path: str | Path = DEFAULT_META_PATH,
        format_spec: str = "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/best[vcodec^=avc1]/best",
    ) -> None:
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded videos.
            meta_path: Path to the persistent meta.json state file.
            format_spec: yt-dlp format specification.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format_spec = format_spec

        self._meta_path = Path(meta_path)
        self._meta: DownloadMeta | None = None

    def download_all(
        self,
        urls_path: str | Path = DEFAULT_URLS_PATH,
        resume: bool = True,
        verbose: bool = True,
    ) -> dict[str, str]:
        """Download all videos from urls.yaml.

        Args:
            urls_path: Path to urls.yaml file.
            resume: If True, skip already downloaded videos.
            verbose: Show progress messages.

        Returns:
            Dictionary mapping URL to status ("completed", "failed", "skipped").

        """
        urls_path = Path(urls_path)
        if not urls_path.exists():
            raise FileNotFoundError(f"URLs file not found: {urls_path}")

        # Load URLs (includes duplicate detection)
        entries = self._load_urls(urls_path, verbose=verbose)
        if verbose:
            print(f"Loaded {len(entries)} unique video(s) from {urls_path}")

        # Load or create meta
        if resume and self._meta_path.exists():
            self._load_meta()
            if verbose:
                print(f"Resuming from {self._meta_path}")
        else:
            self._create_meta(urls_path)
            if verbose:
                print(f"Created new meta at {self._meta_path}")

        # Detect new URLs
        new_entries = self._update_download_status(entries)
        if verbose and new_entries:
            print(f"Detected {len(new_entries)} new URL(s)")

        # Build download queue
        queue = self._build_queue(entries)
        if verbose:
            print(f"Download queue: {len(queue)} video(s)")

        # Download videos
        results = {}

        for entry in queue:
            url_hash = entry.url_hash
            status = self._meta.downloads[url_hash]

            if verbose:
                name = entry.name or url_hash
                print(f"\nDownloading: {name}")
                print(f"  URL: {entry.url}")

            # Mark as in progress
            status.status = "in_progress"
            self._save_meta()

            try:
                # Download video
                filename = self._download_video(entry, verbose=verbose)

                # Update status
                file_path = self.output_dir / filename
                status.status = "completed"
                status.filename = filename
                status.downloaded_at = datetime.now().isoformat()
                status.file_size = (
                    file_path.stat().st_size if file_path.exists() else None
                )
                status.error_message = None
                results[entry.url] = "completed"

                if verbose:
                    size_mb = (status.file_size or 0) / (1024 * 1024)
                    print(f"  Saved: {filename} ({size_mb:.1f} MB)")

            except Exception as e:
                status.status = "failed"
                status.error_message = str(e)
                results[entry.url] = "failed"

                if verbose:
                    print(f"  Error: {e}")

            self._save_meta()

        # Count skipped
        for entry in entries:
            if entry.url not in results:
                results[entry.url] = "skipped"

        if verbose:
            self._print_summary(results)

        return results

    def get_status(self) -> DownloadMeta | None:
        """Get current download status."""
        if self._meta is None and self._meta_path.exists():
            self._load_meta()
        return self._meta

    def reset(
        self,
        url: str | None = None,
        failed_only: bool = False,
        all_downloads: bool = False,
    ) -> int:
        """Reset download status.

        Args:
            url: Reset specific URL.
            failed_only: Reset only failed downloads.
            all_downloads: Reset all downloads.

        Returns:
            Number of downloads reset.

        """
        if self._meta is None:
            self._load_meta()

        count = 0
        for _key, status in self._meta.downloads.items():
            should_reset = False

            if (
                all_downloads
                or failed_only
                and status.status == "failed"
                or url
                and status.url == url
            ):
                should_reset = True

            if should_reset:
                status.status = "pending"
                status.filename = None
                status.downloaded_at = None
                status.error_message = None
                status.file_size = None
                count += 1

        if count > 0:
            self._save_meta()

        return count

    def _load_urls(self, urls_path: Path, verbose: bool = True) -> list[VideoEntry]:
        """Load video entries from urls.yaml.

        Detects and warns about duplicate URLs, keeping only the first occurrence.
        """
        with urls_path.open("r") as f:
            data = yaml.safe_load(f) or {}

        entries = []
        seen_urls: dict[str, int] = {}  # url -> line index (1-based)
        duplicates: list[tuple[str, int, int]] = []  # (url, first_line, dup_line)

        videos = data.get("videos", [])

        if videos is None:
            return entries

        for idx, item in enumerate(videos, start=1):
            if isinstance(item, str):
                url = item
                entry = VideoEntry(url=url)
            elif isinstance(item, dict):
                url = item["url"]
                entry = VideoEntry(
                    url=url,
                    name=item.get("name"),
                    start=item.get("start"),
                    end=item.get("end"),
                )
            else:
                continue

            # Check for duplicate
            if url in seen_urls:
                duplicates.append((url, seen_urls[url], idx))
            else:
                seen_urls[url] = idx
                entries.append(entry)

        # Report duplicates
        if duplicates and verbose:
            print(
                f"\n⚠️  Warning: Found {len(duplicates)} duplicate URL(s) in {urls_path}:"
            )
            for url, first_idx, dup_idx in duplicates:
                short_url = url[:60] + "..." if len(url) > 60 else url
                print(f"   Entry #{dup_idx} duplicates #{first_idx}: {short_url}")
            print("   Only the first occurrence of each URL will be downloaded.\n")

        return entries

    def _compute_urls_hash(self, urls_path: Path) -> str:
        """Compute hash of urls.yaml content."""
        content = urls_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def _load_meta(self) -> None:
        """Load meta.json."""
        with self._meta_path.open("r") as f:
            data = json.load(f)
        self._meta = DownloadMeta.from_dict(data)

    def _save_meta(self) -> None:
        """Save meta.json."""
        self._meta.updated_at = datetime.now().isoformat()
        with self._meta_path.open("w") as f:
            json.dump(self._meta.to_dict(), f, indent=2)

    def _create_meta(self, urls_path: Path) -> None:
        """Create new meta."""
        now = datetime.now().isoformat()
        self._meta = DownloadMeta(
            version=META_VERSION,
            created_at=now,
            updated_at=now,
            urls_hash=self._compute_urls_hash(urls_path),
            downloads={},
        )
        self._save_meta()

    def _update_download_status(self, entries: list[VideoEntry]) -> list[VideoEntry]:
        """Update meta with new entries."""
        new_entries = []

        for entry in entries:
            url_hash = entry.url_hash

            if url_hash not in self._meta.downloads:
                self._meta.downloads[url_hash] = DownloadStatus(
                    status="pending",
                    url=entry.url,
                )
                new_entries.append(entry)

        if new_entries:
            self._save_meta()

        return new_entries

    def _build_queue(self, entries: list[VideoEntry]) -> list[VideoEntry]:
        """Build download queue."""
        queue = []

        for entry in entries:
            url_hash = entry.url_hash
            status = self._meta.downloads.get(url_hash)

            if status and status.status in ("pending", "in_progress"):
                queue.append(entry)

        return queue

    def _download_video(self, entry: VideoEntry, verbose: bool = True) -> str:
        """Download a single video using yt-dlp.

        Returns:
            Downloaded filename.

        """
        output_template = f"{entry.name}.%(ext)s" if entry.name else "%(title)s.%(ext)s"

        output_path = self.output_dir / output_template

        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "-f",
            self.format_spec,
            "-o",
            str(output_path),
            "--no-playlist",
            "--restrict-filenames",
        ]

        # Add time range if specified
        if entry.start or entry.end:
            postprocessor_args = []
            if entry.start:
                postprocessor_args.extend(["-ss", str(entry.start)])
            if entry.end:
                postprocessor_args.extend(["-to", str(entry.end)])

            cmd.extend(
                [
                    "--postprocessor-args",
                    f"ffmpeg:{' '.join(postprocessor_args)}",
                ]
            )

        # Add progress output
        if verbose:
            cmd.append("--progress")
        else:
            cmd.append("--quiet")

        cmd.append(entry.url)

        # Run yt-dlp
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            error_msg = (
                result.stderr
                if result.stderr
                else f"yt-dlp exited with code {result.returncode}"
            )
            raise RuntimeError(error_msg)

        # Find downloaded file
        # yt-dlp may change extension, so search for matching files
        pattern = f"{entry.name}.*" if entry.name else "*"

        matching_files = sorted(
            self.output_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Filter out meta.json and urls.yaml
        matching_files = [
            f
            for f in matching_files
            if f.suffix not in (".json", ".yaml", ".yml") and f.is_file()
        ]

        if not matching_files:
            raise RuntimeError("Downloaded file not found")

        return matching_files[0].name

    def _print_summary(self, results: dict[str, str]) -> None:
        """Print download summary."""
        completed = sum(1 for v in results.values() if v == "completed")
        failed = sum(1 for v in results.values() if v == "failed")
        skipped = sum(1 for v in results.values() if v == "skipped")

        print("\n" + "=" * 50)
        print("Download Summary")
        print("=" * 50)
        print(f"Total URLs: {len(results)}")
        print(f"Downloaded: {completed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")


def show_status(meta_path: Path) -> int:
    """Show download status."""
    if not meta_path.exists():
        print(f"No meta.json found in {meta_path}")
        return 1

    with meta_path.open("r") as f:
        meta = json.load(f)

    print(f"Meta version: {meta.get('version', 'unknown')}")
    print(f"Created: {meta.get('created_at', 'unknown')}")
    print(f"Updated: {meta.get('updated_at', 'unknown')}")
    print()

    downloads = meta.get("downloads", {})
    if not downloads:
        print("No downloads registered.")
        return 0

    # Count by status
    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
    for status in downloads.values():
        status_counts[status["status"]] = status_counts.get(status["status"], 0) + 1

    print(f"Downloads: {len(downloads)} total")
    print(f"  - Completed: {status_counts['completed']}")
    print(f"  - Pending: {status_counts['pending']}")
    print(f"  - In progress: {status_counts['in_progress']}")
    print(f"  - Failed: {status_counts['failed']}")
    print()

    # Show details
    print("Download details:")
    for _key, status in sorted(downloads.items()):
        status_str = status["status"].upper()
        filename = status.get("filename", "-")
        size = status.get("file_size")
        size_str = f"{size / (1024 * 1024):.1f} MB" if size else "-"

        print(f"  [{status_str}] {filename} ({size_str})")
        print(f"    URL: {status['url'][:60]}...")

        if status["status"] == "failed" and status.get("error_message"):
            print(f"    Error: {status['error_message'][:80]}")

    return 0


def _run(cfg: DictConfig) -> int:
    mode = str(cfg.mode)
    meta_path = Path(str(cfg.download.meta_path))

    if mode == "status":
        return show_status(meta_path)

    downloader = VideoDownloader(
        output_dir=str(cfg.download.output_dir),
        meta_path=str(cfg.download.meta_path),
        format_spec=str(cfg.download.format_spec),
    )

    if mode == "reset_failed":
        count = downloader.reset(failed_only=True)
        print(f"Reset {count} failed download(s) to pending.")
        return 0

    if mode == "reset_all":
        count = downloader.reset(all_downloads=True)
        print(f"Reset {count} download(s) to pending.")
        return 0

    if mode == "reset_url":
        if cfg.reset_url is None:
            print("Error: reset_url must be provided when mode=reset_url", file=sys.stderr)
            return 1
        count = downloader.reset(url=str(cfg.reset_url))
        if count > 0:
            print(f"Reset URL to pending: {cfg.reset_url}")
            return 0
        print(f"URL not found: {cfg.reset_url}")
        return 1

    if mode != "download":
        print(
            f"Error: unknown mode '{mode}' (expected download|status|reset_failed|reset_all|reset_url)",
            file=sys.stderr,
        )
        return 1

    try:
        results = downloader.download_all(
            urls_path=str(cfg.urls_path),
            resume=bool(cfg.resume),
            verbose=bool(cfg.verbose),
        )
        failed = sum(1 for v in results.values() if v == "failed")
        return 0 if failed == 0 else 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


@hydra.main(config_path="../../configs", config_name="download_videos", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    raise SystemExit(_run(cfg))


if __name__ == "__main__":
    main()
