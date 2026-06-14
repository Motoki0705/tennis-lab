"""Image collectors for the DINOv3 tennis SSL data pipeline.

Each collector turns one configured *source* into a list of saved JPEG images
plus provenance metadata. Collectors are intentionally dependency-light and
degrade gracefully: a missing local path or an unreachable URL is reported and
skipped rather than aborting the whole collection run.

Supported source types:
    - ``video_frames``: sample frames from a local or remote video.
    - ``image_dir``: ingest images from a local directory tree.
    - ``image_urls``: download images from a list of URLs (network required).
"""

from __future__ import annotations

import hashlib
import io
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class CollectedImage:
    """A single saved image and where it came from."""

    path: Path
    source_type: str
    provenance: str


def _save_image(image: Image.Image, out_dir: Path, key: str) -> Path:
    """Save ``image`` as JPEG under ``out_dir`` using a content-stable name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    path = out_dir / f"{digest}.jpg"
    image.convert("RGB").save(path, format="JPEG", quality=95)
    return path


def _resolve_video_source(path: str) -> str:
    """Resolve a web video URL to a direct stream URL via yt-dlp.

    Local paths are returned unchanged. Remote pages (e.g. YouTube) are resolved
    to a direct media URL that OpenCV can read. Resolution failures fall back to
    the original path so the caller can report and skip it.
    """
    if not str(path).lower().startswith(("http://", "https://")):
        return path
    try:
        import yt_dlp
    except ImportError:  # pragma: no cover - yt-dlp is a project dependency
        print("[collect] video_frames: yt-dlp unavailable; trying raw URL.")
        return path
    try:
        options = {"quiet": True, "skip_download": True, "format": "best[ext=mp4]/best"}
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(path, download=False)
        if isinstance(info, dict) and info.get("url"):
            return str(info["url"])
    except Exception as exc:  # noqa: BLE001 - best-effort URL resolution
        print(f"[collect] video_frames: yt-dlp could not resolve {path!r} ({exc}).")
    return path


def _collect_video_frames(
    *,
    path: str,
    out_dir: Path,
    stride: int = 10,
    max_frames: int | None = None,
    min_size: int = 0,
) -> list[CollectedImage]:
    """Sample frames from a video at a fixed stride."""
    resolved = _resolve_video_source(str(path))
    capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        print(f"[collect] video_frames: could not open {path!r}; skipping.")
        return []

    collected: list[CollectedImage] = []
    frame_index = 0
    kept = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(stride, 1) == 0:
                height, width = frame.shape[:2]
                if min(height, width) >= min_size:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    saved = _save_image(image, out_dir, f"{path}:{frame_index}")
                    collected.append(
                        CollectedImage(
                            path=saved,
                            source_type="video_frames",
                            provenance=f"{path}#frame={frame_index}",
                        )
                    )
                    kept += 1
                    if max_frames is not None and kept >= max_frames:
                        break
            frame_index += 1
    finally:
        capture.release()

    print(f"[collect] video_frames: kept {len(collected)} frame(s) from {path!r}.")
    return collected


def _iter_image_files(root: Path, pattern: str) -> Iterable[Path]:
    if pattern:
        yield from sorted(root.rglob(pattern))
        return
    for candidate in sorted(root.rglob("*")):
        if candidate.suffix.lower() in _IMAGE_EXTENSIONS:
            yield candidate


def _collect_image_dir(
    *,
    path: str,
    out_dir: Path,
    pattern: str = "",
    max_images: int | None = None,
    min_size: int = 0,
) -> list[CollectedImage]:
    """Ingest images from a local directory tree."""
    root = Path(path)
    if not root.is_dir():
        print(f"[collect] image_dir: directory not found {path!r}; skipping.")
        return []

    collected: list[CollectedImage] = []
    for source_path in _iter_image_files(root, pattern):
        if max_images is not None and len(collected) >= max_images:
            break
        try:
            with Image.open(source_path) as image:
                image.load()
                if min(image.size) < min_size:
                    continue
                saved = _save_image(image, out_dir, str(source_path))
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[collect] image_dir: skipping {source_path} ({exc}).")
            continue
        collected.append(
            CollectedImage(
                path=saved,
                source_type="image_dir",
                provenance=str(source_path),
            )
        )

    print(f"[collect] image_dir: ingested {len(collected)} image(s) from {path!r}.")
    return collected


def _collect_image_urls(
    *,
    urls: Iterable[str],
    out_dir: Path,
    timeout: float = 15.0,
    min_size: int = 0,
) -> list[CollectedImage]:
    """Download images from a list of URLs (best effort, network required)."""
    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a transitive dep
        print("[collect] image_urls: 'requests' unavailable; skipping.")
        return []

    collected: list[CollectedImage] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.load()
        except Exception as exc:  # noqa: BLE001 - best-effort network ingestion
            print(f"[collect] image_urls: skipping {url} ({exc}).")
            continue
        if min(image.size) < min_size:
            continue
        saved = _save_image(image, out_dir, url)
        collected.append(
            CollectedImage(
                path=saved,
                source_type="image_urls",
                provenance=url,
            )
        )

    print(f"[collect] image_urls: downloaded {len(collected)} image(s).")
    return collected


def collect_from_source(source: dict[str, Any], out_dir: Path) -> list[CollectedImage]:
    """Dispatch one configured source to the matching collector."""
    source_type = str(source.get("type", "")).strip()
    if source_type == "video_frames":
        return _collect_video_frames(
            path=str(source["path"]),
            out_dir=out_dir,
            stride=int(source.get("stride", 10)),
            max_frames=source.get("max_frames"),
            min_size=int(source.get("min_size", 0)),
        )
    if source_type == "image_dir":
        return _collect_image_dir(
            path=str(source["path"]),
            out_dir=out_dir,
            pattern=str(source.get("pattern", "")),
            max_images=source.get("max_images"),
            min_size=int(source.get("min_size", 0)),
        )
    if source_type == "image_urls":
        return _collect_image_urls(
            urls=list(source.get("urls", [])),
            out_dir=out_dir,
            timeout=float(source.get("timeout", 15.0)),
            min_size=int(source.get("min_size", 0)),
        )
    raise ValueError(f"Unknown DINOv3 SSL collection source type: {source_type!r}")


def deduplicate_images(images: list[CollectedImage]) -> list[CollectedImage]:
    """Drop byte-identical duplicates, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[CollectedImage] = []
    for item in images:
        digest = hashlib.sha1(item.path.read_bytes()).hexdigest()
        if digest in seen:
            item.path.unlink(missing_ok=True)
            continue
        seen.add(digest)
        unique.append(item)
    return unique


def reset_image_dir(out_dir: Path) -> None:
    """Remove a previously collected image directory if present."""
    if out_dir.exists():
        shutil.rmtree(out_dir)


__all__ = [
    "CollectedImage",
    "collect_from_source",
    "deduplicate_images",
    "reset_image_dir",
]


# Re-exported for callers that want to fabricate a tiny synthetic dataset
# without any external data (used by tests and smoke runs).
def write_synthetic_images(
    *, out_dir: Path, count: int, size: int = 256, seed: int = 0
) -> list[CollectedImage]:
    """Generate ``count`` deterministic synthetic images (offline fallback)."""
    rng = np.random.default_rng(seed)
    collected: list[CollectedImage] = []
    for index in range(count):
        array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        saved = _save_image(image, out_dir, f"synthetic:{seed}:{index}")
        collected.append(
            CollectedImage(
                path=saved,
                source_type="synthetic",
                provenance=f"synthetic#seed={seed}#index={index}",
            )
        )
    return collected
