"""Preview frame providers for the clip studio GUI.

Long match videos cannot be decoded eagerly, so previews are served by
:class:`PreviewSource`: a smart-seek reader (sequential ``grab()`` for small
forward jumps, absolute seek otherwise) that downscales each decoded frame to
tile size and keeps a per-source LRU cache so scrubbing back and forth over
the same region does not re-decode. :class:`PreviewSourcePool` fetches all
cameras in parallel with one worker thread per source.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.video import RandomAccessVideoReader, VideoInfo, probe_video_info


class PreviewSource:
    """Cached, downscaled random access to one source video.

    Thread safety: all public methods are serialized by an internal lock, so
    one instance may be used from a worker thread while others run in
    parallel.

    Args:
        video_path: Source video file.
        tile_width: Width in pixels of decoded preview frames (height keeps
            the source aspect ratio). Sources narrower than this are kept at
            native size.
        cache_frames: Maximum number of preview frames kept in the LRU cache.
        seek_grab_threshold: Forwarded to :class:`RandomAccessVideoReader`.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        tile_width: int = 640,
        cache_frames: int = 96,
        seek_grab_threshold: int = 24,
    ) -> None:
        if tile_width <= 0:
            raise ValueError(f"tile_width must be positive, got {tile_width}")
        if cache_frames <= 0:
            raise ValueError(f"cache_frames must be positive, got {cache_frames}")
        self.video_path = Path(video_path)
        self.info: VideoInfo = probe_video_info(self.video_path)
        self.tile_width = min(tile_width, self.info.width)
        self.cache_frames = cache_frames
        self._reader = RandomAccessVideoReader(
            self.video_path, seek_grab_threshold=seek_grab_threshold
        )
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._lock = threading.Lock()

    def get_frame(self, frame_index: int) -> NDArray[np.uint8]:
        """Return the downscaled BGR preview frame at ``frame_index``."""
        if not 0 <= frame_index < self.info.frame_count:
            raise ValueError(
                f"frame_index {frame_index} out of range "
                f"[0, {self.info.frame_count}) for {self.video_path}"
            )
        with self._lock:
            cached = self._cache.get(frame_index)
            if cached is not None:
                self._cache.move_to_end(frame_index)
                return cached
            frame_bgr = self._reader.read(frame_index)
            preview = self._downscale(frame_bgr)
            self._cache[frame_index] = preview
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
            return preview

    def _downscale(self, frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame_bgr.shape[:2]
        if width <= self.tile_width:
            return frame_bgr
        scale = self.tile_width / width
        resized = cv2.resize(
            frame_bgr,
            (self.tile_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.uint8)

    def close(self) -> None:
        with self._lock:
            self._reader.close()
            self._cache.clear()

    def __enter__(self) -> PreviewSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class PreviewSourcePool:
    """Fetch preview frames from several sources in parallel."""

    def __init__(self, sources: list[PreviewSource]) -> None:
        if not sources:
            raise ValueError("sources must contain at least one PreviewSource")
        self.sources = sources
        self._executor = ThreadPoolExecutor(
            max_workers=len(sources), thread_name_prefix="preview"
        )

    @property
    def infos(self) -> list[VideoInfo]:
        return [source.info for source in self.sources]

    def fetch(
        self, frame_indices: list[int | None]
    ) -> list[NDArray[np.uint8] | None]:
        """Fetch one preview frame per source (``None`` skips a source)."""
        if len(frame_indices) != len(self.sources):
            raise ValueError(
                f"frame_indices length {len(frame_indices)} must match sources "
                f"{len(self.sources)}"
            )

        def fetch_one(
            source: PreviewSource, index: int | None
        ) -> NDArray[np.uint8] | None:
            if index is None:
                return None
            return source.get_frame(index)

        futures = [
            self._executor.submit(fetch_one, source, index)
            for source, index in zip(self.sources, frame_indices, strict=True)
        ]
        return [future.result() for future in futures]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for source in self.sources:
            source.close()

    def __enter__(self) -> PreviewSourcePool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["PreviewSource", "PreviewSourcePool"]
