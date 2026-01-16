"""Video metadata catalog for MAE cached-batch training.

This module centralizes video discovery and lightweight metadata extraction
so that epoch planning can avoid repeatedly opening video files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

try:
    from decord import VideoReader, cpu

    HAS_DECORD = True
except ImportError:  # pragma: no cover
    HAS_DECORD = False


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass(frozen=True)
class VideoMeta:
    video_id: int
    path: str
    num_frames: int
    width: int
    height: int

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


@dataclass(frozen=True)
class VideoCatalog:
    videos: tuple[VideoMeta, ...]

    @classmethod
    def discover(cls, video_dir: str | Path) -> list[Path]:
        video_dir = Path(video_dir)
        paths: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            paths.extend(video_dir.glob(f"*{ext}"))
            paths.extend(video_dir.glob(f"*{ext.upper()}"))
        return sorted({p for p in paths if p.is_file()})

    @classmethod
    def from_paths(
        cls,
        video_paths: Sequence[str | Path],
        *,
        use_decord: bool = True,
        min_frames: int = 10,
    ) -> "VideoCatalog":
        paths = [Path(p) for p in video_paths]
        videos: list[VideoMeta] = []
        use_decord = use_decord and HAS_DECORD
        if not use_decord and not HAS_CV2:
            raise RuntimeError("Neither decord nor cv2 is available for video metadata.")

        for idx, path in enumerate(paths):
            try:
                if use_decord:
                    vr = VideoReader(str(path), ctx=cpu(0))
                    num_frames = len(vr)
                    height, width = vr[0].shape[:2]
                else:
                    cap = cv2.VideoCapture(str(path))
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
            except Exception:
                continue

            if num_frames < min_frames or width <= 0 or height <= 0:
                continue

            videos.append(
                VideoMeta(
                    video_id=len(videos),
                    path=str(path),
                    num_frames=num_frames,
                    width=width,
                    height=height,
                )
            )

        return cls(videos=tuple(videos))

    @classmethod
    def from_video_dir(
        cls,
        video_dir: str | Path,
        *,
        use_decord: bool = True,
        min_frames: int = 10,
    ) -> "VideoCatalog":
        paths = cls.discover(video_dir)
        return cls.from_paths(paths, use_decord=use_decord, min_frames=min_frames)

    def filter_by_paths(self, allowed_paths: Iterable[str | Path]) -> "VideoCatalog":
        allowed = {str(Path(p)) for p in allowed_paths}
        videos = [v for v in self.videos if str(Path(v.path)) in allowed]
        return VideoCatalog(videos=tuple(videos))

    def to_json(self) -> dict:
        return {"videos": [asdict(v) for v in self.videos]}

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VideoCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        videos = tuple(VideoMeta(**v) for v in data.get("videos", []))
        return cls(videos=videos)


if __name__ == "__main__":  # pragma: no cover
    catalog = VideoCatalog.from_video_dir("data/tennis/raw/videos", use_decord=True)
    print(f"videos={len(catalog.videos)}")

