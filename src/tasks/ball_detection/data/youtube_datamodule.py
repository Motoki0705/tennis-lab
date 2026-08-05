"""Lightning DataModule for annotated YouTube ball detection datasets."""

from __future__ import annotations

from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule


class YouTubeDataModule(TrackNetDataModule):
    """Reuse TrackNet loading while changing the dataset path connection.

    ``data.data_dir`` is the sole source root. The split files select the
    ``data`` role explicitly, and every split entry is relative to this source
    root (for example ``frames/video_000001/clip_000001``). Parent-relative
    ``youtube/...`` entries from the former contract are intentionally rejected.
    """


__all__ = ["YouTubeDataModule"]
