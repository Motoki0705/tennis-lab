"""Lightning DataModule for annotated YouTube ball detection datasets."""

from __future__ import annotations

from pathlib import Path

from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule


class YouTubeDataModule(TrackNetDataModule):
    """Reuse TrackNet loading while changing the dataset path connection.

    ``data.data_dir`` points to the YouTube dataset root, for example
    ``data/tennis/youtube``. Split paths are resolved below that root, while
    split entries remain relative to the root's parent (for example
    ``youtube/frames/video_000001/clip_000001``).
    """

    def _resolve_entry_path(self, entry: str) -> Path:
        """Resolve entries written relative to the tennis dataset root."""
        return self.data_dir.parent / entry


__all__ = ["YouTubeDataModule"]
