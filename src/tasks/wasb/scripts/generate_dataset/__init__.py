"""Dataset generation and download script entrypoints for WASB (Hydra-based).

This package keeps legacy imports stable for unit tests by re-exporting the
batch reset helpers from `src.tasks.wasb.scripts.generate_dataset.batch`.
"""

from __future__ import annotations

from src.tasks.wasb.scripts.generate_dataset.batch import META_FILENAME, reset_videos

__all__ = ["META_FILENAME", "reset_videos"]

