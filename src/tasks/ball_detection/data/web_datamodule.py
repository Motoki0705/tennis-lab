"""Lightning DataModule for the unified web ball-detection frame store.

This serves the converted ``data/tennis/web`` datasets (see
:mod:`web_store` and ``scripts/convert_web_dataset``). Only frames that carry
a ball annotation are stored, so every sample is a positive detection example.

Each annotated frame is served as a *static clip*: the single frame is
replicated to ``model.num_frames`` so existing temporal models (which require
``T >= 8``) can be pre-trained for spatial detection on this data. A later
phase can switch to real multi-frame sequences; the per-sample ``temporal``
flag and ``frame_index`` provenance are preserved in the store to make that
switch cheap. Set ``data.temporal_only=true`` to keep only video-sourced
samples once multi-frame windows are introduced.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.augmentation import BallDetectionAugmentation
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import ClipWindow
from src.tasks.ball_detection.data.web_store import WebFrameStore

if TYPE_CHECKING:
    from omegaconf import DictConfig

_WEB_CLIP_DIR = Path("__web__")


class WebBallDetectionDataset(BallDetectionDataset):
    """Static-clip dataset backed by a :class:`WebFrameStore`.

    Reuses the heatmap / coordinate / augmentation pipeline of
    :class:`BallDetectionDataset` and only changes where pixels come from.
    """

    def __init__(
        self,
        *,
        store: WebFrameStore,
        sample_indices: Iterable[int],
        config: DictConfig | None = None,
        augmentation: BallDetectionAugmentation | None = None,
    ) -> None:
        self.store = store
        cfg: Any = config or {}
        num_frames = int((cfg.get("model", {}) or {}).get("num_frames", 8))
        windows = [
            ClipWindow(
                clip_dir=_WEB_CLIP_DIR,
                frame_names=(str(index),) * num_frames,
                labels={str(index): store.labels(index)},
                original_size=store.original_size(index),
                start_index=0,
            )
            for index in (int(value) for value in sample_indices)
        ]
        super().__init__(windows=windows, config=config, augmentation=augmentation)

    def _load_frame(self, path: Path) -> np.ndarray:
        image_h, image_w = self.image_size
        image = self.store.decode_bgr(int(path.name))
        image = cv2.resize(image, (image_w, image_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized: np.ndarray = image.astype(np.float32) / 255.0
        return normalized


class WebBallDataModule(pl.LightningDataModule):
    """Lightning DataModule for the converted web ball-detection store.

    Args:
        config: Full Hydra configuration dictionary.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.data_dir = Path(str(data_cfg.get("data_dir", "data/tennis/web/unified")))
        self.batch_size = int(data_cfg.get("batch_size", 4))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.temporal_only = bool(data_cfg.get("temporal_only", False))
        sources = data_cfg.get("sources", None)
        self.sources = (
            None
            if sources in (None, "all")
            else [str(name) for name in sources]
        )

        self.store: WebFrameStore | None = None
        self.train_dataset: WebBallDetectionDataset | None = None
        self.val_dataset: WebBallDetectionDataset | None = None
        self.test_dataset: WebBallDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open the store and build datasets for each stage."""
        if self.store is None:
            self.store = WebFrameStore(self.data_dir)
        aug_cfg = self.config.get("data", {}).get("augmentation", {})

        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(
                "train",
                BallDetectionAugmentation(aug_cfg),
            )
            self.val_dataset = self._create_dataset(
                "val",
                BallDetectionAugmentation.from_eval_config(aug_cfg),
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(
                "test",
                BallDetectionAugmentation.from_eval_config(aug_cfg),
            )

    def _create_dataset(
        self,
        split: str,
        augmentation: BallDetectionAugmentation | None,
    ) -> WebBallDetectionDataset:
        assert self.store is not None
        indices = self.store.split_indices(
            split,
            temporal_only=self.temporal_only,
            sources=self.sources,
        )
        if indices.size == 0:
            raise RuntimeError(
                f"No web ball detection samples for split={split!r} "
                f"(data_dir={self.data_dir}, temporal_only={self.temporal_only}, "
                f"sources={self.sources})."
            )
        return WebBallDetectionDataset(
            store=self.store,
            sample_indices=indices,
            config=self.config,
            augmentation=augmentation,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["WebBallDataModule", "WebBallDetectionDataset"]
