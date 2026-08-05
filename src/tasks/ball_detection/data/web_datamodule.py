"""Lightning DataModule for the unified web ball-detection frame store.

This serves the converted ``data/tennis/web`` datasets through the web
data-access layer. The store includes positive frames and explicitly annotated
negatives while excluding unknown annotation states.

``data.sampling.mode=static`` repeats one labeled frame to ``model.num_frames``.
``data.sampling.mode=temporal`` builds bidirectional-ready ordered windows from
one split-safe video sequence. The model receives only local order; original
FPS and frame intervals are retained as provenance but are not positional
inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.components.web.data_access_layer.web_store import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    SPLIT_CODES,
    WebFrameStore,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import ClipWindow

if TYPE_CHECKING:
    from omegaconf import DictConfig

_WEB_CLIP_DIR = Path("__web__")


class WebBallDetectionDataset(BallDetectionDataset):
    """Static or temporal windows backed by a :class:`WebFrameStore`.

    Reuses the heatmap / coordinate / augmentation pipeline of
    :class:`BallDetectionDataset` and only changes where pixels come from.
    """

    def __init__(
        self,
        *,
        store: WebFrameStore,
        sample_windows: Iterable[Sequence[int]],
        config: DictConfig,
        augmentation: BallDetectionAugmentation | None = None,
    ) -> None:
        self.store = store
        windows: list[ClipWindow] = []
        for raw_window in sample_windows:
            indices = tuple(int(value) for value in raw_window)
            if not indices:
                raise ValueError("Web sample windows must not be empty.")
            original_sizes = {store.original_size(index) for index in indices}
            if len(original_sizes) != 1:
                raise ValueError(
                    "All frames in a web temporal window must share one size: "
                    f"indices={indices}, sizes={sorted(original_sizes)}."
                )
            frame_names = tuple(str(index) for index in indices)
            windows.append(
                ClipWindow(
                    clip_dir=_WEB_CLIP_DIR,
                    frame_names=frame_names,
                    labels={
                        frame_name: store.labels(index)
                        for frame_name, index in zip(
                            frame_names,
                            indices,
                            strict=True,
                        )
                    },
                    original_size=next(iter(original_sizes)),
                    start_index=0,
                )
            )
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

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        paths = BallRuntimePaths.from_config(config)
        data_cfg = validate_data(config, paths=paths)
        self.data_dir = paths.data(str(data_cfg["data_dir"]))
        self.batch_size = int(data_cfg["batch_size"])
        self.num_workers = int(data_cfg["num_workers"])
        self.pin_memory = bool(data_cfg["pin_memory"])
        sources = data_cfg["sources"]
        self.sources = (
            None
            if sources == "all"
            else [str(name) for name in cast(Sequence[object], sources)]
        )
        sampling_cfg = data_cfg["sampling"]
        if not isinstance(sampling_cfg, Mapping):
            raise TypeError("data.sampling must be a mapping after validation.")
        self.sample_mode = str(sampling_cfg["mode"]).lower()
        if self.sample_mode not in {"static", "temporal"}:
            raise ValueError(
                "data.sampling.mode must be one of ['static', 'temporal'], "
                f"got {self.sample_mode!r}."
            )
        self.sampling_seed = int(sampling_cfg["seed"])
        self.train_negative_fraction = self._parse_negative_fraction(
            sampling_cfg["train_negative_fraction"]
        )
        temporal_cfg = sampling_cfg["temporal"]
        if not isinstance(temporal_cfg, Mapping):
            raise TypeError(
                "data.sampling.temporal must be a mapping after validation."
            )
        self.temporal_frame_step = int(temporal_cfg["frame_step"])
        self.temporal_sample_stride = int(temporal_cfg["sample_stride"])
        max_frame_gap = temporal_cfg["max_frame_gap"]
        self.temporal_max_frame_gap = (
            None if max_frame_gap is None else int(max_frame_gap)
        )
        self.num_frames = int(config.model.num_frames)
        self.augmentation_config = data_cfg["augmentation"]
        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")

        self.store: WebFrameStore | None = None
        self.train_dataset: WebBallDetectionDataset | None = None
        self.val_dataset: WebBallDetectionDataset | None = None
        self.test_dataset: WebBallDetectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open the store and build datasets for each stage."""
        if self.store is None:
            self.store = WebFrameStore(self.data_dir)
        aug_cfg = self.augmentation_config

        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(
                "train",
                BallDetectionAugmentation(aug_cfg),
            )
        if stage in {"fit", "validate"} or stage is None:
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
        sample_windows = self._sample_windows(split)
        if not sample_windows:
            raise RuntimeError(
                f"No web ball detection samples for split={split!r} "
                f"(data_dir={self.data_dir}, sample_mode={self.sample_mode!r}, "
                f"sources={self.sources}, num_frames={self.num_frames})."
            )
        return WebBallDetectionDataset(
            store=self.store,
            sample_windows=sample_windows,
            config=self.config,
            augmentation=augmentation,
        )

    def _sample_windows(self, split: str) -> list[tuple[int, ...]]:
        assert self.store is not None
        if self.sample_mode == "temporal":
            windows: list[tuple[int, ...]] = self.store.temporal_windows(
                split,
                num_frames=self.num_frames,
                frame_step=self.temporal_frame_step,
                sample_stride=self.temporal_sample_stride,
                max_frame_gap=self.temporal_max_frame_gap,
                sources=self.sources,
            )
            return windows

        indices = self.store.split_indices(split, sources=self.sources)
        if split == "train" and self.train_negative_fraction is not None:
            indices = self._limit_negative_fraction(
                indices,
                fraction=self.train_negative_fraction,
                seed=self.sampling_seed + SPLIT_CODES[split],
            )
        return [(int(index),) * self.num_frames for index in indices.tolist()]

    def _limit_negative_fraction(
        self,
        indices: np.ndarray,
        *,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        assert self.store is not None
        positive: np.ndarray = np.asarray(
            [
                index
                for index in indices.tolist()
                if self.store.label_state(index) == LABEL_POSITIVE
            ],
            dtype=np.int64,
        )
        negative: np.ndarray = np.asarray(
            [
                index
                for index in indices.tolist()
                if self.store.label_state(index) == LABEL_NEGATIVE
            ],
            dtype=np.int64,
        )
        if fraction == 0.0 or negative.size == 0:
            return positive
        if positive.size == 0:
            raise RuntimeError(
                "Cannot enforce a negative fraction without positive samples."
            )
        max_negative = int(np.floor(positive.size * fraction / (1.0 - fraction)))
        if negative.size > max_negative:
            rng = np.random.default_rng(seed)
            negative = rng.choice(
                negative,
                size=max_negative,
                replace=False,
            )
        balanced: np.ndarray = np.sort(np.concatenate([positive, negative]))
        return balanced

    @staticmethod
    def _parse_negative_fraction(value: Any) -> float | None:
        if value is None:
            return None
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(
                "data.sampling.train_negative_fraction must be a number or null."
            )
        fraction = float(value)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                "data.sampling.train_negative_fraction must be in [0, 1), "
                f"got {fraction}."
            )
        return fraction

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
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
        if self.val_dataset is None:
            raise RuntimeError("setup('validate') must run before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["WebBallDataModule", "WebBallDetectionDataset"]
