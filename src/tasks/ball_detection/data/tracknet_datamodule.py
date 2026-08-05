"""Lightning DataModule for TrackNet-style ball detection datasets."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader

from src.tasks.ball_detection.configuration import BallRuntimePaths, validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import ClipWindow, FrameLabel
from src.utils.configuration import PathRole

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrackNetDataModule(pl.LightningDataModule):
    """Lightning DataModule for TrackNet-style ball detection data.

    Discovers source files, parses annotations, builds normalized windows, and
    connects them to :class:`BallDetectionDataset`.

    Args:
        config: Full Hydra configuration dictionary.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        paths = BallRuntimePaths.from_config(config)
        data_cfg = validate_data(config, paths=paths)
        self.path_resolver = paths.resolver
        self.data_relative_dir = str(data_cfg["data_dir"])
        self.data_dir = paths.data(self.data_relative_dir)
        self.batch_size = int(data_cfg["batch_size"])
        self.num_workers = int(data_cfg["num_workers"])
        self.pin_memory = bool(data_cfg["pin_memory"])
        self.num_frames = int(config.model.num_frames)
        self.sample_stride = int(data_cfg["sample_stride"])

        split_cfg = data_cfg["split"]
        if not isinstance(split_cfg, Mapping):
            raise TypeError("data.split must be a mapping after validation.")
        self.split_root_role = PathRole(str(split_cfg["root_role"]))
        self.train_split_file = self.path_resolver.resolve(
            self.split_root_role, str(split_cfg["train_file"])
        )
        self.val_split_file = self.path_resolver.resolve(
            self.split_root_role, str(split_cfg["val_file"])
        )
        self.test_split_file = self.path_resolver.resolve(
            self.split_root_role, str(split_cfg["test_file"])
        )
        self.augmentation_config = data_cfg["augmentation"]

        self.train_dataset: BallDetectionDataset | None = None
        self.val_dataset: BallDetectionDataset | None = None
        self.test_dataset: BallDetectionDataset | None = None

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sample_stride <= 0:
            raise ValueError("data.sample_stride must be positive.")

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        aug_cfg = self.augmentation_config

        if stage == "fit" or stage is None:
            train_aug = BallDetectionAugmentation(aug_cfg)
            self.train_dataset = self.create_dataset(
                split_name="train",
                split_file=self.train_split_file,
                augmentation=train_aug,
            )
        if stage in {"fit", "validate"} or stage is None:
            self.val_dataset = self.create_dataset(
                split_name="val",
                split_file=self.val_split_file,
                augmentation=BallDetectionAugmentation.from_eval_config(aug_cfg),
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.create_dataset(
                split_name="test",
                split_file=self.test_split_file,
                augmentation=BallDetectionAugmentation.from_eval_config(aug_cfg),
            )

    def create_dataset(
        self,
        *,
        split_name: str,
        split_file: str | Path,
        augmentation: BallDetectionAugmentation | None,
    ) -> BallDetectionDataset:
        """Build windows and connect them to the shared dataset."""
        windows = self.create_windows(
            split_name=split_name,
            split_file=split_file,
        )
        return BallDetectionDataset(
            windows=windows,
            config=self.config,
            augmentation=augmentation,
        )

    def create_windows(
        self,
        *,
        split_name: str,
        split_file: str | Path,
    ) -> list[ClipWindow]:
        """Build normalized windows for one dataset split.

        New source formats can override this method, or a narrower helper such
        as :meth:`_resolve_entry_path`, while reusing dataset construction.
        """
        resolved_split_file = self._resolve_split_file(Path(split_file).expanduser())
        windows = self._build_labeled_windows(resolved_split_file)
        if not windows:
            raise RuntimeError(
                "No supervised ball detection windows were found. "
                f"data_dir={self.data_dir}, split_file={resolved_split_file}"
            )
        return windows

    def _resolve_split_file(self, split_file: Path) -> Path:
        """Validate the one role-resolved split path selected at composition."""
        if not split_file.is_absolute():
            raise ValueError(
                "TrackNet split paths must be resolved by data.split.root_role "
                f"before dataset construction; got {split_file}."
            )
        resolved: Path = self.path_resolver.validate(
            self.split_root_role, split_file
        )
        if not resolved.exists():
            raise FileNotFoundError(f"Split file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Split path must be a file: {resolved}")
        return resolved

    def _build_labeled_windows(self, split_file: Path) -> list[ClipWindow]:
        windows: list[ClipWindow] = []
        for entry in self._read_split_entries(split_file):
            for clip_dir in self._expand_entry(entry):
                label_path = clip_dir / "Label.csv"
                if not label_path.exists():
                    continue
                frame_names = tuple(
                    sorted(path.name for path in clip_dir.glob("*.jpg"))
                )
                if len(frame_names) < self.num_frames:
                    continue
                original_size = self._resolve_original_size(
                    clip_dir,
                    frame_names[0],
                )
                labels = self._read_label_csv(label_path)
                max_start = len(frame_names) - self.num_frames
                for start_index in range(0, max_start + 1, self.sample_stride):
                    windows.append(
                        ClipWindow(
                            clip_dir=clip_dir,
                            frame_names=frame_names,
                            labels=labels,
                            original_size=original_size,
                            start_index=start_index,
                        )
                    )
        return windows

    def _read_split_entries(self, split_file: Path) -> list[str]:
        with split_file.open("r", encoding="utf-8") as handle:
            entries = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if not entries:
            raise RuntimeError(f"Split file is empty: {split_file}")
        return entries

    def _resolve_entry_path(self, entry: str) -> Path:
        """Resolve one split entry below the declared data source root."""
        entry_path = Path(entry)
        if entry_path.is_absolute() or ".." in entry_path.parts:
            raise ValueError(
                "TrackNet split entries must be relative children of "
                f"data.data_dir; got {entry!r}."
            )
        resolved: Path = self.path_resolver.resolve(
            PathRole.DATA,
            self.data_relative_dir,
            entry_path,
        )
        return resolved

    def _expand_entry(self, entry: str) -> list[Path]:
        entry_path = self._resolve_entry_path(entry)
        if not entry_path.exists():
            raise FileNotFoundError(
                f"Split entry '{entry}' does not exist at {entry_path}."
            )
        if entry_path.is_dir() and self._is_clip_dir(entry_path):
            return [entry_path]
        if entry_path.is_dir():
            clip_dirs = sorted(
                path
                for path in entry_path.iterdir()
                if path.is_dir() and self._is_clip_dir(path)
            )
            if clip_dirs:
                return clip_dirs
        raise RuntimeError(
            "Split entries must point to either a game directory containing "
            f"clip directories or a specific clip directory. Invalid entry: {entry}"
        )

    @staticmethod
    def _is_clip_dir(path: Path) -> bool:
        """Return whether a directory follows a supported clip convention."""
        return path.name.startswith("Clip") or path.name.startswith("clip_")

    @staticmethod
    def _resolve_original_size(
        clip_dir: Path,
        first_frame_name: str,
    ) -> tuple[int, int]:
        with Image.open(clip_dir / first_frame_name) as image:
            return int(image.width), int(image.height)

    @staticmethod
    def _read_label_csv(path: Path) -> dict[str, tuple[FrameLabel, ...]]:
        labels: dict[str, list[FrameLabel]] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_fields = {
                "file name",
                "visibility",
                "x-coordinate",
                "y-coordinate",
            }
            missing = required_fields.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Missing required CSV columns in {path}: {sorted(missing)}"
                )
            for row in reader:
                frame_name = str(row["file name"]).strip()
                visibility = float(row["visibility"] or 0)
                instance_id = str(row.get("instance id") or "").strip()
                if not instance_id and visibility <= 0:
                    labels.setdefault(frame_name, [])
                    continue
                labels.setdefault(frame_name, []).append(
                    FrameLabel(
                        visibility=visibility,
                        x=float(row["x-coordinate"] or 0),
                        y=float(row["y-coordinate"] or 0),
                        instance_id=instance_id or "b001",
                        role=str(row.get("role") or "target").strip() or "target",
                        state=str(
                            row.get("ball state")
                            or ("visible" if visibility > 0 else "absent")
                        ),
                    )
                )
        return {
            frame_name: tuple(frame_labels)
            for frame_name, frame_labels in labels.items()
        }

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
            raise RuntimeError("setup('fit') must run before val_dataloader().")
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


__all__ = ["TrackNetDataModule"]
