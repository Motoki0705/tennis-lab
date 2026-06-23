"""Lightning DataModule for TrackNet-style ball detection datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.types import ClipWindow, FrameLabel

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrackNetDataModule(pl.LightningDataModule):
    """Lightning DataModule for TrackNet-style ball detection data.

    Discovers source files, parses annotations, builds normalized windows, and
    connects them to :class:`BallDetectionDataset`.

    Args:
        config: Full Hydra configuration dictionary.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.data_dir = Path(str(data_cfg.get("data_dir", "data/tennis/tracknet")))
        self.batch_size = int(data_cfg.get("batch_size", 4))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.num_frames = int(self.config.get("model", {}).get("num_frames", 8))
        self.sample_stride = int(data_cfg.get("sample_stride", 1))

        split_cfg = data_cfg.get("split", {})
        self.train_split_file = str(split_cfg.get("train_file", ""))
        self.val_split_file = str(split_cfg.get("val_file", ""))
        self.test_split_file = str(split_cfg.get("test_file", ""))

        self.train_dataset: BallDetectionDataset | None = None
        self.val_dataset: BallDetectionDataset | None = None
        self.test_dataset: BallDetectionDataset | None = None

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sample_stride <= 0:
            raise ValueError("data.sample_stride must be positive.")

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        aug_cfg = self.config.get("data", {}).get("augmentation", {})

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
        manifest_paths = self._pseudo_manifest_paths_for_split(split_name)
        if manifest_paths:
            windows.extend(self._build_pseudo_windows(manifest_paths))
        if not windows:
            raise RuntimeError(
                "No supervised ball detection windows were found. "
                f"data_dir={self.data_dir}, split_file={resolved_split_file}"
            )
        return windows

    def _resolve_split_file(self, split_file: Path) -> Path:
        if split_file.is_absolute():
            resolved = split_file
        else:
            candidates = [
                Path.cwd() / split_file,
                self.data_dir / split_file,
                self.data_dir / "splits" / split_file,
            ]
            resolved = next(
                (candidate for candidate in candidates if candidate.exists()),
                split_file,
            )
        if not resolved.exists():
            raise FileNotFoundError(f"Split file not found: {resolved}")
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

    def _build_pseudo_windows(
        self,
        manifest_paths: Sequence[Path],
    ) -> list[ClipWindow]:
        windows: list[ClipWindow] = []
        for manifest_path in manifest_paths:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line.strip()
                    if not record:
                        continue
                    entry = json.loads(record)
                    image_dir = Path(str(entry["image_dir"])).expanduser()
                    label_csv = Path(str(entry["label_csv"])).expanduser()
                    frame_count = int(entry["frame_count"])
                    accepted_starts = [
                        int(start) for start in entry.get("accepted_starts", [])
                    ]
                    original_size = (
                        int(entry["original_width"]),
                        int(entry["original_height"]),
                    )
                    if frame_count < self.num_frames or not accepted_starts:
                        continue
                    if not image_dir.exists():
                        raise FileNotFoundError(
                            f"Pseudo image_dir not found: {image_dir}"
                        )
                    if not label_csv.exists():
                        raise FileNotFoundError(
                            f"Pseudo label_csv not found: {label_csv}"
                        )
                    frame_names = tuple(
                        f"{frame_index:06d}.jpg"
                        for frame_index in range(frame_count)
                    )
                    labels = self._read_label_csv(label_csv)
                    for start_index in accepted_starts:
                        if (
                            start_index < 0
                            or start_index + self.num_frames > frame_count
                        ):
                            continue
                        windows.append(
                            ClipWindow(
                                clip_dir=image_dir,
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
        """Resolve one split entry to its path."""
        return self.data_dir / entry

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

    def _pseudo_manifest_paths_for_split(self, split_name: str) -> list[Path]:
        data_cfg = self.config.get("data", {}) or {}
        configured: Any = data_cfg.get("pseudo_manifest_paths", [])
        if hasattr(configured, "get") and not isinstance(
            configured,
            (str, bytes),
        ):
            configured = configured.get(split_name, [])
        elif split_name != "train":
            configured = []
        if isinstance(configured, (str, Path)):
            configured = [configured]

        resolved_paths: list[Path] = []
        for manifest_path in configured or []:
            candidate = Path(str(manifest_path)).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Pseudo manifest not found: {candidate}"
                )
            resolved_paths.append(candidate)
        return resolved_paths

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


__all__ = ["TrackNetDataModule"]
