"""Labeled supervised dataset for ball detection."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.argumentation import (
    BallDetectionArgumentation,
    make_sample_rng,
)
from src.tasks.ball_detection.data.types import BallDetectionSample
from src.utils.data.heatmaps import generate_gaussian_heatmaps

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class FrameLabel:
    """One frame label parsed from ``Label.csv``.

    Attributes:
        visibility: Visibility flag from the annotation file.
        x: Ball x coordinate in original image pixels.
        y: Ball y coordinate in original image pixels.
    """

    visibility: float
    x: float
    y: float
    instance_id: str = ""
    role: str = "target"
    state: str = "visible"


@dataclass(frozen=True)
class ClipWindow:
    """One fixed-length temporal training window.

    Attributes:
        clip_dir: Directory containing frame images and ``Label.csv``.
        frame_names: Ordered frame names available in the clip.
        labels: Per-frame instance labels keyed by frame name.
        original_size: Original frame size in ``(width, height)`` ordering.
        start_index: Inclusive start index of the temporal window.
    """

    clip_dir: Path
    frame_names: tuple[str, ...]
    labels: dict[str, tuple[FrameLabel, ...]]
    original_size: tuple[int, int]
    start_index: int


class BallDetectionDataset(Dataset[BallDetectionSample]):
    """Dataset for supervised spatio-temporal ball detection."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        split_file: str | Path,
        config: DictConfig | None = None,
        argumentation: BallDetectionArgumentation | None = None,
        pseudo_manifest_paths: Sequence[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.argumentation = argumentation

        data_cfg = self.config.get("data", {}) or {}
        model_cfg = self.config.get("model", {}) or {}

        self.data_dir = Path(str(data_dir)).expanduser()
        self.split_file = self._resolve_split_file(Path(str(split_file)).expanduser())
        self.pseudo_manifest_paths = self._resolve_manifest_paths(pseudo_manifest_paths or [])
        self.num_frames = int(model_cfg.get("num_frames", 8))
        self.sample_stride = int(data_cfg.get("sample_stride", 1))
        self.image_size = self._parse_size(data_cfg.get("image_size", [288, 512]), name="data.image_size")
        self.heatmap_size = self._parse_size(data_cfg.get("heatmap_size", [144, 256]), name="data.heatmap_size")
        self.sigma_ratio = float(data_cfg.get("sigma_ratio", 0.0066))
        self.max_instances = int(data_cfg.get("max_instances", 8))

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sample_stride <= 0:
            raise ValueError("data.sample_stride must be positive.")
        if self.sigma_ratio <= 0:
            raise ValueError("data.sigma_ratio must be positive.")
        if self.max_instances <= 0:
            raise ValueError("data.max_instances must be positive.")

        self.windows = self._build_windows()
        if self.pseudo_manifest_paths:
            self.windows.extend(self._build_pseudo_windows())
        if not self.windows:
            raise RuntimeError(
                "No supervised ball detection windows were found. "
                f"data_dir={self.data_dir}, split_file={self.split_file}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> BallDetectionSample:
        window = self.windows[index]
        image_h, image_w = self.image_size
        heatmap_h, heatmap_w = self.heatmap_size
        original_w, original_h = window.original_size

        frames_hwc: list[np.ndarray] = []
        coords_image: list[list[tuple[float, float]]] = []
        visibility: list[list[float]] = []

        for offset in range(self.num_frames):
            frame_name = window.frame_names[window.start_index + offset]
            frame_path = window.clip_dir / frame_name
            frames_hwc.append(self._load_frame(frame_path))

            labels = [
                label
                for label in window.labels.get(frame_name, ())
                if label.role != "distractor"
            ]
            if len(labels) > self.max_instances:
                raise ValueError(
                    f"{window.clip_dir / 'Label.csv'} frame={frame_name} has "
                    f"{len(labels)} trainable instances, exceeding "
                    f"data.max_instances={self.max_instances}."
                )
            frame_coords: list[tuple[float, float]] = []
            frame_visibility: list[float] = []
            for label in labels:
                if label.visibility > 0:
                    frame_coords.append((
                        label.x * image_w / max(original_w, 1),
                        label.y * image_h / max(original_h, 1),
                    ))
                    frame_visibility.append(1.0)
                else:
                    frame_coords.append((0.0, 0.0))
                    frame_visibility.append(0.0)
            coords_image.append(frame_coords)
            visibility.append(frame_visibility)

        if self.argumentation is not None:
            frames_hwc, coords_image, visibility = self.argumentation.forward(
                frames_hwc,
                coords_image,
                visibility,
                rng=make_sample_rng(index),
            )

        image_tensors: list[np.ndarray] = []
        heatmaps: list[np.ndarray] = []
        coords_original: list[tuple[float, float]] = []
        primary_visibility: list[float] = []
        instance_coords_original: list[list[tuple[float, float]]] = []
        instance_visibility: list[list[float]] = []
        for frame, frame_coords, frame_visibility in zip(
            frames_hwc,
            coords_image,
            visibility,
            strict=True,
        ):
            image_tensors.append(np.transpose(frame, (2, 0, 1)))
            normalized_centers = [
                self._to_normalized_xy(
                    x_img=x_img,
                    y_img=y_img,
                    width=image_w,
                    height=image_h,
                )
                for x_img, y_img in frame_coords
            ]
            if normalized_centers:
                instance_heatmaps = generate_gaussian_heatmaps(
                    size_hw=self.heatmap_size,
                    centers_xy=normalized_centers,
                    sigma_ratio=self.sigma_ratio,
                    visibility=frame_visibility,
                )
                heatmaps.append(instance_heatmaps.amax(dim=0).cpu().numpy())
            else:
                heatmaps.append(np.zeros(self.heatmap_size, dtype=np.float32))

            original_points = [
                (
                    x_img * original_w / max(image_w, 1),
                    y_img * original_h / max(image_h, 1),
                )
                if vis > 0
                else (0.0, 0.0)
                for (x_img, y_img), vis in zip(
                    frame_coords,
                    frame_visibility,
                    strict=True,
                )
            ]
            padded_points = original_points + [(0.0, 0.0)] * (
                self.max_instances - len(original_points)
            )
            padded_visibility = frame_visibility + [0.0] * (
                self.max_instances - len(frame_visibility)
            )
            instance_coords_original.append(padded_points)
            instance_visibility.append(padded_visibility)
            primary_index = next(
                (idx for idx, vis in enumerate(frame_visibility) if vis > 0),
                None,
            )
            if primary_index is None:
                coords_original.append((0.0, 0.0))
                primary_visibility.append(0.0)
            else:
                coords_original.append(original_points[primary_index])
                primary_visibility.append(1.0)

        sample: BallDetectionSample = {
            "images": torch.from_numpy(np.stack(image_tensors)).to(torch.float32),
            "heatmaps": torch.from_numpy(np.stack(heatmaps)).to(torch.float32),
            "coords": torch.tensor(coords_original, dtype=torch.float32),
            "visibility": torch.tensor(primary_visibility, dtype=torch.float32),
            "instance_coords": torch.tensor(
                instance_coords_original,
                dtype=torch.float32,
            ),
            "instance_visibility": torch.tensor(
                instance_visibility,
                dtype=torch.float32,
            ),
            "original_size": torch.tensor([original_w, original_h], dtype=torch.float32),
            "heatmap_size": torch.tensor([heatmap_w, heatmap_h], dtype=torch.float32),
        }
        return sample

    def _resolve_split_file(self, split_file: Path) -> Path:
        if split_file.is_absolute():
            resolved = split_file
        else:
            candidates = [
                Path.cwd() / split_file,
                self.data_dir / split_file,
                self.data_dir / "splits" / split_file,
            ]
            resolved = next((candidate for candidate in candidates if candidate.exists()), split_file)
        if not resolved.exists():
            raise FileNotFoundError(f"Split file not found: {resolved}")
        return resolved

    def _build_windows(self) -> list[ClipWindow]:
        entries = self._read_split_entries()
        windows: list[ClipWindow] = []
        for entry in entries:
            for clip_dir in self._expand_entry(entry):
                label_path = clip_dir / "Label.csv"
                if not label_path.exists():
                    continue
                frame_names = tuple(sorted(path.name for path in clip_dir.glob("*.jpg")))
                if len(frame_names) < self.num_frames:
                    continue
                original_size = self._resolve_original_size(clip_dir, frame_names[0])
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

    def _build_pseudo_windows(self) -> list[ClipWindow]:
        windows: list[ClipWindow] = []
        for manifest_path in self.pseudo_manifest_paths:
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
                    original_width = int(entry["original_width"])
                    original_height = int(entry["original_height"])
                    if frame_count < self.num_frames or not accepted_starts:
                        continue
                    if not image_dir.exists():
                        raise FileNotFoundError(f"Pseudo image_dir not found: {image_dir}")
                    if not label_csv.exists():
                        raise FileNotFoundError(f"Pseudo label_csv not found: {label_csv}")
                    frame_names = tuple(
                        f"{frame_index:06d}.jpg" for frame_index in range(frame_count)
                    )
                    labels = self._read_label_csv(label_csv)
                    for start_index in accepted_starts:
                        if start_index < 0 or start_index + self.num_frames > frame_count:
                            continue
                        windows.append(
                            ClipWindow(
                                clip_dir=image_dir,
                                frame_names=frame_names,
                                labels=labels,
                                original_size=(original_width, original_height),
                                start_index=start_index,
                            )
                        )
        return windows

    def _read_split_entries(self) -> list[str]:
        with self.split_file.open("r", encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
        if not entries:
            raise RuntimeError(f"Split file is empty: {self.split_file}")
        return entries

    def _resolve_manifest_paths(self, manifest_paths: Sequence[str | Path]) -> list[Path]:
        resolved_paths: list[Path] = []
        for manifest_path in manifest_paths:
            candidate = Path(str(manifest_path)).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if not candidate.exists():
                raise FileNotFoundError(f"Pseudo manifest not found: {candidate}")
            resolved_paths.append(candidate)
        return resolved_paths

    def _expand_entry(self, entry: str) -> list[Path]:
        entry_path = self.data_dir / entry
        if not entry_path.exists():
            raise FileNotFoundError(
                f"Split entry '{entry}' does not exist under data_dir={self.data_dir}."
            )
        if entry_path.is_dir() and self._is_clip_dir(entry_path):
            return [entry_path]
        if entry_path.is_dir():
            clip_dirs = sorted(
                path for path in entry_path.iterdir() if path.is_dir() and self._is_clip_dir(path)
            )
            if clip_dirs:
                return clip_dirs
        raise RuntimeError(
            "Split entries must point to either a game directory containing Clip* subdirectories "
            f"or a specific clip directory. Invalid entry: {entry}"
        )

    @staticmethod
    def _is_clip_dir(path: Path) -> bool:
        """Return whether a directory follows a supported clip naming convention."""
        return path.name.startswith("Clip") or path.name.startswith("clip_")

    def _resolve_original_size(self, clip_dir: Path, first_frame_name: str) -> tuple[int, int]:
        frame_path = clip_dir / first_frame_name
        with Image.open(frame_path) as image:
            width, height = image.size
        return width, height

    def _read_label_csv(self, path: Path) -> dict[str, tuple[FrameLabel, ...]]:
        labels: dict[str, list[FrameLabel]] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"file name", "visibility", "x-coordinate", "y-coordinate"}
            missing = required_fields.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required CSV columns in {path}: {sorted(missing)}")
            for row in reader:
                frame_name = str(row["file name"]).strip()
                visibility = float(row["visibility"] or 0)
                x_value = float(row["x-coordinate"] or 0)
                y_value = float(row["y-coordinate"] or 0)
                instance_id = str(row.get("instance id") or "").strip()
                state = str(row.get("ball state") or ("visible" if visibility > 0 else "absent"))
                role = str(row.get("role") or "target").strip() or "target"
                if not instance_id and visibility <= 0:
                    labels.setdefault(frame_name, [])
                    continue
                labels.setdefault(frame_name, []).append(
                    FrameLabel(
                        visibility=visibility,
                        x=x_value,
                        y=y_value,
                        instance_id=instance_id or "b001",
                        role=role,
                        state=state,
                    )
                )
        return {
            frame_name: tuple(frame_labels)
            for frame_name, frame_labels in labels.items()
        }

    def _load_frame(self, path: Path) -> np.ndarray:
        image_h, image_w = self.image_size
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        image = cv2.resize(image, (image_w, image_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _to_normalized_xy(
        *,
        x_img: float,
        y_img: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        x_norm = 0.0 if width <= 1 else x_img / float(width - 1)
        y_norm = 0.0 if height <= 1 else y_img / float(height - 1)
        return x_norm, y_norm

    @staticmethod
    def _parse_size(value: Any, *, name: str) -> tuple[int, int]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name} must be a list or tuple with length 2.")
        return int(value[0]), int(value[1])
