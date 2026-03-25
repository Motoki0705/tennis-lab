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
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.argumentation import (
    BallDetectionArgumentation,
    make_sample_rng,
)
from src.tasks.ball_detection.data.types import BallDetectionSample

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


@dataclass(frozen=True)
class ClipWindow:
    """One fixed-length temporal training window.

    Attributes:
        clip_dir: Directory containing frame images and ``Label.csv``.
        frame_names: Ordered frame names available in the clip.
        labels: Per-frame labels keyed by frame name.
        original_size: Original frame size in ``(width, height)`` ordering.
        start_index: Inclusive start index of the temporal window.
    """

    clip_dir: Path
    frame_names: tuple[str, ...]
    labels: dict[str, FrameLabel]
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
        self.gaussian_size = int(data_cfg.get("gaussian_size", 3))
        self.gaussian_variance = float(data_cfg.get("gaussian_variance", 1.0))

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sample_stride <= 0:
            raise ValueError("data.sample_stride must be positive.")
        if self.gaussian_size < 0:
            raise ValueError("data.gaussian_size must be non-negative.")
        if self.gaussian_variance <= 0:
            raise ValueError("data.gaussian_variance must be positive.")

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
        coords_image: list[tuple[float, float]] = []
        visibility: list[float] = []

        for offset in range(self.num_frames):
            frame_name = window.frame_names[window.start_index + offset]
            frame_path = window.clip_dir / frame_name
            frames_hwc.append(self._load_frame(frame_path))

            label = window.labels.get(frame_name, FrameLabel(visibility=0.0, x=0.0, y=0.0))
            if label.visibility > 0:
                x_img = label.x * image_w / max(original_w, 1)
                y_img = label.y * image_h / max(original_h, 1)
                coords_image.append((x_img, y_img))
                visibility.append(1.0)
            else:
                coords_image.append((0.0, 0.0))
                visibility.append(0.0)

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
        for frame, (x_img, y_img), vis in zip(frames_hwc, coords_image, visibility):
            image_tensors.append(np.transpose(frame, (2, 0, 1)))
            if vis > 0:
                x_hm = x_img * heatmap_w / image_w
                y_hm = y_img * heatmap_h / image_h
                heatmaps.append(
                    self._generate_heatmap(
                        height=heatmap_h,
                        width=heatmap_w,
                        center_x=x_hm,
                        center_y=y_hm,
                    )
                )
                coords_original.append(
                    (
                        x_img * original_w / max(image_w, 1),
                        y_img * original_h / max(image_h, 1),
                    )
                )
            else:
                heatmaps.append(np.zeros((heatmap_h, heatmap_w), dtype=np.float32))
                coords_original.append((0.0, 0.0))

        sample: BallDetectionSample = {
            "images": torch.from_numpy(np.stack(image_tensors)).to(torch.float32),
            "heatmaps": torch.from_numpy(np.stack(heatmaps)).to(torch.float32),
            "coords": torch.tensor(coords_original, dtype=torch.float32),
            "visibility": torch.tensor(visibility, dtype=torch.float32),
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
        if entry_path.is_dir() and entry_path.name.startswith("Clip"):
            return [entry_path]
        if entry_path.is_dir():
            clip_dirs = sorted(
                path for path in entry_path.iterdir() if path.is_dir() and path.name.startswith("Clip")
            )
            if clip_dirs:
                return clip_dirs
        raise RuntimeError(
            "Split entries must point to either a game directory containing Clip* subdirectories "
            f"or a specific clip directory. Invalid entry: {entry}"
        )

    def _resolve_original_size(self, clip_dir: Path, first_frame_name: str) -> tuple[int, int]:
        frame_path = clip_dir / first_frame_name
        with Image.open(frame_path) as image:
            width, height = image.size
        return width, height

    def _read_label_csv(self, path: Path) -> dict[str, FrameLabel]:
        labels: dict[str, FrameLabel] = {}
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
                labels[frame_name] = FrameLabel(visibility=visibility, x=x_value, y=y_value)
        return labels

    def _load_frame(self, path: Path) -> np.ndarray:
        image_h, image_w = self.image_size
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        image = cv2.resize(image, (image_w, image_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _generate_heatmap(
        self,
        *,
        height: int,
        width: int,
        center_x: float,
        center_y: float,
    ) -> np.ndarray:
        heatmap = np.zeros((height, width), dtype=np.float32)
        cx_int = int(round(center_x))
        cy_int = int(round(center_y))
        if cx_int < 0 or cy_int < 0 or cx_int >= width or cy_int >= height:
            return heatmap

        size = self.gaussian_size
        yy, xx = np.mgrid[-size : size + 1, -size : size + 1]
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * self.gaussian_variance)).astype(np.float32)
        kernel /= max(float(kernel.max()), 1e-8)

        y_start = max(0, cy_int - size)
        y_end = min(height, cy_int + size + 1)
        x_start = max(0, cx_int - size)
        x_end = min(width, cx_int + size + 1)

        kernel_y_start = size - (cy_int - y_start)
        kernel_y_end = kernel_y_start + (y_end - y_start)
        kernel_x_start = size - (cx_int - x_start)
        kernel_x_end = kernel_x_start + (x_end - x_start)
        heatmap[y_start:y_end, x_start:x_end] = kernel[
            kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
        ]
        return heatmap

    @staticmethod
    def _parse_size(value: Any, *, name: str) -> tuple[int, int]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name} must be a list or tuple with length 2.")
        return int(value[0]), int(value[1])
