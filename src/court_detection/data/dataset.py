"""Dataset for court keypoint detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.schema.keypoint_schema import NUM_COURT_KP

NUM_KEYPOINTS = NUM_COURT_KP  # CourtKP20 specification


class CourtKeypointDataset(Dataset):
    """Dataset for court keypoint detection.

    Loads images and their corresponding keypoint annotations.
    Supports manual annotations (json format) created by the annotation tool.

    Args:
        data_dir: Path to data directory.
        split: Data split ('train', 'val', 'test').
        input_size: Input image size [H, W].
        heatmap_size: Output heatmap size [H, W].
        transform: Optional image transform.
        augmentation: Augmentation config dict.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        input_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        transform: Any = None,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.transform = transform
        self.augmentation = augmentation or {}

        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load sample list from data directory."""
        samples = []

        # Check for manual annotations (json files)
        json_files = sorted(self.data_dir.glob("*_keypoints.json"))
        if json_files:
            n_files = len(json_files)
            if self.split == "train":
                files = json_files[: int(n_files * 0.8)]
            elif self.split == "val":
                files = json_files[int(n_files * 0.8) : int(n_files * 0.9)]
            else:
                files = json_files[int(n_files * 0.9) :]

            for json_file in files:
                samples.append({"path": json_file})
            return samples

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        return self._load_json_sample(sample["path"])

    def _load_json_sample(self, path: Path) -> dict[str, Tensor]:
        """Load a manually annotated sample from json file."""
        with open(path) as f:
            data = json.load(f)

        # Load image
        image_path = Path(data["image_path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.input_size[1], self.input_size[0]))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        # Load keypoints
        keypoints = torch.zeros(NUM_KEYPOINTS, 2)
        visibility = torch.zeros(NUM_KEYPOINTS)

        for i, kp in enumerate(data.get("keypoints", [])[:NUM_KEYPOINTS]):
            if kp["visibility"] > 0:
                # Normalize coordinates to [0, 1]
                keypoints[i, 0] = kp["x"] / orig_w
                keypoints[i, 1] = kp["y"] / orig_h
                visibility[i] = 1.0

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints, visibility)

        return {
            "image": image,
            "keypoints": keypoints,
            "visibility": visibility,
            "heatmaps": heatmaps,
        }

    def _generate_heatmaps(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        sigma: float = 2.0,
    ) -> Tensor:
        """Generate Gaussian heatmaps from keypoint coordinates.

        Args:
            keypoints: Normalized keypoint coordinates (K, 2).
            visibility: Visibility flags (K,).
            sigma: Gaussian sigma in heatmap pixels.

        Returns:
            Heatmaps of shape (K, Hm, Wm).
        """
        K = keypoints.shape[0]
        H, W = self.heatmap_size

        heatmaps = torch.zeros(K, H, W)

        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32)
        x_coords = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        for k in range(K):
            if visibility[k] > 0:
                # Convert normalized coords to heatmap coords
                cx = keypoints[k, 0] * W
                cy = keypoints[k, 1] * H

                # Generate Gaussian
                heatmap = torch.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
                )
                heatmaps[k] = heatmap

        return heatmaps
