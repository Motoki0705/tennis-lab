"""Self-supervised image dataset backed by a collection manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.tasks.dino_ssl.data.augmentation import (
    DataAugmentationDINO,
    MaskingGenerator,
)
from src.tasks.dino_ssl.generate_dataset.manifest import read_manifest


class SSLImageDataset(Dataset):
    """Yield multi-crop views and iBOT masks for each collected image."""

    def __init__(
        self,
        *,
        root: str | Path,
        augmentation: DataAugmentationDINO,
        masking: MaskingGenerator,
    ) -> None:
        manifest = read_manifest(Path(root))
        self.image_paths = manifest.image_paths()
        if not self.image_paths:
            raise RuntimeError(f"No images found in DINOv3 SSL manifest at {root!r}.")
        self.augmentation = augmentation
        self.masking = masking
        self.num_global_crops = 2

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        views = self.augmentation(image)
        masks = [self.masking() for _ in range(self.num_global_crops)]
        return {
            "global_crops": views["global_crops"],
            "local_crops": views["local_crops"],
            "masks": masks,
        }


def ssl_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate crop-major batches: each crop index becomes one stacked tensor."""
    num_global = len(batch[0]["global_crops"])
    num_local = len(batch[0]["local_crops"])

    global_crops = [
        torch.stack([sample["global_crops"][i] for sample in batch], dim=0)
        for i in range(num_global)
    ]
    local_crops = [
        torch.stack([sample["local_crops"][i] for sample in batch], dim=0)
        for i in range(num_local)
    ]
    masks = [
        torch.stack([sample["masks"][i] for sample in batch], dim=0)
        for i in range(num_global)
    ]
    return {
        "global_crops": global_crops,
        "local_crops": local_crops,
        "masks": masks,
    }


__all__ = ["SSLImageDataset", "ssl_collate_fn"]
