"""Base dataset writer for NPZ-based datasets.

This module provides a common base class for dataset writers used in PLCS and BLCS,
reducing code duplication and ensuring consistency across dataset generation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDatasetWriter(ABC):
    """Base class for NPZ dataset writers (PLCS/BLCS unified format).

    Provides common functionality for:
    - Directory management
    - Train/val/test split generation
    - Metadata JSON generation
    - Dataset statistics tracking

    Subclasses must implement save_scene() for module-specific data serialization.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize dataset writer.

        Args:
            output_dir: Output directory for dataset.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.scenes_dir = self.output_dir / "scenes"
        self.scenes_dir.mkdir(exist_ok=True)

        # Track scenes for meta.json
        self.scene_records: list[dict] = []
        self.scene_counter = 0

    @abstractmethod
    def save_scene(self, scene_data) -> Path:  # type: ignore[no-untyped-def]
        """Save a single scene to npz file.

        This method must be implemented by subclasses to handle module-specific
        data structures (PLCS vs BLCS).

        Args:
            scene_data: Scene data to save (module-specific type).

        Returns:
            Path to saved NPZ file.

        """
        pass

    def save_split_info(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Save train/val/test split information.

        Creates split files (train.txt, val.txt, test.txt) and split_info.json
        with metadata about the splits.

        Args:
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for testing set.
            seed: Random seed for reproducibility.

        """
        import random

        # Get all scene files
        scene_files = [r["file"] for r in self.scene_records]

        # Shuffle with seed
        random.seed(seed)
        random.shuffle(scene_files)

        # Calculate split sizes
        n_total = len(scene_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Create splits
        splits = {
            "train": scene_files[:n_train],
            "val": scene_files[n_train : n_train + n_val],
            "test": scene_files[n_train + n_val :],
        }

        # Save split files
        for split_name, filenames in splits.items():
            split_file = self.output_dir / f"{split_name}.txt"
            with open(split_file, "w") as f:
                for filename in filenames:
                    f.write(f"{filename}\n")

            logger.info("Saved %s split: %s scenes", split_name, len(filenames))

        # Save split info as JSON
        split_info = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "n_scenes": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            },
        }

        with open(self.output_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    def save_meta_json(self, config: dict | None = None) -> None:
        """Save meta.json with all scene information.

        Args:
            config: Generator configuration (optional).

        """
        # Calculate statistics
        total_cameras = sum(r["num_cameras"] for r in self.scene_records)
        avg_cameras = (
            total_cameras / len(self.scene_records) if self.scene_records else 0
        )

        meta = {
            "generated_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {
                "total_scenes": len(self.scene_records),
                "total_cameras": total_cameras,
                "avg_cameras_per_scene": avg_cameras,
            },
            "scenes": self.scene_records,
        }

        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "meta.json saved: %s scenes, %s cameras",
            len(self.scene_records),
            total_cameras,
        )
