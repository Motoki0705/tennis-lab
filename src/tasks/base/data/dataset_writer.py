"""Base dataset writer for npy + json scene datasets.

This module provides a common base class for dataset writers used in PLCS and BLCS,
reducing code duplication and ensuring consistency across dataset generation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class BaseDatasetWriter(ABC):
    """Base class for scene dataset writers (PLCS/BLCS unified format).

    Each scene is saved as a directory containing npy array files and json
    metadata files.  Provides common functionality for:
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
        """Save a single scene as a directory with npy + json files.

        This method must be implemented by subclasses to handle module-specific
        data structures (PLCS vs BLCS).

        Args:
            scene_data: Scene data to save (module-specific type).

        Returns:
            Path to saved scene directory.

        """
        pass

    def _write_scene_files(
        self,
        scene_path: Path,
        scene_meta: Any,
        scalars: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Write meta.json, scalars.json, and each array as ``{key}.npy``.

        Args:
            scene_path: Scene output directory (already created).
            scene_meta: Scene metadata object exposing ``to_dict()``.
            scalars: Scalar values written to scalars.json.
            arrays: Mapping of array key to numpy array, written as npy files.
        """
        import numpy as np

        # Write meta.json
        with open(scene_path / "meta.json", "w") as f:
            json.dump(scene_meta.to_dict(), f, indent=2)

        # Write scalars.json
        with open(scene_path / "scalars.json", "w") as f:
            json.dump(scalars, f, indent=2)

        # Write array files
        for key, arr in arrays.items():
            np.save(scene_path / f"{key}.npy", arr)

    def _append_court_camera_arrays(
        self,
        arrays: dict[str, np.ndarray],
        scalars: dict[str, Any],
        cam: Any,
        prefix: str,
    ) -> None:
        """Append shared court camera params/arrays for a single camera.

        Writes the ``{prefix}params`` scalar plus the ``{prefix}court_kp_uv``,
        ``{prefix}court_kp_visible`` and ``{prefix}court_visibility_count``
        arrays using the dtypes shared by PLCS and BLCS.

        Args:
            arrays: Array accumulator mutated in place.
            scalars: Scalar accumulator mutated in place.
            cam: Camera record exposing ``camera_params``, ``court_kp_uv``,
                ``court_kp_visible`` and ``court_visibility_count``.
            prefix: Per-camera key prefix (e.g. ``"cam_0_"``).
        """
        import numpy as np

        scalars[f"{prefix}params"] = cam.camera_params
        arrays[f"{prefix}court_kp_uv"] = cam.court_kp_uv.astype(np.float32)
        arrays[f"{prefix}court_kp_visible"] = cam.court_kp_visible.astype(bool)
        arrays[f"{prefix}court_visibility_count"] = np.array(
            cam.court_visibility_count,
            dtype=np.float32,
        )

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
