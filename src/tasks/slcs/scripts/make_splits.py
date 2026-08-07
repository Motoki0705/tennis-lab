"""
Generate the reproducible recording-level train/val/test split file for an
issue #634 dataset.

Usage:
    python -m src.tasks.slcs.scripts.make_splits
    python -m src.tasks.slcs.scripts.make_splits data.dataset_root=tennis_scene_dataset
    python -m src.tasks.slcs.scripts.make_splits splits.val_ratio=0.2 splits.seed=1

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/make_splits.yaml`.
    - Dataset and split paths are relative to `paths.data_root`.
    - The split unit is recording_id (clips of one recording never straddle
      splits); assignment is deterministic in (seed, ratios, recordings).
    - Refuses to overwrite an existing split file unless splits.overwrite=true,
      because retraining against a silently changed split invalidates
      comparisons.
"""

from __future__ import annotations

from collections import Counter

from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSSplitConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.splits import (
    generate_overfit_splits,
    generate_recording_splits,
    save_split_file,
)
from src.tennis_scene.generate_dataset.manifest import DatasetManifestError
from src.utils.hydra import hydra_main


def run(config: DictConfig) -> None:
    """Generate and save the split file."""
    runtime = SLCSSplitConfig.from_config(config)
    split_file = runtime.data.split_file
    if split_file.exists() and not runtime.overwrite:
        raise DatasetManifestError(
            f"split file already exists: {split_file}. Set splits.overwrite=true "
            "to regenerate (this changes train/val/test membership)."
        )

    index = SLCSDataIndex.load(runtime.data.dataset_root)
    assignments = (
        generate_overfit_splits(index)
        if runtime.overfit
        else generate_recording_splits(
            index,
            val_ratio=runtime.val_ratio,
            test_ratio=runtime.test_ratio,
            seed=runtime.seed,
        )
    )
    val_ratio = 0.0 if runtime.overfit else runtime.val_ratio
    test_ratio = 0.0 if runtime.overfit else runtime.test_ratio
    save_split_file(
        split_file,
        assignments,
        seed=runtime.seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    counts = Counter(assignments.values())
    print(f"wrote {split_file}: {dict(counts)} over {len(assignments)} recordings")


@hydra_main(
    config_path="../configs",
    config_name="make_splits",
    version_base="1.3",
    validation_boundary="slcs.make_splits",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for split generation."""
    run(config)


if __name__ == "__main__":
    main()
