"""
Generate the reproducible recording-level train/val/test split file for an
issue #634 dataset.

Usage:
    python -m src.tasks.slcs.scripts.make_splits
    python -m src.tasks.slcs.scripts.make_splits data.dataset_root=data/tennis_scene_dataset
    python -m src.tasks.slcs.scripts.make_splits splits.val_ratio=0.2 splits.seed=1

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/make_splits.yaml`.
    - The split unit is recording_id (clips of one recording never straddle
      splits); assignment is deterministic in (seed, ratios, recordings).
    - Refuses to overwrite an existing split file unless splits.overwrite=true,
      because retraining against a silently changed split invalidates
      comparisons.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from omegaconf import DictConfig

from src.tasks.slcs.data.contract import DatasetContractError, DatasetIndex
from src.tasks.slcs.data.splits import generate_recording_splits, save_split_file
from src.utils.hydra import hydra_main


def run(config: DictConfig) -> None:
    """Generate and save the split file."""
    dataset_root = str(config.data.dataset_root)
    split_cfg = config.splits
    split_file = Path(str(config.data.split_file))
    if split_file.exists() and not bool(split_cfg.overwrite):
        raise DatasetContractError(
            f"split file already exists: {split_file}. Set splits.overwrite=true "
            "to regenerate (this changes train/val/test membership)."
        )

    index = DatasetIndex.load(dataset_root)
    assignments = generate_recording_splits(
        index,
        val_ratio=float(split_cfg.val_ratio),
        test_ratio=float(split_cfg.test_ratio),
        seed=int(split_cfg.seed),
    )
    save_split_file(
        split_file,
        assignments,
        seed=int(split_cfg.seed),
        val_ratio=float(split_cfg.val_ratio),
        test_ratio=float(split_cfg.test_ratio),
    )
    counts = Counter(assignments.values())
    print(f"wrote {split_file}: {dict(counts)} over {len(assignments)} recordings")


@hydra_main(config_path="../configs", config_name="make_splits", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for split generation."""
    run(config)


if __name__ == "__main__":
    main()  # type: ignore[call-arg, unused-ignore]
