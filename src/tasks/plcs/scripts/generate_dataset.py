"""Generate PLCS scenes for training with Hydra-managed configuration.

Usage:
    .venv/bin/python -m src.tasks.plcs.scripts.generate_dataset
    .venv/bin/python -m src.tasks.plcs.scripts.generate_dataset generation=multi_object run.output_dir=plcs/multi_object
    .venv/bin/python -m src.tasks.plcs.scripts.generate_dataset camera=broadcast run.output_dir=plcs/single_object_broadcast
    .venv/bin/python -m src.tasks.plcs.scripts.generate_dataset generation=multi_object camera=broadcast run.output_dir=plcs/multi_object_broadcast

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/generate_dataset.yaml`.
    - The script uses Hydra for configuration loading.
    - The default output is `data/plcs/single_object`; overrides are relative to `paths.data_root`.
    - Parallel scene generation uses worker processes for scene synthesis only.
    - `generation` changes only object cardinality; both modes use the same simulator and writer.
"""

from __future__ import annotations

import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)
from src.utils.hydra import hydra_main
from src.utils.io import save_json
from src.utils.seeding import seed_everything


@hydra_main(
    config_path="../configs",
    config_name="generate_dataset",
    version_base="1.3",
    validation_boundary="plcs.generate_dataset",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Generate scenes and write them to disk."""
    runtime = PLCSGenerationConfig.from_config(cfg)
    cfg = runtime.config
    device = runtime.device

    # Set random seeds
    seed = runtime.seed
    seed_everything(seed)

    # Create output directory
    output_dir = runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Initialize components
    writer = PLCSDatasetWriter(
        output_dir,
        court_keypoint_contract=runtime.court_keypoint_contract,
    )
    resolved_meta = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_meta, dict):
        raise TypeError("Resolved generation metadata must be a mapping.")
    writer.save_meta_json(config=resolved_meta)

    # Generate scenes
    num_scenes = runtime.num_scenes
    num_workers = runtime.num_workers
    results = generate_parallel_scenes(
        config=cfg,
        device=device,
        start_index=0,
        num_scenes=num_scenes,
        num_workers=num_workers,
    )

    print(f"\nGenerating {num_scenes} scenes...")
    print(f"Scene generation mode: parallel (workers={num_workers})")

    successful = 0
    failed = 0
    total_cameras = 0

    stats: dict[str, Any] = {
        "categories": {},
        "total_frames": 0,
        "cameras_per_scene": [],
    }
    scenes_meta: list[dict] = []

    for scene in tqdm(results, desc="Generating scenes", total=num_scenes):
        # Save scene
        writer.save_scene(scene)

        # Update statistics
        successful += 1
        total_cameras += len(scene.cameras)
        stats["total_frames"] += scene.meta["num_frames"]
        stats["cameras_per_scene"].append(len(scene.cameras))

        category = scene.meta.get("motion_category", "unknown")
        stats["categories"].setdefault(category, 0)
        stats["categories"][category] += 1

        scenes_meta.append(scene.meta)

    # Save statistics
    stats["successful_scenes"] = successful
    stats["failed_scenes"] = failed
    stats["avg_cameras"] = total_cameras / successful if successful > 0 else 0

    stats_path = output_dir / "stats.json"
    save_json(stats, stats_path)

    meta_path = output_dir / "scenes_meta.json"
    save_json(scenes_meta, meta_path, default=str)

    writer.save_meta_json(config=resolved_meta)
    writer.save_split_info(
        train_ratio=runtime.train_ratio,
        val_ratio=runtime.val_ratio,
        test_ratio=runtime.test_ratio,
        seed=seed,
    )

    print("\nGeneration complete!")
    print(f"  Successful scenes: {successful}")
    print(f"  Failed scenes:     {failed}")
    print(f"  Avg cameras/scene: {stats['avg_cameras']:.2f}")
    print(f"  Stats saved to:    {stats_path}")
    print(f"  Metadata saved to: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
