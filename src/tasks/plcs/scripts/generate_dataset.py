"""Generate PLCS scenes for training with Hydra-managed configuration.

Usage:
    python -m src.tasks.plcs.scripts.generate_dataset
    python -m src.tasks.plcs.scripts.generate_dataset run.output_dir=data/plcs simulation.num_scenes=10
    python -m src.tasks.plcs.scripts.generate_dataset run.device=cpu run.num_workers=4

Notes:
    - Configuration is loaded from `src/tasks/plcs/configs/generate_dataset.yaml`.
    - The script uses Hydra for configuration loading.
    - Parallel scene generation uses worker processes for scene synthesis only.
    - `generation` changes only object cardinality; both modes use the same simulator and writer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)
from src.utils.device import resolve_device
from src.utils.hydra import hydra_main
from src.utils.io import save_json
from src.utils.seeding import seed_everything


def _prepare_paths(cfg: DictConfig) -> DictConfig:
    cfg.run.device = str(resolve_device(cfg.run.device))
    cfg.paths.smplh_model_path = to_absolute_path(str(cfg.paths.smplh_model_path))
    cfg.run.output_dir = to_absolute_path(str(cfg.run.output_dir))

    for _category, source in cfg.motion_sources.items():
        if source is None:
            continue
        source.paths = [to_absolute_path(str(p)) for p in source.paths]

    return cfg


@hydra_main(config_path="../configs", config_name="generate_dataset", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Generate scenes and write them to disk."""
    cfg = _prepare_paths(cfg)
    device = str(cfg.run.device)

    # Set random seeds
    seed = int(cfg.run.seed)
    seed_everything(seed)

    # Create output directory
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    generation_mode = str(cfg.generation.mode)
    if generation_mode not in {"single_object", "multi_object"}:
        raise ValueError(
            f"Unsupported generation.mode='{generation_mode}'. "
            "Supported: ['single_object', 'multi_object']"
        )

    # Initialize components
    writer = PLCSDatasetWriter(output_dir)

    # Generate scenes
    num_scenes = int(cfg.simulation.num_scenes)
    num_workers = int(cfg.run.get("num_workers", 1))
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel PLCS dataset generation requires run.device=cpu when "
            f"run.num_workers={num_workers}"
        )

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

    writer.save_meta_json(config=OmegaConf.to_container(cfg, resolve=True))
    writer.save_split_info(
        train_ratio=float(cfg.run.get("train_ratio", 0.8)),
        val_ratio=float(cfg.run.get("val_ratio", 0.1)),
        test_ratio=float(cfg.run.get("test_ratio", 0.1)),
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
