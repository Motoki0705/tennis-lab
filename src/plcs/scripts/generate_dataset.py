"""Generate PLCS scenes for training using Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.generate_dataset`
    `uv run python -m src.plcs.scripts.generate_dataset run.output_dir=data/plcs simulation.num_scenes=10`

Config entry point: `src/plcs/configs/generate_dataset.yaml`
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.plcs.generate_dataset.scene_generator import SceneGenerator


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_paths(cfg: DictConfig) -> DictConfig:
    cfg.run.device = _resolve_device(str(cfg.run.device))
    cfg.paths.smplh_model_path = to_absolute_path(str(cfg.paths.smplh_model_path))
    cfg.run.output_dir = to_absolute_path(str(cfg.run.output_dir))

    for _category, source in cfg.motion_sources.items():
        if source is None:
            continue
        source.paths = [to_absolute_path(str(p)) for p in source.paths]

    return cfg


@hydra.main(config_path="../configs", config_name="generate_dataset", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Generate scenes and write them to disk."""
    cfg = _prepare_paths(cfg)

    # Set random seeds
    seed = int(cfg.run.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Initialize components
    print("\nInitializing motion sampler...")
    motion_sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg.paths.smplh_model_path,
        device=cfg.run.device,
    )

    print(f"Available categories: {motion_sampler.get_available_categories()}")

    print("\nInitializing scene generator...")
    scene_generator = SceneGenerator(
        config=cfg,
        motion_sampler=motion_sampler,
        device=cfg.run.device,
    )
    writer = PLCSDatasetWriter(output_dir)

    # Generate scenes
    num_scenes = int(cfg.simulation.num_scenes)
    print(f"\nGenerating {num_scenes} scenes...")

    successful = 0
    failed = 0
    total_cameras = 0

    stats: dict[str, object] = {
        "categories": {},
        "total_frames": 0,
        "cameras_per_scene": [],
    }
    scenes_meta: list[dict] = []

    for i in tqdm(range(num_scenes), desc="Generating scenes"):
        try:
            scene_id = f"scene_{i:06d}"

            # Generate scene
            scene = scene_generator.generate_scene(
                scene_id=scene_id,
                category=cfg.run.category,
            )

            # Check if we have valid cameras after filtering
            if len(scene.cameras) == 0:
                print(f"\nWarning: Scene {scene_id} has no valid cameras, skipping...")
                failed += 1
                continue

            # Save scene (NPZ only)
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

        except Exception as exc:  # pragma: no cover - logging only
            print(f"\nError generating scene {i}: {exc}")
            failed += 1
            continue

    # Save statistics
    stats["successful_scenes"] = successful
    stats["failed_scenes"] = failed
    stats["avg_cameras"] = total_cameras / successful if successful > 0 else 0

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    meta_path = output_dir / "scenes_meta.json"
    with open(meta_path, "w") as f:
        json.dump(scenes_meta, f, indent=2, default=str)

    writer.save_meta_json(config=OmegaConf.to_container(cfg, resolve=True))
    writer.save_dataset_info(
        {
            "total_cameras": total_cameras,
            "avg_cameras_per_scene": stats["avg_cameras"],
            "human_keypoint_visibility_threshold": (
                scene_generator.human_keypoint_visibility_threshold
            ),
            "human_visibility_threshold": scene_generator.human_visibility_threshold,
            "court_visibility_threshold": scene_generator.court_visibility_threshold,
        }
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
