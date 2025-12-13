"""Generate PLCS training dataset using Hydra configurations."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from src.plcs.configs import SimulationConfig, register_configs
from src.plcs.data.motion_sampler import MotionSampler
from src.plcs.data.scene_generator import SceneGenerator

register_configs()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_paths(cfg: SimulationConfig) -> SimulationConfig:
    cfg.device = _resolve_device(cfg.device)
    cfg.smplh_model_path = to_absolute_path(cfg.smplh_model_path)
    cfg.simulation.output_dir = to_absolute_path(cfg.simulation.output_dir)

    for source in cfg.motion_sources.values():
        source.paths = [to_absolute_path(p) for p in source.paths]

    return cfg


@hydra.main(version_base=None, config_name="plcs_simulation")
def main(cfg: SimulationConfig) -> int:  # pragma: no cover - CLI entry
    """Main function."""

    cfg = _prepare_paths(cfg)

    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenes").mkdir(parents=True, exist_ok=True)

    # Save resolved config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Initialize components
    print("\nInitializing motion sampler...")
    motion_sampler = MotionSampler(
        config=cfg,
        smplh_model_path=cfg.smplh_model_path,
        device=cfg.device,
    )

    print(f"Available categories: {motion_sampler.get_available_categories()}")

    print("\nInitializing scene generator...")
    scene_generator = SceneGenerator(
        config=cfg,
        motion_sampler=motion_sampler,
        device=cfg.device,
    )

    # Generate scenes
    num_scenes = cfg.simulation.num_scenes
    print(f"\nGenerating {num_scenes} scenes...")

    successful = 0
    failed = 0
    total_cameras = 0

    stats: Dict[str, object] = {
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
                category=cfg.category,
            )

            # Check if we have valid cameras after filtering
            if len(scene.cameras) == 0:
                print(f"\nWarning: Scene {scene_id} has no valid cameras, skipping...")
                failed += 1
                continue

            # Save scene (NPZ only)
            output_path = output_dir / "scenes" / f"{scene_id}.npz"
            scene_generator.save_scene(scene, output_path)

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

    print("\nGeneration complete!")
    print(f"  Successful scenes: {successful}")
    print(f"  Failed scenes:     {failed}")
    print(f"  Avg cameras/scene: {stats['avg_cameras']:.2f}")
    print(f"  Stats saved to:    {stats_path}")
    print(f"  Metadata saved to: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
