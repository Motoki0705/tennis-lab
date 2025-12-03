#!/usr/bin/env python
"""Generate PLCS training dataset.

This script generates training data by simulating player motion on tennis court
and projecting to multiple camera views.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.plcs.data.motion_sampler import MotionSampler
from src.plcs.data.scene_generator import SceneGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PLCS training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/plcs"),
        help="Output directory for generated scenes",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="Number of scenes to generate",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=15,
        help="Number of cameras to sample per scene",
    )
    parser.add_argument(
        "--human-visibility-threshold",
        type=float,
        default=0.8,
        help="Minimum human visibility ratio for camera filtering",
    )
    parser.add_argument(
        "--court-visibility-threshold",
        type=int,
        default=15,
        help="Minimum court keypoints visible for camera filtering",
    )
    parser.add_argument(
        "--smplh-model-path",
        type=Path,
        default=Path("data/smplx/smplh"),
        help="Path to SMPL-H model directory",
    )
    parser.add_argument(
        "--motion-source-dir",
        type=Path,
        default=Path("data/ACCAD"),
        help="Root directory for motion data",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Specific motion category to sample from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )

    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build config
    config = OmegaConf.create(
        {
            "simulation": {
                "num_scenes": args.num_scenes,
                "num_cameras": args.num_cameras,
                "human_visibility_threshold": args.human_visibility_threshold,
                "court_visibility_threshold": args.court_visibility_threshold,
                "output_dir": str(args.output_dir),
            },
            "camera": {
                "z_min": 3.0,
                "z_max": 5.0,
                "hfov_deg": 60.0,
                "image_size": [1280, 720],
            },
            "motion_sources": {
                "default": {
                    "paths": [str(args.motion_source_dir)],
                    "weight": 1.0,
                },
            },
            "smplh_model_path": str(args.smplh_model_path),
        }
    )

    # Load config file if provided
    if args.config:
        file_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, file_config)

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Create output directory
    output_dir = Path(config.simulation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print(f"\nInitializing motion sampler from {args.motion_source_dir}...")
    motion_sampler = MotionSampler(
        config=config,
        smplh_model_path=config.smplh_model_path,
        device=args.device,
    )

    print(f"Available categories: {motion_sampler.get_available_categories()}")

    print("\nInitializing scene generator...")
    scene_generator = SceneGenerator(
        config=config,
        motion_sampler=motion_sampler,
        device=args.device,
    )

    # Generate scenes
    num_scenes = config.simulation.num_scenes
    print(f"\nGenerating {num_scenes} scenes...")

    successful = 0
    failed = 0
    total_cameras = 0

    stats = {
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
                category=args.category,
            )

            # Check if we have valid cameras after filtering
            if len(scene.cameras) == 0:
                print(f"\nWarning: Scene {scene_id} has no valid cameras, skipping...")
                failed += 1
                continue

            # Save scene (NPZ only)
            output_path = output_dir / "scenes" / f"{scene_id}.npz"
            scene_generator.save_scene(scene, output_path)

            # Collect scene metadata for dataset meta.json
            scene_meta = scene_generator.build_scene_meta(scene)
            scene_meta["file"] = f"{scene_id}.npz"
            scenes_meta.append(scene_meta)

            # Update stats
            successful += 1
            total_cameras += len(scene.cameras)
            stats["total_frames"] += scene.meta["num_frames"]
            stats["cameras_per_scene"].append(len(scene.cameras))

            cat = scene.meta["motion_category"]
            if cat not in stats["categories"]:
                stats["categories"][cat] = 0
            stats["categories"][cat] += 1

        except Exception as e:
            print(f"\nError generating scene {i}: {e}")
            failed += 1
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Successful scenes: {successful}")
    print(f"Failed scenes: {failed}")
    print(f"Total cameras: {total_cameras}")
    print(f"Average cameras per scene: {total_cameras / max(1, successful):.1f}")
    print(f"Total frames: {stats['total_frames']}")
    print("\nCategory distribution:")
    for cat, count in sorted(stats["categories"].items()):
        print(f"  {cat}: {count}")
    print(f"\nOutput saved to: {output_dir}")

    # Save unified meta.json for the entire dataset
    meta_path = output_dir / "meta.json"
    dataset_meta = {
        "generated_at": datetime.now().isoformat(),
        "num_scenes": successful,
        "failed_scenes": failed,
        "total_cameras": total_cameras,
        "total_frames": stats["total_frames"],
        "avg_cameras_per_scene": total_cameras / max(1, successful),
        "categories": dict(stats["categories"]),
        "config": {
            "simulation": OmegaConf.to_container(config.get("simulation", {})),
            "camera": OmegaConf.to_container(config.get("camera", {})),
            "smplh_model_path": config.get("smplh_model_path", ""),
        },
        "scenes": scenes_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)
    print(f"Dataset metadata saved to: {meta_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
