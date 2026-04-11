"""Generate PLCS scenes from AthletePose3D 3D-pose sequences.

This script creates synthetic training data for the PLCS (Player Localisation
in Court System) model using pre-computed 3D pose sequences from the
AthletePose3D MoCap dataset, bypassing the SMPL-H body model entirely.

Usage:
    python -m src.tasks.plcs.scripts.generate_dataset_athlete_pose
    python -m src.tasks.plcs.scripts.generate_dataset_athlete_pose simulation.num_scenes=10

Notes:
    - Configuration is loaded from ``src/tasks/plcs/configs/generate_dataset_athlete_pose.yaml``.
    - AthletePose3D data must be extracted under ``data/AthletePose3D/pose_3d_v3/``.
    - No SMPL-H model files are required for this pipeline.
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

from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.sampling.athlete_pose_sampler import (
    AthletePose3DSampler,
)
from src.tasks.plcs.generate_dataset.scene_generator import SceneGenerator


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prepare_paths(cfg: DictConfig) -> DictConfig:
    cfg.run.device = _resolve_device(str(cfg.run.device))
    cfg.run.output_dir = to_absolute_path(str(cfg.run.output_dir))
    cfg.athlete_pose.data_dir = to_absolute_path(str(cfg.athlete_pose.data_dir))
    return cfg


@hydra.main(
    config_path="../configs",
    config_name="generate_dataset_athlete_pose",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Generate scenes from AthletePose3D and write them to disk."""
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

    # --- Initialise AthletePose3D sampler (no SMPL-H) ---
    print("\nInitializing AthletePose3D sampler...")
    sampler = AthletePose3DSampler(config=cfg)
    print(f"  {len(sampler)} sequences available ({sampler.split} split)")

    # --- We still need a SceneGenerator for camera logic.
    #     Build one *without* a motion_sampler (we won't call generate_scene). ---
    print("\nInitializing scene generator (camera-only)...")
    # Create a lightweight SceneGenerator; pass a dummy motion_sampler.
    # We only use its camera_projector, court_kp_3d, _sample_initial_pose, etc.
    scene_generator = SceneGenerator.__new__(SceneGenerator)
    scene_generator.config = cfg
    scene_generator.device = torch.device(cfg.run.device)
    scene_generator.motion_sampler = None  # type: ignore[assignment]  # not used
    scene_generator.court_kp_3d = None

    sim_cfg = cfg.get("simulation", {})
    scene_generator.num_cameras = int(sim_cfg.get("num_cameras", 5))
    scene_generator.human_visibility_threshold = float(
        sim_cfg.get("human_visibility_threshold", 0.8)
    )
    scene_generator.court_visibility_threshold = float(
        sim_cfg.get("court_visibility_threshold", 15)
    )

    from src.utils.projection.camera_projector import CameraConfig, CameraProjector

    cam_cfg = cfg.get("camera", {})
    camera_config = CameraConfig(
        z_min=float(cam_cfg.get("z_min", 3.0)),
        z_max=float(cam_cfg.get("z_max", 5.0)),
        hfov_deg=float(cam_cfg.get("hfov_deg", 60.0)),
        image_size=tuple(cam_cfg.get("image_size", [1280, 720])),
        target_x_range=tuple(cam_cfg.get("target_x_range", [-2.0, 2.0])),
        target_y_range=tuple(cam_cfg.get("target_y_range", [-2.0, 2.0])),
        target_z_range=tuple(cam_cfg.get("target_z_range", [0.5, 1.5])),
    )
    scene_generator.camera_projector = CameraProjector(camera_config)
    scene_generator.image_size = scene_generator.camera_projector.config.image_size

    writer = PLCSDatasetWriter(output_dir)

    # --- Generate scenes ---
    num_scenes = int(cfg.simulation.num_scenes)
    print(f"\nGenerating {num_scenes} scenes from AthletePose3D...")

    successful = 0
    failed = 0
    total_cameras = 0
    total_frames = 0
    cameras_per_scene: list[int] = []
    scenes_meta: list[dict[str, object]] = []

    for i in tqdm(range(num_scenes), desc="Generating scenes"):
        try:
            scene_id = f"scene_{i:06d}"

            motion = sampler.sample()
            scene = scene_generator.generate_scene_from_athlete_pose(
                motion,
                scene_id=scene_id,
            )

            if len(scene.cameras) == 0:
                print(f"\nWarning: Scene {scene_id} has no valid cameras, skipping...")
                failed += 1
                continue

            writer.save_scene(scene)

            successful += 1
            total_cameras += len(scene.cameras)
            total_frames += scene.meta["num_frames"]
            cameras_per_scene.append(len(scene.cameras))
            scenes_meta.append(scene.meta)

        except Exception as exc:  # pragma: no cover - logging only
            print(f"\nError generating scene {i}: {exc}")
            failed += 1
            continue

    # --- Save statistics ---
    avg_cameras = total_cameras / successful if successful else 0.0
    stats: dict[str, object] = {
        "successful_scenes": successful,
        "failed_scenes": failed,
        "total_frames": total_frames,
        "cameras_per_scene": cameras_per_scene,
        "avg_cameras": avg_cameras,
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "scenes_meta.json", "w") as f:
        json.dump(scenes_meta, f, indent=2, default=str)

    resolved_cfg: dict[str, object] | None = OmegaConf.to_container(  # type: ignore[assignment]
        cfg, resolve=True,
    )
    writer.save_meta_json(config=resolved_cfg)
    writer.save_dataset_info(
        {
            "total_cameras": total_cameras,
            "avg_cameras_per_scene": stats["avg_cameras"],
            "human_visibility_threshold": scene_generator.human_visibility_threshold,
            "court_visibility_threshold": scene_generator.court_visibility_threshold,
        }
    )
    writer.save_split_info(
        train_ratio=float(cfg.run.get("train_ratio", 0.8)),
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Successful scenes: {successful}")
    print(f"  Failed scenes:     {failed}")
    print(f"  Total cameras:     {total_cameras}")
    if successful:
        print(f"  Avg cameras/scene: {total_cameras / successful:.1f}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'=' * 60}")

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
