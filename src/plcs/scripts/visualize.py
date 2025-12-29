"""Visualize PLCS scenes and optionally run model predictions (Hydra-based).

Example commands:
    `uv run python -m src.plcs.scripts.visualize`
    `uv run python -m src.plcs.scripts.visualize visualization.scene_path=data/plcs/scenes/scene_000000.npz visualization.info=true`

Config entry point: `src/plcs/configs/visualize.yaml`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.plcs.generate_dataset.io.dataset_io import load_scene
from src.utils.rendering import PLCSSceneRenderer as SceneRenderer

if TYPE_CHECKING:
    from src.plcs.generate_dataset.scene_generator import SceneData



@dataclass
class RuntimeConfig:
    """Resolved configuration values for visualization workflows."""

    mode: str
    scene_path: Path
    frame: int
    view: str
    camera: int
    animation_view: str
    fps: float | None
    save: Path | None
    save_input: Path | None
    info: bool
    checkpoint: str | None
    device: str


# =============================================================================
# Helpers
# =============================================================================


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Convert Hydra config into runtime-friendly values."""
    vis = cfg.visualization

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        frame=int(vis.frame),
        view=str(vis.view),
        camera=int(vis.camera),
        animation_view=str(vis.animation_view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        save_input=Path(to_absolute_path(str(vis.save_input))) if vis.save_input else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        device=_resolve_device(str(vis.device)),
    )


def _require_checkpoint(cfg: RuntimeConfig) -> bool:
    if cfg.checkpoint is None:
        print("Error: checkpoint must be provided for prediction modes.")
        return False
    return True


# =============================================================================
# Core logic
# =============================================================================


def print_scene_info(scene: SceneData) -> None:
    """Print scene metadata and statistics."""
    meta = scene.meta
    print("=" * 60)
    print("Scene Information")
    print("=" * 60)
    print(f"Scene ID:        {meta['scene_id']}")
    print(f"Motion source:   {meta['motion_source']}")
    print(f"Category:        {meta['motion_category']}")
    print(f"Gender:          {meta['gender']}")
    print(f"FPS:             {meta['fps']}")
    print(f"Num frames:      {meta['num_frames']}")
    print(f"Duration:        {meta['num_frames'] / meta['fps']:.2f} seconds")
    print(
        f"Initial pos:     ({meta['initial_position'][0]:.2f}, {meta['initial_position'][1]:.2f})"
    )
    print(f"Initial yaw:     {np.degrees(meta['initial_yaw']):.1f}°")
    print(f"Cameras sampled: {meta['num_cameras_sampled']}")
    num_cameras = meta.get("num_cameras", meta.get("num_cameras_filtered", 0))
    print(f"Cameras kept:    {num_cameras}")
    print()
    print("Position statistics (normalized):")
    print(
        f"  X range: [{scene.position[:, 0].min():.3f}, {scene.position[:, 0].max():.3f}]"
    )
    print(
        f"  Y range: [{scene.position[:, 1].min():.3f}, {scene.position[:, 1].max():.3f}]"
    )
    print(
        f"  Z range: [{scene.position[:, 2].min():.3f}, {scene.position[:, 2].max():.3f}]"
    )
    print()
    print("Camera visibility:")
    for i, cam in enumerate(scene.cameras):
        print(
            f"  Camera {i}: Human {cam.human_visibility_ratio:.1%}, Court {cam.court_visibility_count:.1f}/20"
        )


def save_input_scene(scene: SceneData, cfg: RuntimeConfig) -> None:
    """Save 2D input scene animation (camera view)."""
    if cfg.save_input is None:
        return

    renderer = SceneRenderer()
    meta = getattr(scene, "meta", {})
    fps = cfg.fps or meta.get("fps", 30.0)

    print("Creating 2D input scene animation (camera view)...")
    anim = renderer.create_animation(
        scene,
        view="camera",
        camera_idx=cfg.camera,
        fps=fps,
    )

    cfg.save_input.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving input scene animation to {cfg.save_input}...")
    anim.save(str(cfg.save_input), fps=fps)
    plt.close()
    print("Done!")


def validate_frame_and_camera(scene: SceneData, cfg: RuntimeConfig) -> int | None:
    """Validate frame and camera indices."""
    num_frames = scene.meta["num_frames"]
    if cfg.frame >= num_frames:
        print(f"Error: Frame {cfg.frame} out of range (0-{num_frames - 1})")
        return 1

    num_cameras = len(scene.cameras)
    if cfg.camera >= num_cameras:
        print(f"Error: Camera {cfg.camera} out of range (0-{num_cameras - 1})")
        return 1

    return None


def render_scene(scene: SceneData, cfg: RuntimeConfig) -> int:
    """Render scene based on view type."""
    renderer = SceneRenderer()

    if cfg.view == "animation":
        print(f"Creating animation ({cfg.animation_view} view)...")
        anim = renderer.create_animation(
            scene,
            view=cfg.animation_view,
            camera_idx=cfg.camera,
            fps=cfg.fps,
        )

        if cfg.save:
            print(f"Saving animation to {cfg.save}...")
            anim.save(str(cfg.save), fps=cfg.fps or scene.meta["fps"])
            print("Done!")
        else:
            plt.show()

    elif cfg.view == "3d":
        print(f"Rendering 3D view (frame {cfg.frame})...")
        fig, ax = renderer.render_frame_3d(scene, cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "2d":
        print(f"Rendering 2D top-down view (frame {cfg.frame})...")
        fig, ax = renderer.render_frame_2d_topdown(scene, cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "camera":
        print(f"Rendering camera {cfg.camera} view (frame {cfg.frame})...")
        fig, ax = renderer.render_camera_view(scene, cfg.frame, cfg.camera)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    elif cfg.view == "multi":
        print(f"Rendering multi-view (frame {cfg.frame})...")
        fig, axes = renderer.render_multi_view(scene, cfg.frame)

        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()

    return 0


# =============================================================================
# Subcommand: visualize (Ground Truth)
# =============================================================================


def main_visualize(cfg: RuntimeConfig) -> int:
    """Visualize ground truth scene data."""
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, cfg)
    if err is not None:
        return err

    save_input_scene(scene, cfg)
    return render_scene(scene, cfg)


# =============================================================================
# Subcommand: predict (Single-Frame Model)
# =============================================================================


def main_predict(cfg: RuntimeConfig) -> int:
    """Run single-frame model predictions and visualize."""
    from src.plcs.inference.predictor import PLCSPredictor

    if not _require_checkpoint(cfg):
        return 1

    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = PLCSPredictor.load_from_checkpoint(cfg.checkpoint, device=cfg.device)

    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, cfg)
    if err is not None:
        return err

    # Run predictions and overwrite SceneData
    num_frames = scene.meta["num_frames"]
    cam = scene.cameras[cfg.camera]

    save_input_scene(scene, cfg)

    print(f"Running predictions for {num_frames} frames using camera {cfg.camera}...")
    for frame_idx in range(num_frames):
        human_kp = torch.from_numpy(cam.human_kp_uv[frame_idx]).float().unsqueeze(0)  # (1, 17, 2)
        court_kp = torch.from_numpy(cam.court_kp_uv[frame_idx]).float().unsqueeze(0)  # (1, 20, 2)
        human_vis = torch.from_numpy(cam.human_kp_visible[frame_idx].astype(np.float32)).unsqueeze(0)  # (1, 17)
        court_vis = torch.from_numpy(cam.court_kp_visible[frame_idx].astype(np.float32)).unsqueeze(0)  # (1, 20)

        pred = predictor.predict(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )

        # Overwrite normalized position and rotation
        scene.position[frame_idx] = pred["position"].numpy()
        scene.rotation[frame_idx] = pred["rotation"].numpy()

    return render_scene(scene, cfg)


# =============================================================================
# Subcommand: predict-seq (Sequence Model)
# =============================================================================


def main_predict_sequence(cfg: RuntimeConfig) -> int:
    """Run sequence model predictions and visualize."""
    from src.plcs.inference.sequence_predictor import PLCSSequencePredictor

    if not _require_checkpoint(cfg):
        return 1

    print(f"Loading sequence checkpoint from {cfg.checkpoint}...")
    predictor = PLCSSequencePredictor.load_from_checkpoint(cfg.checkpoint, device=cfg.device)

    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(cfg.scene_path)

    if cfg.info:
        print_scene_info(scene)
        return 0

    err = validate_frame_and_camera(scene, cfg)
    if err is not None:
        return err

    save_input_scene(scene, cfg)

    # Run sequence prediction and overwrite SceneData
    cam = scene.cameras[cfg.camera]

    # Prepare sequence inputs: (T, K, 2) and (T, K)
    human_kp_seq = torch.from_numpy(cam.human_kp_uv).float()  # (T, 17, 2)
    court_kp_raw = torch.from_numpy(cam.court_kp_uv).float()  # (T, 20, 2)
    human_vis_seq = torch.from_numpy(cam.human_kp_visible).float()  # (T, 17)
    court_vis_raw = torch.from_numpy(cam.court_kp_visible).float()  # (T, 20)

    # Aggregate court keypoints: take mean across temporal dimension
    # Model expects (B, 1, 20, 2) for court, pre-aggregated anchor
    court_kp_agg = court_kp_raw.mean(dim=0, keepdim=True)  # (1, 20, 2)
    court_vis_agg = court_vis_raw.mean(dim=0, keepdim=True)  # (1, 20)

    # Add batch dimension for predictor
    human_kp_seq = human_kp_seq.unsqueeze(0)  # (1, T, 17, 2)
    court_kp_seq = court_kp_agg.unsqueeze(0)  # (1, 1, 20, 2)
    human_vis_seq = human_vis_seq.unsqueeze(0)  # (1, T, 17)
    court_vis_seq = court_vis_agg.unsqueeze(0)  # (1, 1, 20)

    num_frames = scene.meta["num_frames"]
    print(
        f"Running sequence prediction for {num_frames} frames using camera {cfg.camera}..."
    )

    pred = predictor.predict(
        human_kp=human_kp_seq,
        court_kp=court_kp_seq,
        human_vis=human_vis_seq,
        court_vis=court_vis_seq,
        denormalize=False,  # Keep normalized for SceneData
    )

    # Overwrite SceneData with predictions (squeeze batch dimension)
    scene.position[...] = pred["position"].squeeze(0).cpu().numpy()  # (T, 3)
    scene.rotation[...] = pred["rotation"].squeeze(0).cpu().numpy()  # (T, 2)

    return render_scene(scene, cfg)


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(config_path="../configs", config_name="visualize", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    """Hydra entry point for visualization and prediction."""
    runtime_cfg = build_runtime_config(cfg)

    if runtime_cfg.mode == "visualize":
        return main_visualize(runtime_cfg)
    if runtime_cfg.mode == "predict":
        return main_predict(runtime_cfg)
    if runtime_cfg.mode in {"predict-seq", "predict_seq"}:
        return main_predict_sequence(runtime_cfg)

    print(f"Unknown mode: {runtime_cfg.mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
