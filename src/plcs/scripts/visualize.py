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
from src.plcs.generate_dataset.scene_generator import CameraData
from src.utils.rendering import PLCSSceneRenderer as SceneRenderer
from src.utils.rendering.skeleton_renderer import SkeletonRenderer

if TYPE_CHECKING:
    from src.plcs.generate_dataset.scene_generator import SceneData


def _copy_scene(scene: SceneData) -> SceneData:
    """Create a deep copy of SceneData without modifying the original.
    
    Args:
        scene: Original scene data.
        
    Returns:
        Deep copy of the scene.
    """
    from src.plcs.generate_dataset.scene_generator import SceneData
    
    # Copy metadata
    meta_copy = scene.meta.copy()
    
    # Deep copy arrays
    position_copy = scene.position.copy()
    rotation_copy = scene.rotation.copy()
    canonical_pose_3d_copy = scene.canonical_pose_3d.copy()
    
    # Deep copy cameras
    cameras_copy = []
    for cam in scene.cameras:
        # Camera params may be stored as 'params' or 'camera_params'
        cam_params = getattr(cam, 'camera_params', None) or getattr(cam, 'params', None)
        
        cam_copy = CameraData(
            camera_params=cam_params.copy() if isinstance(cam_params, dict) else cam_params,
            human_kp_uv=cam.human_kp_uv.copy(),
            court_kp_uv=cam.court_kp_uv.copy(),
            human_kp_visible=cam.human_kp_visible.copy(),
            court_kp_visible=cam.court_kp_visible.copy(),
            human_visibility_ratio=cam.human_visibility_ratio,
            court_visibility_count=cam.court_visibility_count,
        )
        cameras_copy.append(cam_copy)
    
    # Create new SceneData instance
    scene_copy = SceneData(
        meta=meta_copy,
        position=position_copy,
        rotation=rotation_copy,
        canonical_pose_3d=canonical_pose_3d_copy,
        cameras=cameras_copy,
    )
    
    return scene_copy



@dataclass
class CompareConfig:
    """Configuration for GT vs pred comparison."""

    layout: str  # overlay | side-by-side
    gt_color: str
    pred_color: str
    show_error: bool


@dataclass
class RenderingConfig:
    """Rendering configuration for visualization."""

    human_kp_size: float
    court_kp_size: float
    bone_width: float


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
    compare: CompareConfig
    rendering: RenderingConfig


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
        compare=CompareConfig(
            layout=str(vis.compare.layout),
            gt_color=str(vis.compare.gt_color),
            pred_color=str(vis.compare.pred_color),
            show_error=bool(vis.compare.show_error),
        ),
        rendering=RenderingConfig(
            human_kp_size=float(vis.rendering.human_kp_size),
            court_kp_size=float(vis.rendering.court_kp_size),
            bone_width=float(vis.rendering.bone_width),
        ),
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
    from src.utils.rendering.skeleton_renderer import SkeletonStyle
    
    # Create renderer with custom rendering settings
    renderer = SceneRenderer(
        skeleton_renderer=SkeletonRenderer(
            skeleton_type="coco17",
            style=SkeletonStyle(
                joint_size=cfg.rendering.human_kp_size,
                bone_width=cfg.rendering.bone_width,
            ),
        ),
    )

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
        fig, ax = renderer.render_camera_view(
            scene, cfg.frame, cfg.camera, court_kp_size=cfg.rendering.court_kp_size
        )

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
# Subcommand: compare (GT vs Prediction Comparison)
# =============================================================================


def _run_prediction(scene: SceneData, cfg: RuntimeConfig) -> SceneData:
    """Run predictions and return a copy of scene with predicted positions/rotations.
    
    Args:
        scene: Ground truth scene data (not modified).
        cfg: Runtime configuration.
        
    Returns:
        Scene copy with predicted positions and rotations.
    """
    from src.plcs.inference.predictor import PLCSPredictor
    
    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = PLCSPredictor.load_from_checkpoint(cfg.checkpoint, device=cfg.device)
    
    # Create a copy for predictions
    scene_pred = _copy_scene(scene)
    
    num_frames = scene.meta["num_frames"]
    cam = scene.cameras[cfg.camera]
    
    print(f"Running predictions for {num_frames} frames using camera {cfg.camera}...")
    for frame_idx in range(num_frames):
        human_kp = torch.from_numpy(cam.human_kp_uv[frame_idx]).float().unsqueeze(0)
        court_kp = torch.from_numpy(cam.court_kp_uv[frame_idx]).float().unsqueeze(0)
        human_vis = torch.from_numpy(cam.human_kp_visible[frame_idx].astype(np.float32)).unsqueeze(0)
        court_vis = torch.from_numpy(cam.court_kp_visible[frame_idx].astype(np.float32)).unsqueeze(0)
        
        pred = predictor.predict(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )
        
        # Update prediction scene
        scene_pred.position[frame_idx] = pred["position"].numpy()
        scene_pred.rotation[frame_idx] = pred["rotation"].numpy()
    
    return scene_pred


def render_comparison(scene_gt: SceneData, scene_pred: SceneData, cfg: RuntimeConfig) -> int:
    """Render GT vs pred comparison based on view type.
    
    Args:
        scene_gt: Ground truth scene.
        scene_pred: Predicted scene.
        cfg: Runtime configuration.
        
    Returns:
        Exit code (0 for success).
    """
    from src.utils.rendering.skeleton_renderer import SkeletonStyle
    
    # Create renderers with custom colors
    gt_renderer = SceneRenderer(
        skeleton_renderer=SkeletonRenderer(
            skeleton_type="coco17",
            style=SkeletonStyle(
                joint_color=cfg.compare.gt_color,
                bone_color=cfg.compare.gt_color,
                joint_size=cfg.rendering.human_kp_size,
                bone_width=cfg.rendering.bone_width,
            ),
        ),
    )
    
    pred_renderer = SceneRenderer(
        skeleton_renderer=SkeletonRenderer(
            skeleton_type="coco17",
            style=SkeletonStyle(
                joint_color=cfg.compare.pred_color,
                bone_color=cfg.compare.pred_color,
                joint_size=cfg.rendering.human_kp_size,
                bone_width=cfg.rendering.bone_width,
            ),
        ),
    )
    
    # Calculate error if enabled
    error_str = ""
    if cfg.compare.show_error:
        pos_error = np.linalg.norm(
            scene_gt.position[cfg.frame] - scene_pred.position[cfg.frame]
        )
        error_str = f" | L2 error: {pos_error:.4f}"
    
    if cfg.view == "2d":
        print(f"Rendering 2D comparison (frame {cfg.frame})...")
        
        if cfg.compare.layout == "overlay":
            # Overlay GT and pred on same axes
            fig, ax = gt_renderer.render_frame_2d_topdown(scene_gt, cfg.frame)
            # Render pred on same axes without clearing
            ax.set_prop_cycle(None)  # Reset color cycle
            pred_renderer.render_frame_2d_topdown(
                scene_pred, cfg.frame, ax=ax, clear_axes=False, show_trail=False
            )
            ax.set_title(f"Frame {cfg.frame} | GT (green) vs Pred (magenta){error_str}")
            
        else:  # side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            gt_renderer.render_frame_2d_topdown(scene_gt, cfg.frame, ax=ax1)
            ax1.set_title(f"GT | Frame {cfg.frame}")
            pred_renderer.render_frame_2d_topdown(scene_pred, cfg.frame, ax=ax2)
            ax2.set_title(f"Pred | Frame {cfg.frame}{error_str}")
            fig.tight_layout()
        
        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()
    
    elif cfg.view == "3d":
        print(f"Rendering 3D comparison (frame {cfg.frame})...")
        
        if cfg.compare.layout == "overlay":
            fig, ax = gt_renderer.render_frame_3d(scene_gt, cfg.frame)
            # Render pred on same axes (without clearing)
            pred_renderer.render_frame_3d(scene_pred, cfg.frame, ax=ax, clear_axes=False)
            ax.set_title(f"Frame {cfg.frame} | GT (green) vs Pred (magenta){error_str}")
            
        else:  # side-by-side
            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
            gt_renderer.render_frame_3d(scene_gt, cfg.frame, ax=ax1)
            ax1.set_title(f"GT | Frame {cfg.frame}")
            pred_renderer.render_frame_3d(scene_pred, cfg.frame, ax=ax2)
            ax2.set_title(f"Pred | Frame {cfg.frame}{error_str}")
            fig.tight_layout()
        
        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()
    
    elif cfg.view in {"multi", "camera"}:
        # For multi and camera views, use overlay for now
        print(f"Rendering {cfg.view} comparison (frame {cfg.frame})...")
        print("Note: comparison for multi/camera views uses overlay layout")
        
        if cfg.view == "multi":
            fig, axes = gt_renderer.render_multi_view(scene_gt, cfg.frame)
            # TODO: Overlay pred on multi view (requires accessing individual axes)
        else:
            fig, ax = gt_renderer.render_camera_view(scene_gt, cfg.frame, cfg.camera)
            # TODO: Overlay pred on camera view
        
        if cfg.save:
            fig.savefig(str(cfg.save), dpi=150, bbox_inches="tight")
            print(f"Saved to {cfg.save}")
        else:
            plt.show()
    
    return 0


def main_compare(cfg: RuntimeConfig) -> int:
    """Run predictions and compare with ground truth.
    
    Args:
        cfg: Runtime configuration.
        
    Returns:
        Exit code.
    """
    if not _require_checkpoint(cfg):
        return 1
    
    print(f"Loading scene from {cfg.scene_path}...")
    scene_gt = load_scene(cfg.scene_path)
    
    if cfg.info:
        print_scene_info(scene_gt)
        return 0
    
    err = validate_frame_and_camera(scene_gt, cfg)
    if err is not None:
        return err
    
    save_input_scene(scene_gt, cfg)
    
    # Run prediction (GT remains unchanged)
    scene_pred = _run_prediction(scene_gt, cfg)
    
    # Render comparison
    return render_comparison(scene_gt, scene_pred, cfg)


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
    if runtime_cfg.mode == "compare":
        return main_compare(runtime_cfg)
    if runtime_cfg.mode in {"predict-seq", "predict_seq"}:
        return main_predict_sequence(runtime_cfg)

    print(f"Unknown mode: {runtime_cfg.mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
