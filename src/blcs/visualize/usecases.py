"""Use cases for BLCS visualization.

This module contains the high-level workflows for:
- Visualizing ground-truth scenes
- Running predictions and visualizing results
- Single-camera and multi-camera modes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.blcs.visualize.config import VisualizationConfig
from src.blcs.visualize.io import (
    load_scene,
    resolve_camera_list,
    validate_camera_index,
    validate_frame_index,
)
from src.blcs.visualize.renderers import (
    create_animation,
    create_comparison_animation,
    save_animation,
    show_animation,
)


def visualize_scene(cfg: VisualizationConfig) -> int:
    """Visualize ground-truth scene.
    
    Args:
        cfg: Visualization configuration.
        
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(str(cfg.scene_path))
    
    if cfg.info:
        from src.utils.rendering import BLCSSceneRenderer
        renderer = BLCSSceneRenderer()
        renderer.print_scene_info(scene)
        return 0
    
    # Validate frame
    err = validate_frame_index(scene, cfg.frame)
    if err is not None:
        return err
    
    # Determine camera index(es) to use
    if cfg.cameras:
        # Multi-camera mode
        err, cameras = resolve_camera_list(scene, cfg.cameras)
        if err is not None:
            return err
        camera_idx = cameras[0] if cameras else 0
        print(f"Using cameras: {cameras} (animation will use camera {camera_idx})")
    else:
        # Single-camera mode
        camera_idx = cfg.camera if cfg.camera is not None else 0
        err = validate_camera_index(scene, camera_idx)
        if err is not None:
            return err
    
    # Create input scene animation if requested
    if cfg.save_input:
        print("Creating input scene animation (2d_camera view)...")
        input_anim = create_animation(
            scene,
            animation_view="2d_camera",
            camera_idx=camera_idx,
            fps=cfg.fps or float(scene["meta"].get("fps_out", 30.0)),
        )
        if input_anim is None:
            print("Error: Failed to create input scene animation")
            return 1
        save_animation(
            input_anim,
            str(cfg.save_input),
            fps=cfg.fps or float(scene["meta"].get("fps_out", 30.0)),
        )
    
    # Create main animation
    fps = cfg.fps or float(scene["meta"].get("fps_out", 30.0))
    print(f"Creating animation ({cfg.animation_view} view)...")
    anim = create_animation(
        scene,
        animation_view=cfg.animation_view,
        camera_idx=camera_idx,
        fps=fps,
    )
    
    if anim is None:
        return 1
    
    if cfg.save:
        save_animation(anim, str(cfg.save), fps=fps)
    else:
        show_animation(anim)
    
    return 0


def visualize_prediction(cfg: VisualizationConfig) -> int:
    """Run prediction and visualize results.
    
    Supports both single-camera and multi-camera prediction modes.
    
    Args:
        cfg: Visualization configuration.
        
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    if cfg.checkpoint is None:
        print("Error: visualization.checkpoint must be set for predict mode.")
        return 1
    
    # Determine if this is multiview mode
    is_multiview = cfg.mode in {"predict-multiview", "predict_multiview"} or bool(cfg.cameras)
    
    if is_multiview:
        return _visualize_prediction_multiview(cfg)
    else:
        return _visualize_prediction_single(cfg)


def _visualize_prediction_single(cfg: VisualizationConfig) -> int:
    """Run single-camera prediction and visualize."""
    from src.blcs.inference.predictor import BLCSPredictor
    
    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint, device=cfg.device
    )
    
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(str(cfg.scene_path))
    
    if cfg.info:
        from src.utils.rendering import BLCSSceneRenderer
        renderer = BLCSSceneRenderer()
        renderer.print_scene_info(scene)
        return 0
    
    # Validate inputs
    err = validate_frame_index(scene, cfg.frame)
    if err is not None:
        return err
    
    camera_idx = cfg.camera if cfg.camera is not None else 0
    err = validate_camera_index(scene, camera_idx)
    if err is not None:
        return err
    
    # Extract camera data
    cam = scene["cameras"][camera_idx]
    ball_uv = torch.from_numpy(cam["ball_uv"]).float()
    court_kp = torch.from_numpy(cam["court_kp_uv"]).float()
    ball_vis = torch.from_numpy(cam["ball_visible"].astype(np.float32))
    court_vis = torch.from_numpy(cam["court_kp_visible"].astype(np.float32))
    
    # Run prediction
    print("Running BLCS prediction...")
    outputs = predictor.predict(
        ball_uv=ball_uv,
        court_kp=court_kp,
        ball_vis=ball_vis,
        court_vis=court_vis,
        denormalize=True,
    )
    
    # Save outputs if requested
    if cfg.output:
        _save_prediction_outputs(outputs, Path(cfg.output))
    
    # Save input scene if requested
    if cfg.save_input:
        print("Creating input scene animation (2d_camera view)...")
        fps = cfg.fps or float(scene["meta"].get("fps_out", 30.0))
        input_anim = create_animation(
            scene,
            animation_view="2d_camera",
            camera_idx=camera_idx,
            fps=fps,
        )
        if input_anim is not None:
            save_animation(input_anim, str(cfg.save_input), fps=fps)
    
    # Get positions for visualization
    gt_pos = scene["ball_pos_world"]
    pred_pos = outputs["position"].squeeze(0).cpu().numpy()
    
    # Create comparison animation
    fps = cfg.fps or float(scene["meta"].get("fps_out", 30.0))
    print(f"Creating comparison animation ({cfg.animation_view} view)...")
    anim = create_comparison_animation(
        gt_positions=gt_pos,
        pred_positions=pred_pos,
        view=cfg.animation_view,
        fps=fps,
        title="GT vs Prediction",
    )
    
    if anim is None:
        return 1
    
    if cfg.save:
        save_animation(anim, str(cfg.save), fps=fps)
    else:
        show_animation(anim)
    
    return 0


def _visualize_prediction_multiview(cfg: VisualizationConfig) -> int:
    """Run multi-camera prediction and visualize."""
    from src.blcs.inference.predictor import BLCSPredictor
    
    print(f"Loading multi-view checkpoint from {cfg.checkpoint}...")
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint, device=cfg.device
    )
    
    print(f"Loading scene from {cfg.scene_path}...")
    scene = load_scene(str(cfg.scene_path))
    
    if cfg.info:
        from src.utils.rendering import BLCSSceneRenderer
        renderer = BLCSSceneRenderer()
        renderer.print_scene_info(scene)
        return 0
    
    # Validate frame
    err = validate_frame_index(scene, cfg.frame)
    if err is not None:
        return err
    
    # Resolve camera list
    err, cameras = resolve_camera_list(scene, cfg.cameras)
    if err is not None:
        return err
    
    if len(cameras) < 2:
        print("Warning: Multi-view prediction works best with >= 2 cameras")
    
    print(f"Using cameras: {cameras}")
    
    # Collect multi-view data
    ball_uv_list = []
    court_kp_list = []
    ball_vis_list = []
    court_vis_list = []
    
    for cam_idx in cameras:
        cam = scene["cameras"][cam_idx]
        ball_uv_list.append(cam["ball_uv"])  # (T, 2)
        court_kp_list.append(cam["court_kp_uv"])  # (20, 2)
        ball_vis_list.append(cam["ball_visible"].astype(np.float32))  # (T,)
        court_vis_list.append(cam["court_kp_visible"].astype(np.float32))  # (20,)
    
    # Stack to (N, T, 2) and (N, 20, 2)
    ball_uv = torch.from_numpy(np.stack(ball_uv_list, axis=0)).float()
    court_kp = torch.from_numpy(np.stack(court_kp_list, axis=0)).float()
    ball_vis = torch.from_numpy(np.stack(ball_vis_list, axis=0)).float()
    ball_mask = torch.ones_like(ball_vis)
    court_vis = torch.from_numpy(np.stack(court_vis_list, axis=0)).float()
    
    # Run prediction
    num_views = len(cameras)
    num_frames = int(scene["ball_pos_world"].shape[0])
    print(f"Running multi-view prediction ({num_views} cameras, {num_frames} frames)...")
    
    outputs = predictor.predict(
        ball_uv=ball_uv,
        court_kp=court_kp,
        ball_vis=ball_vis,
        ball_mask=ball_mask,
        court_vis=court_vis,
        denormalize=True,
    )
    
    # Save outputs if requested
    if cfg.output:
        _save_prediction_outputs(outputs, Path(cfg.output))
    
    # Get positions
    pred_pos = outputs["position"].squeeze(0).cpu().numpy()
    if "position_meters" in outputs:
        pred_pos = outputs["position_meters"].squeeze(0).cpu().numpy()
    
    gt_pos = scene["ball_pos_world"]
    
    # Create comparison animation
    fps = cfg.fps or float(scene["meta"].get("fps_out", 30.0))
    print(f"Creating comparison animation ({cfg.animation_view} view)...")
    anim = create_comparison_animation(
        gt_positions=gt_pos,
        pred_positions=pred_pos,
        view=cfg.animation_view,
        fps=fps,
        title="GT vs Multi-View Prediction",
    )
    
    if anim is None:
        return 1
    
    if cfg.save:
        save_animation(anim, str(cfg.save), fps=fps)
    else:
        show_animation(anim)
    
    return 0


def _save_prediction_outputs(outputs: dict[str, Any], output_path: Path) -> None:
    """Save prediction outputs to file.
    
    Args:
        outputs: Prediction outputs dictionary.
        output_path: Output file path (.pt or .json).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".pt":
        torch.save(outputs, output_path)
    elif output_path.suffix == ".json":
        json_data = {k: v.squeeze(0).cpu().tolist() for k, v in outputs.items()}
        output_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        print(
            f"Warning: Unknown output format '{output_path.suffix}', "
            "only .pt and .json are supported. Skipping save."
        )
        return
    
    print(f"Saved prediction outputs to {output_path}")
