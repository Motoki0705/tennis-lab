"""Rendering utilities for BLCS animations."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from src.utils.rendering import BLCSSceneRenderer


def create_animation(
    scene: dict[str, Any],
    animation_view: str,
    camera_idx: int = 0,
    fps: float = 30.0,
) -> Any:
    """Create animation from scene.
    
    Args:
        scene: Scene dictionary.
        animation_view: View type ('2d_camera' or '3d').
        camera_idx: Camera index (for 2d_camera view).
        fps: Frames per second.
        
    Returns:
        Animation object or None on error.
    """
    renderer = BLCSSceneRenderer()
    
    # Map animation_view to renderer's expected format
    view_mapping = {
        "2d_camera": "camera",
        "3d": "3d",
    }
    
    renderer_view = view_mapping.get(animation_view, animation_view)
    
    return renderer.create_animation(
        scene,
        view=renderer_view,
        camera_idx=camera_idx,
        fps=fps,
    )


def create_comparison_animation(
    gt_positions: Any,
    pred_positions: Any,
    animation_view: str,
    fps: float = 30.0,
    title: str = "GT vs Prediction",
) -> Any:
    """Create comparison animation (GT vs prediction).
    
    Args:
        gt_positions: Ground truth positions array.
        pred_positions: Predicted positions array.
        animation_view: View type ('2d_camera' or '3d').
        fps: Frames per second.
        title: Animation title.
        
    Returns:
        Animation object or None on error.
    """
    renderer = BLCSSceneRenderer()
    
    # For comparison, map 2d_camera to 2d (top-down view)
    # This matches the existing behavior where camera view isn't supported for comparison
    view_mapping = {
        "2d_camera": "2d",
        "3d": "3d",
    }
    
    renderer_view = view_mapping.get(animation_view, animation_view)
    
    return renderer.create_comparison_animation(
        gt_positions=gt_positions,
        pred_positions=pred_positions,
        view=renderer_view,
        fps=fps,
        title=title,
    )


def save_animation(
    animation: Any,
    save_path: str,
    fps: float = 30.0,
) -> None:
    """Save animation to file.
    
    Args:
        animation: Animation object.
        save_path: Output file path.
        fps: Frames per second.
    """
    import os
    from pathlib import Path
    
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving animation to {save_path}...")
    animation.save(str(save_path), fps=fps)
    plt.close()
    print("Done!")


def show_animation(animation: Any) -> None:
    """Display animation interactively.
    
    Args:
        animation: Animation object.
    """
    plt.show()


def print_scene_info(scene: dict[str, Any]) -> None:
    """Print scene information.
    
    Args:
        scene: Scene dictionary.
    """
    renderer = BLCSSceneRenderer()
    renderer.print_scene_info(scene)
