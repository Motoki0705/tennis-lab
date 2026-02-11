"""I/O operations for BLCS scenes."""

from __future__ import annotations

from typing import Any

from src.blcs.generate_dataset.io.dataset_io import load_scene as _load_scene


def load_scene(scene_path: str) -> dict[str, Any]:
    """Load a BLCS scene from file.
    
    Args:
        scene_path: Path to scene file (.npz).
        
    Returns:
        Scene dictionary with keys:
            - ball_pos_world: (T, 3) array
            - cameras: list of camera dicts
            - meta: metadata dict
            - num_cameras: int
    """
    return _load_scene(scene_path)


def validate_frame_index(scene: dict[str, Any], frame_idx: int) -> int | None:
    """Validate frame index against scene.
    
    Args:
        scene: Scene dictionary.
        frame_idx: Frame index to validate.
        
    Returns:
        None if valid, error code (1) if invalid.
    """
    num_frames = int(scene["ball_pos_world"].shape[0])
    if frame_idx < 0 or frame_idx >= num_frames:
        print(f"Error: Frame {frame_idx} out of range (0-{num_frames - 1})")
        return 1
    return None


def validate_camera_index(scene: dict[str, Any], camera_idx: int) -> int | None:
    """Validate camera index against scene.
    
    Args:
        scene: Scene dictionary.
        camera_idx: Camera index to validate.
        
    Returns:
        None if valid, error code (1) if invalid.
    """
    num_cameras = int(scene["num_cameras"])
    if camera_idx < 0 or camera_idx >= num_cameras:
        print(f"Error: Camera {camera_idx} out of range (0-{num_cameras - 1})")
        return 1
    return None


def resolve_camera_list(
    scene: dict[str, Any], cameras: list[int]
) -> tuple[int | None, list[int]]:
    """Resolve and validate camera list.
    
    Args:
        scene: Scene dictionary.
        cameras: Camera list (empty list means all cameras).
        
    Returns:
        Tuple of (error_code, resolved_cameras).
        error_code is None if valid, 1 if invalid.
    """
    num_cameras = int(scene["num_cameras"])
    
    if not cameras:
        # Empty list means use all cameras
        return None, list(range(num_cameras))
    
    # Validate each camera index
    for cam_idx in cameras:
        if cam_idx < 0 or cam_idx >= num_cameras:
            print(f"Error: Camera {cam_idx} out of range (0-{num_cameras - 1})")
            return 1, []
    
    return None, cameras
