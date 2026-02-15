"""Model input adapters for PLCS visualization prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_multiview_inputs(scene: Any, cameras: list[int]) -> dict[str, torch.Tensor]:
    """Build predictor inputs by stacking selected camera observations."""
    human_kp = np.stack([scene.cameras[c].human_kp_uv for c in cameras], axis=0)
    court_kp = np.stack([scene.cameras[c].court_kp_uv for c in cameras], axis=0)
    human_vis = np.stack(
        [scene.cameras[c].human_kp_visible.astype(np.float32) for c in cameras],
        axis=0,
    )
    court_vis = np.stack(
        [scene.cameras[c].court_kp_visible.astype(np.float32) for c in cameras],
        axis=0,
    )
    human_mask = np.ones((human_kp.shape[0], human_kp.shape[1]), dtype=np.float32)

    return {
        "human_kp": torch.from_numpy(human_kp).float().unsqueeze(0),
        "court_kp": torch.from_numpy(court_kp).float().unsqueeze(0),
        "human_vis": torch.from_numpy(human_vis).float().unsqueeze(0),
        "human_mask": torch.from_numpy(human_mask).float().unsqueeze(0),
        "court_vis": torch.from_numpy(court_vis).float().unsqueeze(0),
    }


def build_sequence_inputs(scene: Any, camera_idx: int) -> dict[str, torch.Tensor]:
    """Build sequence-model inputs for one camera."""
    cam = scene.cameras[camera_idx]
    num_frames = int(cam.human_kp_uv.shape[0])
    return {
        "human_kp": torch.from_numpy(cam.human_kp_uv).float().unsqueeze(0),
        "court_kp": torch.from_numpy(cam.court_kp_uv).float().unsqueeze(0),
        "human_vis": torch.from_numpy(cam.human_kp_visible.astype(np.float32))
        .float()
        .unsqueeze(0),
        "human_mask": torch.ones((1, num_frames), dtype=torch.float32),
        "court_vis": torch.from_numpy(cam.court_kp_visible.astype(np.float32))
        .float()
        .unsqueeze(0),
    }


def build_frame_inputs(scene: Any, camera_idx: int, frame_idx: int) -> dict[str, torch.Tensor]:
    """Build frame-model inputs for one frame of one camera."""
    cam = scene.cameras[camera_idx]
    return {
        "human_kp": torch.from_numpy(cam.human_kp_uv[frame_idx]).float().unsqueeze(0),
        "court_kp": torch.from_numpy(cam.court_kp_uv[frame_idx]).float().unsqueeze(0),
        "human_vis": torch.from_numpy(cam.human_kp_visible[frame_idx].astype(np.float32))
        .float()
        .unsqueeze(0),
        "human_mask": torch.ones((1,), dtype=torch.float32),
        "court_vis": torch.from_numpy(cam.court_kp_visible[frame_idx].astype(np.float32))
        .float()
        .unsqueeze(0),
    }
