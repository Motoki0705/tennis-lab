"""Canonical-scene dataset adapter for multi-ball BLCS tracking."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)

BLCS_TRACKING_KEYS = (
    "scene_format_version", "ball_uv", "ball_score", "ball_candidate_mask",
    "ball_visible", "court_kp", "court_vis", "frame_mask", "view_mask",
    "position_3d", "ball_present", "target_ball_mask", "ball_uv_gt",
    "ball_visible_gt", "candidate_gt_index",
)


def _shuffle_objects(value: Tensor, camera: int) -> Tensor:
    """Apply a deterministic camera-time permutation to the object axis."""
    num_objects = int(value.shape[1])
    return torch.stack(
        [
            torch.roll(frame, shifts=(camera + frame_index) % num_objects, dims=0)
            for frame_index, frame in enumerate(value)
        ]
    )


class BLCSTrackingDataset(CanonicalTrackingDataset):
    """Load canonical BLCS scenes and expose the track-query tensor contract."""

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        scene_path = self.scenes[index]
        scalars = json.loads((scene_path / "scalars.json").read_text())
        num_cameras = int(scalars["num_cameras"])
        position = torch.from_numpy(np.load(scene_path / "ball_pos_norm.npy")).float()
        if position.ndim == 2:
            position = position[:, None]
        num_frames, max_balls = position.shape[:2]
        present_path = scene_path / "ball_present.npy"
        present = (
            torch.from_numpy(np.load(present_path)).bool()
            if present_path.exists()
            else torch.ones((num_frames, max_balls), dtype=torch.bool)
        )
        uv_rows, visible_rows, present_rows, court_rows, court_vis_rows = [], [], [], [], []
        for camera in range(num_cameras):
            prefix = scene_path / f"cam_{camera}_"
            uv = torch.from_numpy(np.load(f"{prefix}ball_uv.npy")).float()
            visible = torch.from_numpy(np.load(f"{prefix}ball_visible.npy")).bool()
            if uv.ndim == 2:
                uv, visible = uv[:, None], visible[:, None]
            uv_rows.append(_shuffle_objects(uv, camera))
            visible_rows.append(_shuffle_objects(visible, camera))
            present_rows.append(_shuffle_objects(present, camera))
            court = torch.from_numpy(np.load(f"{prefix}court_kp_uv.npy")).float()[:14]
            court_visible = torch.from_numpy(
                np.load(f"{prefix}court_kp_visible.npy")
            ).bool()[:14]
            court_rows.append(court[None].expand(num_frames, -1, -1))
            court_vis_rows.append(court_visible[None].expand(num_frames, -1))
        uv_tensor = torch.stack(uv_rows)
        visible_tensor = torch.stack(visible_rows)
        gt_index = torch.stack(
            [
                _shuffle_objects(
                    torch.arange(max_balls).expand(num_frames, -1), camera
                )
                for camera in range(num_cameras)
            ]
        )
        candidate_mask = visible_tensor & torch.stack(present_rows)
        return {
            "scene_format_version": torch.tensor(2),
            "ball_uv": uv_tensor,
            "ball_score": visible_tensor.float(),
            "ball_candidate_mask": candidate_mask,
            "ball_visible": visible_tensor,
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "frame_mask": torch.ones(num_frames, dtype=torch.bool),
            "view_mask": torch.ones(num_cameras, dtype=torch.bool),
            "position_3d": position,
            "ball_present": present,
            "target_ball_mask": present.any(0),
            "ball_uv_gt": torch.stack([
                torch.from_numpy(np.load(scene_path / f"cam_{camera}_ball_uv.npy")).float().reshape(num_frames, max_balls, 2)
                for camera in range(num_cameras)
            ]),
            "ball_visible_gt": torch.stack([
                torch.from_numpy(np.load(scene_path / f"cam_{camera}_ball_visible.npy")).bool().reshape(num_frames, max_balls)
                for camera in range(num_cameras)
            ]),
            "candidate_gt_index": gt_index,
        }


def collate_blcs_tracking_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable rally durations and stack canonical BLCS scenes."""
    return pad_and_stack_tracking_batch(batch, time_dimensions={
        "ball_uv": 1, "ball_score": 1, "ball_candidate_mask": 1,
        "ball_visible": 1, "court_kp": 1, "court_vis": 1, "frame_mask": 0,
        "position_3d": 0, "ball_present": 0, "ball_uv_gt": 1,
        "ball_visible_gt": 1, "candidate_gt_index": 1,
    })


__all__ = ["BLCS_TRACKING_KEYS", "BLCSTrackingDataset", "collate_blcs_tracking_batch"]
