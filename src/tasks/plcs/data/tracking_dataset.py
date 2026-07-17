"""Canonical-scene dataset adapter for multi-person PLCS tracking."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)

PLCS_TRACKING_KEYS = (
    "scene_format_version", "human_kp", "human_vis", "detection_mask",
    "detection_score", "bbox", "court_kp", "court_vis", "frame_mask",
    "view_mask", "position", "rotation", "canonical_pose_3d", "human_kp_3d",
    "person_present", "target_person_mask", "clean_human_kp",
    "clean_human_visible", "detection_gt_index",
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


def _bounding_boxes(keypoints: Tensor, visible: Tensor) -> Tensor:
    boxes = torch.zeros((*keypoints.shape[:-2], 4), dtype=keypoints.dtype)
    flat_points = keypoints.reshape(-1, keypoints.shape[-2], 2)
    flat_visible = visible.reshape(-1, visible.shape[-1])
    flat_boxes = boxes.reshape(-1, 4)
    for index, (points, mask) in enumerate(zip(flat_points, flat_visible, strict=True)):
        if mask.any():
            selected = points[mask]
            flat_boxes[index] = torch.cat([selected.min(0).values, selected.max(0).values])
    return boxes


class PLCSTrackingDataset(CanonicalTrackingDataset):
    """Load canonical PLCS scenes and expose the track-query tensor contract."""

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        scene_path = self.scenes[index]
        scalars = json.loads((scene_path / "scalars.json").read_text())
        num_cameras = int(scalars["num_cameras"])
        position = torch.from_numpy(np.load(scene_path / "position.npy")).float()
        rotation = torch.from_numpy(np.load(scene_path / "rotation.npy")).float()
        canonical_pose = torch.from_numpy(
            np.load(scene_path / "canonical_pose_3d.npy")
        ).float()
        human_kp_3d = torch.from_numpy(
            np.load(scene_path / "human_kp_3d.npy")
        ).float()
        if position.ndim == 2:
            position, rotation = position[:, None], rotation[:, None]
            canonical_pose, human_kp_3d = canonical_pose[:, None], human_kp_3d[:, None]
        num_frames, max_persons = position.shape[:2]
        present_path = scene_path / "person_present.npy"
        present = (
            torch.from_numpy(np.load(present_path)).bool()
            if present_path.exists()
            else torch.ones((num_frames, max_persons), dtype=torch.bool)
        )

        kp_rows, visible_rows, present_rows, court_rows, court_vis_rows, index_rows = [], [], [], [], [], []
        for camera in range(num_cameras):
            prefix = scene_path / f"cam_{camera}_"
            keypoints = torch.from_numpy(np.load(f"{prefix}human_kp_uv.npy")).float()
            visible = torch.from_numpy(np.load(f"{prefix}human_kp_visible.npy")).bool()
            if keypoints.ndim == 3:
                keypoints, visible = keypoints[:, None], visible[:, None]
            kp_rows.append(_shuffle_objects(keypoints, camera))
            visible_rows.append(_shuffle_objects(visible, camera))
            present_rows.append(_shuffle_objects(present, camera))
            index_rows.append(
                _shuffle_objects(
                    torch.arange(max_persons).expand(num_frames, -1), camera
                )
            )
            court = torch.from_numpy(np.load(f"{prefix}court_kp_uv.npy")).float()
            court_visible = torch.from_numpy(
                np.load(f"{prefix}court_kp_visible.npy")
            ).bool()
            court_rows.append(court[:, :14])
            court_vis_rows.append(court_visible[:, :14])
        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        detection_mask = human_vis.any(-1) & torch.stack(present_rows)
        return {
            "scene_format_version": torch.tensor(2),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "detection_mask": detection_mask,
            "detection_score": human_vis.float().mean(-1),
            "bbox": _bounding_boxes(human_kp, human_vis),
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "frame_mask": torch.ones(num_frames, dtype=torch.bool),
            "view_mask": torch.ones(num_cameras, dtype=torch.bool),
            "position": position,
            "rotation": rotation,
            "canonical_pose_3d": canonical_pose,
            "human_kp_3d": human_kp_3d,
            "person_present": present,
            "target_person_mask": present.any(0),
            "clean_human_kp": torch.stack([
                torch.from_numpy(np.load(scene_path / f"cam_{camera}_human_kp_uv.npy")).float().reshape(num_frames, max_persons, 17, 2)
                for camera in range(num_cameras)
            ]),
            "clean_human_visible": torch.stack([
                torch.from_numpy(np.load(scene_path / f"cam_{camera}_human_kp_visible.npy")).bool().reshape(num_frames, max_persons, 17)
                for camera in range(num_cameras)
            ]),
            "detection_gt_index": torch.stack(index_rows),
        }


def collate_plcs_tracking_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable motion durations and stack canonical PLCS scenes."""
    return pad_and_stack_tracking_batch(batch, time_dimensions={
        "human_kp": 1, "human_vis": 1, "detection_mask": 1,
        "detection_score": 1, "bbox": 1, "court_kp": 1, "court_vis": 1,
        "frame_mask": 0, "position": 0, "rotation": 0,
        "canonical_pose_3d": 0, "human_kp_3d": 0, "person_present": 0,
        "clean_human_kp": 1, "clean_human_visible": 1,
        "detection_gt_index": 1,
    })


__all__ = ["PLCS_TRACKING_KEYS", "PLCSTrackingDataset", "collate_plcs_tracking_batch"]
