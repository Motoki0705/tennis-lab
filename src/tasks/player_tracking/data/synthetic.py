"""Synthetic articulated multi-person clips with unordered detections."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tasks.player_tracking.data.types import PlayerTrackingBatch

FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]
BoolArray: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]
IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]


_SKELETON = np.array(
    [
        [0.00, 0.00, 0.74],
        [-0.02, 0.00, 0.78],
        [0.02, 0.00, 0.78],
        [-0.05, 0.00, 0.76],
        [0.05, 0.00, 0.76],
        [-0.13, 0.00, 0.60],
        [0.13, 0.00, 0.60],
        [-0.21, 0.00, 0.45],
        [0.21, 0.00, 0.45],
        [-0.24, 0.00, 0.30],
        [0.24, 0.00, 0.30],
        [-0.08, 0.00, 0.36],
        [0.08, 0.00, 0.36],
        [-0.09, 0.00, 0.18],
        [0.09, 0.00, 0.18],
        [-0.10, 0.02, 0.00],
        [0.10, 0.02, 0.00],
    ],
    dtype=np.float32,
)


class SyntheticPlayerTrackingDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate padded player scenes with independent camera-time shuffles."""

    def __init__(self, config: Any, *, split: str) -> None:
        self.config = config
        self.num_samples = int(config.split_sizes[split])
        self.seed = (
            int(config.seed) + {"train": 0, "val": 100_000, "test": 200_000}[split]
        )
        self.max_frames = int(config.max_frames)
        self.min_frames = int(config.min_frames)
        self.max_views = int(config.max_views)
        self.min_views = int(config.min_views)
        self.max_persons = int(config.max_persons)
        self.min_persons = int(config.min_persons)
        self.max_detections = int(config.max_detections)
        self.num_joints = int(config.num_joints)
        if self.num_joints != len(_SKELETON):
            raise ValueError(f"Synthetic generator requires {len(_SKELETON)} joints.")
        self.detection_dropout_probability = float(config.detection_dropout_probability)
        self.joint_dropout_probability = float(config.joint_dropout_probability)
        self.false_positive_probability = float(config.false_positive_probability)
        self.keypoint_noise_std = float(config.keypoint_noise_std)
        self.num_court_keypoints = int(config.num_court_keypoints)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _project(points: FloatArray, camera_index: int) -> FloatArray:
        angle = 2.0 * np.pi * camera_index / 8.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = (points[..., 0] - 0.5) * cos_a - (points[..., 1] - 0.5) * sin_a
        depth = 2.2 + (points[..., 0] - 0.5) * sin_a + (points[..., 1] - 0.5) * cos_a
        u = 0.5 + 0.7 * x / depth
        v = 0.88 - 0.75 * points[..., 2] / depth
        return cast(FloatArray, np.asarray(np.stack([u, v], axis=-1), dtype=np.float32))

    @staticmethod
    def _motion(
        rng: np.random.Generator, length: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)
        start = np.asarray(
            rng.uniform([0.18, 0.18, 0.0], [0.82, 0.82, 0.0]), dtype=np.float32
        )
        velocity = np.asarray(
            rng.uniform([-0.30, -0.30], [0.30, 0.30]), dtype=np.float32
        )
        position: FloatArray = np.zeros((length, 3), dtype=np.float32)
        position[:, :2] = start[:2] + time[:, None] * velocity[None]
        position[:, :2] = np.clip(position[:, :2], 0.08, 0.92)
        yaw = np.arctan2(velocity[1], velocity[0] + 1e-6) + 0.15 * np.sin(
            2 * np.pi * time
        )
        rotation = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)
        pose = np.broadcast_to(_SKELETON, (length, len(_SKELETON), 3)).copy()
        gait = 0.04 * np.sin(4.0 * np.pi * time + rng.uniform(0.0, 2.0 * np.pi))
        pose[:, [9, 14], 1] += gait[:, None]
        pose[:, [10, 13], 1] -= gait[:, None]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        local_x = pose[..., 0].copy()
        local_y = pose[..., 1].copy()
        pose[..., 0] = local_x * cos_yaw[:, None] - local_y * sin_yaw[:, None]
        pose[..., 1] = local_x * sin_yaw[:, None] + local_y * cos_yaw[:, None]
        pose += position[:, None]
        return position, rotation, pose.astype(np.float32)

    def __getitem__(self, index: int) -> PlayerTrackingBatch:
        rng = np.random.default_rng(self.seed + index)
        valid_frames = int(rng.integers(self.min_frames, self.max_frames + 1))
        valid_views = int(rng.integers(self.min_views, self.max_views + 1))
        num_persons = int(rng.integers(self.min_persons, self.max_persons + 1))

        position: FloatArray = np.zeros(
            (self.max_frames, self.max_persons, 3), dtype=np.float32
        )
        rotation: FloatArray = np.zeros(
            (self.max_frames, self.max_persons, 2), dtype=np.float32
        )
        rotation[..., 0] = 1.0
        canonical_pose: FloatArray = np.zeros(
            (self.max_frames, self.max_persons, self.num_joints, 3), dtype=np.float32
        )
        human_kp_3d = np.zeros_like(canonical_pose)
        person_present: BoolArray = np.zeros(
            (self.max_frames, self.max_persons), dtype=np.bool_
        )
        clean_uv: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.max_persons, self.num_joints, 2),
            dtype=np.float32,
        )
        clean_visible: BoolArray = np.zeros(clean_uv.shape[:-1], dtype=np.bool_)
        for person_index in range(num_persons):
            start = int(rng.integers(0, max(valid_frames // 3, 1)))
            end = int(
                rng.integers(max(start + 2, valid_frames * 2 // 3), valid_frames + 1)
            )
            person_position, person_rotation, world_pose = self._motion(
                rng, end - start
            )
            position[start:end, person_index] = person_position
            rotation[start:end, person_index] = person_rotation
            canonical_pose[start:end, person_index] = (
                world_pose - person_position[:, None]
            )
            human_kp_3d[start:end, person_index] = world_pose
            person_present[start:end, person_index] = True
            for view in range(valid_views):
                projected = self._project(world_pose, view)
                joint_visible = np.logical_and(
                    (projected >= 0.0).all(-1), (projected <= 1.0).all(-1)
                )
                clean_uv[view, start:end, person_index] = projected
                clean_visible[view, start:end, person_index] = joint_visible

        human_kp: FloatArray = np.zeros(
            (
                self.max_views,
                self.max_frames,
                self.max_detections,
                self.num_joints,
                2,
            ),
            dtype=np.float32,
        )
        human_vis: BoolArray = np.zeros(human_kp.shape[:-1], dtype=np.bool_)
        detection_mask: BoolArray = np.zeros(
            (self.max_views, self.max_frames, self.max_detections), dtype=np.bool_
        )
        detection_score = np.zeros_like(detection_mask, dtype=np.float32)
        bbox: FloatArray = np.zeros((*detection_mask.shape, 4), dtype=np.float32)
        detection_gt_index: IntArray = np.full_like(detection_score, -1, dtype=np.int64)

        for view in range(valid_views):
            for frame in range(valid_frames):
                detections: list[
                    tuple[np.ndarray, np.ndarray, float, np.ndarray, int]
                ] = []
                for person_index in range(num_persons):
                    if not person_present[frame, person_index]:
                        continue
                    if rng.random() < self.detection_dropout_probability:
                        continue
                    points = np.clip(
                        clean_uv[view, frame, person_index]
                        + rng.normal(
                            0.0,
                            self.keypoint_noise_std,
                            size=(self.num_joints, 2),
                        ),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    joint_vis = clean_visible[view, frame, person_index].copy()
                    joint_vis &= (
                        rng.random(self.num_joints) >= self.joint_dropout_probability
                    )
                    visible_points = points[joint_vis]
                    if visible_points.size:
                        bounds = np.concatenate(
                            [visible_points.min(0), visible_points.max(0)]
                        ).astype(np.float32)
                    else:
                        bounds = np.zeros(4, dtype=np.float32)
                    detections.append(
                        (
                            points,
                            joint_vis,
                            float(joint_vis.mean()),
                            bounds,
                            person_index,
                        )
                    )
                if rng.random() < self.false_positive_probability:
                    center = rng.uniform(0.15, 0.85, size=2).astype(np.float32)
                    scale = rng.uniform(0.04, 0.16)
                    false_points = np.clip(
                        center + rng.normal(0.0, scale, size=(self.num_joints, 2)),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    false_vis = rng.random(self.num_joints) > 0.25
                    false_bounds = np.concatenate(
                        [false_points.min(0), false_points.max(0)]
                    ).astype(np.float32)
                    detections.append(
                        (
                            false_points,
                            false_vis,
                            float(rng.uniform(0.1, 0.55)),
                            false_bounds,
                            -1,
                        )
                    )
                rng.shuffle(detections)
                for detection, item in enumerate(detections[: self.max_detections]):
                    human_kp[view, frame, detection] = item[0]
                    human_vis[view, frame, detection] = item[1]
                    detection_mask[view, frame, detection] = True
                    detection_score[view, frame, detection] = item[2]
                    bbox[view, frame, detection] = item[3]
                    detection_gt_index[view, frame, detection] = item[4]

        court_kp: FloatArray = np.zeros(
            (self.max_views, self.max_frames, self.num_court_keypoints, 2),
            dtype=np.float32,
        )
        base_court = np.stack(
            [
                np.linspace(0.1, 0.9, self.num_court_keypoints, dtype=np.float32),
                np.tile(
                    np.array([0.2, 0.8], dtype=np.float32),
                    self.num_court_keypoints // 2 + 1,
                )[: self.num_court_keypoints],
            ],
            axis=-1,
        )
        court_kp[:valid_views, :valid_frames] = base_court
        court_vis: BoolArray = np.zeros(court_kp.shape[:-1], dtype=np.bool_)
        court_vis[:valid_views, :valid_frames] = True
        frame_mask = np.arange(self.max_frames) < valid_frames
        view_mask = np.arange(self.max_views) < valid_views
        target_person_mask = np.arange(self.max_persons) < num_persons

        return {
            "scene_format_version": torch.tensor(2, dtype=torch.int64),
            "human_kp": torch.from_numpy(human_kp),
            "human_vis": torch.from_numpy(human_vis),
            "detection_mask": torch.from_numpy(detection_mask),
            "detection_score": torch.from_numpy(detection_score),
            "bbox": torch.from_numpy(bbox),
            "court_kp": torch.from_numpy(court_kp),
            "court_vis": torch.from_numpy(court_vis),
            "frame_mask": torch.from_numpy(frame_mask),
            "view_mask": torch.from_numpy(view_mask),
            "position": torch.from_numpy(position),
            "rotation": torch.from_numpy(rotation),
            "canonical_pose_3d": torch.from_numpy(canonical_pose),
            "human_kp_3d": torch.from_numpy(human_kp_3d),
            "person_present": torch.from_numpy(person_present),
            "target_person_mask": torch.from_numpy(target_person_mask),
            "clean_human_kp": torch.from_numpy(clean_uv),
            "clean_human_visible": torch.from_numpy(clean_visible),
            "detection_gt_index": torch.from_numpy(detection_gt_index),
        }
