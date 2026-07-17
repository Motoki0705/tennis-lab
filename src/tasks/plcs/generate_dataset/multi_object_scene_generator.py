"""Compose multiple simulated motions into one canonical PLCS scene."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Protocol, cast

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.scene_generator import (
    CameraData,
    SceneData,
)
from src.utils.projection.camera_projector import CameraProjector, camera_from_mapping


class _PLCSSceneSource(Protocol):
    """Single-person scene generator contract used by the compositor."""

    camera_projector: CameraProjector

    def generate_scene(self, scene_id: str) -> SceneData: ...


class MultiPersonSceneGenerator:
    """Generate independent players and observe them with one shared camera rig."""

    def __init__(
        self,
        scene_generator: _PLCSSceneSource,
        *,
        min_persons: int,
        max_persons: int,
    ) -> None:
        if min_persons < 1 or max_persons < min_persons:
            raise ValueError(
                "Person count must satisfy 1 <= min_persons <= max_persons."
            )
        self.scene_generator = scene_generator
        self.min_persons = min_persons
        self.max_persons = max_persons

    def generate_scene(self, scene_id: str) -> SceneData:
        """Generate one multi-person scene in the normal PLCS scene schema."""
        num_persons = random.randint(self.min_persons, self.max_persons)
        objects = [
            self.scene_generator.generate_scene(
                scene_id=f"{scene_id}_person_{index:02d}"
            )
            for index in range(num_persons)
        ]
        if any(scene.human_kp_3d is None for scene in objects):
            raise RuntimeError("PLCS scene generation must provide COCO17 world joints.")
        base = objects[0]
        num_frames = min(int(scene.position.shape[0]) for scene in objects)

        def _stack_padded(values: list[np.ndarray]) -> np.ndarray:
            shape = (num_frames, self.max_persons, *values[0].shape[1:])
            result = np.zeros(shape, dtype=np.float32)
            for index, value in enumerate(values):
                result[:, index] = value[:num_frames]
            return cast(np.ndarray, result)

        position = _stack_padded([scene.position for scene in objects])
        rotation = _stack_padded([scene.rotation for scene in objects])
        rotation[:, num_persons:, 0] = 1.0
        canonical_pose = _stack_padded(
            [scene.canonical_pose_3d for scene in objects]
        )
        world_joints = _stack_padded(
            [np.asarray(scene.human_kp_3d) for scene in objects]
        )
        person_present: np.ndarray = np.zeros(
            (num_frames, self.max_persons), dtype=np.bool_
        )
        person_present[:, :num_persons] = True

        cameras: list[CameraData] = []
        projector = self.scene_generator.camera_projector
        for base_camera in base.cameras:
            camera = camera_from_mapping(base_camera.camera_params)
            human_uv: np.ndarray = np.zeros(
                (num_frames, self.max_persons, 17, 2), dtype=np.float32
            )
            human_visible: np.ndarray = np.zeros(
                (num_frames, self.max_persons, 17), dtype=np.bool_
            )
            for person_index in range(num_persons):
                points = torch.from_numpy(world_joints[:, person_index]).float()
                projected, visible = projector.project_points_to_uv(points, camera)
                human_uv[:, person_index] = projected.cpu().numpy()
                human_visible[:, person_index] = visible.cpu().numpy()
            cameras.append(
                CameraData(
                    camera_params=base_camera.camera_params,
                    human_kp_uv=human_uv,
                    court_kp_uv=base_camera.court_kp_uv[:num_frames],
                    human_kp_visible=human_visible,
                    court_kp_visible=base_camera.court_kp_visible[:num_frames],
                    human_visibility_ratio=float(
                        human_visible[:, :num_persons].any(axis=-1).mean()
                    ),
                    court_visibility_count=base_camera.court_visibility_count,
                )
            )

        base.meta = {
            **base.meta,
            "scene_id": scene_id,
            "num_frames": num_frames,
            "num_persons": num_persons,
            "motion_sources": [scene.meta["motion_source"] for scene in objects],
        }
        base.position = position
        base.rotation = rotation
        base.canonical_pose_3d = canonical_pose
        base.human_kp_3d = world_joints
        base.cameras = cameras
        base.person_present = person_present
        base.num_persons = num_persons
        return base

    def generate(self, num_scenes: int) -> Iterator[SceneData]:
        """Yield canonical multi-person scenes."""
        for index in range(num_scenes):
            yield self.generate_scene(f"scene_{index:06d}")
