"""Compose multiple physical rallies into one canonical BLCS scene."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    CameraData,
)
from src.utils.projection.camera_projector import (
    CameraProjector,
    camera_from_mapping,
)


class _BLCSSceneSource(Protocol):
    """Physical single-ball generator contract used by the compositor."""

    config: Any

    def sample_from_cell(self) -> int: ...

    def sample_side(self) -> str: ...

    def generate_scene(
        self, from_cell: int, side: str, scene_id: str
    ) -> BLCSSceneData | None: ...


class MultiBallSceneGenerator:
    """Generate independent rallies and observe them with one shared camera rig."""

    def __init__(
        self,
        scene_generator: _BLCSSceneSource,
        *,
        min_balls: int,
        max_balls: int,
    ) -> None:
        if min_balls < 1 or max_balls < min_balls:
            raise ValueError("Ball count must satisfy 1 <= min_balls <= max_balls.")
        self.scene_generator = scene_generator
        self.min_balls = min_balls
        self.max_balls = max_balls

    def _generate_ball(self, scene_id: str) -> BLCSSceneData:
        scene = self.scene_generator.generate_scene(
            self.scene_generator.sample_from_cell(),
            self.scene_generator.sample_side(),
            scene_id,
        )
        if scene is None:
            raise RuntimeError("BLCS physical scene generation returned no scene.")
        return scene

    def generate_scene(self, scene_id: str) -> BLCSSceneData:
        """Generate one multi-ball scene in the normal BLCS scene schema."""
        num_balls = random.randint(self.min_balls, self.max_balls)
        objects = [
            self._generate_ball(f"{scene_id}_ball_{index:02d}")
            for index in range(num_balls)
        ]
        base = objects[0]
        num_frames = min(int(scene.ball_pos_world.shape[0]) for scene in objects)

        def _stack_padded(values: list[Tensor]) -> Tensor:
            shape = (num_frames, self.max_balls, *values[0].shape[1:])
            result = values[0].new_zeros(shape)
            for index, value in enumerate(values):
                result[:, index] = value[:num_frames]
            return result

        ball_pos_world = _stack_padded([scene.ball_pos_world for scene in objects])
        ball_pos_norm = _stack_padded([scene.ball_pos_norm for scene in objects])
        ball_vel_world = _stack_padded([scene.ball_vel_world for scene in objects])
        ball_present = torch.zeros(
            (num_frames, self.max_balls), dtype=torch.bool, device=ball_pos_world.device
        )
        ball_present[:, :num_balls] = True

        cameras: list[CameraData] = []
        for base_camera in base.cameras:
            camera = camera_from_mapping(base_camera.camera_params)
            projector = CameraProjector(
                self.scene_generator.config.camera,
                court_config=self.scene_generator.config.court,
            )
            uv: np.ndarray = np.zeros(
                (num_frames, self.max_balls, 2), dtype=np.float32
            )
            visible: np.ndarray = np.zeros(
                (num_frames, self.max_balls), dtype=np.bool_
            )
            for object_index, scene in enumerate(objects):
                projected, projected_visible = projector.project_points_to_uv(
                    scene.ball_pos_world[:num_frames], camera
                )
                uv[:, object_index] = projected.cpu().numpy()
                visible[:, object_index] = projected_visible.cpu().numpy()
            cameras.append(
                CameraData(
                    camera_params=base_camera.camera_params,
                    ball_uv=uv,
                    ball_visible=visible,
                    ball_visibility_ratio=float(visible[:, :num_balls].mean()),
                    court_kp_uv=base_camera.court_kp_uv,
                    court_kp_visible=base_camera.court_kp_visible,
                    court_visibility_count=base_camera.court_visibility_count,
                )
            )

        base.scene_id = scene_id
        base.ball_pos_world = ball_pos_world
        base.ball_pos_norm = ball_pos_norm
        base.ball_vel_world = ball_vel_world
        base.cameras = cameras
        base.ball_present = ball_present
        base.num_balls = num_balls
        base.shots = [
            {"ball_index": index, "shots": scene.shots}
            for index, scene in enumerate(objects)
        ]
        return base

    def generate(self, num_scenes: int) -> Iterator[BLCSSceneData]:
        """Yield canonical multi-ball scenes."""
        for index in range(num_scenes):
            yield self.generate_scene(f"scene_{index:06d}")
