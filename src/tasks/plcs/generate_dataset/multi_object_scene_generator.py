"""Compose simulated motions as lifecycle instances on one global timeline."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset.timeline_composer import (
    TimelineComposer,
    TimelineConfig,
)
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.utils.projection.camera_projector import CameraProjector, camera_from_mapping


class _PLCSSceneSource(Protocol):
    """Single-person scene generator contract used by the compositor."""

    camera_projector: CameraProjector

    def generate_scene(self, scene_id: str) -> SceneData: ...


class MultiPersonSceneGenerator:
    """Generate independent players with reusable-query lifecycle intervals."""

    def __init__(
        self,
        scene_generator: _PLCSSceneSource,
        *,
        timeline: TimelineConfig | Mapping[str, Any],
        rng: random.Random | None = None,
    ) -> None:
        self.scene_generator = scene_generator
        self.timeline = (
            timeline
            if isinstance(timeline, TimelineConfig)
            else TimelineConfig.from_mapping(timeline)
        )
        self.composer = TimelineComposer(self.timeline, rng=rng)

    def generate_scene(self, scene_id: str) -> SceneData:
        """Generate one fixed-length multi-person lifecycle scene."""
        num_persons = self.composer.sample_num_tracks()
        objects = [
            self.scene_generator.generate_scene(
                scene_id=f"{scene_id}_person_{index:02d}"
            )
            for index in range(num_persons)
        ]
        if any(scene.human_kp_3d is None for scene in objects):
            raise RuntimeError(
                "PLCS scene generation must provide COCO17 world joints."
            )
        composition = self.composer.compose(
            [str(scene.meta["scene_id"]) for scene in objects],
            [int(scene.position.shape[0]) for scene in objects],
        )
        base = objects[0]
        position = composition.compose_numpy([scene.position for scene in objects])
        rotation = composition.compose_numpy([scene.rotation for scene in objects])
        rotation[~composition.present] = np.array([1.0, 0.0], dtype=np.float32)
        canonical_pose = composition.compose_numpy(
            [scene.canonical_pose_3d for scene in objects]
        )
        world_joints = composition.compose_numpy(
            [np.asarray(scene.human_kp_3d) for scene in objects]
        )

        cameras: list[CameraData] = []
        projector = self.scene_generator.camera_projector
        for base_camera in base.cameras:
            camera = camera_from_mapping(base_camera.camera_params)
            human_uv: NDArray[np.float32] = np.zeros(
                (self.timeline.num_frames, self.timeline.max_tracks, 17, 2),
                dtype=np.float32,
            )
            human_visible: NDArray[np.bool_] = np.zeros(
                (self.timeline.num_frames, self.timeline.max_tracks, 17),
                dtype=np.bool_,
            )
            for track_index in range(num_persons):
                points = torch.from_numpy(world_joints[:, track_index]).float()
                projected, visible = projector.project_points_to_uv(points, camera)
                active = composition.present[:, track_index]
                track_uv = projected.cpu().numpy()
                track_visible = visible.cpu().numpy() & active[:, None]
                track_uv[~active] = 0.0
                human_uv[:, track_index] = track_uv
                human_visible[:, track_index] = track_visible
            court_uv = np.repeat(
                base_camera.court_kp_uv[0:1], self.timeline.num_frames, axis=0
            )
            court_visible = np.repeat(
                base_camera.court_kp_vis[0:1], self.timeline.num_frames, axis=0
            )
            active_count = max(int(composition.present[:, :num_persons].sum()), 1)
            cameras.append(
                CameraData(
                    camera_params=base_camera.camera_params,
                    human_kp_uv=human_uv,
                    court_kp_uv=court_uv,
                    human_kp_vis=human_visible,
                    court_kp_vis=court_visible,
                    human_visibility_ratio=float(
                        human_visible[:, :num_persons].any(axis=-1).sum() / active_count
                    ),
                    court_visibility_count=base_camera.court_visibility_count,
                )
            )

        base.meta = {
            **base.meta,
            "scene_id": scene_id,
            "num_frames": self.timeline.num_frames,
            "num_persons": num_persons,
            "motion_sources": [scene.meta["motion_source"] for scene in objects],
        }
        base.position = position
        base.rotation = rotation
        base.canonical_pose_3d = canonical_pose
        base.human_kp_3d = world_joints
        base.cameras = cameras
        base.person_present = composition.present
        base.num_persons = num_persons
        base.track_instances = [
            placement.to_metadata() for placement in composition.placements
        ]
        return base

    def generate(self, num_scenes: int) -> Iterator[SceneData]:
        """Yield canonical multi-person lifecycle scenes."""
        for index in range(num_scenes):
            yield self.generate_scene(f"scene_{index:06d}")
