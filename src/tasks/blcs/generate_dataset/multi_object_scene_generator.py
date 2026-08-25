"""Compose physical rallies as lifecycle instances on one global timeline."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset.timeline_composer import (
    TimelineComposer,
    TimelineConfig,
    TrackPlacement,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    CameraData,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    is_retryable_full_physics_rejection,
)
from src.utils.projection.camera_projector import (
    CameraProjector,
    camera_from_mapping,
)

logger = logging.getLogger(__name__)


class _BLCSSceneSource(Protocol):
    """Physical single-ball generator contract used by the compositor."""

    config: Any

    def sample_from_cell(self) -> int: ...

    def sample_side(self) -> str: ...

    def generate_scene(
        self, from_cell: int, side: str, scene_id: str
    ) -> BLCSSceneData | None: ...


def _shift_shots(scene: BLCSSceneData, placement: TrackPlacement) -> list[dict]:
    shifted: list[dict] = []
    timestamp_keys = (
        "t_start",
        "t_net",
        "t_bounce1",
        "t_bounce2",
        "t_bounce3",
        "t_return",
    )
    for shot in scene.shots:
        start = shot.get("t_start")
        if (
            start is None
            or not placement.source_start <= int(start) < placement.source_end
        ):
            continue
        entry = dict(shot)
        for key in timestamp_keys:
            value = entry.get(key)
            if value is not None:
                entry[key] = int(value) - placement.source_start + placement.birth_frame
        shifted.append(entry)
    return shifted


class MultiBallSceneGenerator:
    """Generate independent rallies with reusable-query lifecycle intervals."""

    def __init__(
        self,
        scene_generator: _BLCSSceneSource,
        *,
        timeline: TimelineConfig | Mapping[str, Any],
        maximum_physics_attempts_per_object: int,
        rng: random.Random | None = None,
    ) -> None:
        if (
            isinstance(maximum_physics_attempts_per_object, bool)
            or not isinstance(maximum_physics_attempts_per_object, int)
            or maximum_physics_attempts_per_object <= 0
        ):
            raise ValueError(
                "maximum_physics_attempts_per_object must be a positive integer."
            )
        self.scene_generator = scene_generator
        self.maximum_physics_attempts_per_object = maximum_physics_attempts_per_object
        self.timeline = (
            timeline
            if isinstance(timeline, TimelineConfig)
            else TimelineConfig.from_mapping(timeline)
        )
        self.composer = TimelineComposer(self.timeline, rng=rng)

    def _generate_ball(self, scene_id: str) -> BLCSSceneData:
        last_rejection: RuntimeError | None = None
        last_reason = "BLCS physical scene generation returned no scene."
        for attempt in range(1, self.maximum_physics_attempts_per_object + 1):
            try:
                scene = self.scene_generator.generate_scene(
                    self.scene_generator.sample_from_cell(),
                    self.scene_generator.sample_side(),
                    scene_id,
                )
            except RuntimeError as error:
                if not is_retryable_full_physics_rejection(error):
                    raise
                last_rejection = error
                last_reason = str(error)
                continue
            if scene is not None:
                if attempt > 1:
                    logger.info(
                        "Accepted BLCS physics proposal for %s after bounded "
                        "resampling (attempt %s/%s); last_rejection=%s",
                        scene_id,
                        attempt,
                        self.maximum_physics_attempts_per_object,
                        last_reason,
                    )
                return scene
            last_rejection = None
            last_reason = "BLCS physical scene generation returned no scene."

        exhausted = RuntimeError(
            "BLCS physical scene generation exhausted "
            f"{self.maximum_physics_attempts_per_object} bounded attempts for "
            f"{scene_id!r}; last_rejection={last_reason}"
        )
        if last_rejection is not None:
            raise exhausted from last_rejection
        raise exhausted

    def generate_scene(self, scene_id: str) -> BLCSSceneData:
        """Generate one fixed-length multi-ball lifecycle scene."""
        num_balls = self.composer.sample_num_tracks()
        objects = [
            self._generate_ball(f"{scene_id}_ball_{index:02d}")
            for index in range(num_balls)
        ]
        composition = self.composer.compose(
            [scene.scene_id for scene in objects],
            [int(scene.ball_pos_world.shape[0]) for scene in objects],
        )
        base = objects[0]
        ball_pos_world = composition.compose_tensor(
            [scene.ball_pos_world for scene in objects]
        )
        ball_pos_norm = composition.compose_tensor(
            [scene.ball_pos_norm for scene in objects]
        )
        ball_vel_world = composition.compose_tensor(
            [scene.ball_vel_world for scene in objects]
        )
        ball_vel_norm = composition.compose_tensor(
            [scene.ball_vel_norm for scene in objects]
        )
        ball_present = torch.from_numpy(composition.present).to(
            device=ball_pos_world.device
        )

        cameras: list[CameraData] = []
        projector = CameraProjector(
            self.scene_generator.config.camera,
            court_config=self.scene_generator.config.court,
        )
        for base_camera in base.cameras:
            camera = camera_from_mapping(base_camera.camera_params)
            uv: NDArray[np.float32] = np.zeros(
                (self.timeline.num_frames, self.timeline.max_tracks, 2),
                dtype=np.float32,
            )
            visible: NDArray[np.bool_] = np.zeros(
                (self.timeline.num_frames, self.timeline.max_tracks),
                dtype=np.bool_,
            )
            for track_index in range(num_balls):
                projected, projected_visible = projector.project_points_to_uv(
                    ball_pos_world[:, track_index], camera
                )
                active = composition.present[:, track_index]
                track_uv = projected.cpu().numpy()
                track_visible = projected_visible.cpu().numpy() & active
                track_uv[~active] = 0.0
                uv[:, track_index] = track_uv
                visible[:, track_index] = track_visible
            active_count = max(int(composition.present[:, :num_balls].sum()), 1)
            cameras.append(
                CameraData(
                    camera_params=base_camera.camera_params,
                    ball_uv=uv,
                    ball_vis=visible,
                    ball_visibility_ratio=float(
                        visible[:, :num_balls].sum() / active_count
                    ),
                    court_kp_uv=base_camera.court_kp_uv,
                    court_kp_vis=base_camera.court_kp_vis,
                    court_visibility_count=base_camera.court_visibility_count,
                )
            )

        base.scene_id = scene_id
        base.rally_length = sum(len(scene.shots) for scene in objects)
        base.ball_pos_world = ball_pos_world
        base.ball_pos_norm = ball_pos_norm
        base.ball_vel_world = ball_vel_world
        base.ball_vel_norm = ball_vel_norm
        base.cameras = cameras
        base.ball_present = ball_present
        base.num_balls = num_balls
        base.track_instances = [
            placement.to_metadata() for placement in composition.placements
        ]
        base.shots = [
            {
                "track_id": placement.track_id,
                "source_scene_id": placement.source_scene_id,
                "shots": _shift_shots(scene, placement),
            }
            for scene, placement in zip(objects, composition.placements, strict=True)
        ]
        return base

    def generate(self, num_scenes: int) -> Iterator[BLCSSceneData]:
        """Yield canonical multi-ball lifecycle scenes."""
        for index in range(num_scenes):
            yield self.generate_scene(f"scene_{index:06d}")
