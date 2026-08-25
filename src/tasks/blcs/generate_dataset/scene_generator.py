"""Scene generator for BLCS dataset generation (rally-only scene format).

Per-scene variation:
- Physics constants (gravity, drag, magnus, restitution, friction)
- Wind (speed + direction)
- Court geometry (net post position)
- Camera intrinsics and look-at direction
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    PhysicsConfig,
)
from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
    NUM_CELLS_PER_SIDE,
    CellManager,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyConfig,
    RallySimulator,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
)
from src.utils.projection.camera_projector import (
    CameraConfig,
    CameraProjector,
    CameraView,
)
from src.utils.schema.court import CourtConfig
from src.utils.schema.court_normalization import normalize_court_velocity

logger = logging.getLogger(__name__)


@dataclass
class CameraData:
    """Data for a single valid camera view."""

    camera_params: dict

    ball_uv: np.ndarray  # [T, 2] normalized UV coordinates
    ball_vis: np.ndarray  # [T] visibility flags
    ball_visibility_ratio: float  # Ratio of visible frames

    court_kp_uv: np.ndarray  # [20, 2] UV coordinates
    court_kp_vis: np.ndarray  # [20] visibility flags
    court_visibility_count: float  # Average visible keypoints


@dataclass
class BLCSSceneData:
    """Complete BLCS scene data (1 scene = 1 rally = 1 file with N cameras).

    A rally is a sequence of shots. The trajectory is continuous across all shots,
    with per-shot event timing recorded in the shots list.
    """

    scene_id: str

    initial_from_cell: int
    initial_from_side: str  # "near" or "far"

    rally_length: int  # Number of shots in rally
    end_reason: str  # RallyEndReason value
    winner_side: str | None  # "near", "far", or None

    shots: list[dict]

    ball_pos_world: Tensor  # [T, 3] world coordinates (meters)
    ball_pos_norm: Tensor  # [T, 3] normalized coordinates
    ball_vel_world: Tensor  # [T, 3] velocities (m/s)
    ball_vel_norm: Tensor  # [T, 3] velocities divided by fixed scale

    cameras: list[CameraData]
    num_cameras_sampled: int  # Total cameras tried (before filtering)

    fps_out: int
    sim_fps: int

    # Per-scene variation metadata
    physics_config_dict: dict  # serialized PhysicsConfig (sampled values)
    court_config_dict: dict  # serialized CourtConfig (sampled values)

    # Present for multi-object scenes. Object arrays then use shape [T, O, ...].
    num_balls: int
    ball_present: Tensor | None = None
    track_instances: list[dict] = field(default_factory=list)


@dataclass
class GeneratorConfig:
    """Configuration for scene generator."""

    physics: PhysicsConfig
    rally: RallyConfig
    camera: CameraConfig
    targeted_velocity: TargetedVelocityConfig
    court: CourtConfig


class BLCSSceneGenerator:
    """Generate rally-based BLCS scenes with per-scene variation."""

    def __init__(
        self,
        *,
        config: GeneratorConfig,
        device: str | torch.device,
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        self.cell_manager = CellManager()

        # Track statistics
        self.total_scenes_generated = 0
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0

    def _sample_physics_config(self) -> PhysicsConfig:
        """Sample stochastic physics parameters for one scene."""
        return self.config.physics.sample()

    def _sample_court_config(self) -> CourtConfig:
        """Sample stochastic court geometry for one scene."""
        return self.config.court.sample()

    def sample_from_cell(self) -> int:
        """Sample an initial launch cell index."""
        return int(torch.randint(0, NUM_CELLS_PER_SIDE, (1,)).item())

    def sample_side(self) -> str:
        """Sample an initial player side."""
        return "near" if torch.rand(1).item() < 0.5 else "far"

    def _build_rally_simulator(
        self,
        physics_config: PhysicsConfig,
    ) -> RallySimulator:
        """Build a rally simulator with sampled physics config."""
        return RallySimulator(
            physics_config=physics_config,
            rally_config=self.config.rally,
            cell_manager=self.cell_manager,
            targeted_velocity_config=self.config.targeted_velocity,
            device=self.device,
        )

    def _camera_view_to_data(self, view: CameraView) -> CameraData:
        """Convert CameraView to CameraData with visibility metrics."""
        if view.points_uv is None or view.points_vis is None:
            raise ValueError("BLCS camera views must include projected ball points.")

        ball_vis = view.points_vis.numpy()
        T = len(ball_vis)
        ball_visibility_ratio = float(ball_vis.sum()) / T if T > 0 else 0.0

        court_vis = view.court_kp_vis.numpy()
        court_visibility_count = float(court_vis.sum())

        return CameraData(
            camera_params=view.camera_params,
            ball_uv=view.points_uv.numpy(),
            ball_vis=ball_vis,
            ball_visibility_ratio=ball_visibility_ratio,
            court_kp_uv=view.court_kp_uv.numpy(),
            court_kp_vis=court_vis,
            court_visibility_count=court_visibility_count,
        )

    def _generate_valid_cameras(
        self,
        trajectory: Tensor,
        court_config: CourtConfig,
    ) -> list[CameraData]:
        """Generate cameras and filter by visibility threshold.

        Args:
            trajectory: Ball trajectory [T, 3].
            court_config: Sampled court config (for keypoints with varied post position).

        Returns:
            Valid CameraData instances that pass visibility threshold.
        """
        cfg = self.config
        # Create projector with court config that has the sampled net post position
        projector = CameraProjector(cfg.camera, court_config=court_config)
        valid_cameras: list[CameraData] = []
        for camera in projector.cameras():
            self.total_cameras_tried += 1
            view = projector.generate_camera_view(trajectory, camera=camera)
            cam_data = self._camera_view_to_data(view)
            valid_cameras.append(cam_data)
            self.total_cameras_accepted += 1
        return valid_cameras

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData | None:
        """Generate a single BLCS scene (one rally) with per-scene variation."""
        # 1. Sample per-scene variation
        physics_config = self._sample_physics_config()
        court_config = self._sample_court_config()

        # 2. Build rally simulator with sampled physics
        rally_simulator = self._build_rally_simulator(physics_config)
        physics = BallPhysics(physics_config)

        # 3. Generate rally
        rally_result = rally_simulator.generate_rally(from_cell, side)

        # 4. Generate cameras (with sampled court geometry)
        valid_cameras = self._generate_valid_cameras(
            rally_result.trajectory, court_config
        )

        # 5. Convert shot events to metadata dicts
        shots_meta: list[dict] = []
        for event in rally_result.shot_events:
            shots_meta.append(
                {
                    "shot_index": event.shot_index,
                    "from_side": event.from_side,
                    "from_cell": event.from_cell,
                    "category": event.category.value,
                    "t_start": event.t_start,
                    "t_net": event.t_net,
                    "t_bounce1": event.t_bounce1,
                    "t_bounce2": event.t_bounce2,
                    "t_bounce3": event.t_bounce3,
                    "t_return": event.t_return,
                    "to_cell": event.to_cell,
                    "shot_type": event.shot_type,
                    "return_type": event.return_type,
                }
            )

        # 6. Normalize trajectory
        ball_pos_norm = physics.normalize_position(rally_result.trajectory)
        num_cameras_sampled = len(valid_cameras)

        return BLCSSceneData(
            scene_id=scene_id,
            initial_from_cell=rally_result.initial_from_cell,
            initial_from_side=rally_result.initial_from_side,
            rally_length=rally_result.rally_length,
            end_reason=rally_result.end_reason.value,
            winner_side=rally_result.winner_side,
            shots=shots_meta,
            ball_pos_world=rally_result.trajectory,
            ball_pos_norm=ball_pos_norm,
            ball_vel_world=rally_result.velocities,
            ball_vel_norm=normalize_court_velocity(rally_result.velocities),
            cameras=valid_cameras,
            num_cameras_sampled=num_cameras_sampled,
            fps_out=rally_result.fps_out,
            sim_fps=rally_result.sim_fps,
            physics_config_dict=physics_config.to_dict(),
            court_config_dict={
                "net_post_offset_x": court_config.net_post_offset_x,
            },
            num_balls=1,
        )

    def generate(
        self,
        num_scenes: int,
    ) -> Iterator[BLCSSceneData]:
        """Generate BLCS scenes (rally-only)."""
        if num_scenes < 0:
            raise ValueError(f"num_scenes must be >= 0, got {num_scenes}")

        scene_counter = 0
        for _ in range(num_scenes):
            from_cell = self.sample_from_cell()
            side = self.sample_side()

            scene_id = f"scene_{scene_counter:06d}"
            scene = self.generate_scene(from_cell, side, scene_id)

            if scene is not None:
                scene_counter += 1
                self.total_scenes_generated += 1
                yield scene

                if scene_counter % 100 == 0:
                    logger.info(
                        f"Progress: {scene_counter}/{num_scenes} scenes generated"
                    )

        if scene_counter < num_scenes:
            logger.warning(f"Only generated {scene_counter}/{num_scenes} scenes ")

        stats = self.get_statistics()
        logger.info(f"Generation complete. Total scenes: {scene_counter}")
        logger.info(f"Total cameras: {stats['total_cameras']}")

    def get_statistics(self) -> dict:
        """Get current generation statistics."""
        total_scenes = self.total_scenes_generated
        acceptance_rate = (
            self.total_cameras_accepted / self.total_cameras_tried
            if self.total_cameras_tried > 0
            else 0.0
        )

        return {
            "total_scenes": total_scenes,
            "total_scenes_generated": total_scenes,
            "total_cameras": self.total_cameras_accepted,
            "total_cameras_tried": self.total_cameras_tried,
            "camera_acceptance_rate": acceptance_rate,
            "avg_cameras_per_scene": (
                self.total_cameras_accepted / total_scenes if total_scenes > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset generator state for new generation."""
        self.total_scenes_generated = 0
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0
