"""Scene generator for BLCS dataset generation (rally-only scene format)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from src.tasks.blcs.generate_dataset.sampling.distribution_sampler import (
    DistributionSampler,
    SamplingConfig,
)
from src.tasks.blcs.simulation.ball_physics import BallPhysics, PhysicsConfig
from src.tasks.blcs.simulation.cell_manager import CellManager
from src.tasks.blcs.simulation.rally_simulator import (
    RallyConfig,
    RallySimulator,
)
from src.tasks.blcs.simulation.shot_simulator import ShotConfig
from src.tasks.blcs.simulation.targeted_velocity_sampler import TargetedVelocityConfig
from src.utils.projection.camera_projector import (
    CameraConfig,
    CameraProjector,
    CameraView,
)

logger = logging.getLogger(__name__)

@dataclass
class CameraData:
    """Data for a single valid camera view."""

    camera_params: dict

    ball_uv: np.ndarray  # [T, 2] normalized UV coordinates
    ball_visible: np.ndarray  # [T] visibility flags
    ball_visibility_ratio: float  # Ratio of visible frames

    court_kp_uv: np.ndarray  # [20, 2] UV coordinates
    court_kp_visible: np.ndarray  # [20] visibility flags
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

    cameras: list[CameraData]
    num_cameras_sampled: int  # Total cameras tried (before filtering)

    fps_out: int
    sim_fps: int


@dataclass
class GeneratorConfig:
    """Configuration for scene generator."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    targeted_velocity: TargetedVelocityConfig = field(
        default_factory=TargetedVelocityConfig
    )

    # Camera sampling parameters
    num_cameras_sampled: int = 15  # Number of cameras to try per scene
    ball_visibility_threshold: float = 0.8  # Min ratio of visible ball frames

    max_attempts_multiplier: int = 10


class BLCSSceneGenerator:
    """Generate rally-based BLCS scenes with distribution-controlled sampling."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or GeneratorConfig()
        self.device = torch.device(device)

        self.cell_manager = CellManager()
        self.distribution_sampler = DistributionSampler(self.config.sampling)

        self.rally_simulator = RallySimulator(
            physics_config=self.config.physics,
            shot_config=self.config.shot,
            rally_config=self.config.rally,
            cell_manager=self.cell_manager,
            targeted_velocity_config=self.config.targeted_velocity,
            distribution_sampler=self.distribution_sampler,
            device=device,
        )
        self.camera_projector = CameraProjector(self.config.camera)
        self.physics = BallPhysics(self.config.physics)

        # Track statistics
        self.total_scenes_generated = 0
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0

    def _camera_view_to_data(self, view: CameraView) -> CameraData:
        """Convert CameraView to CameraData with visibility metrics."""
        # Compute ball visibility ratio
        ball_vis = view.points_visible.numpy()
        T = len(ball_vis)
        ball_visibility_ratio = float(ball_vis.sum()) / T if T > 0 else 0.0

        # Compute court visibility count (average visible keypoints)
        court_vis = view.court_kp_visible.numpy()
        court_visibility_count = float(court_vis.sum())

        return CameraData(
            camera_params=view.camera_params,
            ball_uv=view.points_uv.numpy(),
            ball_visible=ball_vis,
            ball_visibility_ratio=ball_visibility_ratio,
            court_kp_uv=view.court_kp_uv.numpy(),
            court_kp_visible=court_vis,
            court_visibility_count=court_visibility_count,
        )

    def _generate_valid_cameras(
        self,
        trajectory: Tensor,
    ) -> list[CameraData]:
        """Generate cameras and filter by visibility threshold.

        Args:
            trajectory: Ball trajectory [T, 3].

        Returns:
            list: Valid CameraData instances that pass visibility threshold.

        """
        cfg = self.config
        valid_cameras: list[CameraData] = []

        if cfg.camera.placement_mode == "fixed_8":
            for camera in self.camera_projector.fixed_cameras():
                self.total_cameras_tried += 1
                view = self.camera_projector.generate_camera_view(trajectory, camera=camera)
                cam_data = self._camera_view_to_data(view)
                valid_cameras.append(cam_data)
                self.total_cameras_accepted += 1
            return valid_cameras

        for _ in range(cfg.num_cameras_sampled):
            self.total_cameras_tried += 1

            # Generate camera view
            view = self.camera_projector.generate_camera_view(trajectory)
            cam_data = self._camera_view_to_data(view)

            # Check visibility threshold
            if cam_data.ball_visibility_ratio >= cfg.ball_visibility_threshold:
                valid_cameras.append(cam_data)
                self.total_cameras_accepted += 1

        return valid_cameras

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData | None:
        """Generate a single BLCS scene (one rally)."""
        cfg = self.config

        # 1. Generate rally
        rally_result = self.rally_simulator.generate_rally(from_cell, side)

        # 2. Generate and filter cameras
        valid_cameras = self._generate_valid_cameras(rally_result.trajectory)

        # 3. If no valid cameras, return None
        if not valid_cameras:
            return None

        # 4. Convert shot events to metadata dicts
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
                    "t_return": event.t_return,
                    "to_cell": event.to_cell,
                }
            )

        # 5. Normalize trajectory
        ball_pos_norm = self.physics.normalize_position(rally_result.trajectory)
        num_cameras_sampled = (
            len(valid_cameras)
            if cfg.camera.placement_mode == "fixed_8"
            else cfg.num_cameras_sampled
        )

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
            cameras=valid_cameras,
            num_cameras_sampled=num_cameras_sampled,
            fps_out=rally_result.fps_out,
            sim_fps=rally_result.sim_fps,
        )

    def generate(
        self,
        num_scenes: int,
    ) -> Iterator[BLCSSceneData]:
        """Generate BLCS scenes (rally-only)."""
        if num_scenes < 0:
            raise ValueError(f"num_scenes must be >= 0, got {num_scenes}")

        cfg = self.config
        scene_counter = 0
        attempts = 0
        max_attempts = num_scenes * max(1, int(cfg.max_attempts_multiplier))

        while scene_counter < num_scenes and attempts < max_attempts:
            attempts += 1

            # Random starting position
            from_cell = int(torch.randint(0, 20, (1,)).item())
            side = "near" if torch.rand(1).item() < 0.5 else "far"

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
            logger.warning(
                f"Only generated {scene_counter}/{num_scenes} scenes "
                f"after {attempts} attempts"
            )

        stats = self.get_statistics()
        logger.info(f"Generation complete. Total scenes: {scene_counter}")
        logger.info(f"Total cameras: {stats['total_cameras']}")

    def get_statistics(self) -> dict:
        """Get current generation statistics.

        Returns:
            dict: Statistics including camera acceptance rate.

        """
        base_stats = self.distribution_sampler.get_statistics()

        total_shots_sampled = int(base_stats.get("total_samples", 0))
        total_scenes = self.total_scenes_generated
        acceptance_rate = (
            self.total_cameras_accepted / self.total_cameras_tried
            if self.total_cameras_tried > 0
            else 0.0
        )

        return {
            **base_stats,
            "total_samples": total_shots_sampled,
            "total_shots_sampled": total_shots_sampled,
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
        self.distribution_sampler.reset()
        self.total_scenes_generated = 0
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0
