"""Scene generator for BLCS dataset (blcs.md §7 compliant).

Orchestrates:
- Shot simulation
- Camera projection with visibility filtering
- Distribution-controlled sampling
- Scene data packaging (PLCS-unified format)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from src.blcs.data.camera_projector import CameraConfig, CameraProjector, CameraView
from src.blcs.data.distribution_sampler import DistributionSampler, SamplingConfig
from src.blcs.simulation.ball_physics import BallPhysics, PhysicsConfig
from src.blcs.simulation.cell_manager import CellManager, ShotCategory
from src.blcs.simulation.shot_simulator import ShotConfig, ShotSimulator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CameraData:
    """Data for a single valid camera view (PLCS-aligned structure)."""

    # Camera parameters (serializable)
    camera_params: dict

    # Ball trajectory projection
    ball_uv: np.ndarray  # [T, 2] normalized UV coordinates
    ball_visible: np.ndarray  # [T] visibility flags
    ball_visibility_ratio: float  # Ratio of visible frames

    # Court keypoints projection
    court_kp_uv: np.ndarray  # [20, 2] UV coordinates
    court_kp_visible: np.ndarray  # [20] visibility flags
    court_visibility_count: float  # Average visible keypoints


@dataclass
class BLCSSceneData:
    """Complete scene data for BLCS (1 scene = 1 file with N cameras)."""

    # Identifiers
    scene_id: str

    # Origin info
    from_cell: int
    from_side: str  # "near" or "far"

    # Classification
    category: ShotCategory
    to_cell: int | None  # None for DIRECT_NET/DIRECT_FENCE

    # 3D trajectory data
    ball_pos_world: Tensor  # [T, 3] world coordinates (meters)
    ball_pos_norm: Tensor  # [T, 3] normalized coordinates
    ball_vel_world: Tensor  # [T, 3] velocities (m/s)

    # Event times (frame index, -1 if not occurred)
    t_net: int
    t_fence: int
    t_bounce1: int
    t_bounce2: int

    # Multiple valid camera views
    cameras: list[CameraData]
    num_cameras_sampled: int  # Total cameras tried (before filtering)

    # Simulation metadata
    fps_out: int
    sim_fps: int


@dataclass
class GeneratorConfig:
    """Configuration for scene generator."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Camera sampling parameters
    num_cameras_sampled: int = 15  # Number of cameras to try per scene
    ball_visibility_threshold: float = 0.8  # Min ratio of visible ball frames

    max_attempts_per_cell: int = 10000


class BLCSSceneGenerator:
    """Generates BLCS training scenes with controlled distribution.

    New workflow (PLCS-aligned):
    1. Generate shot via physics simulation
    2. Sample multiple candidate cameras (num_cameras_sampled)
    3. Filter by ball visibility threshold
    4. If any valid cameras, yield scene with all valid cameras
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or GeneratorConfig()
        self.device = torch.device(device)

        self.cell_manager = CellManager()
        self.shot_simulator = ShotSimulator(
            physics_config=self.config.physics,
            shot_config=self.config.shot,
            cell_manager=self.cell_manager,
            device=device,
        )
        self.camera_projector = CameraProjector(self.config.camera)
        self.distribution_sampler = DistributionSampler(self.config.sampling)
        self.physics = BallPhysics(self.config.physics)

        # Track statistics
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0

    def _camera_view_to_data(self, view: CameraView) -> CameraData:
        """Convert CameraView to CameraData with visibility metrics."""
        # Compute ball visibility ratio
        ball_vis = view.ball_visible.numpy()
        T = len(ball_vis)
        ball_visibility_ratio = float(ball_vis.sum()) / T if T > 0 else 0.0

        # Compute court visibility count (average visible keypoints)
        court_vis = view.court_kp_visible.numpy()
        court_visibility_count = float(court_vis.sum())

        return CameraData(
            camera_params=view.camera_params,
            ball_uv=view.ball_uv.numpy(),
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
        valid_cameras = []

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

    def generate_scenes_for_cell(
        self,
        from_cell: int,
        side: str,
    ) -> Iterator[BLCSSceneData]:
        """Generate scenes for a specific from_cell until targets met.

        Args:
            from_cell: Origin cell ID (0-19).
            side: "near" or "far".

        Yields:
            BLCSSceneData for each generated scene (1 scene = 1 file with N cameras).

        """
        cfg = self.config
        sampler = self.distribution_sampler
        attempts = 0
        scene_count = 0

        while not sampler.is_from_cell_complete(from_cell, side):
            if attempts >= cfg.max_attempts_per_cell:
                logger.warning(
                    f"Max attempts ({cfg.max_attempts_per_cell}) reached for "
                    f"from_cell={from_cell}, side={side}. "
                    f"Completion: {sampler.get_completion_ratio(from_cell, side):.2%}"
                )
                break

            attempts += 1

            # 1. Generate shot
            shot_result = self.shot_simulator.generate_shot(from_cell, side)

            # 2. Check if this shot category/cell is needed
            if not sampler.should_accept(
                from_cell, side, shot_result.category, shot_result.to_cell
            ):
                continue

            # 3. Generate and filter cameras
            valid_cameras = self._generate_valid_cameras(shot_result.trajectory)

            # 4. If no valid cameras, skip this shot (don't count it)
            if not valid_cameras:
                continue

            # 5. Record sample and yield scene
            sampler.record_sample(
                from_cell, side, shot_result.category, shot_result.to_cell
            )

            scene_id = f"scene_{scene_count:06d}"
            ball_pos_norm = self.physics.normalize_position(shot_result.trajectory)

            yield BLCSSceneData(
                scene_id=scene_id,
                from_cell=from_cell,
                from_side=side,
                category=shot_result.category,
                to_cell=shot_result.to_cell,
                ball_pos_world=shot_result.trajectory,
                ball_pos_norm=ball_pos_norm,
                ball_vel_world=shot_result.velocities,
                t_net=shot_result.t_net,
                t_fence=shot_result.t_fence,
                t_bounce1=shot_result.t_bounce1,
                t_bounce2=shot_result.t_bounce2,
                cameras=valid_cameras,
                num_cameras_sampled=cfg.num_cameras_sampled,
                fps_out=cfg.shot.output_fps,
                sim_fps=cfg.shot.sim_fps,
            )
            scene_count += 1

        logger.info(
            f"Completed from_cell={from_cell}, side={side}: "
            f"{scene_count} scenes from {attempts} attempts, "
            f"completion={sampler.get_completion_ratio(from_cell, side):.2%}"
        )

    def generate_all_scenes(self) -> Iterator[BLCSSceneData]:
        """Generate scenes for all from_cells and sides.

        Yields:
            BLCSSceneData for each generated scene.

        """
        scene_counter = 0
        for side in ["near", "far"]:
            for from_cell in range(20):
                logger.info(f"Generating scenes for from_cell={from_cell}, side={side}")
                for scene in self.generate_scenes_for_cell(from_cell, side):
                    # Assign global scene ID
                    scene.scene_id = f"scene_{scene_counter:06d}"
                    scene_counter += 1
                    yield scene

        stats = self.get_statistics()
        logger.info(f"Generation complete. Total scenes: {stats['total_scenes']}")
        logger.info(f"Total cameras: {stats['total_cameras']}")
        logger.info(f"Avg cameras per scene: {stats['avg_cameras_per_scene']:.2f}")

    def get_statistics(self) -> dict:
        """Get current generation statistics.

        Returns:
            dict: Statistics including camera acceptance rate.

        """
        base_stats = self.distribution_sampler.get_statistics()

        # Add camera statistics
        total_scenes = base_stats.get("total_samples", 0)
        acceptance_rate = (
            self.total_cameras_accepted / self.total_cameras_tried
            if self.total_cameras_tried > 0
            else 0.0
        )

        return {
            **base_stats,
            "total_scenes": total_scenes,
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
        self.total_cameras_tried = 0
        self.total_cameras_accepted = 0
