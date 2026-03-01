"""Rally simulator for BLCS - Multi-shot rally simulation.

Generates rally sequences by chaining multiple shots:
- Optional serve as first shot
- Return shots: volley (no bounce), normal (1st-2nd bounce), or late (2nd-3rd bounce)
- Rally termination on net hit, out-of-bounds, or max rallies
- No retry loop: each shot is simulated once with targeted velocity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)
from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
    CellManager,
    NUM_CELLS_PER_SIDE,
    NUM_IN_COURT_CELLS,
    ShotCategory,
)
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)
from src.utils.schema.court import HALF_LENGTH, HALF_SINGLES_WIDTH

if TYPE_CHECKING:
    pass


class RallyEndReason(Enum):
    """Reason for rally termination."""

    ONGOING = "ongoing"
    NET_FAULT = "net_fault"
    OWN_SIDE_BOUNCE = "own_side_bounce"
    OUT = "out"
    MAX_RALLIES = "max_rallies"
    MAX_FRAMES = "max_frames"
    DOUBLE_BOUNCE = "double_bounce"


class ShotType(Enum):
    """Type of shot."""

    SERVE = "serve"
    GROUNDSTROKE = "groundstroke"
    VOLLEY = "volley"


@dataclass
class RallyConfig:
    """Configuration for rally simulation."""

    # Initial condition ranges
    z_range: tuple[float, float] = (0.8, 1.4)
    speed_range: tuple[float, float] = (15.0, 35.0)
    azimuth_range_deg: tuple[float, float] = (-30.0, 30.0)
    elevation_range_deg: tuple[float, float] = (5.0, 25.0)
    spin_x_range: tuple[float, float] = (-20.0, 20.0)
    spin_y_range: tuple[float, float] = (-80.0, -40.0)
    spin_z_range: tuple[float, float] = (-20.0, 20.0)

    # Simulation parameters
    max_sim_frames: int = 2000
    output_fps: int = 30
    sim_fps: int = 240

    max_rallies: int = 10
    max_total_frames: int = 12000
    court_margin: float = 0.5
    hit_timing_range: tuple[float, float] = (0.2, 0.8)
    return_z_range: tuple[float, float] = (0.8, 1.4)
    min_rally_length: int = 2
    net_fault_accept_prob: float = 0.05

    # --- Serve ---
    serve_probability: float = 0.3
    serve_speed_range: tuple[float, float] = (30.0, 55.0)
    serve_elevation_range_deg: tuple[float, float] = (2.0, 10.0)
    serve_z_range: tuple[float, float] = (2.0, 2.8)
    serve_azimuth_range_deg: tuple[float, float] = (-15.0, 15.0)

    # --- Volley / multi-bounce ---
    volley_probability: float = 0.05
    normal_return_probability: float = 0.85
    late_return_probability: float = 0.10


@dataclass
class ShotEventInfo:
    """Event information for a single shot within a rally."""

    shot_index: int
    from_side: str
    from_cell: int

    t_start: int
    t_net: int
    t_bounce1: int
    t_bounce2: int
    t_bounce3: int
    t_return: int

    bounce1_pos: Tensor | None
    bounce2_pos: Tensor | None
    bounce3_pos: Tensor | None

    category: ShotCategory
    to_cell: int
    shot_type: str  # ShotType.value
    return_type: str  # "volley" | "normal" | "late_return" | "none"


@dataclass
class RallyResult:
    """Result of a rally simulation."""

    trajectory: Tensor
    velocities: Tensor
    trajectory_sim: Tensor

    shot_events: list[ShotEventInfo]

    rally_length: int
    end_reason: RallyEndReason
    total_frames: int
    winner_side: str | None

    initial_from_cell: int
    initial_from_side: str

    fps_out: int
    sim_fps: int


@dataclass
class ShotResult:
    """Result of a single-shot simulation."""

    trajectory: Tensor
    velocities: Tensor
    trajectory_sim: Tensor
    initial_state: BallState

    t_net: int
    t_fence: int
    t_bounce1: int
    t_bounce2: int

    net_pos: Tensor | None
    bounce1_pos: Tensor | None
    bounce2_pos: Tensor | None

    category: ShotCategory
    to_cell: int | None

    from_cell: int
    from_side: str
    target_side: str
    shot_type: ShotType


class RallySimulator:
    """Simulates tennis rallies as sequences of shots.

    Key changes from previous version:
    - Supports serve as first shot (configurable probability)
    - Supports volley (no bounce) and late return (2nd-3rd bounce) with
      configurable probabilities
    - No retry loop: each shot computed once via targeted velocity
    - Shot type and return type recorded per shot
    """

    COURT_X_LIMIT = HALF_SINGLES_WIDTH
    COURT_Y_LIMIT = HALF_LENGTH

    def __init__(
        self,
        physics_config: PhysicsConfig | None = None,
        rally_config: RallyConfig | None = None,
        cell_manager: CellManager | None = None,
        targeted_velocity_config: TargetedVelocityConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize rally simulator.

        Args:
            physics_config: Physics parameters (already sampled for this rally).
            rally_config: Rally-specific parameters.
            cell_manager: Cell manager for position sampling.
            targeted_velocity_config: Configuration for targeted velocity sampling.
            device: Torch device.
        """
        self.physics = BallPhysics(physics_config)
        self.physics_config = physics_config or PhysicsConfig()
        self.rally_config = rally_config or RallyConfig()
        self.cell_manager = cell_manager or CellManager()
        self.device = torch.device(device)

        self.targeted_velocity_sampler = TargetedVelocitySampler(
            cell_manager=self.cell_manager,
            config=targeted_velocity_config,
            device=device,
        )

    def sample_initial_condition(
        self,
        from_cell: int,
        from_side: str,
        shot_type: ShotType = ShotType.GROUNDSTROKE,
    ) -> BallState:
        """Sample initial conditions for a shot."""
        cfg = self.rally_config
        if shot_type == ShotType.SERVE:
            return self._sample_serve_condition(from_side)

        position = self.cell_manager.sample_position_in_cell(
            cell_id=from_cell,
            side=from_side,
            z_range=cfg.z_range,
            device=self.device,
        )
        velocity = self._sample_velocity(from_side)
        spin = self._sample_spin()
        return BallState(position=position, velocity=velocity, spin=spin)

    def _sample_serve_condition(self, from_side: str) -> BallState:
        """Sample initial conditions for a serve."""
        cfg = self.rally_config
        baseline_y = HALF_LENGTH if from_side == "far" else -HALF_LENGTH
        x = (torch.rand(1).item() - 0.5) * 2.0
        behind = 0.5 + torch.rand(1).item() * 1.0
        y = baseline_y + behind if from_side == "far" else baseline_y - behind
        z = cfg.serve_z_range[0] + torch.rand(1).item() * (
            cfg.serve_z_range[1] - cfg.serve_z_range[0]
        )
        position = torch.tensor([x, y, z], device=self.device)

        velocity = self._sample_velocity(
            from_side,
            speed_range=cfg.serve_speed_range,
            elevation_range_deg=cfg.serve_elevation_range_deg,
            azimuth_range_deg=cfg.serve_azimuth_range_deg,
        )
        spin = self._sample_spin()
        return BallState(position=position, velocity=velocity, spin=spin)

    def _sample_velocity(
        self,
        from_side: str,
        speed_range: tuple[float, float] | None = None,
        elevation_range_deg: tuple[float, float] | None = None,
        azimuth_range_deg: tuple[float, float] | None = None,
    ) -> Tensor:
        """Sample initial velocity vector in m/s."""
        cfg = self.rally_config
        sr = speed_range or cfg.speed_range
        er = elevation_range_deg or cfg.elevation_range_deg
        ar = azimuth_range_deg or cfg.azimuth_range_deg

        speed = sr[0] + torch.rand(1).item() * (sr[1] - sr[0])
        azimuth_deg = ar[0] + torch.rand(1).item() * (ar[1] - ar[0])
        elevation_deg = er[0] + torch.rand(1).item() * (er[1] - er[0])
        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)
        base_dir = 1.0 if from_side == "near" else -1.0

        vx = speed * math.cos(elevation_rad) * math.sin(azimuth_rad)
        vy = speed * math.cos(elevation_rad) * math.cos(azimuth_rad) * base_dir
        vz = speed * math.sin(elevation_rad)
        return torch.tensor([vx, vy, vz], device=self.device)

    def _sample_spin(self) -> Tensor:
        """Sample spin angular velocity in rad/s."""
        cfg = self.rally_config
        spin_x = cfg.spin_x_range[0] + torch.rand(1).item() * (
            cfg.spin_x_range[1] - cfg.spin_x_range[0]
        )
        spin_y = cfg.spin_y_range[0] + torch.rand(1).item() * (
            cfg.spin_y_range[1] - cfg.spin_y_range[0]
        )
        spin_z = cfg.spin_z_range[0] + torch.rand(1).item() * (
            cfg.spin_z_range[1] - cfg.spin_z_range[0]
        )
        return torch.tensor([spin_x, spin_y, spin_z], device=self.device)

    def simulate_shot(
        self,
        initial_state: BallState,
        from_cell: int,
        from_side: str,
        shot_type: ShotType = ShotType.GROUNDSTROKE,
    ) -> ShotResult:
        """Simulate a single shot until second bounce/fence/max frames."""
        cfg = self.rally_config
        target_side = "far" if from_side == "near" else "near"

        state = initial_state.clone()
        trajectory_sim: list[Tensor] = [state.position.clone()]
        velocities_sim: list[Tensor] = [state.velocity.clone()]

        bounce_count = 0
        t_net_sim = -1
        t_fence_sim = -1
        t_bounce1_sim = -1
        t_bounce2_sim = -1
        net_pos: Tensor | None = None
        bounce1_pos: Tensor | None = None
        bounce2_pos: Tensor | None = None
        hit_net_before_bounce = False
        hit_fence_before_bounce = False

        for frame in range(cfg.max_sim_frames - 1):
            prev_pos = state.position.clone()
            state = self.physics.step(state)

            if bounce_count == 0 and t_net_sim < 0:
                hit_net, pos_at_net = self.physics.check_net_collision(
                    prev_pos, state.position
                )
                if hit_net:
                    t_net_sim = frame + 1
                    net_pos = pos_at_net
                    hit_net_before_bounce = True
                    state = self.physics.apply_net_collision(state, net_pos=pos_at_net)

            if self.physics.check_fence_collision(state.position):
                if bounce_count == 0:
                    hit_fence_before_bounce = True
                t_fence_sim = frame + 1
                trajectory_sim.append(state.position.clone())
                velocities_sim.append(state.velocity.clone())
                break

            state, bounced = self.physics.handle_bounce(state)
            if bounced:
                bounce_count += 1
                if bounce_count == 1:
                    t_bounce1_sim = frame + 1
                    bounce1_pos = state.position.clone()
                elif bounce_count == 2:
                    t_bounce2_sim = frame + 1
                    bounce2_pos = state.position.clone()
                    trajectory_sim.append(state.position.clone())
                    velocities_sim.append(state.velocity.clone())
                    break

            trajectory_sim.append(state.position.clone())
            velocities_sim.append(state.velocity.clone())

        trajectory_sim_tensor = torch.stack(trajectory_sim, dim=0)
        velocities_sim_tensor = torch.stack(velocities_sim, dim=0)
        downsample_factor = cfg.sim_fps // cfg.output_fps
        trajectory = trajectory_sim_tensor[::downsample_factor]
        velocities = velocities_sim_tensor[::downsample_factor]

        def convert_time(t_sim: int) -> int:
            return -1 if t_sim < 0 else t_sim // downsample_factor

        category, to_cell = self.cell_manager.classify_shot(
            hit_net_before_bounce=hit_net_before_bounce,
            hit_fence_before_bounce=hit_fence_before_bounce,
            bounce_pos=bounce1_pos,
            target_side=target_side,
        )

        return ShotResult(
            trajectory=trajectory,
            velocities=velocities,
            trajectory_sim=trajectory_sim_tensor,
            initial_state=initial_state,
            t_net=convert_time(t_net_sim),
            t_fence=convert_time(t_fence_sim),
            t_bounce1=convert_time(t_bounce1_sim),
            t_bounce2=convert_time(t_bounce2_sim),
            net_pos=net_pos,
            bounce1_pos=bounce1_pos,
            bounce2_pos=bounce2_pos,
            category=category,
            to_cell=to_cell,
            from_cell=from_cell,
            from_side=from_side,
            target_side=target_side,
            shot_type=shot_type,
        )

    # ------------------------------------------------------------------
    # Rally end detection
    # ------------------------------------------------------------------

    def check_rally_end(
        self,
        bounce_pos: Tensor | None,
        hit_net_before_bounce: bool,
        target_side: str,
    ) -> tuple[bool, RallyEndReason]:
        """Check if rally should end based on current shot result."""
        margin = self.rally_config.court_margin

        if hit_net_before_bounce:
            return True, RallyEndReason.NET_FAULT

        if bounce_pos is None:
            return True, RallyEndReason.NET_FAULT

        x, y, _ = bounce_pos.tolist()
        is_target_side = y > 0 if target_side == "far" else y < 0
        if not is_target_side:
            return True, RallyEndReason.OWN_SIDE_BOUNCE

        x_limit = self.COURT_X_LIMIT + margin
        y_limit = self.COURT_Y_LIMIT + margin

        if abs(x) > x_limit or abs(y) > y_limit:
            return True, RallyEndReason.OUT

        return False, RallyEndReason.ONGOING

    # ------------------------------------------------------------------
    # Return type sampling
    # ------------------------------------------------------------------

    def _decide_return_type(self) -> str:
        """Decide how the opponent returns the ball."""
        cfg = self.rally_config
        r = torch.rand(1).item()
        if r < cfg.volley_probability:
            return "volley"
        elif r < cfg.volley_probability + cfg.normal_return_probability:
            return "normal"
        else:
            return "late_return"

    # ------------------------------------------------------------------
    # Return timing
    # ------------------------------------------------------------------

    def _sample_return_timing(
        self,
        return_type: str,
        t_net_sim: int,
        t_bounce1_sim: int,
        t_bounce2_sim: int,
        t_bounce3_sim: int,
        trajectory_sim: list[Tensor],
    ) -> int:
        """Sample frame for return hit based on return_type.

        Args:
            return_type: "volley", "normal", or "late_return".
            t_net_sim: Frame when ball crosses net.
            t_bounce1_sim: First bounce frame.
            t_bounce2_sim: Second bounce frame.
            t_bounce3_sim: Third bounce frame.
            trajectory_sim: Full-res trajectory.

        Returns:
            Frame index for return hit (sim fps).
        """
        cfg = self.rally_config
        max_idx = len(trajectory_sim) - 1
        if max_idx <= 0:
            return 0

        if return_type == "volley":
            # Return between net crossing and first bounce (or end)
            start = max(0, t_net_sim + 1)
            end = t_bounce1_sim if t_bounce1_sim >= 0 else max_idx
            end = min(end - 1, max_idx)  # before bounce
            if start > end:
                start = max(0, t_net_sim)
                end = max(start, max_idx)
        elif return_type == "late_return":
            # Return between 2nd and 3rd bounce
            if t_bounce2_sim >= 0:
                start = t_bounce2_sim
                end = t_bounce3_sim if t_bounce3_sim >= 0 else min(
                    t_bounce2_sim + 240, max_idx
                )
            else:
                # No 2nd bounce -> fallback to normal timing
                start = max(0, t_bounce1_sim)
                estimated_b2 = t_bounce1_sim + 240 if t_bounce1_sim >= 0 else max_idx
                end = min(estimated_b2, max_idx)
        else:  # normal
            start = max(0, t_bounce1_sim)
            if t_bounce2_sim >= 0:
                end = t_bounce2_sim
            else:
                end = min(start + 240, max_idx)

        end = min(end, max_idx)
        if start > end:
            return min(start, max_idx)

        # Sample timing within range
        min_frac, max_frac = cfg.hit_timing_range
        frac = min_frac + torch.rand(1).item() * (max_frac - min_frac)
        t_return = int(start + frac * (end - start))

        # Prefer z within return_z_range
        z_min, z_max = cfg.return_z_range
        candidates = []
        for i in range(start, min(end + 1, max_idx + 1)):
            z_val = float(trajectory_sim[i][2].item())
            if z_min <= z_val <= z_max:
                candidates.append(i)
        if candidates:
            t_return = min(candidates, key=lambda i: abs(i - t_return))

        return min(t_return, max_idx)

    # ------------------------------------------------------------------
    # Return shot generation
    # ------------------------------------------------------------------

    def _sample_return_initial_state(
        self,
        ball_pos_at_return: Tensor,
        from_side: str,
        target_cell: int,
    ) -> BallState:
        """Sample initial state for return shot (single attempt, no retry)."""
        target_side = "far" if from_side == "near" else "near"

        position = ball_pos_at_return.clone()
        spin = self._sample_spin()

        velocity = self.targeted_velocity_sampler.sample_velocity_for_target_cell(
            start_pos=position,
            target_cell=target_cell,
            target_side=target_side,
            from_side=from_side,
            physics=self.physics,
            spin=spin,
        )

        return BallState(position=position, velocity=velocity, spin=spin)

    def _sample_target_cell(self) -> int:
        """Sample a target cell for the next shot.

        Returns a random in-court cell with bias toward service boxes and
        back court, with small probability of out-court targeting.
        """
        r = torch.rand(1).item()
        if r < 0.85:
            # In-court (0-5)
            return int(torch.randint(0, NUM_IN_COURT_CELLS, (1,)).item())
        else:
            # Out-court (6-8) — intentional faults for variation
            return int(
                NUM_IN_COURT_CELLS
                + torch.randint(0, NUM_CELLS_PER_SIDE - NUM_IN_COURT_CELLS, (1,)).item()
            )

    # ------------------------------------------------------------------
    # Single shot simulation (extended for 3 bounces)
    # ------------------------------------------------------------------

    def _simulate_single_shot(
        self,
        initial_state: BallState,
        from_cell: int,
        from_side: str,
        max_frames: int,
        max_bounces: int = 3,
    ) -> dict:
        """Simulate a single shot within a rally.

        Extended to track up to 3 bounces for late-return support.
        """
        cfg = self.rally_config
        target_side = "far" if from_side == "near" else "near"

        state = initial_state.clone()
        trajectory_sim: list[Tensor] = [state.position.clone()]
        velocities_sim: list[Tensor] = [state.velocity.clone()]

        bounce_count = 0
        t_net_sim = -1
        t_bounce1_sim = -1
        t_bounce2_sim = -1
        t_bounce3_sim = -1
        bounce1_pos: Tensor | None = None
        bounce2_pos: Tensor | None = None
        bounce3_pos: Tensor | None = None
        hit_net_before_bounce = False

        actual_max_frames = min(cfg.max_sim_frames, max_frames)

        for frame in range(actual_max_frames - 1):
            prev_pos = state.position.clone()

            state = self.physics.step(state)

            # Check net collision
            if bounce_count == 0 and t_net_sim < 0:
                hit_net, pos_at_net = self.physics.check_net_collision(
                    prev_pos, state.position
                )
                if hit_net:
                    t_net_sim = frame + 1
                    hit_net_before_bounce = True
                    state = self.physics.apply_net_collision(state, net_pos=pos_at_net)

            # Record net crossing (even when ball clears net)
            if t_net_sim < 0 and bounce_count == 0:
                cleared = self.physics.compute_net_clearance(prev_pos, state.position)
                if cleared is not None and cleared >= 0:
                    t_net_sim = frame + 1

            # Check fence collision
            if self.physics.check_fence_collision(state.position):
                trajectory_sim.append(state.position.clone())
                velocities_sim.append(state.velocity.clone())
                break

            # Handle bounce
            state, bounced = self.physics.handle_bounce(state)
            if bounced:
                bounce_count += 1
                if bounce_count == 1:
                    t_bounce1_sim = frame + 1
                    bounce1_pos = state.position.clone()
                elif bounce_count == 2:
                    t_bounce2_sim = frame + 1
                    bounce2_pos = state.position.clone()
                elif bounce_count == 3:
                    t_bounce3_sim = frame + 1
                    bounce3_pos = state.position.clone()
                    trajectory_sim.append(state.position.clone())
                    velocities_sim.append(state.velocity.clone())
                    break

            trajectory_sim.append(state.position.clone())
            velocities_sim.append(state.velocity.clone())

        category, to_cell = self.cell_manager.classify_shot(
            hit_net_before_bounce=hit_net_before_bounce,
            hit_fence_before_bounce=False,
            bounce_pos=bounce1_pos,
            target_side=target_side,
        )

        return {
            "trajectory_sim": trajectory_sim,
            "velocities_sim": velocities_sim,
            "t_net_sim": t_net_sim,
            "t_bounce1_sim": t_bounce1_sim,
            "t_bounce2_sim": t_bounce2_sim,
            "t_bounce3_sim": t_bounce3_sim,
            "bounce1_pos": bounce1_pos,
            "bounce2_pos": bounce2_pos,
            "bounce3_pos": bounce3_pos,
            "hit_net_before_bounce": hit_net_before_bounce,
            "category": category,
            "to_cell": to_cell,
        }

    # ------------------------------------------------------------------
    # Main rally loop
    # ------------------------------------------------------------------

    def simulate_rally(
        self,
        from_cell: int,
        from_side: str,
    ) -> RallyResult:
        """Simulate a complete rally.

        Key design decisions:
        - First shot may be a serve (configurable probability)
        - Each return is a single targeted-velocity attempt (no retry loop)
        - Return type (volley/normal/late_return) sampled per return
        - Shot always accepted regardless of landing cell
        """
        cfg = self.rally_config
        all_positions_sim: list[Tensor] = []
        all_velocities_sim: list[Tensor] = []
        shot_events: list[ShotEventInfo] = []

        current_side = from_side
        current_cell = from_cell
        total_sim_frames = 0
        rally_count = 0
        end_reason = RallyEndReason.ONGOING
        winner_side: str | None = None

        # Decide if first shot is a serve
        is_serve = torch.rand(1).item() < cfg.serve_probability
        first_shot_type = ShotType.SERVE if is_serve else ShotType.GROUNDSTROKE

        while rally_count < cfg.max_rallies:
            target_side = "far" if current_side == "near" else "near"

            # --- Generate shot ---
            if rally_count == 0:
                # First shot
                shot_type = first_shot_type
                if shot_type == ShotType.SERVE:
                    # Serve: start from behind baseline, target service box
                    initial_state = self.sample_initial_condition(
                        from_cell=current_cell,
                        from_side=current_side,
                        shot_type=ShotType.SERVE,
                    )
                else:
                    initial_state = self.sample_initial_condition(
                        from_cell=current_cell,
                        from_side=current_side,
                        shot_type=ShotType.GROUNDSTROKE,
                    )
            else:
                shot_type = ShotType.GROUNDSTROKE

            shot_result = self._simulate_single_shot(
                initial_state=initial_state,
                from_cell=current_cell,
                from_side=current_side,
                max_frames=cfg.max_total_frames - total_sim_frames,
                max_bounces=3,
            )

            # --- Decide return type ---
            if rally_count < cfg.max_rallies - 1:
                return_type = self._decide_return_type()
            else:
                return_type = "none"

            # --- Calculate frame offsets ---
            t_offset = len(all_positions_sim)
            downsample = cfg.sim_fps // cfg.output_fps

            shot_info = ShotEventInfo(
                shot_index=rally_count,
                from_side=current_side,
                from_cell=current_cell,
                t_start=t_offset // downsample,
                t_net=self._convert_time(shot_result["t_net_sim"], downsample, t_offset),
                t_bounce1=self._convert_time(
                    shot_result["t_bounce1_sim"], downsample, t_offset
                ),
                t_bounce2=self._convert_time(
                    shot_result["t_bounce2_sim"], downsample, t_offset
                ),
                t_bounce3=self._convert_time(
                    shot_result["t_bounce3_sim"], downsample, t_offset
                ),
                t_return=-1,
                bounce1_pos=shot_result["bounce1_pos"],
                bounce2_pos=shot_result["bounce2_pos"],
                bounce3_pos=shot_result["bounce3_pos"],
                category=shot_result["category"],
                to_cell=(
                    shot_result["to_cell"] if shot_result["to_cell"] is not None else -1
                ),
                shot_type=shot_type.value,
                return_type=return_type,
            )

            # --- Check rally termination ---
            should_end, reason = self.check_rally_end(
                bounce_pos=shot_result["bounce1_pos"],
                hit_net_before_bounce=shot_result["hit_net_before_bounce"],
                target_side=target_side,
            )

            if should_end:
                all_positions_sim.extend(shot_result["trajectory_sim"])
                all_velocities_sim.extend(shot_result["velocities_sim"])
                total_sim_frames += len(shot_result["trajectory_sim"])
                end_reason = reason
                winner_side = target_side
                shot_events.append(shot_info)
                break

            # --- Sample return timing ---
            t_return_sim = self._sample_return_timing(
                return_type=return_type,
                t_net_sim=shot_result["t_net_sim"],
                t_bounce1_sim=shot_result["t_bounce1_sim"],
                t_bounce2_sim=shot_result["t_bounce2_sim"],
                t_bounce3_sim=shot_result["t_bounce3_sim"],
                trajectory_sim=shot_result["trajectory_sim"],
            )

            # Check double/triple bounce before return
            if return_type == "normal":
                if (
                    shot_result["t_bounce2_sim"] >= 0
                    and shot_result["t_bounce2_sim"] <= t_return_sim
                ):
                    all_positions_sim.extend(shot_result["trajectory_sim"])
                    all_velocities_sim.extend(shot_result["velocities_sim"])
                    total_sim_frames += len(shot_result["trajectory_sim"])
                    end_reason = RallyEndReason.DOUBLE_BOUNCE
                    winner_side = current_side
                    shot_events.append(shot_info)
                    break

            t_return_sim = min(t_return_sim, len(shot_result["trajectory_sim"]) - 1)

            ball_pos_at_return = shot_result["trajectory_sim"][t_return_sim]

            trajectory_to_add = shot_result["trajectory_sim"][: t_return_sim + 1]
            velocities_to_add = shot_result["velocities_sim"][: t_return_sim + 1]
            all_positions_sim.extend(trajectory_to_add)
            all_velocities_sim.extend(velocities_to_add)
            total_sim_frames += len(trajectory_to_add)

            if total_sim_frames >= cfg.max_total_frames:
                end_reason = RallyEndReason.MAX_FRAMES
                shot_events.append(shot_info)
                break

            shot_info.t_return = self._convert_time(t_return_sim, downsample, t_offset)
            shot_events.append(shot_info)

            # --- Setup next shot ---
            rally_count += 1
            current_side = target_side
            current_cell = self.cell_manager.position_to_cell_id(
                ball_pos_at_return, target_side
            )

            # Sample target and initial state for next shot (no retry)
            target_cell = self._sample_target_cell()

            # For volley, the shot type is VOLLEY
            if return_type == "volley":
                shot_type = ShotType.VOLLEY
            else:
                shot_type = ShotType.GROUNDSTROKE

            initial_state = self._sample_return_initial_state(
                ball_pos_at_return=ball_pos_at_return,
                from_side=current_side,
                target_cell=target_cell,
            )

        if end_reason == RallyEndReason.ONGOING:
            end_reason = RallyEndReason.MAX_RALLIES

        if len(all_positions_sim) == 0:
            return RallyResult(
                trajectory=torch.zeros(1, 3),
                velocities=torch.zeros(1, 3),
                trajectory_sim=torch.zeros(1, 3),
                shot_events=[],
                rally_length=0,
                end_reason=end_reason,
                total_frames=0,
                winner_side=winner_side,
                initial_from_cell=from_cell,
                initial_from_side=from_side,
                fps_out=cfg.output_fps,
                sim_fps=cfg.sim_fps,
            )

        trajectory_sim = torch.stack(all_positions_sim, dim=0)
        velocities_sim = torch.stack(all_velocities_sim, dim=0)

        downsample = cfg.sim_fps // cfg.output_fps
        trajectory = trajectory_sim[::downsample]
        velocities = velocities_sim[::downsample]

        return RallyResult(
            trajectory=trajectory,
            velocities=velocities,
            trajectory_sim=trajectory_sim,
            shot_events=shot_events,
            rally_length=len(shot_events),
            end_reason=end_reason,
            total_frames=len(trajectory),
            winner_side=winner_side,
            initial_from_cell=from_cell,
            initial_from_side=from_side,
            fps_out=cfg.output_fps,
            sim_fps=cfg.sim_fps,
        )

    def _convert_time(self, t_sim: int, downsample: int, offset: int) -> int:
        """Convert simulation frame to output frame with offset."""
        if t_sim < 0:
            return -1
        return (offset + t_sim) // downsample

    def generate_rally(self, from_cell: int, from_side: str) -> RallyResult:
        """Generate a complete rally from sampling to simulation."""
        return self.simulate_rally(from_cell, from_side)


if __name__ == "__main__":
    torch.manual_seed(0)

    simulator = RallySimulator(device="cpu")
    assert simulator.COURT_X_LIMIT == HALF_SINGLES_WIDTH

    result = simulator.generate_rally(from_cell=0, from_side="near")
    assert result.rally_length == len(result.shot_events)

    for idx, ev in enumerate(result.shot_events):
        assert ev.shot_index == idx
        if ev.category == ShotCategory.IN_COURT:
            assert 0 <= ev.to_cell < NUM_IN_COURT_CELLS
        elif ev.category == ShotCategory.OUT_COURT:
            assert NUM_IN_COURT_CELLS <= ev.to_cell < NUM_CELLS_PER_SIDE
        elif ev.category == ShotCategory.DIRECT_NET:
            assert ev.to_cell == -1

    print(
        "smoke ok:",
        f"rally_length={result.rally_length}",
        f"end_reason={result.end_reason.value}",
    )
