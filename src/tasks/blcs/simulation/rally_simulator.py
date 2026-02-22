"""Rally simulator for BLCS - Multi-shot rally simulation.

Generates rally sequences by chaining multiple shots:
- Initial shot from one side
- Return shots triggered between 1st and 2nd bounce
- Rally termination on net hit, out-of-bounds, or max rallies

A rally is defined as a sequence of shots where each player
returns the ball between the 1st and 2nd bounce. The rally ends when:
1. Ball hits the net before first bounce (net fault)
2. Ball bounces outside court + margin (out)
3. Maximum rally count reached
4. Maximum simulation frames reached
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.blcs.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)
from src.blcs.simulation.cell_manager import CellManager, ShotCategory
from src.blcs.simulation.shot_simulator import ShotConfig, ShotSimulator
from src.blcs.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)
from src.utils.schema.court import HALF_LENGTH, HALF_SINGLES_WIDTH

if TYPE_CHECKING:
    from src.blcs.generate_dataset.sampling.distribution_sampler import (
        DistributionSampler,
    )


class RallyEndReason(Enum):
    """Reason for rally termination."""

    ONGOING = "ongoing"  # Rally still in progress
    NET_FAULT = "net_fault"  # Ball hit net before first bounce
    OWN_SIDE_BOUNCE = "own_side_bounce"  # Ball bounced on hitter's side
    OUT = "out"  # Ball bounced outside court + margin
    MAX_RALLIES = "max_rallies"  # Maximum rally count reached
    MAX_FRAMES = "max_frames"  # Maximum simulation frames reached
    DOUBLE_BOUNCE = "double_bounce"  # Ball bounced twice (valid point end)


@dataclass
class RallyConfig:
    """Configuration for rally simulation.

    Extends ShotConfig with rally-specific parameters.
    """

    # Maximum number of rallies (shots) before forced termination
    max_rallies: int = 10

    # Maximum total simulation frames across the entire rally
    # At 240Hz, 12000 frames = 50 seconds
    max_total_frames: int = 12000

    # Court margin for out-of-bounds detection (meters)
    # Ball is considered "in" if it bounces within court + margin
    court_margin: float = 0.5

    # Hit timing range: fraction of time between 1st and 2nd bounce
    # (0.0 = at 1st bounce, 1.0 = at 2nd bounce)
    hit_timing_range: tuple[float, float] = (0.2, 0.8)

    # Height range for return shot (meters)
    return_z_range: tuple[float, float] = (0.8, 1.4)

    # Maximum retries for sampling a valid return shot (2nd shot onwards)
    # A valid return shot is one that doesn't immediately end the rally
    max_return_retries: int = 100

    # Minimum rally length to accept (for filtering short rallies)
    min_rally_length: int = 2

    # Probability to accept a net-fault return shot (0.0 - 1.0)
    net_fault_accept_prob: float = 0.05


@dataclass
class ShotEventInfo:
    """Event information for a single shot within a rally."""

    shot_index: int  # 0-indexed shot number in rally
    from_side: str  # "near" or "far"
    from_cell: int  # Starting cell ID

    # Frame indices (relative to rally start, at output_fps)
    t_start: int  # Frame when this shot starts
    t_net: int  # Frame when ball crosses net (-1 if not crossed)
    t_bounce1: int  # First bounce frame (-1 if not bounced)
    t_bounce2: int  # Second bounce frame (-1 if not bounced)
    t_return: int  # Frame when return hit occurs (-1 if rally ended)

    # Positions
    bounce1_pos: Tensor | None  # First bounce position
    bounce2_pos: Tensor | None  # Second bounce position

    # Shot classification
    category: ShotCategory
    to_cell: int  # Target cell (-1 if out/net)


@dataclass
class RallyResult:
    """Result of a rally simulation."""

    # Trajectory data (at output_fps, concatenated across all shots)
    trajectory: Tensor  # [T_total, 3] positions in meters
    velocities: Tensor  # [T_total, 3] velocities in m/s
    trajectory_sim: Tensor  # [T_sim_total, 3] full sim resolution

    # Per-shot event information
    shot_events: list[ShotEventInfo]

    # Rally-level metadata
    rally_length: int  # Number of shots in rally
    end_reason: RallyEndReason  # Why rally ended
    total_frames: int  # Total frames at output_fps
    winner_side: str | None  # "near", "far", or None (if max_frames/rallies)

    # Initial conditions
    initial_from_cell: int
    initial_from_side: str

    # Simulation parameters
    fps_out: int
    sim_fps: int


class RallySimulator:
    """Simulates tennis rallies as sequences of shots.

    Workflow:
    1. Generate initial shot from specified starting position
    2. After 1st bounce, sample return timing before 2nd bounce
    3. Generate return shot from approximate ball position
    4. Repeat until rally termination condition is met

    Rally terminates when:
    - Ball hits net before reaching opponent's court
    - Ball bounces outside court + margin
    - Maximum rally count reached
    - Maximum simulation frames reached
    """

    # Court boundaries for out detection (with margin applied at runtime)
    COURT_X_LIMIT = HALF_SINGLES_WIDTH
    COURT_Y_LIMIT = HALF_LENGTH

    def __init__(
        self,
        physics_config: PhysicsConfig | None = None,
        shot_config: ShotConfig | None = None,
        rally_config: RallyConfig | None = None,
        cell_manager: CellManager | None = None,
        targeted_velocity_config: TargetedVelocityConfig | None = None,
        distribution_sampler: DistributionSampler | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize rally simulator.

        Args:
            physics_config: Physics parameters.
            shot_config: Shot generation parameters.
            rally_config: Rally-specific parameters.
            cell_manager: Cell manager for position sampling.
            targeted_velocity_config: Configuration for targeted velocity sampling.
            distribution_sampler: Distribution sampler for controlled shot distribution.
                When provided, 2nd+ shots will use distribution-controlled sampling.
            device: Torch device.

        """
        self.physics = BallPhysics(physics_config)
        self.physics_config = physics_config or PhysicsConfig()
        self.shot_config = shot_config or ShotConfig()
        self.rally_config = rally_config or RallyConfig()
        self.cell_manager = cell_manager or CellManager()
        self.device = torch.device(device)

        # Create shot simulator for initial condition sampling
        self.shot_simulator = ShotSimulator(
            physics_config=physics_config,
            shot_config=shot_config,
            cell_manager=cell_manager,
            device=device,
        )

        # Create targeted velocity sampler for cell-to-cell shots
        self.targeted_velocity_sampler = TargetedVelocitySampler(
            cell_manager=self.cell_manager,
            config=targeted_velocity_config,
            device=device,
        )

        # Distribution sampler for controlled shot distribution
        self.distribution_sampler = distribution_sampler

    def check_rally_end(
        self,
        bounce_pos: Tensor | None,
        hit_net_before_bounce: bool,
        target_side: str,
    ) -> tuple[bool, RallyEndReason]:
        """Check if rally should end based on current shot result.

        Args:
            bounce_pos: Position of first bounce (None if no bounce).
            hit_net_before_bounce: Whether ball hit net before bouncing.
            target_side: Expected landing side ("near" or "far").

        Returns:
            Tuple of (should_end, reason).

        """
        margin = self.rally_config.court_margin

        # Net fault: ball hit net before first bounce
        if hit_net_before_bounce:
            return True, RallyEndReason.NET_FAULT

        # No bounce occurred (shouldn't happen in normal simulation)
        if bounce_pos is None:
            return True, RallyEndReason.NET_FAULT

        # Check if bounce occurred on the correct (target) side
        x, y, _ = bounce_pos.tolist()
        is_target_side = y > 0 if target_side == "far" else y < 0
        if not is_target_side:
            return True, RallyEndReason.OWN_SIDE_BOUNCE

        # Check if bounce is outside court + margin
        x_limit = self.COURT_X_LIMIT + margin
        y_limit = self.COURT_Y_LIMIT + margin

        if abs(x) > x_limit or abs(y) > y_limit:
            return True, RallyEndReason.OUT

        return False, RallyEndReason.ONGOING

    def _sample_return_timing(
        self,
        t_bounce1_sim: int,
        t_bounce2_sim: int,
        trajectory_sim: list[Tensor],
    ) -> int:
        """Sample the frame at which return hit occurs.

        Args:
            t_bounce1_sim: Frame of first bounce (sim fps).
            t_bounce2_sim: Frame of second bounce (sim fps, -1 if not occurred).
            trajectory_sim: Full-resolution trajectory for this shot.

        Returns:
            Frame index for return hit (sim fps).

        """
        cfg = self.rally_config

        # If no second bounce, use a reasonable estimate
        # (assume ~1 second between bounces at 240Hz)
        if t_bounce2_sim < 0:
            estimated_t_bounce2 = t_bounce1_sim + 240
        else:
            estimated_t_bounce2 = t_bounce2_sim

        # Sample timing within range
        min_frac, max_frac = cfg.hit_timing_range
        frac = min_frac + torch.rand(1).item() * (max_frac - min_frac)

        t_return = int(t_bounce1_sim + frac * (estimated_t_bounce2 - t_bounce1_sim))

        # Prefer return timing where Z is within return_z_range (continuity)
        z_min, z_max = cfg.return_z_range
        max_idx = len(trajectory_sim) - 1
        if max_idx >= 0:
            start = max(0, t_bounce1_sim)
            if t_bounce2_sim >= 0:
                end = min(t_bounce2_sim, max_idx)
            else:
                end = min(estimated_t_bounce2, max_idx)

            if start <= end:
                candidates = []
                for i in range(start, end + 1):
                    z_val = float(trajectory_sim[i][2].item())
                    if z_min <= z_val <= z_max:
                        candidates.append(i)
                if candidates:
                    t_return = min(candidates, key=lambda i: abs(i - t_return))

        return min(t_return, max_idx)

    def _sample_return_initial_state(
        self,
        ball_pos_at_return: Tensor,
        from_side: str,
        target_cell: int | None = None,
    ) -> BallState:
        """Sample initial state for return shot.

        The return shot starts from approximately where the ball would be hit,
        with a new velocity directed toward the opponent's court.

        Args:
            ball_pos_at_return: Ball position at return timing.
            from_side: Side of the court making the return ("near" or "far").
            target_cell: Optional target cell ID to aim for. If provided,
                uses TargetedVelocitySampler for velocity calculation.

        Returns:
            BallState for the return shot.

        """
        target_side = "far" if from_side == "near" else "near"

        # Keep the exact position at return timing to preserve trajectory continuity
        position = ball_pos_at_return.clone()

        # Sample spin (used by refinement if enabled)
        spin = self.shot_simulator._sample_spin()

        # Sample velocity - targeted if target_cell specified, random otherwise
        if target_cell is not None:
            velocity = self.targeted_velocity_sampler.sample_velocity_for_target_cell(
                start_pos=position,
                target_cell=target_cell,
                target_side=target_side,
                from_side=from_side,
                physics=self.physics,
                spin=spin,
            )
        else:
            velocity = self.shot_simulator._sample_velocity(from_side)

        return BallState(position=position, velocity=velocity, spin=spin)

    def _find_cell_for_position(self, position: Tensor, side: str) -> int:
        """Find the cell ID for a given position.

        Args:
            position: 3D position tensor.
            side: "near" or "far".

        Returns:
            Cell ID (0-19).

        """
        return self.cell_manager.position_to_cell_id(position, side)

    def _sample_valid_first_shot(
        self,
        from_cell: int,
        from_side: str,
        max_frames: int,
    ) -> tuple[dict, BallState] | None:
        """Sample a first shot that continues the rally (doesn't end immediately).

        For the 1st shot, we retry sampling until we find a shot that:
        - Doesn't hit the net before first bounce
        - Bounces within court + margin

        Args:
            from_cell: Starting cell ID.
            from_side: Side making the shot.
            max_frames: Maximum frames for this shot.

        Returns:
            Tuple of (shot_result, initial_state) if valid shot found, None otherwise.

        """
        cfg = self.rally_config

        for _ in range(cfg.max_return_retries):
            # Sample initial state using standard shot simulator
            initial_state = self.shot_simulator.sample_initial_condition(
                from_cell=from_cell,
                from_side=from_side,
            )

            # Simulate the shot
            shot_result = self._simulate_single_shot(
                initial_state=initial_state,
                from_cell=from_cell,
                from_side=from_side,
                max_frames=max_frames,
            )

            # Check if this shot continues the rally
            should_end, reason = self.check_rally_end(
                bounce_pos=shot_result["bounce1_pos"],
                hit_net_before_bounce=shot_result["hit_net_before_bounce"],
                target_side="far" if from_side == "near" else "near",
            )

            # Accept shots that don't end the rally immediately
            if not should_end:
                return shot_result, initial_state

        # Failed to find valid first shot after max retries
        return None

    def _get_needed_target_cells(
        self,
        from_cell: int,
        from_side: str,
    ) -> list[int]:
        """Get target cells that need more samples based on distribution.

        If distribution_sampler is not set, returns all in-court cells.

        Args:
            from_cell: Origin cell ID.
            from_side: Side making the shot.

        Returns:
            List of target cell IDs that need more samples.

        """
        if self.distribution_sampler is None:
            # No distribution control - return all in-court cells
            return list(range(9))

        # Get cells that still need samples
        needed: list[tuple[int, int]] = []  # (cell_id, deficit)

        # Check IN_COURT cells (0-8) - prioritize these
        for to_cell in range(9):
            if self.distribution_sampler.should_accept(
                from_cell, from_side, ShotCategory.IN_COURT, to_cell
            ):
                current = self.distribution_sampler.get_current_count(
                    from_cell, from_side, ShotCategory.IN_COURT, to_cell
                )
                target = self.distribution_sampler.get_target_count(
                    from_cell, from_side, ShotCategory.IN_COURT, to_cell
                )
                deficit = target - current
                needed.append((to_cell, deficit))

        # Check OUT_COURT cells (9-19) with lower priority
        for to_cell in range(9, 20):
            if self.distribution_sampler.should_accept(
                from_cell, from_side, ShotCategory.OUT_COURT, to_cell
            ):
                current = self.distribution_sampler.get_current_count(
                    from_cell, from_side, ShotCategory.OUT_COURT, to_cell
                )
                target = self.distribution_sampler.get_target_count(
                    from_cell, from_side, ShotCategory.OUT_COURT, to_cell
                )
                deficit = target - current
                # Lower priority for out-court
                needed.append((to_cell, deficit // 2))

        if not needed:
            # All targets met - return random in-court cells
            return list(range(9))

        # Sort by deficit (highest first) and return cell IDs
        needed.sort(key=lambda x: -x[1])
        return [cell_id for cell_id, _ in needed]

    def _should_accept_net_fault(self) -> bool:
        """Decide whether to accept a net-fault return shot."""
        prob = self.rally_config.net_fault_accept_prob
        if prob <= 0.0:
            return False
        return torch.rand(1).item() < prob

    def _sample_valid_return_shot(
        self,
        ball_pos_at_return: Tensor,
        from_side: str,
        from_cell: int,
        max_frames: int,
    ) -> tuple[dict, BallState] | None:
        """Sample a return shot with distribution control.

        For 2nd shot onwards, uses TargetedVelocitySampler to aim at specific
        cells and DistributionSampler to control the distribution of shots.

        Args:
            ball_pos_at_return: Ball position at return timing.
            from_side: Side making the return.
            from_cell: Cell ID for the return.
            max_frames: Maximum frames for this shot.

        Returns:
            Tuple of (shot_result, initial_state) if valid shot found, None otherwise.

        """
        cfg = self.rally_config

        # Get target cells that need more samples
        needed_targets = self._get_needed_target_cells(from_cell, from_side)

        for attempt in range(cfg.max_return_retries):
            # Cycle through needed targets
            target_cell = needed_targets[attempt % len(needed_targets)]

            # Sample initial state with targeted velocity
            initial_state = self._sample_return_initial_state(
                ball_pos_at_return=ball_pos_at_return,
                from_side=from_side,
                target_cell=target_cell,
            )

            # Simulate the shot
            shot_result = self._simulate_single_shot(
                initial_state=initial_state,
                from_cell=from_cell,
                from_side=from_side,
                max_frames=max_frames,
            )

            # Check if this shot continues the rally
            should_end, reason = self.check_rally_end(
                bounce_pos=shot_result["bounce1_pos"],
                hit_net_before_bounce=shot_result["hit_net_before_bounce"],
                target_side="far" if from_side == "near" else "near",
            )

            # Get actual category and to_cell
            category = shot_result["category"]
            to_cell = shot_result["to_cell"]

            # Check distribution acceptance
            should_accept_distribution = True
            if self.distribution_sampler is not None:
                should_accept_distribution = self.distribution_sampler.should_accept(
                    from_cell, from_side, category, to_cell
                )

            # Accept shots that:
            # 1. Don't end the rally immediately AND meet distribution requirements
            # 2. Or end by net fault with probabilistic acceptance
            # 3. Or have double bounce (valid rally end)
            if not should_end and should_accept_distribution:
                # Record sample if distribution sampler is set
                if self.distribution_sampler is not None:
                    self.distribution_sampler.record_sample(
                        from_cell, from_side, category, to_cell
                    )
                return shot_result, initial_state

            if reason == RallyEndReason.OWN_SIDE_BOUNCE:
                continue

            if reason == RallyEndReason.NET_FAULT:
                if self._should_accept_net_fault():
                    if self.distribution_sampler is not None:
                        self.distribution_sampler.record_sample(
                            from_cell, from_side, category, to_cell
                        )
                    return shot_result, initial_state
                continue

            # Also accept if double bounce occurred (opponent didn't return)
            # These are valid rally endings regardless of distribution
            if shot_result["t_bounce2_sim"] >= 0:
                if self.distribution_sampler is not None:
                    self.distribution_sampler.record_sample(
                        from_cell, from_side, category, to_cell
                    )
                return shot_result, initial_state

        # Failed to find valid return shot after max retries
        return None

    def simulate_rally(
        self,
        from_cell: int,
        from_side: str,
    ) -> RallyResult:
        """Simulate a complete rally.

        For the first shot, we use the standard sampling from from_cell.
        For subsequent shots (2nd onwards), we sample multiple times until
        we find a shot that continues the rally (doesn't immediately end
        with net fault or out).

        Args:
            from_cell: Starting cell ID (0-19).
            from_side: "near" or "far".

        Returns:
            RallyResult with complete rally data.

        """
        cfg = self.rally_config
        shot_cfg = self.shot_config

        # Accumulated trajectory data
        all_positions_sim: list[Tensor] = []
        all_velocities_sim: list[Tensor] = []
        shot_events: list[ShotEventInfo] = []

        # Rally state
        current_side = from_side
        current_cell = from_cell
        total_sim_frames = 0
        rally_count = 0
        end_reason = RallyEndReason.ONGOING
        winner_side: str | None = None

        # For first shot, use standard sampling
        # For 2nd+ shots, use pre-simulated valid return
        pending_shot_result: dict | None = None
        current_state: BallState | None = None

        while rally_count < cfg.max_rallies:
            target_side = "far" if current_side == "near" else "near"

            # Get shot result - either from pending (pre-simulated) or new simulation
            if pending_shot_result is not None:
                # Use pre-simulated valid return shot
                shot_result = pending_shot_result
                pending_shot_result = None
            else:
                # First shot: sample a valid shot that continues the rally
                valid_first = self._sample_valid_first_shot(
                    from_cell=from_cell,
                    from_side=from_side,
                    max_frames=cfg.max_total_frames - total_sim_frames,
                )
                if valid_first is None:
                    # Failed to find valid first shot
                    end_reason = RallyEndReason.OUT
                    winner_side = "far" if from_side == "near" else "near"
                    break
                shot_result, current_state = valid_first

            # Calculate frame offsets for this shot
            t_offset = len(all_positions_sim)
            downsample = shot_cfg.sim_fps // shot_cfg.output_fps

            # Record shot events
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
                t_return=-1,  # Will be set if return occurs
                bounce1_pos=shot_result["bounce1_pos"],
                bounce2_pos=shot_result["bounce2_pos"],
                category=shot_result["category"],
                to_cell=(
                    shot_result["to_cell"] if shot_result["to_cell"] is not None else -1
                ),
            )

            # Check rally termination (net fault or out)
            should_end, reason = self.check_rally_end(
                bounce_pos=shot_result["bounce1_pos"],
                hit_net_before_bounce=shot_result["hit_net_before_bounce"],
                target_side=target_side,
            )

            if should_end:
                # Shot failed - append full trajectory and end
                all_positions_sim.extend(shot_result["trajectory_sim"])
                all_velocities_sim.extend(shot_result["velocities_sim"])
                total_sim_frames += len(shot_result["trajectory_sim"])
                end_reason = reason
                winner_side = target_side  # Opponent wins if we fault
                shot_events.append(shot_info)
                break

            # Shot is valid (landed in court) - now determine return timing
            # Sample return timing between 1st and 2nd bounce
            t_return_sim = self._sample_return_timing(
                t_bounce1_sim=shot_result["t_bounce1_sim"],
                t_bounce2_sim=shot_result["t_bounce2_sim"],
                trajectory_sim=shot_result["trajectory_sim"],
            )

            # Check if 2nd bounce occurred BEFORE return timing
            # This means opponent couldn't reach the ball in time
            if (
                shot_result["t_bounce2_sim"] >= 0
                and shot_result["t_bounce2_sim"] <= t_return_sim
            ):
                # Double bounce before return - point ends
                # Append trajectory up to 2nd bounce
                all_positions_sim.extend(shot_result["trajectory_sim"])
                all_velocities_sim.extend(shot_result["velocities_sim"])
                total_sim_frames += len(shot_result["trajectory_sim"])
                end_reason = RallyEndReason.DOUBLE_BOUNCE
                winner_side = current_side  # We win if opponent doesn't return
                shot_events.append(shot_info)
                break

            # Clamp return timing to available trajectory
            t_return_sim = min(t_return_sim, len(shot_result["trajectory_sim"]) - 1)

            # Get ball position at return time
            ball_pos_at_return = shot_result["trajectory_sim"][t_return_sim]

            # Append trajectory only up to return point
            trajectory_to_add = shot_result["trajectory_sim"][: t_return_sim + 1]
            velocities_to_add = shot_result["velocities_sim"][: t_return_sim + 1]
            all_positions_sim.extend(trajectory_to_add)
            all_velocities_sim.extend(velocities_to_add)
            total_sim_frames += len(trajectory_to_add)

            # Check max frames
            if total_sim_frames >= cfg.max_total_frames:
                end_reason = RallyEndReason.MAX_FRAMES
                shot_events.append(shot_info)
                break

            # Update shot info with return frame
            shot_info.t_return = self._convert_time(t_return_sim, downsample, t_offset)
            shot_events.append(shot_info)

            # Setup next shot
            rally_count += 1
            current_side = target_side
            current_cell = self._find_cell_for_position(ball_pos_at_return, target_side)

            # Sample a valid return shot (one that continues the rally)
            # This pre-simulates the shot and only accepts ones that don't
            # immediately end the rally (net fault or out)
            valid_return = self._sample_valid_return_shot(
                ball_pos_at_return=ball_pos_at_return,
                from_side=current_side,
                from_cell=current_cell,
                max_frames=cfg.max_total_frames - total_sim_frames,
            )

            if valid_return is None:
                # Failed to find valid return shot after max retries
                # End rally here (opponent couldn't return)
                end_reason = RallyEndReason.OUT
                winner_side = "far" if current_side == "near" else "near"
                break

            # Store pre-simulated shot for next iteration
            pending_shot_result, current_state = valid_return

        # Check if we exited due to max rallies
        if end_reason == RallyEndReason.ONGOING:
            end_reason = RallyEndReason.MAX_RALLIES

        # Handle empty trajectory case
        if len(all_positions_sim) == 0:
            # Return a minimal result
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
                fps_out=shot_cfg.output_fps,
                sim_fps=shot_cfg.sim_fps,
            )

        # Stack trajectories
        trajectory_sim = torch.stack(all_positions_sim, dim=0)
        velocities_sim = torch.stack(all_velocities_sim, dim=0)

        # Downsample to output fps
        downsample = shot_cfg.sim_fps // shot_cfg.output_fps
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
            fps_out=shot_cfg.output_fps,
            sim_fps=shot_cfg.sim_fps,
        )

    def _convert_time(self, t_sim: int, downsample: int, offset: int) -> int:
        """Convert simulation frame to output frame with offset.

        Args:
            t_sim: Frame index in simulation FPS.
            downsample: Downsample factor.
            offset: Frame offset (in sim fps).

        Returns:
            Frame index in output FPS, or -1 if t_sim < 0.

        """
        if t_sim < 0:
            return -1
        return (offset + t_sim) // downsample

    def _simulate_single_shot(
        self,
        initial_state: BallState,
        from_cell: int,
        from_side: str,
        max_frames: int,
    ) -> dict:
        """Simulate a single shot within a rally.

        Similar to ShotSimulator.simulate_shot but returns raw data
        for rally integration.

        Args:
            initial_state: Initial ball state.
            from_cell: Starting cell ID.
            from_side: "near" or "far".
            max_frames: Maximum frames for this shot.

        Returns:
            Dictionary with trajectory and event data.

        """
        cfg = self.shot_config
        target_side = "far" if from_side == "near" else "near"

        # Simulation state
        state = initial_state.clone()
        trajectory_sim: list[Tensor] = [state.position.clone()]
        velocities_sim: list[Tensor] = [state.velocity.clone()]

        # Event tracking
        bounce_count = 0
        t_net_sim = -1
        t_bounce1_sim = -1
        t_bounce2_sim = -1
        bounce1_pos: Tensor | None = None
        bounce2_pos: Tensor | None = None
        hit_net_before_bounce = False

        actual_max_frames = min(cfg.max_sim_frames, max_frames)

        for frame in range(actual_max_frames - 1):
            prev_pos = state.position.clone()

            # Physics step
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

            # Check fence collision (terminates shot)
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
                    trajectory_sim.append(state.position.clone())
                    velocities_sim.append(state.velocity.clone())
                    break

            trajectory_sim.append(state.position.clone())
            velocities_sim.append(state.velocity.clone())

        # Classify shot
        category, to_cell = self.cell_manager.classify_shot(
            hit_net_before_bounce=hit_net_before_bounce,
            hit_fence_before_bounce=False,  # Fence doesn't affect rally end
            bounce_pos=bounce1_pos,
            target_side=target_side,
        )

        return {
            "trajectory_sim": trajectory_sim,
            "velocities_sim": velocities_sim,
            "t_net_sim": t_net_sim,
            "t_bounce1_sim": t_bounce1_sim,
            "t_bounce2_sim": t_bounce2_sim,
            "bounce1_pos": bounce1_pos,
            "bounce2_pos": bounce2_pos,
            "hit_net_before_bounce": hit_net_before_bounce,
            "category": category,
            "to_cell": to_cell,
        }

    def generate_rally(self, from_cell: int, from_side: str) -> RallyResult:
        """Generate a complete rally from sampling to simulation.

        Args:
            from_cell: Starting cell ID (0-19).
            from_side: "near" or "far".

        Returns:
            RallyResult: Complete rally result.

        """
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
            assert 0 <= ev.to_cell <= 8
        elif ev.category == ShotCategory.OUT_COURT:
            assert 9 <= ev.to_cell <= 19
        elif ev.category == ShotCategory.DIRECT_NET:
            assert ev.to_cell == -1

    print(
        "smoke ok:",
        f"rally_length={result.rally_length}",
        f"end_reason={result.end_reason.value}",
    )
