"""Tests for BLCS physics simulation."""

import pytest
import torch

from src.blcs.utils.constants import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
)
from src.blcs.utils.physics import BallPhysics, ShotType, generate_random_shot


class TestBallPhysics:
    """Tests for BallPhysics."""

    def test_init(self) -> None:
        """Test initialization."""
        physics = BallPhysics()
        assert physics.gravity == pytest.approx(9.81)

    def test_init_custom_gravity(self) -> None:
        """Test initialization with custom gravity."""
        physics = BallPhysics(gravity=10.0)
        assert physics.gravity == pytest.approx(10.0)

    def test_compute_acceleration(self) -> None:
        """Test acceleration computation."""
        physics = BallPhysics(use_drag=False)

        vel = torch.tensor([10.0, 5.0, 0.0])
        accel = physics.compute_acceleration(vel)

        # Should have gravity in z direction
        assert accel[2] < 0
        assert accel[2] == pytest.approx(-9.81)

    def test_step(self) -> None:
        """Test single physics step."""
        from src.blcs.utils.physics import BallState

        physics = BallPhysics(use_drag=False)

        state = BallState(
            position=torch.tensor([0.0, 0.0, 2.0]),
            velocity=torch.tensor([10.0, 5.0, 0.0]),
        )

        new_state = physics.step(state, dt=1 / 30)

        # Position should change
        assert not torch.allclose(new_state.position, state.position)
        # Velocity z should decrease due to gravity
        assert new_state.velocity[2] < state.velocity[2]

    def test_simulate_trajectory(self) -> None:
        """Test full trajectory simulation."""
        physics = BallPhysics()

        pos = torch.tensor([0.0, -5.0, 1.5])
        vel = torch.tensor([5.0, 15.0, 3.0])

        trajectory = physics.simulate_trajectory(
            initial_pos=pos,
            initial_vel=vel,
            num_frames=60,
        )

        assert trajectory.shape == (60, 3)

    def test_trajectory_gravity(self) -> None:
        """Test that gravity affects trajectory."""
        physics = BallPhysics(use_drag=False)

        pos = torch.tensor([0.0, 0.0, 5.0])
        vel = torch.tensor([0.0, 0.0, 0.0])  # No initial velocity

        trajectory = physics.simulate_trajectory(
            initial_pos=pos,
            initial_vel=vel,
            num_frames=30,
        )

        # Ball should fall due to gravity
        assert trajectory[-1, 2] < trajectory[0, 2]

    def test_handle_bounce(self) -> None:
        """Test bounce handling."""
        from src.blcs.utils.physics import BallState

        physics = BallPhysics(cor=0.8)

        # Ball below ground with downward velocity
        state = BallState(
            position=torch.tensor([0.0, 0.0, -0.1]),
            velocity=torch.tensor([5.0, 5.0, -10.0]),
        )

        new_state, bounced = physics.handle_bounce(state)

        assert bounced
        # After bounce, vertical velocity should be reversed and reduced
        assert new_state.velocity[2] > 0
        assert abs(new_state.velocity[2]) < abs(state.velocity[2])

    def test_normalize_trajectory(self) -> None:
        """Test trajectory normalization."""
        physics = BallPhysics()

        trajectory = torch.tensor([[5.485, 11.885, 1.07]])  # At court boundary
        normalized = physics.normalize_trajectory(trajectory)

        assert normalized[0, 0] == pytest.approx(1.0)
        assert normalized[0, 1] == pytest.approx(1.0)
        assert normalized[0, 2] == pytest.approx(1.0)

    def test_denormalize_trajectory(self) -> None:
        """Test trajectory denormalization."""
        physics = BallPhysics()

        normalized = torch.tensor([[0.5, 0.5, 0.5]])
        trajectory = physics.denormalize_trajectory(normalized)

        assert trajectory[0, 0] == pytest.approx(5.485 * 0.5)
        assert trajectory[0, 1] == pytest.approx(11.885 * 0.5)
        assert trajectory[0, 2] == pytest.approx(1.07 * 0.5)

    def test_net_collision(self) -> None:
        """Test net collision detection."""
        physics = BallPhysics()

        # Ball crossing net at low height
        pos1 = torch.tensor([0.0, -0.1, 0.3])  # Before net
        pos2 = torch.tensor([0.0, 0.1, 0.3])  # After net

        hit_net = physics.check_net_collision(pos1, pos2)
        # At 0.3m, below net height, should detect collision
        assert hit_net


class TestInitialConditions:
    """Tests for initial condition generators."""

    def test_generate_random_shot_flat(self) -> None:
        """Test flat shot initial conditions generation."""
        pos, vel = generate_random_shot(shot_type=ShotType.FLAT)

        # Position should be in court area
        assert abs(pos[0]) < HALF_DOUBLES_WIDTH + 2
        assert abs(pos[1]) < HALF_LENGTH + 2
        assert pos[2] > 0  # Above ground

    def test_generate_random_shot_topspin(self) -> None:
        """Test topspin shot initial conditions generation."""
        pos, vel = generate_random_shot(shot_type=ShotType.TOPSPIN)

        # Position should be in court area
        assert abs(pos[0]) < HALF_DOUBLES_WIDTH + 2
        assert abs(pos[1]) < HALF_LENGTH + 2
        assert pos[2] > 0  # Above ground

    def test_generate_random_shot_lob(self) -> None:
        """Test lob shot initial conditions generation."""
        pos, vel = generate_random_shot(shot_type=ShotType.LOB)

        # Position should be in court area
        assert abs(pos[0]) < HALF_DOUBLES_WIDTH + 2
        assert pos[2] > 0  # Above ground

    def test_random_variation(self) -> None:
        """Test that random generators produce variation."""
        positions = []
        for _ in range(10):
            pos, _ = generate_random_shot()
            positions.append(pos.clone())

        positions = torch.stack(positions)
        # Should have some variation
        assert positions.std(dim=0).sum() > 0


class TestPhysicsConsistency:
    """Tests for physics consistency."""

    def test_trajectory_continuity(self) -> None:
        """Test that trajectory is continuous (no teleportation)."""
        physics = BallPhysics()

        pos = torch.tensor([0.0, -5.0, 1.5])
        vel = torch.tensor([5.0, 15.0, 3.0])

        trajectory = physics.simulate_trajectory(
            initial_pos=pos,
            initial_vel=vel,
            num_frames=60,
        )

        # Check maximum distance between consecutive frames
        diffs = (trajectory[1:] - trajectory[:-1]).norm(dim=-1)
        dt = 1 / 30  # Default fps
        max_speed = 60.0  # m/s reasonable max for tennis ball
        max_dist = max_speed * dt

        assert (diffs < max_dist * 2).all()  # Allow some margin for bounces

    def test_gravity_only(self) -> None:
        """Test pure gravity motion without drag."""
        physics = BallPhysics(use_drag=False)

        pos = torch.tensor([0.0, 0.0, 10.0])
        vel = torch.tensor([0.0, 0.0, 0.0])

        trajectory = physics.simulate_trajectory(
            initial_pos=pos,
            initial_vel=vel,
            num_frames=30,
            stop_on_second_bounce=False,
        )

        # Ball should fall (z decreases)
        assert trajectory[10, 2] < trajectory[0, 2]
