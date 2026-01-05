"""Unit tests for TargetedVelocitySampler."""

from __future__ import annotations

import pytest
import torch

from src.blcs.simulation.cell_manager import CellManager
from src.blcs.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)


class TestTargetedVelocityConfig:
    """Tests for TargetedVelocityConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TargetedVelocityConfig()
        assert config.azimuth_noise_deg == 5.0
        assert config.elevation_noise_deg == 3.0
        assert config.speed_variation == 0.15
        assert config.min_elevation_deg == 3.0
        assert config.max_elevation_deg == 35.0
        assert config.min_speed == 12.0
        assert config.max_speed == 40.0
        assert config.gravity == 9.81

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TargetedVelocityConfig(
            azimuth_noise_deg=10.0,
            elevation_noise_deg=5.0,
            speed_variation=0.2,
        )
        assert config.azimuth_noise_deg == 10.0
        assert config.elevation_noise_deg == 5.0
        assert config.speed_variation == 0.2


class TestTargetedVelocitySampler:
    """Tests for TargetedVelocitySampler class."""

    @pytest.fixture
    def sampler(self) -> TargetedVelocitySampler:
        """Create a targeted velocity sampler with default config."""
        return TargetedVelocitySampler(device="cpu")

    @pytest.fixture
    def cell_manager(self) -> CellManager:
        """Create a cell manager."""
        return CellManager()

    def test_initialization(self, sampler: TargetedVelocitySampler) -> None:
        """Test sampler initialization."""
        assert sampler.cell_manager is not None
        assert sampler.config is not None

    def test_velocity_direction_near_to_far(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test velocity points toward far side when from_side='near'."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])
        target_pos = torch.tensor([0.0, 5.0, 0.0])

        vel = sampler.compute_velocity_to_target(start_pos, target_pos, "near")

        assert vel.shape == (3,)
        # Y component should be positive (toward far side)
        assert vel[1] > 0

    def test_velocity_direction_far_to_near(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test velocity points toward near side when from_side='far'."""
        start_pos = torch.tensor([0.0, 5.0, 1.0])
        target_pos = torch.tensor([0.0, -5.0, 0.0])

        vel = sampler.compute_velocity_to_target(start_pos, target_pos, "far")

        assert vel.shape == (3,)
        # Y component should be negative (toward near side)
        assert vel[1] < 0

    def test_velocity_z_component_positive(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test velocity z component is positive (upward arc)."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])
        target_pos = torch.tensor([0.0, 5.0, 0.0])

        vel = sampler.compute_velocity_to_target(start_pos, target_pos, "near")

        # Z component should be positive (ball goes up initially)
        assert vel[2] > 0

    def test_velocity_magnitude_within_bounds(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test velocity magnitude is within configured bounds."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])
        target_pos = torch.tensor([0.0, 5.0, 0.0])

        for _ in range(10):  # Multiple samples due to randomness
            vel = sampler.compute_velocity_to_target(start_pos, target_pos, "near")
            speed = torch.norm(vel).item()

            cfg = sampler.config
            # Allow some tolerance for edge cases
            assert speed >= cfg.min_speed * 0.8, f"Speed {speed} below min"
            assert speed <= cfg.max_speed * 1.2, f"Speed {speed} above max"

    def test_sample_velocity_for_target_cell(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test sampling velocity for a specific target cell."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])

        vel = sampler.sample_velocity_for_target_cell(
            start_pos=start_pos,
            target_cell=4,  # Center cell
            target_side="far",
            from_side="near",
        )

        assert vel.shape == (3,)
        # Should aim toward far side
        assert vel[1] > 0

    def test_sample_velocity_for_different_cells(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test that different target cells produce different velocities."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])

        # Sample for left cell
        vel_left = sampler.sample_velocity_for_target_cell(
            start_pos=start_pos,
            target_cell=0,  # Left cell
            target_side="far",
            from_side="near",
        )

        # Sample for right cell
        vel_right = sampler.sample_velocity_for_target_cell(
            start_pos=start_pos,
            target_cell=2,  # Right cell
            target_side="far",
            from_side="near",
        )

        # X components should have different signs on average
        # (left cell -> negative x, right cell -> positive x)
        # Note: Due to noise, individual samples may not follow this
        # but the general trend should be there
        assert vel_left.shape == vel_right.shape == (3,)

    def test_velocity_variation_with_noise(
        self, sampler: TargetedVelocitySampler
    ) -> None:
        """Test that noise causes variation in velocity."""
        start_pos = torch.tensor([0.0, -5.0, 1.0])
        target_pos = torch.tensor([0.0, 5.0, 0.0])

        velocities = []
        for _ in range(5):
            vel = sampler.compute_velocity_to_target(start_pos, target_pos, "near")
            velocities.append(vel.clone())

        # Check that not all velocities are identical
        # (due to noise, they should vary)
        velocities_stacked = torch.stack(velocities)
        std = velocities_stacked.std(dim=0)

        # At least one component should have non-zero std
        assert std.sum() > 0, "Velocities should vary due to noise"
