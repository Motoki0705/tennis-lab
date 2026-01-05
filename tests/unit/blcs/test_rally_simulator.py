"""Unit tests for RallySimulator."""

from __future__ import annotations

import pytest
import torch

from src.blcs.simulation.rally_simulator import (
    RallyConfig,
    RallyEndReason,
    RallyResult,
    RallySimulator,
)
from src.blcs.simulation.shot_simulator import ShotConfig


class TestRallyConfig:
    """Tests for RallyConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RallyConfig()
        assert config.max_rallies == 10
        assert config.max_total_frames == 12000
        assert config.court_margin == 0.5
        assert config.hit_timing_range == (0.2, 0.8)
        assert config.return_z_range == (0.8, 1.4)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RallyConfig(
            max_rallies=5,
            max_total_frames=6000,
            court_margin=1.0,
        )
        assert config.max_rallies == 5
        assert config.max_total_frames == 6000
        assert config.court_margin == 1.0


class TestRallySimulator:
    """Tests for RallySimulator class."""

    @pytest.fixture
    def simulator(self) -> RallySimulator:
        """Create a rally simulator with default config."""
        return RallySimulator(device="cpu")

    @pytest.fixture
    def fast_simulator(self) -> RallySimulator:
        """Create a fast simulator for quick tests."""
        rally_config = RallyConfig(
            max_rallies=3,
            max_total_frames=1000,
        )
        shot_config = ShotConfig(
            max_sim_frames=500,
        )
        return RallySimulator(
            rally_config=rally_config,
            shot_config=shot_config,
            device="cpu",
        )

    def test_initialization(self, simulator: RallySimulator) -> None:
        """Test simulator initialization."""
        assert simulator.physics is not None
        assert simulator.shot_config is not None
        assert simulator.rally_config is not None
        assert simulator.cell_manager is not None

    def test_check_rally_end_net_fault(self, simulator: RallySimulator) -> None:
        """Test rally end detection for net fault."""
        should_end, reason = simulator.check_rally_end(
            bounce_pos=None,
            hit_net_before_bounce=True,
        )
        assert should_end is True
        assert reason == RallyEndReason.NET_FAULT

    def test_check_rally_end_out(self, simulator: RallySimulator) -> None:
        """Test rally end detection for out of bounds."""
        # Bounce far outside court
        out_pos = torch.tensor([10.0, 15.0, 0.0])
        should_end, reason = simulator.check_rally_end(
            bounce_pos=out_pos,
            hit_net_before_bounce=False,
        )
        assert should_end is True
        assert reason == RallyEndReason.OUT

    def test_check_rally_end_in_with_margin(self, simulator: RallySimulator) -> None:
        """Test rally continues for ball within margin."""
        # Bounce just outside court but within margin
        # HALF_DOUBLES_WIDTH = 5.485, margin = 0.5
        # So 5.9 < 5.485 + 0.5 = 5.985 → should be in
        in_pos = torch.tensor([5.9, 10.0, 0.0])
        should_end, reason = simulator.check_rally_end(
            bounce_pos=in_pos,
            hit_net_before_bounce=False,
        )
        assert should_end is False
        assert reason == RallyEndReason.ONGOING

    def test_generate_rally_returns_result(
        self, fast_simulator: RallySimulator
    ) -> None:
        """Test that generate_rally returns a valid RallyResult."""
        result = fast_simulator.generate_rally(from_cell=0, from_side="near")

        assert isinstance(result, RallyResult)
        assert result.trajectory.ndim == 2
        assert result.trajectory.shape[1] == 3
        assert result.velocities.ndim == 2
        assert result.velocities.shape[1] == 3
        assert result.rally_length >= 1
        assert len(result.shot_events) == result.rally_length
        assert result.end_reason != RallyEndReason.ONGOING

    def test_generate_rally_from_both_sides(
        self, fast_simulator: RallySimulator
    ) -> None:
        """Test rally generation from both sides."""
        near_result = fast_simulator.generate_rally(from_cell=0, from_side="near")
        far_result = fast_simulator.generate_rally(from_cell=10, from_side="far")

        assert near_result.initial_from_side == "near"
        assert far_result.initial_from_side == "far"

    def test_shot_events_have_valid_timing(
        self, fast_simulator: RallySimulator
    ) -> None:
        """Test that shot events have monotonically increasing timing."""
        result = fast_simulator.generate_rally(from_cell=0, from_side="near")

        prev_t_start = -1
        for event in result.shot_events:
            assert event.t_start >= prev_t_start
            if event.t_bounce1 >= 0:
                assert event.t_bounce1 >= event.t_start
            if event.t_net >= 0:
                assert event.t_net >= event.t_start
            prev_t_start = event.t_start

    def test_trajectory_length_matches_frames(
        self, fast_simulator: RallySimulator
    ) -> None:
        """Test that trajectory length matches total_frames."""
        result = fast_simulator.generate_rally(from_cell=0, from_side="near")
        assert result.trajectory.shape[0] == result.total_frames

    def test_max_rallies_limit(self) -> None:
        """Test that max_rallies limit is respected."""
        rally_config = RallyConfig(
            max_rallies=2,
            max_total_frames=50000,  # High limit to not interfere
            court_margin=100.0,  # Very large margin to keep ball in
        )
        shot_config = ShotConfig(
            max_sim_frames=500,
        )
        simulator = RallySimulator(
            rally_config=rally_config,
            shot_config=shot_config,
            device="cpu",
        )
        result = simulator.generate_rally(from_cell=0, from_side="near")

        assert result.rally_length <= rally_config.max_rallies
