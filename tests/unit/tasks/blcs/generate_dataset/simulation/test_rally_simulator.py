"""Focused return-side invariants for BLCS rally construction."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyConfig,
    RallySimulator,
)


def _timing_simulator() -> RallySimulator:
    simulator = object.__new__(RallySimulator)
    simulator.rally_config = RallyConfig(
        z_range=(0.8, 1.4),
        spin_x_range=(-1.0, 1.0),
        spin_y_range=(-1.0, 1.0),
        spin_z_range=(-1.0, 1.0),
        max_sim_frames=120,
        output_fps=30,
        sim_fps=120,
        max_rallies=4,
        max_total_frames=240,
        hit_timing_range=(0.2, 0.8),
        return_z_range=(0.8, 1.4),
        serve_probability=0.0,
        serve_z_range=(2.0, 2.5),
        toss_vz_range=(3.0, 4.0),
        toss_xy_noise_range=(-0.1, 0.1),
        toss_max_frames=60,
        toss_z0_tolerance=0.1,
        volley_probability=0.0,
        normal_return_probability=1.0,
        late_return_probability=0.0,
        out_court_target_probability=0.0,
    )
    simulator.cell_manager = CellManager()
    return simulator


@pytest.mark.parametrize(
    ("target_side", "y_values"),
    [
        ("far", (-4.3, -1.0, 0.2, 2.0, 5.4)),
        ("near", (4.3, 1.0, -0.2, -2.0, -5.4)),
    ],
)
def test_volley_timing_with_missing_net_event_uses_requested_side_only(
    target_side: str,
    y_values: tuple[float, ...],
) -> None:
    simulator = _timing_simulator()
    trajectory = [torch.tensor([0.0, y, 1.0]) for y in y_values]

    torch.manual_seed(695)
    first = simulator._sample_return_timing(
        return_type="volley",
        target_side=target_side,
        t_net_sim=-1,
        t_bounce1_sim=len(trajectory),
        t_bounce2_sim=-1,
        t_bounce3_sim=-1,
        trajectory_sim=trajectory,
    )
    torch.manual_seed(695)
    repeated = simulator._sample_return_timing(
        return_type="volley",
        target_side=target_side,
        t_net_sim=-1,
        t_bounce1_sim=len(trajectory),
        t_bounce2_sim=-1,
        t_bounce3_sim=-1,
        trajectory_sim=trajectory,
    )

    assert first == repeated
    assert simulator.cell_manager.is_position_in_cell_grid(
        trajectory[first], target_side
    )


def test_return_timing_rejects_a_window_outside_the_requested_side() -> None:
    simulator = _timing_simulator()
    trajectory = [torch.tensor([0.0, y, 1.0]) for y in (-4.3, -3.0, -2.0)]

    with pytest.raises(RuntimeError, match="requested canonical court side"):
        simulator._sample_return_timing(
            return_type="volley",
            target_side="far",
            t_net_sim=-1,
            t_bounce1_sim=len(trajectory),
            t_bounce2_sim=-1,
            t_bounce3_sim=-1,
            trajectory_sim=trajectory,
        )
