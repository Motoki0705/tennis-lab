"""Rally timeline: every simulation sample advances physical time."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallySimulator

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def test_hits_do_not_repeat_the_contact_sample() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = build_generator_config(compose(config_name="generate_dataset"))
    checked_hits = 0
    rallies = 0
    for seed in range(40):
        torch.manual_seed(seed)
        simulator = RallySimulator(
            physics_config=config.physics.sample(),
            rally_config=config.rally,
            cell_manager=CellManager(),
            targeted_velocity_config=config.targeted_velocity,
            device="cpu",
        )
        try:
            rally = simulator.generate_rally(from_cell=1, from_side="near")
        except RuntimeError:
            continue  # A rejected physics proposal; generation resamples it.
        rallies += 1
        steps = rally.trajectory_sim.diff(dim=0).norm(dim=-1)
        # A ball in flight always moves; a stall would be an exactly zero step.
        assert bool((steps > 0).all())
        checked_hits += max(0, len(rally.shot_events) - 1)
        if rallies == 3:
            break
    assert rallies == 3 and checked_hits > 0
