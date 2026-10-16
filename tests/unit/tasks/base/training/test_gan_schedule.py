"""Boundary contract shared by BLCS/PLCS epochs and refiner update schedules."""

import pytest

from src.tasks.base.training.gan_schedule import gan_weight_at


def test_wait_ramp_plateau_including_exact_target_boundary():
    weights = [gan_weight_at(index, start=2, warmup=4, target=2.0) for index in range(8)]
    assert weights == [0, 0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]


def test_immediate_single_update_ramp_and_zero_target():
    assert gan_weight_at(0, start=0, warmup=1, target=2.0) == 2
    assert gan_weight_at(0, start=1, warmup=1, target=2.0) == 0
    assert gan_weight_at(99, start=0, warmup=4, target=0.0) == 0


@pytest.mark.parametrize("index,start,warmup,target", [(-1, 0, 1, 2), (0, -1, 1, 2), (0, 0, 0, 2), (0, 0, 1, -2)])
def test_invalid_schedule_rejected(index, start, warmup, target):
    with pytest.raises(ValueError):
        gan_weight_at(index, start=start, warmup=warmup, target=target)
