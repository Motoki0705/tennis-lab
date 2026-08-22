"""Contract tests for BLCS court-cell mapping."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
    X_MAX,
    X_MIN,
    Y_MAX,
)


@pytest.mark.parametrize(
    ("position", "side"),
    [
        ((-4.6300435066223145, -0.013850942254066467, 0.0), "far"),
        ((4.6300435066223145, 0.013850942254066467, 0.0), "near"),
        ((-6.0, -1.0, 0.0), "far"),
        ((6.0, 1.0, 0.0), "near"),
    ],
)
def test_position_mapping_rejects_the_wrong_canonical_side(
    position: tuple[float, float, float],
    side: str,
) -> None:
    manager = CellManager()

    with pytest.raises(ValueError, match="requested canonical court side"):
        manager.position_to_cell_id(torch.tensor(position), side=side)


@pytest.mark.parametrize(
    ("canonical_position", "cell_id"),
    [
        ((-HALF_SINGLES_WIDTH, 0.0, 0.0), 0),
        ((0.0, 0.0, 0.0), 1),
        ((HALF_SINGLES_WIDTH, 0.0, 0.0), 1),
        ((-1.0, SERVICE_LINE_DISTANCE, 0.0), 2),
        ((1.0, SERVICE_LINE_DISTANCE, 0.0), 3),
        ((-HALF_DOUBLES_WIDTH, HALF_LENGTH, 0.0), 4),
        ((HALF_DOUBLES_WIDTH, HALF_LENGTH, 0.0), 5),
        ((X_MIN, 1.0, 0.0), 6),
        ((X_MAX, 1.0, 0.0), 7),
        ((0.0, abs(Y_MAX), 0.0), 8),
    ],
)
def test_boundary_ownership_is_exact_and_mirrored_between_sides(
    canonical_position: tuple[float, float, float],
    cell_id: int,
) -> None:
    manager = CellManager()
    x, y, z = canonical_position

    assert manager.position_to_cell_id(torch.tensor([x, y, z]), "far") == cell_id
    assert manager.position_to_cell_id(torch.tensor([-x, -y, z]), "near") == cell_id


def test_net_centre_line_is_a_shared_closed_endpoint_without_tolerance() -> None:
    manager = CellManager()

    far = torch.tensor([-4.6300435066223145, 0.0, 0.0])
    near = torch.tensor([4.6300435066223145, 0.0, 0.0])
    assert manager.position_to_cell_id(far, "far") == 4
    assert manager.position_to_cell_id(near, "near") == 4

    with pytest.raises(ValueError, match="requested canonical court side"):
        manager.position_to_cell_id(torch.tensor([0.0, -1e-6, 0.0]), "far")
    with pytest.raises(ValueError, match="requested canonical court side"):
        manager.position_to_cell_id(torch.tensor([0.0, 1e-6, 0.0]), "near")


@pytest.mark.parametrize("side", ["near", "far"])
def test_sampled_bounces_stay_in_every_requested_cell(side: str) -> None:
    manager = CellManager()
    torch.manual_seed(695)

    for cell_id in manager.get_all_cell_ids():
        position = manager.sample_bounce_position_in_cell(
            cell_id,
            side,
            margin=0.35,
        )
        assert manager.position_to_cell_id(position, side) == cell_id


@pytest.mark.parametrize("side", ["near", "far"])
def test_zero_rng_endpoint_is_constructed_inside_each_requested_cell(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    manager = CellManager()
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: torch.zeros(1))

    for cell_id in manager.get_all_cell_ids():
        launch = manager.sample_position_in_cell(cell_id, side)
        bounce = manager.sample_bounce_position_in_cell(cell_id, side)
        assert manager.position_to_cell_id(launch, side) == cell_id
        assert manager.position_to_cell_id(bounce, side) == cell_id


@pytest.mark.parametrize(
    "position",
    [
        (X_MIN - 0.01, 1.0, 0.0),
        (X_MAX + 0.01, 1.0, 0.0),
        (0.0, abs(Y_MAX) + 0.01, 0.0),
    ],
)
def test_position_mapping_rejects_points_outside_the_bounded_cell_grid(
    position: tuple[float, float, float],
) -> None:
    manager = CellManager()

    with pytest.raises(ValueError, match="outside the canonical half-court cell grid"):
        manager.position_to_cell_id(torch.tensor(position), "far")
