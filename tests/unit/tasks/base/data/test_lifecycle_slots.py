from __future__ import annotations

import pytest
import torch

from src.tasks.base.data.lifecycle_slots import (
    build_fixed_lifecycle_assignment,
    pack_lifecycle_slots,
)


def test_interval_coloring_reuses_slot_after_gap_and_preserves_instance_ids() -> None:
    presence = torch.zeros(24, 4, dtype=torch.bool)
    presence[0:5, 0] = True
    presence[3:9, 1] = True
    presence[12:17, 2] = True
    presence[20:23, 3] = True

    packed = pack_lifecycle_slots(
        presence,
        num_slots=2,
        min_reuse_gap_frames=2,
    )

    assert packed.track_to_slot.tolist() == [0, 1, 0, 0]
    assert packed.target_instance_id[1, 0].item() == 0
    assert packed.target_instance_id[13, 0].item() == 2
    assert packed.target_instance_id[21, 0].item() == 3
    assert (packed.target_instance_id[~packed.target_presence] == -1).all()

    values = torch.arange(24 * 4 * 3).reshape(24, 4, 3).float()
    target = packed.pack_tensor(values, presence)
    torch.testing.assert_close(target[13, 0], values[13, 2])
    assert target[10, 0].eq(0).all()


def test_interval_coloring_is_replayable_and_ties_use_physical_index_order() -> None:
    presence = torch.zeros(12, 3, dtype=torch.bool)
    presence[0:3, 1] = True
    presence[0:3, 2] = True
    presence[6:9, 0] = True

    first = pack_lifecycle_slots(presence, num_slots=4)
    second = pack_lifecycle_slots(presence, num_slots=4)

    assert first.track_to_slot.tolist() == [0, 0, 1]
    torch.testing.assert_close(first.track_to_slot, second.track_to_slot)
    torch.testing.assert_close(first.target_presence, second.target_presence)
    torch.testing.assert_close(first.target_instance_id, second.target_instance_id)


def test_disjoint_physical_track_and_insufficient_slots_are_rejected() -> None:
    disjoint = torch.tensor([[1], [0], [1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        pack_lifecycle_slots(disjoint, num_slots=1)

    overlap = torch.ones(5, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="cannot be packed"):
        pack_lifecycle_slots(overlap, num_slots=1)


def test_fixed_assignment_has_exact_width_and_deterministic_interval_packing() -> None:
    presence = torch.zeros(10, 3, dtype=torch.bool)
    presence[0:4, 0] = True
    presence[1:5, 1] = True
    presence[7:10, 2] = True

    first = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
    )
    second = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
    )

    assert first.track_to_slot.tolist() == [0, 1, 0]
    assert first.target_presence.shape == (10, 4)
    assert first.target_instance_id.shape == (10, 4)
    torch.testing.assert_close(first.track_to_slot, second.track_to_slot)
    torch.testing.assert_close(first.target_presence, second.target_presence)
    torch.testing.assert_close(first.target_instance_id, second.target_instance_id)


def test_random_relabel_arguments_are_destructively_removed() -> None:
    presence = torch.ones(2, 1, dtype=torch.bool)

    with pytest.raises(TypeError, match="randomize_slots"):
        pack_lifecycle_slots(
            presence,
            num_slots=1,
            randomize_slots=True,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="generator"):
        build_fixed_lifecycle_assignment(
            presence,
            num_slots=1,
            min_reuse_gap_frames=0,
            generator=torch.Generator(),  # type: ignore[call-arg]
        )


def test_fixed_assignment_rejects_capacity_overflow_without_truncation() -> None:
    presence = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"cannot be packed.*num_slots=2"):
        build_fixed_lifecycle_assignment(
            presence,
            num_slots=2,
            min_reuse_gap_frames=0,
        )


@pytest.mark.parametrize(
    ("presence", "num_slots", "gap", "message"),
    [
        (torch.ones(2, dtype=torch.bool), 1, 0, "shape"),
        (torch.ones(2, 1, dtype=torch.bool), 0, 0, "positive"),
        (torch.ones(2, 1, dtype=torch.bool), 1, -1, "non-negative"),
    ],
)
def test_lifecycle_assignment_validates_shape_capacity_and_gap(
    presence: torch.Tensor,
    num_slots: int,
    gap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_lifecycle_slots(
            presence,
            num_slots=num_slots,
            min_reuse_gap_frames=gap,
        )
