from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.base.data.lifecycle_slots import pack_lifecycle_slots


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
        randomize_slots=False,
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


def test_train_slot_randomization_is_seeded_and_val_packing_is_deterministic() -> None:
    presence = torch.zeros(12, 2, dtype=torch.bool)
    presence[0:3, 0] = True
    presence[6:9, 1] = True
    first = pack_lifecycle_slots(
        presence,
        num_slots=4,
        randomize_slots=True,
        rng=np.random.default_rng(5),
    )
    second = pack_lifecycle_slots(
        presence,
        num_slots=4,
        randomize_slots=True,
        rng=np.random.default_rng(5),
    )
    deterministic = pack_lifecycle_slots(presence, num_slots=4)

    assert torch.equal(first.track_to_slot, second.track_to_slot)
    assert deterministic.track_to_slot.tolist() == [0, 0]


def test_disjoint_physical_track_and_insufficient_slots_are_rejected() -> None:
    disjoint = torch.tensor([[1], [0], [1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        pack_lifecycle_slots(disjoint, num_slots=1)

    overlap = torch.ones(5, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="cannot be packed"):
        pack_lifecycle_slots(overlap, num_slots=1)
