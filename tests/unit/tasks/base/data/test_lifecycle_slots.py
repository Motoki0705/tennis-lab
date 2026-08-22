from __future__ import annotations

import numpy as np
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


def test_fixed_assignment_has_exact_width_and_deterministic_interval_packing() -> None:
    presence = torch.zeros(10, 3, dtype=torch.bool)
    presence[0:4, 0] = True
    presence[1:5, 1] = True
    presence[7:10, 2] = True

    first = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
        generator=None,
    )
    second = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
        generator=None,
    )

    assert first.track_to_slot.tolist() == [0, 1, 0]
    assert first.target_presence.shape == (10, 4)
    assert first.target_instance_id.shape == (10, 4)
    torch.testing.assert_close(first.track_to_slot, second.track_to_slot)
    torch.testing.assert_close(first.target_presence, second.target_presence)
    torch.testing.assert_close(first.target_instance_id, second.target_instance_id)


def test_fixed_assignment_relabels_all_outputs_with_explicit_torch_generator() -> None:
    presence = torch.ones(3, 3, dtype=torch.bool)
    deterministic = build_fixed_lifecycle_assignment(
        presence,
        num_slots=5,
        min_reuse_gap_frames=0,
        randomize_slots=False,
        generator=None,
    )
    expected_permutation = torch.randperm(
        5,
        generator=torch.Generator().manual_seed(753),
    )

    randomized = build_fixed_lifecycle_assignment(
        presence,
        num_slots=5,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=torch.Generator().manual_seed(753),
    )

    torch.testing.assert_close(
        randomized.track_to_slot,
        expected_permutation[deterministic.track_to_slot],
    )
    inverse_permutation = expected_permutation.argsort()
    torch.testing.assert_close(
        randomized.target_presence,
        deterministic.target_presence[:, inverse_permutation],
    )
    torch.testing.assert_close(
        randomized.target_instance_id,
        deterministic.target_instance_id[:, inverse_permutation],
    )


def test_fixed_assignment_uses_current_torch_rng_for_independent_draws() -> None:
    presence = torch.ones(2, 4, dtype=torch.bool)

    torch.manual_seed(753)
    first = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=None,
    )
    second = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=None,
    )

    torch.manual_seed(753)
    replay_first = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=None,
    )
    replay_second = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=None,
    )

    torch.testing.assert_close(first.track_to_slot, replay_first.track_to_slot)
    torch.testing.assert_close(second.track_to_slot, replay_second.track_to_slot)
    assert not torch.equal(first.track_to_slot, second.track_to_slot)


def test_fixed_assignment_rejects_capacity_overflow_without_truncation() -> None:
    presence = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"cannot be packed.*num_slots=2"):
        build_fixed_lifecycle_assignment(
            presence,
            num_slots=2,
            min_reuse_gap_frames=0,
            randomize_slots=False,
            generator=None,
        )
