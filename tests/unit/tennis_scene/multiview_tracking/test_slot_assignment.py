import pytest

from src.tennis_scene.multiview_tracking.slot_assignment import (
    FragmentLink,
    assign_slots,
)


def test_exclusive_assignment_maximizes_both_slot_paths_together() -> None:
    # Fragments 2 and 3 occupy the same early interval, before both cores.
    # Fragment 2 is locally best for core 0, but is the only option for core 1.
    # Giving 3 to core 0 and 2 to core 1 yields .87 + .90, versus .95 otherwise.
    options = [
        FragmentLink(2, 0, 0.05, 0.1, None, 0.1, 3),
        FragmentLink(2, 1, 0.10, 0.2, None, 0.1, 3),
        FragmentLink(3, 0, 0.13, 0.3, None, 0.1, 3),
    ]
    assignments = assign_slots((0, 1), {i: 1.0 for i in range(4)}, options)
    assert [slot.fragments for slot in assignments] == [(3, 0), (2, 1)]
    assert [slot.links[0][1] for slot in assignments] == pytest.approx([0.82, 0.82])
