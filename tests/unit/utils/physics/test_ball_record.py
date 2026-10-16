"""``ball_physics.v1`` record: segments, exact re-simulation and storage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.utils.physics.ball.record import (
    EVENT_BITS,
    EVENT_KINDS,
    RECORD_SCHEMA,
    BallPhysicsRecord,
    record_keys,
)
from tests.support.physics.ball_record import simulated_record


def test_segments_start_at_the_first_frame_at_or_after_each_event() -> None:
    _, record = simulated_record(40, hits=(50, 132))  # stride 4
    assert [(s.start_frame, s.end_frame, s.event) for s in record.segments] == [
        (0, 13, 0),
        (13, 33, 1),
        (33, 40, 2),
    ]
    segment = record.frame_segment()
    assert segment[12] == 0 and segment[13] == 1 and segment[39] == 2
    spin = record.frame_spin()
    np.testing.assert_array_equal(spin[13], record.event_spin_after[1])
    np.testing.assert_array_equal(spin[0], record.event_spin_after[0])


def test_recorded_states_reproduce_the_simulation_bitwise() -> None:
    positions, record = simulated_record(48, hits=(37, 90, 151))
    assert record.verify(positions, 0.0) == 0.0


def test_a_wrong_field_parameter_is_detected() -> None:
    positions, record = simulated_record(48, hits=(90,))
    with pytest.raises(ValueError, match="does not reproduce"):
        replace(record, k_drag=record.k_drag * 1.5).verify(positions, 1e-4)


def test_event_mask_marks_shots_and_skips_tosses() -> None:
    _, record = simulated_record(20, hits=(30,))
    kinds = record.event_kind.copy()
    kinds[0] = EVENT_KINDS.index("toss")
    tossed = replace(record, event_kind=kinds)
    mask = tossed.event_mask()
    assert mask[0] == 0
    assert mask[8] == EVENT_BITS["shot"]
    assert int(np.count_nonzero(mask)) == 1


def test_truncation_drops_later_events_and_still_reproduces() -> None:
    positions, record = simulated_record(40, hits=(50, 132))
    short = record.truncate(30)
    assert short.frames == 30 and list(short.event_step) == [0, 50]
    assert short.verify(positions[:30], 0.0) == 0.0


def test_npz_round_trip_is_exact(tmp_path: Path) -> None:
    _, record = simulated_record(24, hits=(41,))
    path = tmp_path / "record.npz"
    arrays: dict[str, Any] = {"other": np.zeros(2), **record.to_arrays()}
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as payload:
        loaded = BallPhysicsRecord.from_arrays(
            {key: payload[key] for key in record_keys()}
        )
    assert loaded.surface == record.surface and loaded.dt == record.dt
    for name in ("velocity", "event_step", "event_spin_after", "wind"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(record, name))


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("physics_schema", np.array("ball_physics.v0"), "Unsupported"),
        ("physics_velocity", None, "missing"),
        ("physics_extra", np.zeros(1), "unexpected"),
        ("physics_event_step", np.array([0, 41], np.int32), "int64"),
    ],
)
def test_from_arrays_rejects_incompatible_payloads(
    key: str, value: np.ndarray | None, match: str
) -> None:
    _, record = simulated_record(24, hits=(41,))
    arrays = record.to_arrays()
    assert str(arrays["physics_schema"]) == RECORD_SCHEMA
    if value is None:
        del arrays[key]
    else:
        arrays[key] = value
    with pytest.raises(ValueError, match=match):
        BallPhysicsRecord.from_arrays(arrays)


@pytest.mark.parametrize(
    "change",
    [
        {"surface": "carpet"},
        {"dt": 1 / 100},
        {"event_step": np.array([4, 41], np.int64)},
        {"event_step": np.array([0, 24 * 4], np.int64)},
    ],
)
def test_invalid_records_are_rejected(change: dict[str, Any]) -> None:
    _, record = simulated_record(24, hits=(41,))
    with pytest.raises(ValueError):
        replace(record, **change)
