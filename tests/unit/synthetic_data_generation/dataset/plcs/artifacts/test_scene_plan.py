import numpy as np
import pytest

from src.synthetic_data_generation.dataset.plcs.artifacts.scene_plan import (
    PLCSPersonSchedule,
    build_person_schedule,
)


@pytest.mark.parametrize(("mode", "person_count"), [("single", 1), ("multi", 2)])
def test_person_schedule_is_complete_deterministic_and_bounded(
    mode: str,
    person_count: int,
) -> None:
    first = build_person_schedule(mode=mode, seed=42)  # type: ignore[arg-type]
    second = build_person_schedule(mode=mode, seed=42)  # type: ignore[arg-type]

    assert first.schedule_fingerprint == second.schedule_fingerprint
    assert first.person_count == person_count
    assert first.frame_count == 12
    assert first.present.all()
    assert not first.positions_court_m.flags.writeable
    np.testing.assert_array_equal(first.positions_court_m, second.positions_court_m)
    np.testing.assert_array_equal(first.pose_indices, second.pose_indices)
    assert set(np.unique(first.pose_indices)) == {1, 2}
    assert np.all(np.abs(first.positions_court_m[..., 0]) <= 4.115)
    assert np.all(np.abs(first.positions_court_m[..., 1]) <= 11.885)


def test_seed_changes_motion_but_preserves_identity() -> None:
    first = build_person_schedule(mode="multi", seed=1)
    second = build_person_schedule(mode="multi", seed=2)

    assert first.identity_ids == second.identity_ids
    assert first.schedule_fingerprint != second.schedule_fingerprint
    assert not np.array_equal(first.positions_court_m, second.positions_court_m)


def test_multi_schedule_maintains_collision_clearance() -> None:
    schedule = build_person_schedule(mode="multi", seed=5)
    separation = np.linalg.norm(
        schedule.positions_court_m[:, 0, :2] - schedule.positions_court_m[:, 1, :2],
        axis=1,
    )
    assert float(separation.min()) > 18.0


def test_schedule_rejects_silent_frame_drop() -> None:
    valid = build_person_schedule(mode="single", seed=0)
    present = valid.present.copy()
    present[3, 0] = False
    with pytest.raises(ValueError, match="silently drop"):
        PLCSPersonSchedule.create(
            mode=valid.mode,
            seed=valid.seed,
            fps=valid.fps,
            identity_ids=valid.identity_ids,
            instance_ids=valid.instance_ids.copy(),
            positions_court_m=valid.positions_court_m.copy(),
            velocities_court_mps=valid.velocities_court_mps.copy(),
            yaw_radians=valid.yaw_radians.copy(),
            pose_indices=valid.pose_indices.copy(),
            present=present,
        )
