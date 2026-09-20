"""Persistent false detections retain physical duration and camera-local paths."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from omegaconf import OmegaConf

from src.tasks.base.triangulation_residual.configuration import V2Config
from src.tasks.base.triangulation_residual.persistent import (
    PersistentEvent,
    PersistentKind,
    apply_persistent_events,
    corrupt_persistent,
    sample_persistent_events,
)


def config(**changes):
    path = (
        Path(__file__).resolve().parents[5]
        / "src/tasks/base/triangulation_residual/configs/geometric_residual_v2.yaml"
    )
    return replace(
        V2Config(**OmegaConf.to_container(OmegaConf.load(path).v2)), **changes
    )


def observations(joints=17, frames=60, views=6):
    points = np.full((views, frames, joints, 2), [900.0, 500.0])
    points[..., 0] += np.arange(frames)[None, :, None] * 3.0
    if joints == 17:
        points[:, :, (5, 6), 1] = 400
        points[:, :, (11, 12), 1] = 600
        points[:, :, (7, 9, 15), 0] -= 50
        points[:, :, (8, 10, 16), 0] += 50
    visible = np.ones(points.shape[:-1], dtype=bool)
    return points, visible, np.tile([1920, 1080], (views, 1))


def event(kind, *, views=(0,), joints=(9,), start=5, stop=25, high=True):
    return PersistentEvent(
        kind,
        start,
        stop,
        start / 30,
        (stop - start) / 30,
        views,
        joints,
        high,
        0.85 if high else 0.42,
    )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_event_seconds_and_poisson_rate_do_not_depend_on_sampling_fps(task):
    cfg = config(
        persistent_rate_per_view_second=2.0,
        persistent_min_seconds=0.4,
        persistent_max_seconds=1.2,
    )
    slow = sample_persistent_events(
        150, 6, np.random.default_rng(30), cfg, fps=15, task=task
    )
    fast = sample_persistent_events(
        600, 6, np.random.default_rng(30), cfg, fps=60, task=task
    )
    assert len(slow) == len(fast) > 10
    for first, second in zip(slow, fast, strict=True):
        assert first.start_seconds == second.start_seconds
        assert first.duration_seconds == second.duration_seconds
        assert 0.4 <= first.duration_seconds <= 1.2
        assert first.views == second.views
        assert first.joints == second.joints
        for generated, fps in ((first, 15), (second, 60)):
            assert (
                abs(
                    (generated.stop_frame - generated.start_frame) / fps
                    - generated.duration_seconds
                )
                < 2 / fps
            )
    assert slow == sample_persistent_events(
        150, 6, np.random.default_rng(30), cfg, fps=15, task=task
    )


def test_joint_errors_use_distal_joints_and_include_high_confidence_events():
    cfg = config(persistent_rate_per_view_second=4)
    events = sample_persistent_events(
        1800, 6, np.random.default_rng(48), cfg, fps=30, task="plcs"
    )
    assert {item.kind for item in events} == {
        PersistentKind.SELF_OFFSET,
        PersistentKind.CONTRALATERAL,
        PersistentKind.FROZEN_JOINT,
    }
    assert {len(item.views) for item in events} == {1, 2, 6}
    assert all(not set(item.joints).intersection((11, 12)) for item in events)
    assert sum(item.high_confidence for item in events) > 0.15 * len(events)
    assert sum(
        bool(set(item.joints).intersection((9, 10))) for item in events
    ) > 0.4 * len(events)


def test_contralateral_exchange_uses_visible_opposite_source_when_target_is_hidden():
    reference, visible, sizes = observations()
    visible[0, 5:25, 9] = False
    original = np.where(visible[..., None], reference, np.nan)
    result = apply_persistent_events(
        original,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(5),
        (event(PersistentKind.CONTRALATERAL, joints=(9, 10)),),
        fps=30,
    )
    assert_allclose(result.observations_px[0, 5:25, 9], reference[0, 5:25, 10])
    assert result.mask[0, 5:25, 9].all()
    assert (result.scores[0, 5:25, 9] > 0.7).all()
    assert not result.mask[0, 5:25, 10].any()
    assert result.source_missing[0, 5:25, 10].all()
    assert_allclose(result.observations_px[:, :, 11:13], original[:, :, 11:13])


def test_frozen_detection_persists_after_its_real_source_disappears():
    reference, visible, sizes = observations()
    visible[0, 10:, 9] = False
    reference[0, 10:, 9] = np.nan
    result = apply_persistent_events(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(7),
        (event(PersistentKind.FROZEN_JOINT),),
        fps=30,
    )
    assert result.mask[0, 5:25, 9].all()
    assert_allclose(
        result.observations_px[0, 5:25, 9], np.tile(reference[0, 5, 9], (20, 1))
    )
    assert np.isnan(result.observations_px[0, 25:, 9]).all()


@pytest.mark.parametrize("kind", [PersistentKind.BALL_DECOY, PersistentKind.STUCK_BALL])
def test_ball_false_paths_are_independent_between_views_and_survive_missing_gt(kind):
    reference, visible, sizes = observations(joints=1)
    visible[:, 6:] = False
    reference[:, 6:] = np.nan
    events = (event(kind, views=(0, 1, 2, 3, 4, 5), joints=(0,)),)
    result = apply_persistent_events(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(9),
        events,
        fps=30,
    )
    assert result.mask[:, 5:25].all()
    assert np.isfinite(result.observations_px[:, 5:25]).all()
    for view in range(1, 6):
        assert not np.allclose(
            result.observations_px[0, 5:25], result.observations_px[view, 5:25]
        )
    if kind == PersistentKind.STUCK_BALL:
        assert_allclose(np.diff(result.observations_px[:, 5:25], axis=1), 0)
    else:
        assert (
            np.max(
                np.linalg.norm(
                    np.diff(result.observations_px[:, 5:25, 0], axis=1), axis=-1
                )
            )
            < 3.5
        )


def test_no_ball_source_uses_explicit_image_prior_without_reading_hidden_gt():
    reference, visible, sizes = observations(joints=1)
    visible[:] = False
    reference[:] = np.nan
    result = apply_persistent_events(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(14),
        (event(PersistentKind.BALL_DECOY, joints=(0,)),),
        fps=30,
    )
    assert result.mask[0, 5:25].all()
    assert result.source_missing[0, 5:25].all()
    assert np.isfinite(result.observations_px[0, 5:25]).all()


def test_no_pose_source_is_explicitly_skipped_without_fabricating_a_pose():
    reference, visible, sizes = observations()
    visible[:] = False
    reference[:] = np.nan
    result = apply_persistent_events(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(14),
        (event(PersistentKind.FROZEN_JOINT),),
        fps=30,
    )
    assert not result.mask.any()
    assert result.source_missing[0, 5:25, 9].all()
    assert np.isnan(result.observations_px).all()


def test_forced_persistent_family_has_an_event_even_with_zero_arrival_rate():
    reference, visible, sizes = observations()
    cfg = config(persistent_rate_per_view_second=0.0)
    result = corrupt_persistent(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(7),
        cfg,
        fps=30,
        task="plcs",
        force_event=True,
    )
    assert len(result.events) == 1
    assert result.mask.any()
    repeated = corrupt_persistent(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(7),
        cfg,
        fps=30,
        task="plcs",
        force_event=True,
    )
    assert_array_equal(result.observations_px, repeated.observations_px)
    assert_array_equal(result.scores, repeated.scores)
    assert_array_equal(result.kind, repeated.kind)


def test_high_confidence_ablation_does_not_change_false_paths_or_event_times():
    reference, visible, sizes = observations(joints=1)
    results = [
        corrupt_persistent(
            reference,
            visible.astype(float),
            reference,
            visible,
            sizes,
            np.random.default_rng(5),
            config(
                persistent_rate_per_view_second=4,
                persistent_high_confidence_probability=p,
            ),
            fps=30,
            task="blcs",
        )
        for p in (0.0, 1.0)
    ]
    assert results[0].mask.any()
    assert_array_equal(results[0].observations_px, results[1].observations_px)
    assert_array_equal(results[0].mask, results[1].mask)
    assert (
        results[1].scores[results[1].mask] > results[0].scores[results[0].mask]
    ).all()


def test_self_offset_has_torso_scaled_persistent_displacement():
    reference, visible, sizes = observations()
    result = apply_persistent_events(
        reference,
        visible.astype(float),
        reference,
        visible,
        sizes,
        np.random.default_rng(11),
        (event(PersistentKind.SELF_OFFSET),),
        fps=30,
    )
    delta = result.observations_px[0, 5:25, 9] - reference[0, 5:25, 9]
    assert 50 <= np.linalg.norm(delta[0]) <= 250
    assert np.max(np.linalg.norm(np.diff(delta, axis=0), axis=-1)) < 70 / 30
    assert_allclose(result.observations_px[:, :, 11:13], reference[:, :, 11:13])
