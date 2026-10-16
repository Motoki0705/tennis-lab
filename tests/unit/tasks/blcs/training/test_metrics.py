"""Frame/clip denominators, undefined states and batch/distributed invariance."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.blcs.model_io.contracts import BLCSTrajectoryPrediction
from src.tasks.blcs.training.metrics import BLCSMetrics
from tests.support.tasks.blcs_physics import physics_sample, prepared_samples


def update(tracker, samples, errors):
    batch, target = prepared_samples(samples)
    p = target.position.clone()
    for b, error in enumerate(errors):
        p[b, :, 0] += error / 11.885
    tracker.update(
        BLCSTrajectoryPrediction(p, target.velocity),
        target,
        observed=batch["ball_vis"],
        scene_ids=batch["scene_id"],
    )


def test_hand_computed_rmse_and_batch_padding_invariance():
    a, b = physics_sample(3, name="a"), physics_sample(9, name="b")
    together, split = BLCSMetrics(), BLCSMetrics()
    update(together, [a, b], [1.0, 3.0])
    update(split, [a], [1.0])
    update(split, [b], [3.0])
    result = together.compute()
    assert result["trajectory_rmse_m"] == pytest.approx(np.sqrt(7.0), rel=1e-6)
    assert result["trajectory_mean_m"] == pytest.approx(2.5)
    assert result["clip_trajectory_rmse_mean_m"] == pytest.approx(2.0)
    assert result["trajectory_count"] == 12
    for key, value in result.items():
        other = split.compute()[key]
        assert (np.isnan(value) and np.isnan(other)) or value == pytest.approx(
            other, abs=1e-7
        )


def test_gt_and_jitter_distinguish_position_velocity_head_and_derived_motion():
    batch, t = prepared_samples([physics_sample(20)])
    tracker = BLCSMetrics()
    tracker.update(
        BLCSTrajectoryPrediction(t.position, t.velocity), t, observed=batch["ball_vis"]
    )
    assert tracker.compute()["rollout_position_rmse_m"] < 2e-6
    p = t.position.clone()
    p[:, ::2, 0] += 0.01
    jitter = BLCSMetrics()
    jitter.update(
        BLCSTrajectoryPrediction(p, t.velocity), t, observed=batch["ball_vis"]
    )
    r = jitter.compute()
    assert r["velocity_head_rmse_mps"] == 0
    assert r["position_difference_acceleration_rmse_mps2"] > 100
    assert r["trajectory_rmse_m"] > 0.05


def test_empty_metrics_are_undefined_not_zero_and_missing_is_a_separate_stratum():
    batch, t = prepared_samples([physics_sample(1)])
    tracker = BLCSMetrics()
    tracker.update(
        BLCSTrajectoryPrediction(t.position, None),
        t,
        observed=torch.zeros_like(batch["ball_vis"]),
    )
    r = tracker.report()["metrics"]
    assert r["trajectory__missing_count"] == 1
    assert r["trajectory__observed_count"] == 0
    assert r["rollout_position_rmse_m"] is None
    assert r["velocity_head_rmse_mps"] is None
    empty = replace(t, loss_mask=torch.zeros_like(t.loss_mask))
    tracker.reset()
    tracker.update(
        BLCSTrajectoryPrediction(t.position, t.velocity),
        empty,
        observed=batch["ball_vis"],
    )
    assert tracker.report()["metrics"]["trajectory_rmse_m"] is None


def test_distributed_sampler_duplicate_does_not_change_the_population(monkeypatch):
    tracker = BLCSMetrics()
    update(tracker, [physics_sample(6, name="a")], [2.0])
    expected = tracker.compute()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(destination, records):
        destination[:] = [records, records]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    result = tracker.compute()
    assert result["trajectory_count"] == expected["trajectory_count"]
    assert result["trajectory_rmse_m"] == expected["trajectory_rmse_m"]
