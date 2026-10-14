"""Saved metrics must survive variable clip/view padding without a checkpoint."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.blcs.ablation.evaluation import recompute, save_prediction_rows
from src.tasks.blcs.model_io.contracts import BLCSTrajectoryPrediction
from src.tasks.blcs.training.metrics import BLCSMetrics
from tests.support.tasks.blcs_physics import physics_sample, prepared_samples


def test_full_clip_bundle_recomputes_population_metrics(tmp_path: Path) -> None:
    tracker = BLCSMetrics()
    rows = []
    for length, shift in ((5, 0.0), (13, 0.1)):
        name = f"scene-{length}"
        batch, target = prepared_samples([physics_sample(length, name=name)])
        prediction = BLCSTrajectoryPrediction(target.position + shift, target.velocity)
        tracker.update(prediction, target, observed=batch["ball_vis"], scene_ids=[name])
        row = {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != "seq_len"
        }
        row.update(
            scene_id=name,
            pred_position=prediction.position,
            pred_velocity=prediction.velocity,
        )
        rows.append(row)
    path = tmp_path / "pred_test.npz"
    save_prediction_rows(path, rows)
    expected, actual = tracker.report(), recompute(path)
    assert expected["schema"] == actual["schema"]
    for key, value in expected["metrics"].items():
        if value is None:
            assert actual["metrics"][key] is None
        else:
            assert actual["metrics"][key] == pytest.approx(value, rel=1e-6, abs=1e-6), (
                key
            )
    with np.load(path, allow_pickle=False) as saved:
        assert saved["pred_position"].shape == (2, 13, 3)
        assert saved["camera_w"].shape == (2, 6)
        assert np.all(saved["camera_w"][:, 2:] == 1)


def test_rollout_reports_event_and_short_sequence_exclusions() -> None:
    batch, target = prepared_samples(
        [physics_sample(12, hits=(16,)), physics_sample(1)]
    )
    tracker = BLCSMetrics()
    tracker.update(
        BLCSTrajectoryPrediction(target.position, target.velocity),
        target,
        observed=batch["ball_vis"],
    )
    result = tracker.compute()
    assert result["rollout_h1_excluded_event_count"] == 1
    assert result["rollout_h3_excluded_event_count"] == 3
    assert result["rollout_h3_too_short_clip_count"] == 1
