"""Portable B and real-observation diagnostics keep units and failure support."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.blcs.ablation.evaluation import (
    recompute,
    save_json,
    save_prediction_rows,
)
from src.tasks.blcs.ablation.real_observations import (
    observation_metrics,
    recompute_real,
)
from src.tasks.blcs.ablation.verification import validate_real_metrics
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.projection.camera_projector import make_look_at_camera
from src.utils.schema.court_normalization import denormalize_court_position
from tests.support.tasks.blcs_physics import physics_sample, prepared_samples


@pytest.mark.parametrize("first_oracle_failed", [False, True])
def test_integrated_states_recompute_separate_populations(
    tmp_path: Path, first_oracle_failed: bool
) -> None:
    rows = []
    for index, length in enumerate((7, 11)):
        name = f"scene-{length}"
        batch, target = prepared_samples([physics_sample(length, name=name)])
        row = {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != "seq_len"
        }
        row.update(
            scene_id=name,
            pred_position=target.position,
            pred_velocity=target.velocity,
            physics_parameters=torch.zeros(1, length, 7),
            event_logits=torch.full((1, length, 1), -20.0),
        )
        p, v = denormalize_court_position(target.position), target.velocity * 11.885
        for mode, shift in (("oracle", 0.1), ("predicted", 0.3)):
            row[f"{mode}_integrated_position"] = p + shift
            row[f"{mode}_integrated_velocity"] = v + 2 * shift
        if first_oracle_failed and index == 0:
            row["oracle_integrated_position"].fill_(float("nan"))
            row["oracle_integrated_velocity"].fill_(float("nan"))
        rows.append(row)
    bundle = tmp_path / "pred_test.npz"
    save_prediction_rows(bundle, rows)
    restored = recompute(bundle, include_diagnostics=True)
    assert restored["metrics"]["trajectory_rmse_m"] == 0
    for mode, shift, count in (
        ("oracle", 0.1, 11 if first_oracle_failed else 18),
        ("predicted", 0.3, 18),
    ):
        metrics = restored[f"integrated_{mode}"]["metrics"]
        assert metrics["trajectory_count"] == count
        assert metrics["trajectory_rmse_m"] == pytest.approx(3**0.5 * shift, abs=2e-6)
        assert metrics["velocity_head_rmse_mps"] == pytest.approx(
            3**0.5 * 2 * shift, abs=2e-6
        )
    assert (
        restored["diagnostics"][0]["oracle_integration_failed"] is first_oracle_failed
    )


def test_real_metrics_recompute_without_store_and_detect_camera_corruption(
    tmp_path: Path,
) -> None:
    camera = make_look_at_camera(
        (5, -22, 6), look_at=(0, 0, 0), image_size=(1920, 1080), hfov_deg=60
    )
    K = np.array(
        [[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]],
        dtype=np.float64,
    )
    R = camera.R.double().numpy()
    rig = (PinholeCamera("0", K, R, -R @ camera.C.double().numpy()),)
    time = np.arange(8) / 30
    position = np.stack((time, 2 * time, 1 + 4 * time - 4.9 * time**2), -1)
    uv, visible = rig[0].project(position)
    batch = {
        "ball_uv": torch.from_numpy((uv / [1920, 1080])[None, None]),
        "ball_vis": torch.from_numpy(visible[None, None]),
        "camera_w": torch.tensor([[1920.0]]),
        "camera_h": torch.tensor([[1080.0]]),
    }
    report = {
        "evaluation_fps": 30,
        "gravity_mps2": 9.8,
        "direct": observation_metrics(position, batch, rig, 30, gravity=9.8),
    }
    save_json(tmp_path / "evaluation.json", report)
    arrays = {key: value.numpy() for key, value in batch.items()}
    arrays.update(
        position_m=position,
        projection_K=K[None],
        projection_R=R[None],
        projection_t=rig[0].translation[None],
    )
    np.savez_compressed(tmp_path / "real_predictions.npz", **arrays)
    validate_real_metrics(tmp_path)
    restored = json.loads((tmp_path / "recomputed.json").read_text())
    assert restored["direct"]["reprojection_count"] == 8
    assert restored["direct"]["reprojection_rmse_px"] < 1e-10
    with pytest.raises(ValueError, match="disagrees"):
        recompute_real(tmp_path, gravity=3.7)
    del report["gravity_mps2"]
    save_json(tmp_path / "evaluation.json", report)
    with pytest.raises(ValueError, match="require explicit gravity"):
        recompute_real(tmp_path)
    validate_real_metrics(tmp_path, gravity=9.8)
    report["gravity_mps2"] = 9.8
    save_json(tmp_path / "evaluation.json", report)
    arrays["projection_K"][0, 0, 0] *= 1.1
    np.savez_compressed(tmp_path / "real_predictions.npz", **arrays)
    with pytest.raises(ValueError, match="Saved metric mismatch"):
        validate_real_metrics(tmp_path)
