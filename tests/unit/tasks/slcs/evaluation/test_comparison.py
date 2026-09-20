"""Strict condition pairing and masked domain-level teacher agreement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.slcs.evaluation.comparison import (
    CONDITIONS,
    GAP_CONDITIONS,
    METRICS,
    compare_conditions,
    compare_gap_conditions,
    save_comparison,
)
from src.tasks.slcs.evaluation.evaluate import evaluation_context, save_evaluation
from src.tasks.slcs.model_io import SLCSDecodedOutput, SLCSTrainingTargets
from src.tasks.slcs.training.metrics import SLCSMetrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _arrays() -> dict[str, np.ndarray]:
    player: np.ndarray = np.zeros((2, 1, 2, 3), np.float32)
    player[..., 0] = np.array([[[2, 4]], [[6, 8]]]) / COURT_COORD_SCALE_XYZ[0]
    ball = player[:, 0].copy()
    rotation: np.ndarray = np.zeros((2, 1, 2, 2), np.float32)
    rotation[..., 0] = 1
    predicted_rotation = rotation.copy()
    predicted_rotation[1, ..., 0] = 0
    predicted_rotation[1, ..., 1] = 1
    return {
        "scene_ids": np.array(["video_0/c@cam@0", "shanghai/c@cam@0"]),
        "video_ids": np.array(["video_0", "shanghai"]), "clip_ids": np.array(["video_0/c", "shanghai/c"]),
        "camera_ids": np.array(["cam", "cam"]), "window_start": np.array([0, 0]), "window_length": np.array([2, 2]),
        "frame_idx": np.array([[0, 1], [0, 1]]), "padding_mask": np.zeros((2, 2), bool),
        "player_mask": np.array([[[True, False]], [[True, True]]]), "ball_mask": np.ones((2, 2), bool),
        "player_weight": np.array([[[.1, 0]], [[.8, .2]]], np.float32), "ball_weight": np.array([[.1, .1], [.8, .8]], np.float32),
        "pred_player_position": player, "target_player_position": np.zeros_like(player),
        "pred_player_rotation": predicted_rotation, "target_player_rotation": rotation.copy(),
        "pred_ball_position": ball, "target_ball_position": np.zeros_like(ball),
    }


def _bundles(tmp_path: Path, conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Path]:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"same checkpoint")
    bundles = {mode: tmp_path / mode for mode in conditions}
    for mode, directory in bundles.items():
        arrays = _arrays()
        if mode != conditions[0]:
            arrays["pred_player_position"] *= 2
            arrays["pred_ball_position"] *= 2
        save_evaluation(directory, {}, arrays, context=evaluation_context(checkpoint, input_mode=mode))
    return bundles


DOMAINS = {"video_0": "meiji", "shanghai": "broadcast"}


def test_domain_scores_use_existing_unweighted_metric_contract(tmp_path: Path) -> None:
    bundles = _bundles(tmp_path)
    report = compare_conditions(bundles, DOMAINS)
    full = {row["group"]: row for row in report["rows"] if row["condition"] == "full"}
    assert full["meiji"]["player_position_error_m"] == pytest.approx(2)
    assert full["broadcast"]["player_position_error_m"] == pytest.approx(7)
    assert full["all"]["player_position_error_m"] == pytest.approx(16 / 3)
    assert full["all"]["ball_position_error_m"] == pytest.approx(5)
    assert full["all"]["player_valid_count"] == 3
    assert full["all"]["player_valid_weight_sum"] == pytest.approx(1.1)
    assert full["all"]["player_angular_error_deg"] == pytest.approx(60)
    ablated = next(row for row in report["rows"] if row["group"] == "all" and row["condition"] == "no_rgb")
    assert ablated["full_minus_condition_ball_position_error_m"] == pytest.approx(-5)
    arrays = _arrays()
    targets = SLCSTrainingTargets(**{key: torch.from_numpy(arrays[key]) for key in (
        "target_player_position", "target_player_rotation", "target_ball_position", "player_mask", "ball_mask", "player_weight", "ball_weight", "padding_mask")})
    outputs = SLCSDecodedOutput(
        player_position=torch.from_numpy(arrays["pred_player_position"]), player_rotation=torch.from_numpy(arrays["pred_player_rotation"]),
        ball_position=torch.from_numpy(arrays["pred_ball_position"]), player_position_log_b=torch.zeros((2, 1, 2)),
        player_rotation_log_b=torch.zeros((2, 1, 2)), ball_position_log_b=torch.zeros((2, 2)),
    )
    metrics = SLCSMetrics()
    metrics.update(outputs, targets)
    for key in METRICS:
        assert full["all"][key] == metrics.compute()[key]
    json_path, csv_path = save_comparison(report, tmp_path / "comparison")
    assert json.loads(json_path.read_text()) == report
    assert len(csv_path.read_text().splitlines()) == len(report["rows"]) + 1


@pytest.mark.parametrize("change", ["order", "weight", "sha", "target", "mask"])
@pytest.mark.parametrize("gap", [False, True])
def test_unmatched_conditions_are_rejected(tmp_path: Path, change: str, gap: bool) -> None:
    conditions = GAP_CONDITIONS if gap else CONDITIONS
    bundles = _bundles(tmp_path, conditions)
    directory = bundles[conditions[-1]]
    if change == "sha":
        path = directory / "metrics.json"
        payload = json.loads(path.read_text())
        payload["context"]["checkpoint_sha256"] = "f" * 64
        path.write_text(json.dumps(payload))
    else:
        with np.load(directory / "eval_arrays.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        if change == "order":
            arrays = {key: value[::-1] for key, value in arrays.items()}
        elif change == "weight":
            arrays["player_weight"][0, 0, 0] = .2
        elif change == "mask":
            arrays["player_mask"][0, 0, 0] = False
            arrays["player_weight"][0, 0, 0] = 0
        else:
            arrays["target_ball_position"][0, 0, 0] = .2
        np.savez_compressed(directory / "eval_arrays.npz", **arrays)
    with pytest.raises(ValueError, match="Unmatched|SHA256"):
        (compare_gap_conditions if gap else compare_conditions)(bundles, DOMAINS)


def test_gap_comparison_reports_paired_errors_and_sign(tmp_path: Path) -> None:
    bundles = _bundles(tmp_path, GAP_CONDITIONS)
    report = compare_gap_conditions(bundles, DOMAINS)
    assert "Negative detector_gap-minus-condition error favors detector_gap" in report["interpretation"]
    assert {row["group_type"] for row in report["rows"]} == {"all", "domain", "video"}
    ablated = next(row for row in report["rows"] if row["group"] == "all" and row["condition"] == "detector_gap_no_rgb")
    assert ablated["detector_gap_minus_condition_ball_position_error_m"] == pytest.approx(-5)
    assert ablated["detector_gap_minus_condition_player_position_error_m"] == pytest.approx(-16 / 3)
    assert ablated["player_angular_error_deg"] == pytest.approx(60)
    assert ablated["detector_gap_minus_condition_player_angular_error_deg"] == 0
    json_path, csv_path = save_comparison(report, tmp_path / "gap_rgb_comparison")
    assert json.loads(json_path.read_text()) == report
    assert len(csv_path.read_text().splitlines()) == len(report["rows"]) + 1
    with pytest.raises(ValueError, match="four conditions"):
        compare_conditions(bundles, DOMAINS)
    with pytest.raises(ValueError, match="gap conditions"):
        compare_gap_conditions({"detector_gap": bundles["detector_gap"]}, DOMAINS)


def test_empty_entity_and_explicit_domain_mapping(tmp_path: Path) -> None:
    bundles = _bundles(tmp_path)
    with pytest.raises(ValueError, match="explicit domain"):
        compare_conditions(bundles, {"video_0": "meiji"})
    for directory in bundles.values():
        with np.load(directory / "eval_arrays.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["player_mask"].fill(False)
        arrays["player_weight"].fill(0)
        np.savez_compressed(directory / "eval_arrays.npz", **arrays)
    report = compare_conditions(bundles, DOMAINS)
    assert all(row["player_position_error_m"] is None for row in report["rows"])
