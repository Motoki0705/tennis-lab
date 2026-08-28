"""Matched BLCS selector/selector-zero experiment evidence contract."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.evaluation import compute_paired_reference_position_metrics
from src.tasks.base.models import (
    ReferenceSelectorMode,
    build_compressed_track_query_spatial_coordinates,
)

_CONFIG_DIRS = {
    task: Path(f"src/tasks/{task}/configs").resolve() for task in ("blcs", "plcs")
}
_RUN_DIRS = {
    "reference": Path("knowledge/runs/run-i801-d-reference-seeded"),
    "selector_zero": Path("knowledge/runs/run-i801-d-selector-zero-seeded"),
}
_PARITY_KEYS = {
    "target_position",
    "target_presence",
    "target_instance_id",
    "frame_valid",
    "reference_view_index",
    "view_camera_ids",
    "reference_camera_id",
    "reference_camera_id_string",
    "view_camera_id_strings",
    "reference_from_physical",
    "physical_from_reference",
    "target_frame_contract",
    "court_keypoint_contract",
    "track_query_rope_contract",
    "scene_ids",
}


def _config(profile: str) -> dict[str, object]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIRS["blcs"]), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "data.scene_dir=blcs/multi_object_camera_view_v2",
                "run.seed=42",
            ],
        )
    value = OmegaConf.to_container(config, resolve=True)
    assert isinstance(value, dict)
    return value


def _default_config(task: str) -> dict[str, object]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIRS[task]), version_base="1.3"):
        config = compose(config_name="train_tracking")
    value = OmegaConf.to_container(config, resolve=True)
    assert isinstance(value, dict)
    return value


def _run_metadata(selector_mode: str) -> dict[str, object]:
    value = json.loads((_RUN_DIRS[selector_mode] / "run.json").read_text())
    assert isinstance(value, dict)
    return value


def _bundle(selector_mode: str) -> dict[str, np.ndarray]:
    with np.load(_RUN_DIRS[selector_mode] / "pred_test.npz", allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def test_selector_and_selector_zero_profiles_are_matched_except_selector_identity() -> None:
    selector = _config("track_query_ablation_d_v2_selector")
    selector_zero = _config("track_query_ablation_d_v2_selector_zero")
    selector_model = selector["model"]
    selector_zero_model = selector_zero["model"]
    assert isinstance(selector_model, dict)
    assert isinstance(selector_zero_model, dict)
    assert selector_model["reference_selector_mode"] == "reference"
    assert selector_zero_model["reference_selector_mode"] == "selector_zero"

    normalized_selector = deepcopy(selector)
    normalized_zero = deepcopy(selector_zero)
    assert isinstance(normalized_selector["model"], dict)
    assert isinstance(normalized_zero["model"], dict)
    normalized_selector["model"].pop("reference_selector_mode")
    normalized_zero["model"].pop("reference_selector_mode")
    assert normalized_selector == normalized_zero
    selector_run = selector["run"]
    selector_data = selector["data"]
    zero_data = selector_zero["data"]
    assert isinstance(selector_run, dict)
    assert isinstance(selector_data, dict)
    assert isinstance(zero_data, dict)
    assert selector_run["seed"] == 42
    assert selector_data["scene_dir"] == zero_data["scene_dir"]


def test_registered_runs_are_a_seeded_matched_pair() -> None:
    reference = _run_metadata("reference")
    selector_zero = _run_metadata("selector_zero")
    for field in ("provider", "session", "issue", "commit"):
        assert reference[field] == selector_zero[field]
    assert reference["provider"] == "codex"
    assert reference["issue"] == "801"
    assert reference["commit"] == "b392bbcbab877172b74c190af32b4dcc12366853"

    reference_command = reference["command"]
    selector_zero_command = selector_zero["command"]
    assert isinstance(reference_command, str)
    assert isinstance(selector_zero_command, str)
    reference_argv = shlex.split(reference_command)
    selector_zero_argv = shlex.split(selector_zero_command)

    def normalize(argv: list[str]) -> list[str]:
        return [
            "model=<MATCHED_VARIANT>"
            if token.startswith("model=")
            else "run.output_dir=<MATCHED_VARIANT>"
            if token.startswith("run.output_dir=")
            else token
            for token in argv
        ]

    assert normalize(reference_argv) == normalize(selector_zero_argv)
    assert "model=track_query_ablation_d_v2_selector" in reference_argv
    assert "model=track_query_ablation_d_v2_selector_zero" in selector_zero_argv
    for argv in (reference_argv, selector_zero_argv):
        for invariant in (
            "CUDA_VISIBLE_DEVICES=0",
            "court_keypoints=camera_view_v2",
            "data.scene_dir=blcs/multi_object_camera_view_norm-v2",
            "data.seq_len_range=[128,128]",
            "data.num_views_range=[4,4]",
            "data.batch_size=8",
            "training.trainer.accumulate_grad_batches=4",
            "training.trainer.max_epochs=100",
            "run.seed=42",
            "run.fast_dev_run=false",
            "run.test_after_fit=true",
        ):
            assert invariant in argv

    for selector_mode, metadata in (
        ("reference", reference),
        ("selector_zero", selector_zero),
    ):
        assert "retry4" in str(metadata["run_id"])
        assert "retry4" in str(metadata["name"])
        assert "retry3" not in str(metadata["run_id"])
        assert "retry3" not in str(metadata["name"])
        assert "retry3" not in str(_RUN_DIRS[selector_mode])


def test_registered_prediction_bundles_have_bitwise_matched_evaluation_inputs() -> None:
    reference = _bundle("reference")
    selector_zero = _bundle("selector_zero")
    expected_keys = _PARITY_KEYS | {
        "pred_position",
        "pred_presence_logits",
        "reference_selector_mode",
    }
    assert set(reference) == expected_keys
    assert set(selector_zero) == expected_keys
    for key in _PARITY_KEYS:
        np.testing.assert_array_equal(reference[key], selector_zero[key])

    assert not np.array_equal(reference["pred_position"], selector_zero["pred_position"])
    assert not np.array_equal(
        reference["pred_presence_logits"],
        selector_zero["pred_presence_logits"],
    )
    np.testing.assert_array_equal(
        np.unique(reference["reference_selector_mode"]),
        np.array(["reference"]),
    )
    np.testing.assert_array_equal(
        np.unique(selector_zero["reference_selector_mode"]),
        np.array(["selector_zero"]),
    )

    reference_index = reference["reference_view_index"]
    view_camera_ids = reference["view_camera_ids"]
    view_camera_id_strings = reference["view_camera_id_strings"]
    rows = np.arange(reference_index.shape[0])
    np.testing.assert_array_equal(
        view_camera_ids[rows, reference_index],
        reference["reference_camera_id"],
    )
    np.testing.assert_array_equal(
        view_camera_id_strings[rows, reference_index],
        reference["reference_camera_id_string"],
    )
    np.testing.assert_array_equal(
        np.bincount(reference_index, minlength=view_camera_ids.shape[1]),
        np.array([26, 29, 18, 27]),
    )

    reference_from_physical = reference["reference_from_physical"]
    physical_from_reference = reference["physical_from_reference"]
    identity = np.broadcast_to(
        np.eye(3, dtype=reference_from_physical.dtype),
        reference_from_physical.shape,
    )
    np.testing.assert_array_equal(reference_from_physical @ physical_from_reference, identity)
    np.testing.assert_array_equal(physical_from_reference @ reference_from_physical, identity)
    np.testing.assert_array_equal(
        reference_from_physical @ reference_from_physical.swapaxes(-1, -2),
        identity,
    )
    np.testing.assert_array_equal(
        np.linalg.det(reference_from_physical),
        np.ones(reference_from_physical.shape[0]),
    )


@pytest.mark.parametrize(
    ("task", "model_name", "scene_dir", "evaluation_reference_camera_id"),
    [
        ("blcs", "blcs_track_query", "blcs/multi_object_lifecycle_v2", "cam_1"),
        ("plcs", "plcs_track_query", "plcs/multi_object_lifecycle_v2", "camera_1"),
    ],
)
def test_production_train_tracking_defaults_keep_legacy_model_and_court_contract(
    task: str,
    model_name: str,
    scene_dir: str,
    evaluation_reference_camera_id: str,
) -> None:
    config = _default_config(task)
    model = config["model"]
    court_keypoints = config["court_keypoints"]
    data = config["data"]
    assert isinstance(model, dict)
    assert isinstance(court_keypoints, dict)
    assert isinstance(data, dict)

    assert model["name"] == model_name
    assert model["role_rope_enabled"] is True
    assert "reference_selector_mode" not in model
    assert "court_coordinate_normalization" not in config
    assert court_keypoints == {"selector": "physical_v1"}
    assert data["scene_dir"] == scene_dir
    assert data["evaluation_reference_camera_id"] == evaluation_reference_camera_id


def test_selector_zero_changes_only_axis_three_and_metrics_keep_required_fields() -> None:
    reference_index = torch.tensor([0, 1], dtype=torch.int64)
    selector_coordinates = build_compressed_track_query_spatial_coordinates(
        reference_index,
        num_frames=2,
        num_views=2,
        num_queries=2,
        selector_mode=ReferenceSelectorMode.REFERENCE,
    )
    zero_coordinates = build_compressed_track_query_spatial_coordinates(
        reference_index,
        num_frames=2,
        num_views=2,
        num_queries=2,
        selector_mode=ReferenceSelectorMode.SELECTOR_ZERO,
    )
    torch.testing.assert_close(
        selector_coordinates[..., :2],
        zero_coordinates[..., :2],
    )
    assert not zero_coordinates[..., 2].any()
    assert selector_coordinates[..., 2].any()

    target = torch.tensor(
        [
            [[1.0, 2.0, 0.0], [2.0, -3.0, 1.0]],
            [[-1.0, 4.0, 2.0], [3.0, -2.0, 0.0]],
        ]
    )
    prediction = target + torch.tensor([0.25, -0.5, 1.0])
    prediction[1, 0, 1] = -prediction[1, 0, 1]
    metrics = compute_paired_reference_position_metrics(
        prediction,
        target,
        reference_index,
    )

    assert metrics.y_sign_accuracy == pytest.approx(0.75)
    assert metrics.axis_wise_position_error.x == pytest.approx(0.25)
    assert metrics.axis_wise_position_error.y == pytest.approx(2.25)
    assert metrics.axis_wise_position_error.z == pytest.approx(1.0)
    assert set(metrics.local_reference_index_error) == {0, 1}
