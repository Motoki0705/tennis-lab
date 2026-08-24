"""Matched BLCS selector/selector-zero experiment evidence contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.evaluation import compute_paired_reference_position_metrics
from src.tasks.base.models import (
    ReferenceSelectorMode,
    build_compressed_track_query_spatial_coordinates,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _config(profile: str) -> dict[str, object]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
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
