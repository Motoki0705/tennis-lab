"""Composed-config and canonical metric visibility integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.training.lightning_module import (
    BLCS_TRAJECTORY_METRIC_CONTRACT,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCS_TRACKING_METRIC_CONTRACT,
)
from src.tasks.plcs.training.lightning_module import (
    PLCS_TRAJECTORY_METRIC_CONTRACT,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCS_TRACKING_METRIC_CONTRACT,
)


@pytest.mark.parametrize("task", ["blcs", "plcs"])
def test_standard_monitor_compose_targets_canonical_position_error(task: str) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="train")

    assert config.training.early_stopping.monitor == "val/position_error_m"
    if task == "plcs":
        assert config.training.checkpoint.monitor == "val/position_error_m"
    else:
        assert config.training.checkpoint.monitor == "val/loss"


def test_task_contracts_expose_only_the_required_canonical_headlines() -> None:
    assert BLCS_TRAJECTORY_METRIC_CONTRACT.for_stage("test").headline_keys == (
        "position_error_m",
        "position_accuracy_0.3m",
        "endpoint_error_m",
    )
    assert PLCS_TRAJECTORY_METRIC_CONTRACT.for_stage("test").headline_keys == (
        "position_error_m",
        "angular_error_deg",
        "position_accuracy_0.5m",
        "angle_accuracy_15deg",
    )
    assert BLCS_TRACKING_METRIC_CONTRACT.for_stage("train").headline_keys == ()
    assert BLCS_TRACKING_METRIC_CONTRACT.for_stage("test").headline_keys == (
        "position_error_m",
        "presence_f1",
        "id_switches",
    )
    assert PLCS_TRACKING_METRIC_CONTRACT.for_stage("train").headline_keys == ()
    assert PLCS_TRACKING_METRIC_CONTRACT.for_stage("test").headline_keys == (
        "position_error_m",
        "angular_error_deg",
        "presence_f1",
        "id_switches",
    )
