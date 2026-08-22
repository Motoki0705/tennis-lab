"""Hydra composition coverage for the shared default-on compile contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.integration

_SRC_TASKS = Path(__file__).resolve().parents[4] / "src/tasks"


@pytest.mark.parametrize(
    ("task", "config_name", "overrides"),
    [
        ("ball_detection", "train", []),
        ("ball_detection", "train_staged", ["model=stunet"]),
        ("court_detection", "train", []),
        ("blcs", "train_tracking", []),
        ("plcs", "train", []),
        ("slcs", "train", []),
    ],
)
def test_training_configs_enable_compile_by_default(
    task: str,
    config_name: str,
    overrides: list[str],
) -> None:
    config_dir = _SRC_TASKS / task / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name=config_name, overrides=overrides)

    assert config.training.compile.enabled is True
    assert config.training.compile.backend == "inductor"
    assert config.training.compile.mode == "default"
    assert config.training.compile.fullgraph is False
    assert config.training.compile.dynamic is False
