"""Explicit semantic checkpoint selection survives the curated-root migration."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.court_detection.visualization.orchestrator import build_runtime_config
from src.utils.configuration import MissingConfigurationKeyError, PathRole

CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def test_semantic_preset_requires_an_explicit_compatible_checkpoint() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize", overrides=["visualization=semantic_line"]
        )
    assert OmegaConf.is_missing(config.visualization, "checkpoint")
    with pytest.raises(MissingConfigurationKeyError, match="checkpoint"):
        build_runtime_config(config)


def test_explicit_semantic_checkpoint_keeps_the_training_output_root(
    tmp_path: Path,
) -> None:
    relative = "court_detection/semantic_line/logs/version_0/checkpoints/model.ckpt"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize",
            overrides=[
                "visualization=semantic_line",
                "paths=default",
                f"paths.project_root={tmp_path}",
                f"visualization.checkpoint={relative}",
            ],
        )
    runtime = build_runtime_config(config)
    assert runtime.task == "semantic_line"
    assert (
        runtime.resolver.resolve(PathRole.CHECKPOINT, runtime.checkpoint)
        == tmp_path / "outputs" / relative
    )
