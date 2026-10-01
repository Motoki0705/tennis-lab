"""Explicit semantic checkpoint selection survives the curated-root migration."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.court_detection.visualization.orchestrator import build_runtime_config
from src.utils.configuration import (
    ConfigurationError,
    MissingConfigurationKeyError,
    PathRole,
)

CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def test_semantic_preset_requires_an_explicit_compatible_checkpoint() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize", overrides=["visualization=semantic_line"]
        )
    assert OmegaConf.is_missing(config.visualization, "checkpoint")
    with pytest.raises(MissingConfigurationKeyError, match="checkpoint"):
        build_runtime_config(config)


def test_explicit_semantic_artifact_keeps_pretrained_weights_in_checkpoint_root(
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
                f"visualization.checkpoint={{role:artifact,path:{relative}}}",
            ],
        )
    runtime = build_runtime_config(config)
    assert runtime.task == "semantic_line"
    assert runtime.checkpoint.path == tmp_path / "outputs" / relative
    assert runtime.checkpoint.role is PathRole.ARTIFACT
    assert runtime.resolver.resolve(PathRole.CHECKPOINT, "dinov3/backbone.pth") == (
        tmp_path / "ckpt/dinov3/backbone.pth"
    )


def test_legacy_visualization_checkpoint_string_retains_checkpoint_authority(
    tmp_path: Path,
) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize",
            overrides=[
                f"paths.project_root={tmp_path}",
                "visualization.checkpoint=court_detection/adopted.ckpt",
            ],
        )
    runtime = build_runtime_config(config)
    assert runtime.checkpoint.path == tmp_path / "ckpt/court_detection/adopted.ckpt"
    assert runtime.checkpoint.role is PathRole.CHECKPOINT


@pytest.mark.parametrize(
    "reference",
    [
        "{role:artifact,path:../escaped.ckpt}",
        "{role:output,path:run/model.ckpt}",
        "{role:artifact}",
    ],
)
def test_visualization_rejects_unauthorized_checkpoint_declarations(
    tmp_path: Path, reference: str,
) -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="visualize",
            overrides=[
                f"paths.project_root={tmp_path}",
                f"visualization.checkpoint={reference}",
            ],
        )
    with pytest.raises(ConfigurationError):
        build_runtime_config(config)
