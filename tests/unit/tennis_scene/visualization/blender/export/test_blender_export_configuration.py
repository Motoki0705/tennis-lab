"""Tests for the strict Blender-export entrypoint configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tennis_scene.configuration import parse_blender_export_config
from src.utils.configuration import SemanticConfigurationError
from src.utils.paths import PROJECT_ROOT

_CONFIG_DIR = (PROJECT_ROOT / "src/tennis_scene/configs").resolve()


def _config(root: Path, *overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="blender_export",
            overrides=[
                f"paths.project_root={root.resolve()}",
                "input=tennis_scene/clip/scene.npz",
                "camera_id=cam1",
                *overrides,
            ],
        )


def test_shipped_config_resolves_roles_and_limits(tmp_path: Path) -> None:
    runtime = parse_blender_export_config(_config(tmp_path))

    assert runtime.input_path == (tmp_path / "outputs/tennis_scene/clip/scene.npz")
    assert runtime.output_path.name == "bundle"
    assert runtime.output_path.is_relative_to(
        tmp_path / "outputs/tennis_scene/visualize/blender_export"
    )
    assert runtime.bundle.camera_id == "cam1"
    assert runtime.bundle.player_max_gap_frames == 12
    assert runtime.bundle.ball_max_gap_frames == 3
    assert (
        runtime.smpl_faces_path == tmp_path / "ckpt/body_models/smplh/neutral/model.npz"
    )
    assert runtime.smpl_joint_regressor_path == (
        tmp_path
        / "ckpt/body_models/gvhmr/smpl_neutral_J_regressor.pt"
    )


@pytest.mark.parametrize(
    "override",
    [
        "player_max_gap_frames=-1",
        "plate_jpeg_quality=0",
        "sensor_width_mm=0",
    ],
)
def test_invalid_limits_are_rejected(tmp_path: Path, override: str) -> None:
    with pytest.raises(SemanticConfigurationError):
        parse_blender_export_config(_config(tmp_path, override))


@pytest.mark.parametrize(
    "field,value",
    [
        ("sensor_width_mm", float("nan")),
        ("max_ball_reprojection_median_px", float("inf")),
    ],
)
def test_non_finite_limits_are_rejected(
    tmp_path: Path, field: str, value: float
) -> None:
    cfg = _config(tmp_path)
    cfg[field] = value
    with pytest.raises(SemanticConfigurationError):
        parse_blender_export_config(cfg)
