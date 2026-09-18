"""Quality CLI boundary registration, strict config and independent root routing."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.dataset_pipeline.quality_report import (
    validate_quality_report_config,
)
from src.tennis_scene.scripts import report_slcs_dataset_quality as cli
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES
from src.utils.hydra import validate_boundary
from src.utils.paths import PROJECT_ROOT


def configuration(tmp_path: Path) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        return compose(
            config_name="report_slcs_dataset_quality",
            overrides=[
                f"paths.project_root={tmp_path / 'project'}",
                f"paths.data_root={tmp_path / 'inputs'}",
                f"paths.output_root={tmp_path / 'reports'}",
                "output_dir=tennis_scene/analyze/quality_smoke/s42-001",
            ],
        )


def test_quality_report_has_audited_runtime_boundary() -> None:
    boundary = next(
        item for item in EXPECTED_RUNTIME_BOUNDARIES if item.module == cli.__name__
    )
    assert boundary.validator_key == "tennis_scene.report_slcs_dataset_quality"
    assert (
        boundary.validator_callable
        == "src.tennis_scene.dataset_pipeline.quality_report.validate_quality_report_config"
    )


def test_roots_are_independent_and_validator_does_not_read_inputs(
    tmp_path: Path,
) -> None:
    cfg = configuration(tmp_path)
    with patch(
        "src.tennis_scene.dataset_pipeline.quality_report.load_dataset_manifest",
        side_effect=AssertionError("validator performed IO"),
    ):
        validate_boundary("tennis_scene.report_slcs_dataset_quality", cfg)
    assert not list(tmp_path.iterdir())
    with patch.object(
        cli, "write_quality_report", return_value={"status": "incomplete", "counts": {}}
    ) as report:
        cli.main.__wrapped__(cfg)
    args = report.call_args.args
    assert args[0] == tmp_path / "inputs" / cfg.dataset_directory
    assert args[1] == tmp_path / "inputs/tennis_multivew/processed/meiji_3cam/dataset"
    assert all(path.is_relative_to(tmp_path / "reports") for path in args[2] + args[3])
    output = tmp_path / "reports/tennis_scene/analyze/quality_smoke/s42-001"
    assert args[4] == output
    assert (output / "config.yaml").is_file()
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("allow_incomplete", "false"),
        ("allow_incomplete", 0),
        ("dataset_directory", 7),
        ("generation_directories", "some/path"),
        ("observation_directories", []),
        ("generation_directories", ["a/b", "a/b"]),
        ("excluded_clips", {"video/clip": ""}),
        ("excluded_clips", {"../clip": "reason"}),
        ("quality.min_ball_cameras", True),
        ("quality.min_ball_cameras", 1.5),
        ("quality.label_weight_power", "1.0"),
        ("quality.label_weight_power", float("nan")),
        ("quality.min_player_confidence", -0.1),
        ("quality.extra", 1),
        ("extra", 1),
        ("paths.extra", "unexpected"),
        ("dataset_directory", "../escape"),
    ],
)
def test_invalid_or_extra_config_values_are_not_coerced(
    tmp_path: Path, key: str, value: object
) -> None:
    cfg = configuration(tmp_path)
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises((ValueError, TypeError)):
        validate_quality_report_config(cfg)
