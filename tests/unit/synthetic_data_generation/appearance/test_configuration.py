"""Reject malformed CLI requests before image API, file publication or queue work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from src.synthetic_data_generation.appearance.configuration import (
    APPEARANCE_BOUNDARY,
    AppearanceRuntimeConfig,
)
from src.utils.hydra import registered_boundary_validators, validate_boundary
from src.utils.paths import PROJECT_ROOT


def cli_config() -> DictConfig:
    config = OmegaConf.load(
        PROJECT_ROOT
        / "src/synthetic_data_generation/configs/run_appearance_variant.yaml"
    )
    assert isinstance(config, DictConfig)
    del config.hydra
    return config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "unknown"),
        ("api_retry", "false"),
        ("variant.first_index", True),
        ("variant.output_root", "relative/output"),
        ("comparison.reference", "relative.png"),
        ("derive.parent_root", "../parent"),
        ("report.roots", ["relative/root"]),
        ("result.path", "relative.png"),
        ("batch.concurrency", 0),
        ("batch.indices", [0, 0]),
        ("unexpected", 1),
    ],
)
def test_cli_boundary_rejects_invalid_fields(field: str, value: Any) -> None:
    config = cli_config()
    OmegaConf.update(config, field, value)
    with pytest.raises(ValidationError):
        validate_boundary(APPEARANCE_BOUNDARY, config)


def test_result_action_requires_explicit_review() -> None:
    config = cli_config()
    config.action = "record_result"
    with pytest.raises(ValidationError, match="record_result requires"):
        AppearanceRuntimeConfig.from_config(config)
    config.result.request_id = "frame_000000-r01-01"
    config.result.path = "/tmp/reviewed-api-image.png"
    config.result.accepted = False
    config.result.notes = "Rejected after visual review"
    runtime = AppearanceRuntimeConfig.from_config(config)
    assert runtime.result.accepted is False
    assert runtime.result.path == Path("/tmp/reviewed-api-image.png")


def test_cli_does_not_synthesize_removed_composed_defaults() -> None:
    config = cli_config()
    del config.variant.max_steps
    with pytest.raises(ValidationError, match="missing fields.*max_steps"):
        validate_boundary(APPEARANCE_BOUNDARY, config)


def test_boundary_preserves_selected_venv_symlink(tmp_path: Path) -> None:
    config = cli_config()
    executable = tmp_path / "base-python"
    executable.touch()
    selected = tmp_path / "venv/bin/python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(executable)
    config.variant.training_python = str(selected)
    runtime = AppearanceRuntimeConfig.from_config(config)
    assert runtime.variant.training_python == selected
    assert runtime.variant.training_python != selected.resolve()
    assert registered_boundary_validators()[APPEARANCE_BOUNDARY].callable_symbol == (
        "src.synthetic_data_generation.appearance.configuration.validate_appearance_boundary"
    )
