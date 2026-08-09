"""Deterministic negative matrix for the canonical scene-pipeline config."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Final

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.utils.configuration import ConfigurationError, PathContractError
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT: Final[Path] = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def _compose() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        return compose(config_name="run_scene_pipeline")


def _must_reject(name: str, operation: Callable[[], object]) -> str:
    try:
        operation()
    except (ConfigurationError, PathContractError, FileNotFoundError, TypeError, ValueError):
        return name
    raise AssertionError(f"Canonical scene configuration accepted invalid case: {name}")


def _mutated(config: DictConfig, key: str, value: object) -> DictConfig:
    candidate = deepcopy(config)
    OmegaConf.update(candidate, key, value, merge=False)
    return candidate


def _unknown(config: DictConfig) -> DictConfig:
    candidate = deepcopy(config)
    with open_dict(candidate):
        candidate.pipeline.compatibility_mode = True
    return candidate


def _missing(config: DictConfig) -> DictConfig:
    candidate = deepcopy(config)
    with open_dict(candidate):
        del candidate.request.targets
    return candidate


def run_scene_pipeline_validation_matrix() -> tuple[str, ...]:
    """Exercise representative unknown, missing, path, and semantic failures."""
    valid = _compose()
    ScenePipelineConfiguration.from_config(valid)
    cases: tuple[tuple[str, DictConfig], ...] = (
        ("unknown-key", _unknown(valid)),
        ("missing-targets", _missing(valid)),
        ("source-escape", _mutated(valid, "request.source_video", "../outside.mp4")),
        ("unknown-stage", _mutated(valid, "request.from_stage", "legacy_pipeline")),
        ("implicit-target", _mutated(valid, "request.targets", [])),
        ("camera-count", _mutated(valid, "camera.expected_camera_count", 5)),
        ("court-budget", _mutated(valid, "dataset.court.sampling.proposal_budget", 5001)),
        (
            "court-groups",
            _mutated(valid, "dataset.court.sampling.minimum_trajectory_groups", 23),
        ),
        (
            "short-frame-mode",
            _mutated(valid, "dataset.blcs.timeline.frame_selection", "first_64"),
        ),
        (
            "missing-holdout",
            _mutated(valid, "alignment.evidence.holdout_fraction", 0.0),
        ),
    )
    passed = ["canonical"]
    for name, candidate in cases:
        passed.append(
            _must_reject(
                name,
                partial(ScenePipelineConfiguration.from_config, candidate),
            )
        )
    return tuple(passed)


def canonical_config_root() -> Path:
    """Return the sole Hydra configuration root for external audit tools."""
    return _CONFIG_ROOT


__all__ = ["canonical_config_root", "run_scene_pipeline_validation_matrix"]
