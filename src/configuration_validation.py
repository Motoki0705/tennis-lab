"""Execute the cross-domain strict configuration validation matrix.

This module is an integration boundary, rather than a runtime compatibility layer.
It composes canonical configs and proves that representative missing, unknown,
mistyped, conflicting, legacy, and invalid-path values fail before side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.submodules.configuration import GvhmrDemoConfig
from src.synthetic_data_generation.config_validation import (
    run_validation_matrix as run_synthetic_matrix,
)
from src.tasks.ball_detection.validation import (
    run_negative_matrix as run_ball_matrix,
)
from src.tasks.blcs.configuration import (
    run_negative_matrix as run_blcs_matrix,
)
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.plcs.validation_matrix import run_negative_matrix as run_plcs_matrix
from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.utils.configuration import ConfigurationError, PathContractError
from src.utils.paths import PROJECT_ROOT

_EXPECTED_FAILURES = (
    ConfigurationError,
    PathContractError,
    KeyError,
    TypeError,
    ValueError,
)


def _compose(config_dir: Path, config_name: str) -> DictConfig:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir.resolve())
    ):
        return compose(config_name=config_name)


def _reject(name: str, operation: Callable[[], object]) -> str:
    try:
        operation()
    except _EXPECTED_FAILURES:
        return name
    raise AssertionError(f"Cross-domain negative case was accepted: {name}")


def _mutate(config: DictConfig, operation: Callable[[DictConfig], None]) -> DictConfig:
    candidate = deepcopy(config)
    with open_dict(candidate):
        operation(candidate)
    return candidate


def _domain_cases(
    domain: str,
    canonical: DictConfig,
    validate: Callable[[DictConfig], object],
) -> tuple[str, ...]:
    validate(canonical)
    cases = (
        _reject(
            f"{domain}:unknown-key",
            lambda: validate(
                _mutate(canonical, lambda value: value.__setitem__("typo", True))
            ),
        ),
        _reject(
            f"{domain}:missing-paths",
            lambda: validate(
                _mutate(canonical, lambda value: value.__delitem__("paths"))
            ),
        ),
        _reject(
            f"{domain}:wrong-root-type",
            lambda: validate(
                _mutate(
                    canonical,
                    lambda value: value.paths.__setitem__("data_root", 3),
                )
            ),
        ),
    )
    return (f"{domain}:canonical", *cases)


def _training_conflict(config: DictConfig) -> None:
    config.run.resume = "resume.ckpt"
    config.run.init_weights = "init.ckpt"


def _tennis_scene_conflict(config: DictConfig) -> None:
    config.ball_detection.source = "execute"
    config.ball_detection.load_path = "tennis_scene/legacy-ball.npz"


def _submodule_mapping(config: DictConfig) -> Mapping[str, object]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError("GVHMR demo composition must be a mapping.")
    return cast(Mapping[str, object], value)


def run_cross_domain_matrix() -> tuple[str, ...]:
    """Run canonical and negative validation for every migrated domain."""
    completed = list(run_ball_matrix())
    run_blcs_matrix()
    completed.append("blcs:root-negative-matrix")
    run_plcs_matrix()
    completed.append("plcs:negative-matrix")
    completed.extend(run_synthetic_matrix())

    task_root = PROJECT_ROOT / "src/tasks"
    court = _compose(task_root / "court_detection/configs", "train")
    completed.extend(_domain_cases("court", court, CourtTrainingConfig.from_config))
    completed.append(
        _reject(
            "court:exclusive-run-paths",
            lambda: CourtTrainingConfig.from_config(_mutate(court, _training_conflict)),
        )
    )

    blcs = _compose(task_root / "blcs/configs", "train")
    completed.extend(_domain_cases("blcs", blcs, validate_training_boundary))
    completed.append(
        _reject(
            "blcs:exclusive-run-paths",
            lambda: validate_training_boundary(_mutate(blcs, _training_conflict)),
        )
    )

    slcs = _compose(task_root / "slcs/configs", "train")
    completed.extend(_domain_cases("slcs", slcs, SLCSTrainingRuntimeConfig.from_config))
    completed.append(
        _reject(
            "slcs:exclusive-run-paths",
            lambda: SLCSTrainingRuntimeConfig.from_config(
                _mutate(slcs, _training_conflict)
            ),
        )
    )

    tennis_scene = _compose(PROJECT_ROOT / "src/tennis_scene/configs", "pipeline")
    completed.extend(
        _domain_cases("tennis-scene", tennis_scene, PipelineRuntimeConfig.from_config)
    )
    completed.append(
        _reject(
            "tennis-scene:load-execute-conflict",
            lambda: PipelineRuntimeConfig.from_config(
                _mutate(tennis_scene, _tennis_scene_conflict)
            ),
        )
    )

    submodule = _compose(PROJECT_ROOT / "src/submodules/configs", "demo_gvhmr")

    def validate_submodule(config: DictConfig) -> object:
        return GvhmrDemoConfig.from_mapping(
            _submodule_mapping(config), repository_root=PROJECT_ROOT
        )

    completed.extend(_domain_cases("submodules", submodule, validate_submodule))
    completed.append(
        _reject(
            "submodules:invalid-num-tracks",
            lambda: validate_submodule(
                _mutate(
                    submodule,
                    lambda value: value.__setitem__("num_tracks", 0),
                )
            ),
        )
    )
    return tuple(completed)


def main() -> int:
    """Run the deterministic cross-domain matrix."""
    completed = run_cross_domain_matrix()
    print(f"Cross-domain strict configuration matrix: PASS ({len(completed)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
