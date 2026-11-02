"""Check all required assets for the published pipeline before GPU execution."""

from __future__ import annotations

import json
from dataclasses import asdict

from omegaconf import DictConfig
from yaml import YAMLError

from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    pipeline_asset_configuration,
)
from src.tennis_scene.preflight import AssetCheck, check_assets, configured_requirements
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "tennis_scene.preflight"


def _report(checks: tuple[AssetCheck, ...]) -> int:
    failed = sum(check.status != "ok" for check in checks)
    print(json.dumps({"status": "failed" if failed else "ok", "failed": failed,
                      "inference_executed": False, "checks": [asdict(check) for check in checks]}, indent=2))
    if failed:
        raise SystemExit(1)
    return 0


def validate_preflight_boundary(cfg: DictConfig) -> None:
    """Validate diagnostic inputs; child paths are rejected one asset at a time."""
    try:
        pipeline_asset_configuration(cfg)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _report((AssetCheck("pipeline_configuration", "configuration", "error",
                            f"Asset inventory unavailable: {error}"),))


register_boundary_validator(_BOUNDARY, validate_preflight_boundary)


@hydra_main(version_base="1.3", config_path="../configs", config_name="pipeline", validation_boundary=_BOUNDARY)
def main(cfg: DictConfig) -> int:
    try:
        requirements = configured_requirements(cfg)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _report((AssetCheck("pipeline_configuration", "configuration", "error",
                                  f"Asset inventory unavailable: {error}"),))
    checks = check_assets(requirements)
    if any(requirement.path_error is not None for requirement in requirements):
        checks += (AssetCheck("pipeline_configuration", "configuration", "error",
                              "Full runtime validation unavailable until invalid asset paths or configuration inputs are repaired"),)
    else:
        try:
            PipelineRuntimeConfig.from_config(cfg)
        except (OSError, RuntimeError, TypeError, ValueError, YAMLError) as error:
            checks += (AssetCheck("pipeline_configuration", "configuration", "error", str(error)),)
    # Hydra discards the task function's return value when invoked as a CLI.
    # Raise inside the task so missing assets cannot produce a successful shell exit.
    return _report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
