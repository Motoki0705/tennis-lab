"""Check all required assets for the published pipeline before GPU execution."""

from __future__ import annotations

import json
from dataclasses import asdict

from omegaconf import DictConfig

from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    validate_pipeline_boundary,
)
from src.tennis_scene.preflight import check_assets, standard_requirements
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "tennis_scene.preflight"
register_boundary_validator(_BOUNDARY, validate_pipeline_boundary)


@hydra_main(version_base="1.3", config_path="../configs", config_name="pipeline", validation_boundary=_BOUNDARY)
def main(cfg: DictConfig) -> int:
    runtime = PipelineRuntimeConfig.from_config(cfg)
    checks = check_assets(standard_requirements(runtime))
    failed = sum(check.status != "ok" for check in checks)
    print(json.dumps({"status": "failed" if failed else "ok", "failed": failed,
                      "inference_executed": False, "checks": [asdict(check) for check in checks]}, indent=2))
    # Hydra discards the task function's return value when invoked as a CLI.
    # Raise inside the task so missing assets cannot produce a successful shell exit.
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
