"""Run the isolated COLMAP geometry bridge through shared path roles.

Usage:
    python -m src.synthetic_data_generation.scripts.alignment.geometry_bridge

Notes:
    - Hydra composes `configs/alignment/geometry_bridge.yaml` for direct use.
    - The provider subprocess calls `provider_main` with an explicit roots JSON.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def provider_main(arguments: Sequence[str]) -> int:
    """Run inside the configured provider interpreter without Hydra imports."""
    from src.synthetic_data_generation.alignment.scene_provider.geometry_bridge import (
        _runtime_versions,
        run_geometry_bridge,
    )

    if tuple(arguments) == ("--runtime-json",):
        print(json.dumps(_runtime_versions(), sort_keys=True))
        return 0
    if len(arguments) != 4 or arguments[0] != "--path-roots":
        raise ValueError(
            "Geometry bridge requires --path-roots ROOTS_JSON REQUEST.json OUTPUT.npz."
        )
    from src.synthetic_data_generation.configuration import non_hydra_path_resolver

    resolver = non_hydra_path_resolver(arguments[1])
    run_geometry_bridge(
        Path(arguments[2]),
        Path(arguments[3]),
        resolver=resolver,
    )
    return 0


def _hydra_entry(cfg: DictConfig) -> int:
    from src.synthetic_data_generation.alignment.scene_provider.geometry_bridge import (
        run_geometry_bridge,
    )
    from src.synthetic_data_generation.configuration import validate_config
    from src.utils.configuration import PathRole

    runtime = validate_config("synthetic.alignment.geometry_bridge", cfg)
    run_geometry_bridge(
        runtime.path(PathRole.CACHE, "request"),
        runtime.path(PathRole.CACHE, "output"),
        resolver=runtime.resolver,
    )
    return 0


def hydra_cli() -> int:
    """Compose the direct-use request/output config and invoke the bridge."""
    from src.utils.hydra import hydra_main

    @hydra_main(
        version_base="1.3",
        config_path="../../configs",
        config_name="alignment/geometry_bridge",
        validation_boundary="synthetic.alignment.geometry_bridge",
    )
    def run(cfg: DictConfig) -> int:
        return _hydra_entry(cfg)

    result: int = run()
    return result


if __name__ == "__main__":
    raise SystemExit(hydra_cli())


__all__ = ["hydra_cli", "provider_main"]
