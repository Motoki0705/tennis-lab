"""Serve the BLCS simulator API using strict Hydra configuration.

Usage:
    python -m src.tasks.blcs.generate_dataset.api_server server.port=8001

Notes:
    - Hydra loads `src/tasks/blcs/configs/api_server.yaml`.
    - Simulation defaults come from the canonical BLCS generator config groups.
"""

from __future__ import annotations

import uvicorn
from omegaconf import DictConfig

from src.tasks.blcs.configuration import validate_api_boundary
from src.tasks.blcs.generate_dataset.api_server.app import create_app
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.utils.hydra import hydra_main


@hydra_main(  # type: ignore[untyped-decorator]
    config_path="../../configs",
    config_name="api_server",
    version_base="1.3",
    validation_boundary="blcs.api_server",
)
def main(config: DictConfig) -> int:
    """Validate config before constructing the application or opening a socket."""
    validate_api_boundary(config)
    generator_config = build_generator_config(config)
    uvicorn.run(
        create_app(generator_config),
        host=str(config.server.host),
        port=int(config.server.port),
        log_level=str(config.server.log_level),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
