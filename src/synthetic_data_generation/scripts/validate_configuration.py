"""Execute the strict Synthetic Data configuration and path matrix.

Usage:
    python -m src.synthetic_data_generation.scripts.validate_configuration

Notes:
    - Hydra composes `src/synthetic_data_generation/configs/validate_configuration.yaml`.
    - The matrix performs no model training or GPU work.
"""

from __future__ import annotations

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from src.synthetic_data_generation.config_validation import run_validation_matrix
from src.synthetic_data_generation.configuration import validate_config
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="validate_configuration",
    validation_boundary="synthetic.validation_matrix",
)
def main(cfg: DictConfig) -> int:
    """Validate the explicit command config and execute every negative case."""
    validate_config("synthetic.validation_matrix", cfg)
    GlobalHydra.instance().clear()
    passed = run_validation_matrix()
    print(f"Synthetic strict configuration matrix: PASS ({len(passed)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
