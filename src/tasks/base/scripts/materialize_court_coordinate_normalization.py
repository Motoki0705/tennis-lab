"""Materialize a version-qualified BLCS or PLCS normalization dataset copy.

Usage:
    python -m src.tasks.base.scripts.materialize_court_coordinate_normalization court_coordinate_normalization=v2
    python -m src.tasks.base.scripts.materialize_court_coordinate_normalization court_coordinate_normalization=v2 materialization.dataset_kind=plcs materialization.source_dir=data/plcs_broadcast materialization.output_dir=data/plcs_broadcast_norm_v2

Notes:
    - Hydra loads configuration from ``src/tasks/base/configs/materialize_court_coordinate_normalization.yaml``.
    - The source dataset is validated and never modified or overwritten.
    - Output publication fails if the version-qualified destination already exists.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.base.data.court_coordinate_materializer import (
    CourtCoordinateMaterializationConfig,
    materialize_court_coordinate_normalization_dataset,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "base.materialize_court_coordinate_normalization"


def _validate_boundary(config: DictConfig) -> None:
    CourtCoordinateMaterializationConfig.from_config(config)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="materialize_court_coordinate_normalization",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(config: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Materialize and publish the configured normalization dataset."""
    runtime = CourtCoordinateMaterializationConfig.from_config(config)
    result = materialize_court_coordinate_normalization_dataset(runtime)
    print(
        f"materialized {result.scene_count} scenes at {result.output_dir} "
        f"(max_abs_round_trip_error={result.max_abs_round_trip_error_m:.9g}m; "
        f"manifest={result.manifest_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
