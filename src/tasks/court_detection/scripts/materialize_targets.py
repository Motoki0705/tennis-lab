"""Explicitly precompute selected segmentation/line Court targets.

Usage:
    python -m src.tasks.court_detection.scripts.materialize_targets data/processing=all
    python -m src.tasks.court_detection.scripts.materialize_targets data/source=synthetic_court data/processing=seg_line

Notes:
    - Hydra loads configuration from ``src/tasks/court_detection/configs/train.yaml``.
    - Outputs are written only below ``data.processing.derived_target_root``;
      neither source dataset is modified.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.target_generation.materializer import (
    CourtTargetMaterializer,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.materialize_targets"


def _validate_boundary(config: DictConfig) -> None:
    CourtTrainingConfig.from_config(config)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="train",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(config: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Materialize dense heads selected by data.processing.targets."""
    runtime = CourtTrainingConfig.from_config(config)
    store = CourtDerivedTargetStore(runtime.data.processing.derived_target_root)
    input_layer = build_court_input(runtime.data.source, target_store=store)
    dense = tuple(
        target.kind
        for target in runtime.data.processing.targets
        if target.kind in {"seg", "line"}
    )
    if not dense:
        raise ValueError(
            "Select seg and/or line in data.processing.targets before materialization."
        )
    results = CourtTargetMaterializer(
        input_layer=input_layer,
        target_store=store,
    ).materialize(
        splits=("train", "val", "test"),
        target_kinds=dense,
    )
    for result in results:
        print(
            f"[court-targets] split={result.split} "
            f"target={result.target_kind} written={result.written}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
