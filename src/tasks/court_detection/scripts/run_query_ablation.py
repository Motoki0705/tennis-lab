"""Generate a deterministic ordered manifest of queue-ready Court query commands.

Usage:
    python -m src.tasks.court_detection.scripts.run_query_ablation
    python -m src.tasks.court_detection.scripts.run_query_ablation ablation.selected.encoder_depth=4 ablation.selected.decoder_family=dpt ablation.selected.decoder_size=base

Notes:
    - Hydra loads ``src/tasks/court_detection/configs/run_query_ablation.yaml``.
    - Encoder commands are emitted first; later phases remain explicitly unresolved
      until the prior phase selection is supplied.
    - This script only writes argv records. It never launches GPU work, concurrent
      jobs, or the training-queue worker.
"""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from src.tasks.court_detection.experiments.configuration import (
    QueryAblationConfig,
    validate_ablation_boundary,
)
from src.tasks.court_detection.experiments.query_ablation import (
    build_ablation_manifest,
)
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json_atomic

_BOUNDARY = "court_detection.run_query_ablation"
register_boundary_validator(_BOUNDARY, validate_ablation_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="run_query_ablation",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Write the deterministic manifest without starting any run."""
    runtime = QueryAblationConfig.from_config(cfg)
    manifest = build_ablation_manifest(runtime)
    save_json_atomic(manifest, runtime.output_path)
    runs = cast(list[dict[str, object]], manifest["runs"])
    ready = sum(bool(run["queue_ready"]) for run in runs)
    print(
        f"Saved {len(runs)} ordered runs to {runtime.output_path}; "
        f"{ready} are queue-ready."
    )


if __name__ == "__main__":
    main()
