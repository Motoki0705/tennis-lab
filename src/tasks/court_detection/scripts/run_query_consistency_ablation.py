"""Generate the Issue #790 DPT scaling-grid and consistency manifest.

Usage:
    python -m src.tasks.court_detection.scripts.run_query_consistency_ablation
    python -m src.tasks.court_detection.scripts.run_query_consistency_ablation consistency_ablation.selected.input_long_side=256 consistency_ablation.selected.encoder_depth=8 consistency_ablation.selected.decoder_family=dpt consistency_ablation.selected.decoder_size=base

Notes:
    - Hydra loads ``src/tasks/court_detection/configs/run_query_consistency_ablation.yaml``.
    - The 24-member input-size × depth × DPT-size scaling grid is always queue-ready.
      The optional four-condition consistency phase requires one explicit selected
      grid member.
    - This script writes commands and never launches training, profiling, or a queue worker.
"""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from src.tasks.court_detection.experiments.query_consistency import (
    QueryConsistencyAblationConfig,
    build_query_consistency_manifest,
    validate_query_consistency_ablation_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json_atomic

_BOUNDARY = "court_detection.run_query_consistency_ablation"
register_boundary_validator(
    _BOUNDARY, validate_query_consistency_ablation_boundary
)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="run_query_consistency_ablation",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Write one deterministic scaling manifest without starting work."""
    runtime = QueryConsistencyAblationConfig.from_config(cfg)
    manifest = build_query_consistency_manifest(runtime)
    save_json_atomic(manifest, runtime.output_path)
    runs = cast(list[dict[str, object]], manifest["runs"])
    ready = sum(bool(run["queue_ready"]) for run in runs)
    print(
        f"Saved {len(runs)} ordered #790 runs to {runtime.output_path}; "
        f"{ready} are queue-ready."
    )


if __name__ == "__main__":
    main()
