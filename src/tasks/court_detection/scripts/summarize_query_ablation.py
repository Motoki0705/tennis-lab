"""Validate complete Court query results and write scaling and Pareto artifacts.

Usage:
    python -m src.tasks.court_detection.scripts.summarize_query_ablation summary.adoption.supervision=kp+pose summary.adoption.rationale='reviewed evidence favors joint pose supervision'

Notes:
    - Hydra loads ``src/tasks/court_detection/configs/summarize_query_ablation.yaml``.
    - Missing phases, runs, seeds, metrics, GPU profiles, or adoption decisions fail
      closed before any summary artifact is written.
    - This script reads completed run evidence and never launches training jobs.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.experiments.configuration import (
    QuerySummaryConfig,
    validate_summary_boundary,
)
from src.tasks.court_detection.experiments.query_summary import (
    load_json_mapping,
    summarize_ablation,
    write_summary_artifacts,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.summarize_query_ablation"
register_boundary_validator(_BOUNDARY, validate_summary_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="summarize_query_ablation",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Validate all evidence before atomically writing summary artifacts."""
    runtime = QuerySummaryConfig.from_config(cfg)
    supervision, rationale = runtime.require_adoption_decision()
    manifest = load_json_mapping(runtime.manifest_path)
    results = load_json_mapping(runtime.results_path)
    summary = summarize_ablation(
        manifest,
        results,
        adopted_supervision=supervision,
        adoption_rationale=rationale,
        require_gpu_profiles=runtime.require_gpu_profiles,
    )
    artifacts = write_summary_artifacts(summary, output_dir=runtime.output_dir)
    print(
        f"Saved {len(artifacts)} Court query summary artifacts to {runtime.output_dir}"
    )


if __name__ == "__main__":
    main()
