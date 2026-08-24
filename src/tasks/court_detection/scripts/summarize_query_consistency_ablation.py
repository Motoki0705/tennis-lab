"""Collect and summarize one explicitly completed Issue #790 scaling phase.

Usage:
    python -m src.tasks.court_detection.scripts.summarize_query_consistency_ablation summary.phase=encoder_scaling
    python -m src.tasks.court_detection.scripts.summarize_query_consistency_ablation summary.phase=consistency_ablation

Notes:
    - Hydra loads ``src/tasks/court_detection/configs/summarize_query_consistency_ablation.yaml``.
    - Test metrics, TensorBoard diagnostics/loss curves, and GPU capacity profiles
      are all mandatory; missing evidence fails before summary artifacts are written.
    - This script reads completed evidence and never launches training or profiling.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.experiments.query_consistency_summary import (
    QueryConsistencySummaryConfig,
    collect_query_consistency_results,
    load_json_mapping,
    summarize_query_consistency,
    validate_query_consistency_summary_boundary,
    write_query_consistency_summary_artifacts,
)
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json_atomic

_BOUNDARY = "court_detection.summarize_query_consistency_ablation"
register_boundary_validator(
    _BOUNDARY, validate_query_consistency_summary_boundary
)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="summarize_query_consistency_ablation",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Collect complete requested-phase evidence and write summary artifacts."""
    runtime = QueryConsistencySummaryConfig.from_config(cfg)
    manifest = load_json_mapping(runtime.manifest_path)
    results = collect_query_consistency_results(
        manifest,
        evidence_root=runtime.evidence_root,
        phase=runtime.phase,
        require_gpu_profiles=runtime.require_gpu_profiles,
    )
    summary = summarize_query_consistency(
        manifest,
        results,
        phase=runtime.phase,
    )
    save_json_atomic(results, runtime.results_path)
    artifacts = write_query_consistency_summary_artifacts(
        summary,
        results,
        output_dir=runtime.output_dir,
    )
    print(f"Saved {len(artifacts)} #790 artifacts to {runtime.output_dir}")


if __name__ == "__main__":
    main()
