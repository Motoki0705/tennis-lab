"""
Evaluate a trained SLCS checkpoint on a dataset split and export analysis arrays.

Usage:
    python -m src.tasks.slcs.scripts.evaluate evaluate.checkpoint=slcs/run/last.ckpt
    python -m src.tasks.slcs.scripts.evaluate evaluate.checkpoint=... evaluate.split=val
    python -m src.tasks.slcs.scripts.evaluate court_coordinate_normalization=v2 evaluate.checkpoint=...

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/evaluate.yaml`
      (dataset location comes from the shared `data` group).
    - Checkpoint and output paths are relative to `paths.checkpoint_root` and
      `paths.output_root`, respectively.
    - The selected normalization must match the dataset and checkpoint; legacy
      metadata-free artifacts are accepted only with the default v1 runtime.
    - Writes `metrics.json` (BLCS/PLCS-comparable metric names) and
      `eval_arrays.npz` (per-frame errors/masks/uncertainties) into
      `evaluate.output_dir`; the analysis script consumes the npz.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSEvaluationConfig
from src.tasks.slcs.evaluation.evaluate import evaluate_split, save_evaluation
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.utils.hydra import hydra_main


def run(config: DictConfig) -> None:
    """Run evaluation and persist artifacts."""
    runtime = SLCSEvaluationConfig.from_config(config)
    predictor = SLCSPredictor.load_from_checkpoint(
        runtime.checkpoint,
        resolver=runtime.resolver,
        court_coordinate_normalization=runtime.court_coordinate_normalization,
        device=runtime.device,
        strict=runtime.checkpoint_strict,
        weights_only=runtime.checkpoint_weights_only,
    )
    report, arrays = evaluate_split(
        predictor,
        dataset_root=runtime.data.dataset_root,
        split_file=runtime.data.split_file,
        split=runtime.split,
        data_config=runtime.data.pipeline,
        batch_size=runtime.batch_size,
    )
    metrics_path, arrays_path = save_evaluation(
        runtime.output_dir,
        report,
        arrays,
        context={
            "checkpoint": str(runtime.checkpoint),
            "split": runtime.split,
            "dataset_root": str(runtime.data.dataset_root),
            "court_coordinate_normalization": {
                "version": runtime.court_coordinate_normalization.version,
                "scale_xyz": list(
                    runtime.court_coordinate_normalization.scale_xyz
                ),
            },
        },
    )
    print(f"metrics -> {metrics_path}")
    print(f"arrays  -> {arrays_path}")
    for key in sorted(report):
        print(f"  {key}: {report[key]:.6f}")


@hydra_main(
    config_path="../configs",
    config_name="evaluate",
    version_base="1.3",
    validation_boundary="slcs.evaluate",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for SLCS evaluation."""
    run(config)


if __name__ == "__main__":
    main()
