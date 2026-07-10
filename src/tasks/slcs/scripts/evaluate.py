"""
Evaluate a trained SLCS checkpoint on a dataset split and export analysis arrays.

Usage:
    python -m src.tasks.slcs.scripts.evaluate evaluate.checkpoint=outputs/slcs/.../last.ckpt
    python -m src.tasks.slcs.scripts.evaluate evaluate.checkpoint=... evaluate.split=val
    python -m src.tasks.slcs.scripts.evaluate evaluate.checkpoint=... evaluate.device=cuda

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/evaluate.yaml`
      (dataset location comes from the shared `data` group).
    - Writes `metrics.json` (BLCS/PLCS-comparable metric names) and
      `eval_arrays.npz` (per-frame errors/masks/uncertainties) into
      `evaluate.output_dir`; the analysis script consumes the npz.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.slcs.data.dataset import SLCSDataConfig
from src.tasks.slcs.evaluation.evaluate import evaluate_split, save_evaluation
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.utils.hydra import hydra_main


def run(config: DictConfig) -> None:
    """Run evaluation and persist artifacts."""
    eval_cfg = config.evaluate
    checkpoint = eval_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("evaluate.checkpoint must point to a trained SLCS checkpoint.")
    predictor = SLCSPredictor.load_from_checkpoint(
        str(checkpoint), device=str(eval_cfg.get("device", "cpu"))
    )
    data_config = SLCSDataConfig.from_config(config.data)
    report, arrays = evaluate_split(
        predictor,
        dataset_root=str(config.data.dataset_root),
        split_file=str(config.data.split_file),
        split=str(eval_cfg.get("split", "test")),
        data_config=data_config,
        batch_size=int(eval_cfg.get("batch_size", 4)),
    )
    metrics_path, arrays_path = save_evaluation(
        str(eval_cfg.output_dir),
        report,
        arrays,
        context={
            "checkpoint": str(checkpoint),
            "split": str(eval_cfg.get("split", "test")),
            "dataset_root": str(config.data.dataset_root),
        },
    )
    print(f"metrics -> {metrics_path}")
    print(f"arrays  -> {arrays_path}")
    for key in sorted(report):
        print(f"  {key}: {report[key]:.6f}")


@hydra_main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for SLCS evaluation."""
    run(config)


if __name__ == "__main__":
    main()
