"""Evaluate one BLCS reference-v2 checkpoint against a strict paired side manifest.

Usage:
    TENNIS_REPRO_DIR=/path/to/repro python -m src.tasks.blcs.scripts.evaluate_reference_counterfactual evaluation.checkpoint_path=/path/to/model.ckpt paths.data_root=/path/to/data

Notes:
    - Hydra loads `src/tasks/blcs/configs/evaluate_reference_counterfactual.yaml`.
    - The command performs two checkpoint-only test passes and never calls fit().
    - Queue-compatible `pred_test.npz` and flat `metrics.json`, plus the full
      `reference_counterfactual.json`, are published without overwrite.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.blcs.evaluation import run_blcs_reference_counterfactual
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="evaluate_reference_counterfactual",
    version_base="1.3",
    validation_boundary="blcs.evaluate_reference_counterfactual",
)
def main(config: DictConfig) -> None:
    """Run strict paired checkpoint-only BLCS evaluation."""
    paths = run_blcs_reference_counterfactual(config)
    print(f"Saved BLCS paired report to {paths.json_path}")


if __name__ == "__main__":
    main()
