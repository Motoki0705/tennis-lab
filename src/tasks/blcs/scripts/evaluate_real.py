"""Evaluate a strictly loaded BLCS checkpoint on the fixed held-out split."""

from omegaconf import DictConfig

from src.tasks.blcs.evaluation.configuration import (
    RealEvaluationConfig,
    validate_real_evaluation,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("blcs.evaluate_real", validate_real_evaluation)


@hydra_main(
    config_path="../configs",
    config_name="evaluate_real",
    version_base="1.3",
    validation_boundary="blcs.evaluate_real",
)
def main(cfg: DictConfig) -> None:
    from src.tasks.blcs.evaluation.real import evaluate

    evaluate(RealEvaluationConfig.from_config(cfg))


if __name__ == "__main__":
    main()
