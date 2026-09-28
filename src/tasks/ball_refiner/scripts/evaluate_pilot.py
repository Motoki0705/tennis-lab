"""Diagnose a fixed best checkpoint on one validation partition; queue CUDA execution."""

from omegaconf import DictConfig

from src.tasks.ball_refiner.evaluation.configuration import validate_evaluation_boundary
from src.tasks.ball_refiner.evaluation.runner import run_evaluation
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("ball_refiner.evaluate_pilot", validate_evaluation_boundary)


@hydra_main(version_base=None, config_path="../configs", config_name="evaluate_pilot", validation_boundary="ball_refiner.evaluate_pilot")
def main(config: DictConfig) -> None:
    run_evaluation(config)


if __name__ == "__main__":
    main()
