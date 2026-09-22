"""Resume the interrupted run in fresh workers while retaining its logger history."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import cv2
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.triangulation_residual.training import ResidualTrainingRunner
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main

# This module is imported by spawned workers as well as by the parent process.
cv2.setNumThreads(1)


class ResumeRunner(ResidualTrainingRunner):
    def build_logger(self, config: Any, output_dir: Path) -> TensorBoardLogger:
        runtime = self.validate_runtime_config(config)
        if runtime.run.resume is None:
            raise ValueError("This recovery entrypoint requires an explicit checkpoint")
        checkpoint = torch.load(
            runtime.run.resume, map_location="cpu", weights_only=False
        )
        directories = {
            Path(state["dirpath"])
            for name, state in checkpoint["callbacks"].items()
            if name.startswith("ModelCheckpoint")
        }
        if len(directories) != 1:
            raise ValueError("Expected one original checkpoint directory")
        directory = directories.pop()
        validated = runtime.resolver.validate(PathRole.OUTPUT, output_dir)
        if (
            directory.name != "checkpoints"
            or directory.parent.parent != validated / "logs"
        ):
            raise ValueError("Resume must retain the original run's checkpoint history")
        return TensorBoardLogger(
            save_dir=str(validated), name="logs", version=directory.parent.name
        )


@hydra_main(
    config_path=str(Path(__file__).resolve().parents[3] / "src/tasks/blcs/configs"),
    config_name="train_triangulation_residual_v2",
    version_base="1.3",
    validation_boundary="blcs.triangulation_residual.train",
)
def main(config: DictConfig) -> None:
    ResumeRunner().run(config)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
