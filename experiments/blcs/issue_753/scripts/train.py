"""Train the current BLCS track-query model under the #648 baseline scale.

Usage:
    .venv/bin/python -m experiments.blcs.issue_753.scripts.train

Notes:
    - Hydra reads the experiment recipe from the adjacent configs directory.
    - The training run must be launched through the repository training queue.
"""

from __future__ import annotations

from omegaconf import DictConfig

from experiments.blcs.issue_753.baseline_matched import (
    BaselineMatchedTrainingRunner,
)
from src.tasks.base.configuration import TrainingRuntimeConfig
from src.utils.hydra import hydra_main
from src.utils.paths import PROJECT_ROOT


@hydra_main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:
    """Validate shared runtime paths and execute the matched experiment."""
    TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    BaselineMatchedTrainingRunner().run(config)


if __name__ == "__main__":
    main()
