"""Train a ball detector with the staged multi-frame schedule (issue #579).

Runs one phase of the staged schedule (TrackNet T=1 -> mix T=1 -> TrackNet
multi-frame T in [1,8] -> mix multi-frame). Each phase is launched as a separate
process; phases 2-4 load the previous phase's weights via ``run.init_weights``.

Usage:
    python -m src.tasks.ball_detection.scripts.train_staged
    python -m src.tasks.ball_detection.scripts.train_staged --config-name staged_phase3
    python -m src.tasks.ball_detection.scripts.train_staged run.dry_run=true

Notes:
    - Hydra loads configuration from
      ``src/tasks/ball_detection/configs/train_staged.yaml`` (or a phase config).
    - Forwards the resolved config to ``StagedBallDetectionTrainingRunner``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import hydra
from omegaconf import DictConfig

from src.tasks.ball_detection.training.staged_runner import (
    StagedBallDetectionTrainingRunner,
)


def _hydra_main(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return cast(Callable[[Any], Any], hydra.main(*args, **kwargs))


@_hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_staged",
)
def main(cfg: DictConfig) -> None:
    """Train one staged ball-detection phase."""
    runner = StagedBallDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
