"""Train Court detection with synthetic and real samples in every batch.

Usage:
    python -m src.tasks.court_detection.scripts.train_mixed
    python -m src.tasks.court_detection.scripts.train_mixed \
        loss.pose.enabled=true \
        loss.pose.translation_weight=1.0 \
        loss.pose.rotation_weight=1.0 \
        loss.pose.focal_weight=1.0 \
        loss.consistency.enabled=true \
        loss.consistency.weight=1.0 \
        data/augmentation=pose_safe \
        data.source.keypoint_court_scope=target_court
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_detection.training.runner_mixed import (
    MixedCourtDetectionTrainingRunner,
    validate_mixed_train_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.train_mixed"
register_boundary_validator(_BOUNDARY, validate_mixed_train_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_mixed",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Train with fixed-ratio mixed-source batches."""
    MixedCourtDetectionTrainingRunner().run(cfg)


if __name__ == "__main__":
    main()
