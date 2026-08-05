"""Train a BLCS model with Hydra-managed configuration.

Usage:
    python -m src.tasks.blcs.scripts.train
    python -m src.tasks.blcs.scripts.train training.max_epochs=5 run.gpus=0
    python -m src.tasks.blcs.scripts.train model=multiview data=multiview_sequence
    python -m src.tasks.blcs.scripts.train data=chunked_multiview_sequence_bs4 training=chunked
    python -m src.tasks.blcs.scripts.train --config-name train_chunked_gan
    python -m src.tasks.blcs.scripts.train run.dry_run=true

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/train.yaml`.
    - Experiment configs can be selected with `--config-name`.
    - Chunked training is selected with a chunked data config.
    - GAN training is selected with a GAN training config.
    - The runner handles the full BLCS training loop from the resolved config.
    - Use `--config-name train_tracking` for multi-ball tracking.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.utils.hydra import hydra_main
from src.utils.paths import PROJECT_ROOT


@hydra_main(
    config_path="../configs",
    config_name="train",
    version_base="1.3",
    validation_boundary="blcs.train",
)
def main(config: DictConfig) -> None:
    """Hydra entry point."""
    TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    model = validate_training_boundary(config)
    generator_config = None
    is_tracking = model.name == "blcs_track_query"
    if str(config.data.backend) == "chunked" and not is_tracking:
        generator_config = build_generator_config(config)
    runner = BLCSTrainingRunner(generator_config=generator_config)
    runner.run(config)


if __name__ == "__main__":
    main()
