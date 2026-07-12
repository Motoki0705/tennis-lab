"""SLCS training configuration integration tests."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.tasks.slcs.training.lightning_module import SLCSLightningModule

_CONFIG_DIR = Path(__file__).parents[4] / "src" / "tasks" / "slcs" / "configs"


def test_lightning_schedule_uses_trainer_max_epochs() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=["model=small", "training.trainer.max_epochs=7"],
        )

    module = SLCSLightningModule(config)

    assert module.max_epochs == 7
