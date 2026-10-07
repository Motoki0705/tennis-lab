"""Unsupported task renderers fail before constructing models or using CUDA."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.training.lightning_module import (
    PlayerDetectionLightningModule,
)
from src.tasks.slcs.training.lightning_module import SLCSLightningModule

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("task", ["slcs", "player_detection"])
def test_unsupported_task_rejects_qualitative_logging_before_model_creation(
    task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_model_creation(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsupported qualitative logging must fail before building the model.")

    monkeypatch.setattr(
        "src.tasks.slcs.training.lightning_module.create_slcs_model_io",
        unexpected_model_creation,
    )
    monkeypatch.setattr(
        "src.tasks.player_detection.training.lightning_module.build_player_dino",
        unexpected_model_creation,
    )
    with initialize_config_dir(
        config_dir=str(_ROOT / "src/tasks" / task / "configs"), version_base="1.3"
    ):
        config = compose(
            config_name="train", overrides=["training.qualitative_logging.enabled=true"]
        )
    with pytest.raises(ValueError, match="to implement render_qualitative_samples"):
        if task == "slcs":
            SLCSLightningModule(config)
        else:
            PlayerDetectionLightningModule(config, PlayerTrainingConfig.from_config(config))
