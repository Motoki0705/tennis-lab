"""Hydra CLI: python -m src.tennis_scene.chat_annotation.scripts.prepare."""

from omegaconf import DictConfig

from src.tennis_scene.chat_annotation.configuration import (
    PrepareConfig,
    validate_prepare_config,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator(
    "tennis_scene.chat_annotation.prepare", validate_prepare_config
)


@hydra_main(  # type: ignore[untyped-decorator]
    config_path="../configs",
    config_name="prepare",
    version_base="1.3",
    validation_boundary="tennis_scene.chat_annotation.prepare",
)
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.chat_annotation.preparation import prepare

    destination = prepare(PrepareConfig.from_config(cfg))
    print(f"Prepared: {destination}")


if __name__ == "__main__":
    main()
