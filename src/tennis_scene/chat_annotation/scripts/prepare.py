"""Hydra CLI: python -m src.tennis_scene.chat_annotation.scripts.prepare."""

from collections.abc import Callable

from omegaconf import DictConfig

from src.tennis_scene.chat_annotation.configuration import (
    PrepareConfig,
    validate_prepare_config,
)
from src.utils.hydra import hydra_main as _shared_hydra_main
from src.utils.hydra import register_boundary_validator

register_boundary_validator(
    "tennis_scene.chat_annotation.prepare", validate_prepare_config
)

def hydra_main(
    *,
    config_path: str,
    config_name: str,
    version_base: str,
    validation_boundary: str,
) -> Callable[[Callable[[DictConfig], None]], Callable[[], None]]:
    """Keep this CLI typed when mypy intentionally skips imported modules."""
    decorator: Callable[
        [Callable[[DictConfig], None]], Callable[[], None]
    ] = _shared_hydra_main(
        config_path=config_path,
        config_name=config_name,
        version_base=version_base,
        validation_boundary=validation_boundary,
    )
    return decorator


@hydra_main(
    config_path="../configs",
    config_name="prepare",
    version_base="1.3",
    validation_boundary="tennis_scene.chat_annotation.prepare",
)
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.chat_annotation.batch import prepare_batch
    from src.tennis_scene.chat_annotation.preparation import prepare

    config = PrepareConfig.from_config(cfg)
    destination = prepare_batch(config) if config.urls else prepare(config)
    print(f"Prepared: {destination}")


if __name__ == "__main__":
    main()
