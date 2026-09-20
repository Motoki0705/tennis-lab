"""Import curated original broadcast media and filtered 2D ball observations."""

from omegaconf import DictConfig

from src.tennis_scene.dataset_pipeline.preparation import (
    BroadcastImportConfig,
    import_broadcast,
    validate_broadcast_import_config,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator(
    "tennis_scene.import_broadcast_ball", validate_broadcast_import_config
)


@hydra_main(
    config_path="../configs",
    config_name="import_broadcast_ball",
    version_base="1.3",
    validation_boundary="tennis_scene.import_broadcast_ball",
)
def main(cfg: DictConfig) -> None:
    import_broadcast(BroadcastImportConfig.from_config(cfg))


if __name__ == "__main__":
    main()
