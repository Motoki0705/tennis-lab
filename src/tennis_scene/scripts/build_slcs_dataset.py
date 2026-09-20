"""Build a versioned real-RGB SLCS dataset from clips with one command."""

from omegaconf import DictConfig

from src.tennis_scene.dataset_pipeline.configuration import validate_build_config
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("tennis_scene.build_slcs_dataset", validate_build_config)


@hydra_main(config_path="../configs", config_name="build_slcs_dataset", version_base="1.3", validation_boundary="tennis_scene.build_slcs_dataset")
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.dataset_pipeline.build import build_dataset

    build_dataset(cfg)


if __name__ == "__main__":
    main()
