"""benchmark vitpose precision; execute GPU diagnostics through training queue."""

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.dataset_pipeline.diagnostics.configuration import (
    ViTPoseBenchmarkConfig,
    validate_vitpose,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator(
    "tennis_scene.benchmark_vitpose_precision", validate_vitpose
)


@hydra_main(
    config_path="../configs",
    config_name="benchmark_vitpose_precision",
    version_base="1.3",
    validation_boundary="tennis_scene.benchmark_vitpose_precision",
)
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.dataset_pipeline.diagnostics.vitpose import benchmark

    request = ViTPoseBenchmarkConfig.from_config(cfg)
    benchmark(request)
    OmegaConf.save(cfg, request.output / "diagnostic_config.yaml", resolve=True)


if __name__ == "__main__":
    main()
