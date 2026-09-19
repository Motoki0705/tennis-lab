"""probe meiji court; execute GPU diagnostics through training queue."""

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.dataset_pipeline.diagnostics.configuration import (
    CourtProbeConfig,
    validate_court,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("tennis_scene.probe_meiji_court", validate_court)


@hydra_main(
    config_path="../configs",
    config_name="probe_meiji_court",
    version_base="1.3",
    validation_boundary="tennis_scene.probe_meiji_court",
)
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.dataset_pipeline.diagnostics.court import probe

    request = CourtProbeConfig.from_config(cfg)
    probe(request)
    OmegaConf.save(cfg, request.runtime.output / "diagnostic_config.yaml", resolve=True)


if __name__ == "__main__":
    main()
