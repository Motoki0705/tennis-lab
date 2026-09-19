"""evaluate refinement; execute GPU diagnostics through training queue."""

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.dataset_pipeline.diagnostics.configuration import (
    RefinementEvaluationConfig,
    validate_refinement,
)
from src.utils.hydra import hydra_main, register_boundary_validator

register_boundary_validator("tennis_scene.evaluate_refinement", validate_refinement)


@hydra_main(
    config_path="../configs",
    config_name="evaluate_refinement",
    version_base="1.3",
    validation_boundary="tennis_scene.evaluate_refinement",
)
def main(cfg: DictConfig) -> None:
    from src.tennis_scene.dataset_pipeline.diagnostics.refinement import evaluate

    request = RefinementEvaluationConfig.from_config(cfg)
    evaluate(request)
    OmegaConf.save(cfg, request.output / "diagnostic_config.yaml", resolve=True)


if __name__ == "__main__":
    main()
