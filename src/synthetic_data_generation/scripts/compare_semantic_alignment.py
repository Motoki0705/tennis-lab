"""Generate paired placement/projection diagnostics after model training finishes."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.semantic_comparison import (
    compare_alignment,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="compare_semantic_alignment",
    version_base="1.3",
)
def main(config: DictConfig) -> None:
    """Load explicit source/output authority and execute one paired comparison."""
    options = config.comparison
    expected = {
        "checkpoint",
        "output",
        "grid_spacing_metres",
        "translation_radius_metres",
        "yaw_radius_radians",
        "smoothing_metres",
        "maximum_iterations",
    }
    if set(options) != expected:
        raise ValueError("Unexpected comparison configuration keys.")
    runtime = ScenePipelineConfiguration.from_config(
        OmegaConf.masked_copy(config, [key for key in config if key != "comparison"])
    )
    output = runtime.resolver.resolve(PathRole.OUTPUT, str(options.output))
    compare_alignment(
        runtime,
        checkpoint=runtime.resolver.resolve(PathRole.OUTPUT, str(options.checkpoint)),
        output=output,
        grid_spacing_metres=float(options.grid_spacing_metres),
        translation_radius_metres=float(options.translation_radius_metres),
        yaw_radius_radians=float(options.yaw_radius_radians),
        smoothing_metres=float(options.smoothing_metres),
        maximum_iterations=int(options.maximum_iterations),
    )
    (output / "resolved-config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True)
    )


if __name__ == "__main__":
    main()
