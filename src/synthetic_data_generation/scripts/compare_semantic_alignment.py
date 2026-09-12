"""Generate paired placement/projection diagnostics after model training finishes."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.comparison_configuration import (
    ComparisonRuntime,
)
from src.synthetic_data_generation.alignment.semantic_comparison import (
    compare_alignment,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator


def validate_comparison_boundary(config: DictConfig) -> None:
    """Validate comparison and scene authority before loading a model."""
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
    ComparisonRuntime.from_config(
        OmegaConf.masked_copy(config, [key for key in config if key != "comparison"])
    )


register_boundary_validator(
    "synthetic.semantic_comparison", validate_comparison_boundary
)


@hydra_main(
    config_path="../configs",
    config_name="compare_semantic_alignment",
    version_base="1.3",
    validation_boundary="synthetic.semantic_comparison",
)
def main(config: DictConfig) -> None:
    """Load explicit source/output authority and execute one paired comparison."""
    options = config.comparison
    runtime = ComparisonRuntime.from_config(
        OmegaConf.masked_copy(config, [key for key in config if key != "comparison"])
    )
    output = runtime.resolver.resolve(PathRole.OUTPUT, str(options.output))
    compare_alignment(
        runtime,
        checkpoint=runtime.resolver.resolve(
            PathRole.CHECKPOINT, str(options.checkpoint)
        ),
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
