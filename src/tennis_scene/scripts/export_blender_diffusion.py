"""Export schematic completion assets from an adopted measured rays delivery."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.visualization.blender.export.diffusion_configuration import (
    validate_diffusion_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "tennis_scene.export_blender_diffusion"
register_boundary_validator(_BOUNDARY, validate_diffusion_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="export_blender_diffusion",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    from src.tennis_scene.visualization.blender.export.diffusion import (
        export_diffusion_bundle,
    )
    from src.tennis_scene.visualization.blender.export.diffusion_configuration import (
        parse_diffusion_config,
    )

    runtime = parse_diffusion_config(cfg)
    target = export_diffusion_bundle(
        runtime.input_path,
        runtime.output_path / "bundle",
        font=runtime.font_path,
        config=runtime.config,
        resolved_config=runtime.resolved_config,
    )
    logging.getLogger(__name__).info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    main()
