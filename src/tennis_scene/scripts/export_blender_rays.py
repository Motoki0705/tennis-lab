"""Rank or export saved multi-view ball rays for a Blender explanation shot."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.visualization.blender.export.rays_configuration import (
    validate_rays_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "tennis_scene.export_blender_rays"
register_boundary_validator(_BOUNDARY, validate_rays_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="export_blender_rays",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Consume adopted artifacts only; never run pipeline components."""
    from src.tennis_scene.visualization.blender.export.rays import (
        export_ranking,
        export_rays_bundle,
        load_rays_input,
    )
    from src.tennis_scene.visualization.blender.export.rays_configuration import (
        parse_rays_config,
    )

    runtime = parse_rays_config(cfg)
    inputs = load_rays_input(runtime.input_path)
    if runtime.mode == "rank":
        target = export_ranking(
            inputs,
            runtime.output_path / "ranking.json",
            config=runtime.config,
            resolved_config=runtime.resolved_config,
        )
    else:
        assert (
            runtime.start is not None
            and runtime.stop is not None
            and runtime.t0 is not None
        )
        target = export_rays_bundle(
            inputs,
            runtime.output_path / "bundle",
            start=runtime.start,
            stop=runtime.stop,
            t0=runtime.t0,
            config=runtime.config,
            resolved_config=runtime.resolved_config,
        )
    logging.getLogger(__name__).info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    main()
