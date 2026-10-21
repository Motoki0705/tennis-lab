"""Export one identity's placement trace as a Blender input bundle.

Usage:
    python -m src.tennis_scene.scripts.export_blender_placement \
        store=<component store directory> identity=0 frames=2:120 t0=60 \
        plate_camera_id=cam0 plate_video_path=<clip>/cam0.mp4 \
        human_vis_threshold=0.15

Notes:
    - Configuration is loaded from
      `src/tennis_scene/configs/blender_placement.yaml`.
    - The exporter only reads the adopted store; it never runs pose or GVHMR
      inference. SMPL geometry is reconstructed deterministically on the CPU.
    - Build and render the Blender scene with
      `src/tennis_scene/visualization/blender/shots/player_placement.py`.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import validate_blender_placement_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.blender_placement"
register_boundary_validator(_BOUNDARY, validate_blender_placement_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="blender_placement",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Export the configured identity placement as a Blender bundle."""
    from src.tennis_scene.configuration import parse_blender_placement_config
    from src.tennis_scene.visualization.blender.export.placement import (
        export_blender_placement_bundle,
    )

    runtime = parse_blender_placement_config(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    manifest = export_blender_placement_bundle(
        store_dir=runtime.store_path,
        output_dir=runtime.output_path,
        config=runtime.export,
        smpl_faces_path=runtime.smpl_faces_path,
        resolved_config=cast(dict[str, Any], resolved)
        if isinstance(resolved, dict)
        else None,
    )
    LOGGER.info(f"Wrote Blender placement bundle manifest {manifest}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
