"""Export one identity's placement trace or rank placement windows.

Usage:
    python -m src.tennis_scene.scripts.export_blender_placement \
        mode=export store=<component store directory> identity=0 frames=2:120 t0=60 \
        plate_camera_id=cam0 plate_video_path=<clip>/cam0.mp4 \
        human_vis_threshold=0.15

    python -m src.tennis_scene.scripts.export_blender_placement \
        mode=rank store=<component store directory> \
        plate_camera_id=cam0 plate_video_path=<clip>/cam0.mp4 \
        human_vis_threshold=0.15

Notes:
    - Configuration is loaded from
      `src/tennis_scene/configs/blender_placement.yaml`.
    - The exporter only reads the adopted store; it never runs pose or GVHMR
      inference. SMPL geometry is reconstructed deterministically on the CPU.
    - `mode=rank` writes only `ranking.json`; it never extracts a plate or
      writes a bundle.
    - Build and render the Blender scene with
      `src/tennis_scene/visualization/blender/shots/player_placement.py`.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import (
    blender_placement_mode,
    validate_blender_placement_boundary,
)
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.blender_placement"
register_boundary_validator(_BOUNDARY, validate_blender_placement_boundary)


def _resolved_config(cfg: DictConfig) -> dict[str, Any] | None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else None


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="blender_placement",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Run the mode selected by the composed placement configuration."""
    mode = blender_placement_mode(cfg)
    if mode == "rank":
        from src.tennis_scene.configuration import (
            parse_blender_placement_rank_config,
        )
        from src.tennis_scene.visualization.blender.export.placement_ranking import (
            rank_blender_placement,
        )

        rank_runtime = parse_blender_placement_rank_config(cfg)
        ranking = rank_blender_placement(
            store_dir=rank_runtime.store_path,
            output_path=rank_runtime.rank_output_path,
            config=rank_runtime.rank,
            plate_camera_id=rank_runtime.plate_camera_id,
            plate_video_path=rank_runtime.plate_video_path,
            human_vis_threshold=rank_runtime.human_vis_threshold,
            resolved_config=_resolved_config(cfg),
        )
        LOGGER.info(f"Wrote Blender placement ranking {ranking}")
        return 0

    from src.tennis_scene.configuration import parse_blender_placement_config
    from src.tennis_scene.visualization.blender.export.placement import (
        export_blender_placement_bundle,
    )

    export_runtime = parse_blender_placement_config(cfg)
    manifest = export_blender_placement_bundle(
        store_dir=export_runtime.store_path,
        output_dir=export_runtime.output_path,
        config=export_runtime.export,
        smpl_faces_path=export_runtime.smpl_faces_path,
        resolved_config=_resolved_config(cfg),
    )
    LOGGER.info(f"Wrote Blender placement bundle manifest {manifest}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
