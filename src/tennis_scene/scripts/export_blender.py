"""Export one camera's view of a saved scene as a Blender input bundle.

Usage:
    python -m src.tennis_scene.scripts.export_blender \
        input=<scene.npz fragment under ARTIFACT> camera_id=cam1

Notes:
    - Configuration is loaded from `src/tennis_scene/configs/blender_export.yaml`.
    - The scene archive must carry `player_smpl_valid` / `ball_3d_valid` masks.
    - Build and render the Blender scene with `src/tennis_scene/visualization/blender/shots/reveal.py`.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_blender_export_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.blender_export"
register_boundary_validator(_BOUNDARY, validate_blender_export_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="blender_export",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Export the configured scene/camera as a Blender bundle."""
    from src.tennis_scene.configuration import parse_blender_export_config
    from src.tennis_scene.visualization.blender.export import export_blender_bundle

    runtime = parse_blender_export_config(cfg)
    manifest = export_blender_bundle(
        scene_path=runtime.input_path,
        smpl_faces_path=runtime.smpl_faces_path,
        smpl_joint_regressor_path=runtime.smpl_joint_regressor_path,
        output_dir=runtime.output_path,
        config=runtime.bundle,
    )
    LOGGER.info(f"Wrote Blender bundle manifest {manifest}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
