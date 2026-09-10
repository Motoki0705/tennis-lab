"""
Launch the local browser clip studio with one explicit project-file contract.

Usage:
    python -m src.tennis_scene.scripts.clip_studio project_path=tennis_scene/project.json

Notes:
    - All configured paths are role-relative and resolved by RuntimePathRoots.
    - Existing projects must omit video_paths and camera_ids; new projects require both.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_clip_studio_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.clip_studio"
register_boundary_validator(_BOUNDARY, validate_clip_studio_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="clip_studio",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Validate the full GUI boundary, then load or create one project."""
    import uvicorn

    from src.tennis_scene.clip_studio.initialization import load_or_create_project
    from src.tennis_scene.clip_studio.web.app import create_app
    from src.tennis_scene.configuration import parse_clip_studio_config

    runtime = parse_clip_studio_config(cfg)
    project, notice = load_or_create_project(runtime)
    app = create_app(runtime, project, startup_notice=notice)
    LOGGER.info("Open http://127.0.0.1:%s in your browser", runtime.gui.port)
    uvicorn.run(app, host="127.0.0.1", port=runtime.gui.port)
    return 0


if __name__ == "__main__":
    main()
