"""Review ball pseudo-labels and export human-approved clips.

Usage:
    python -m src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball
    python -m src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball annotate.video_id=video_000002

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/annotate_youtube_ball.yaml`.
    - Shift-click adds a ball; Tab selects another ball; Delete removes the selected ball.
    - Press `c` to approve the current frame and `z` to zoom around the selected ball.
    - Press `f` to finalize a candidate after every frame is completed.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.ball_detection import configuration as _configuration  # noqa: F401
from src.tasks.ball_detection.configuration import BallRuntimePaths
from src.tasks.ball_detection.generate_dataset import (
    BallAnnotationSessionConfig,
    FinalizeConfig,
    ZoomConfig,
    run_annotation_session,
)
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../../configs",
    config_name="annotate_youtube_ball",
    version_base="1.3",
    validation_boundary="ball.annotation",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - interactive CLI
    """Hydra entry point."""
    ann = cfg.annotate
    zoom = ann.zoom
    finalize = ann.finalize
    status: int = run_annotation_session(
        BallAnnotationSessionConfig(
            root=BallRuntimePaths.from_config(cfg).data(str(ann.root)),
            video_id=str(ann.video_id),
            candidate_id=None if ann.candidate_id is None else str(ann.candidate_id),
            start_index=None if ann.start_index is None else int(ann.start_index),
            window_name=str(ann.window_name),
            max_display_width=int(ann.max_display_width),
            max_display_height=int(ann.max_display_height),
            point_radius=int(ann.point_radius),
            point_thickness=int(ann.point_thickness),
            max_balls_per_frame=int(ann.max_balls_per_frame),
            zoom=ZoomConfig(
                key=str(zoom.key),
                factor=float(zoom.factor),
            ),
            finalize=FinalizeConfig(
                key=str(finalize.key),
                overwrite=bool(finalize.overwrite),
            ),
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
