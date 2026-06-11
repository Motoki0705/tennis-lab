"""Select candidate clips or run pseudo-label prediction for one YouTube video.

Usage:
    python -m src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset workflow.video_id=video_000001 workflow.mode=select
    python -m src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset workflow.video_id=video_000001 workflow.mode=predict

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/clip_and_predict_youtube_dataset.yaml`.
    - Exactly one `workflow.video_id` is processed per invocation.
    - Selection resumes after the last candidate endpoint; prediction starts only in explicit `mode=predict`.
    - Prediction stores every local heatmap peak above the configured threshold after NMS.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.ball_detection.generate_dataset import (
    CandidatePredictionConfig,
    CandidateSelectionConfig,
    predict_candidates,
    run_candidate_selection,
)

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(
    config_path="../../configs",
    config_name="clip_and_predict_youtube_dataset",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    workflow = cfg.workflow
    if workflow.video_id is None or not str(workflow.video_id).strip():
        raise ValueError(
            "workflow.video_id is required. Example: workflow.video_id=video_000001"
        )
    root = Path(to_absolute_path(str(workflow.root))).resolve()
    video_id = str(workflow.video_id)
    mode = str(workflow.mode)
    paths = workflow.paths
    raw_dir = root / str(paths.frames_dir) / video_id / str(paths.raw_dir)
    staging_dir = root / str(paths.staging_dir) / video_id

    if mode == "select":
        select = workflow.select
        return run_candidate_selection(
            root=root,
            video_id=video_id,
            raw_dir=raw_dir,
            staging_dir=staging_dir,
            config=CandidateSelectionConfig(
                resume=bool(select.resume),
                start_index=None if select.start_index is None else int(select.start_index),
                window_name=str(select.window_name),
                max_display_width=int(select.max_display_width),
                max_display_height=int(select.max_display_height),
                min_frames=int(select.min_frames),
                copy_mode=str(select.copy_mode),
                overwrite=bool(select.overwrite),
                skip_small=int(select.skip_small),
                skip_medium=int(select.skip_medium),
                skip_large=int(select.skip_large),
            ),
        )
    if mode == "predict":
        prediction = workflow.prediction
        return predict_candidates(
            root=root,
            video_id=video_id,
            staging_dir=staging_dir,
            config=CandidatePredictionConfig(
                checkpoint=Path(to_absolute_path(str(prediction.checkpoint))).resolve(),
                device=str(prediction.device),
                sequence_length=int(prediction.sequence_length),
                window_stride=int(prediction.window_stride),
                batch_size=int(prediction.batch_size),
                image_size=tuple(int(value) for value in prediction.image_size),
                normalize_imagenet=bool(prediction.normalize_imagenet),
                imagenet_mean=tuple(float(value) for value in prediction.imagenet_mean),
                imagenet_std=tuple(float(value) for value in prediction.imagenet_std),
                peak_threshold=float(prediction.peak_threshold),
                nms_kernel=int(prediction.nms_kernel),
                max_candidates_per_frame=int(prediction.max_candidates_per_frame),
                aggregation=str(prediction.aggregation),
                overwrite=bool(prediction.overwrite),
            ),
        )
    raise ValueError(f"Unsupported workflow.mode={mode!r}; expected select or predict.")


if __name__ == "__main__":
    raise SystemExit(main())
