"""
Run SLCS clip inference on one clip camera and render qualitative outputs.

Usage:
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=rec-a/clip_000
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=rec-a/clip_000 predict.camera_id=cam0
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=... predict.render_3d=false

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/predict_clip.yaml`;
      dataset location comes from the shared `data` group.
    - Checkpoint and output paths are relative to `paths.checkpoint_root` and
      `paths.output_root`, respectively.
    - Writes `predictions.npz` (full-timeline positions/yaw/uncertainty in
      meters and normalized units), an optional 3D+top-down comparison video
      and an optional 2D overlay video (observations + homography-based
      ground projections; requires the clip media).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSPredictConfig
from src.tasks.slcs.data.contract import CLIPS_DIR_NAME, ClipManifest, split_clip_id
from src.tasks.slcs.data.dataset import load_clip_arrays
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.visualization.overlay_2d import render_overlay_video
from src.tasks.slcs.visualization.renderer_3d import (
    SceneRenderInputs,
    SLCSSceneRenderer,
)
from src.utils.hydra import hydra_main
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

SavezCompressed = Callable[..., None]


def run(config: DictConfig) -> None:
    """Predict one clip camera and write artifacts."""
    runtime = SLCSPredictConfig.from_config(config)
    recording_id, clip_name = split_clip_id(runtime.clip_id)
    clip_dir = runtime.data.dataset_root / CLIPS_DIR_NAME / recording_id / clip_name

    data_config = runtime.data.pipeline
    manifest = ClipManifest.load(clip_dir)
    camera_id = runtime.camera_id
    manifest.camera_index(camera_id)

    predictor = SLCSPredictor.load_from_checkpoint(
        runtime.checkpoint,
        resolver=runtime.resolver,
        device=runtime.device,
        strict=runtime.checkpoint_strict,
        weights_only=runtime.checkpoint_weights_only,
    )
    result = predictor.predict_clip(
        clip_dir,
        camera_id,
        data_config=data_config,
        stride=data_config.eval_stride,
        batch_size=runtime.batch_size,
        denormalize=True,
    )

    output_dir = runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "predictions.npz"
    prediction_arrays = {key: value.numpy() for key, value in result.items()}
    prediction_arrays["clip_id"] = np.asarray(manifest.clip_id)
    prediction_arrays["camera_id"] = np.asarray(camera_id)
    savez_compressed = cast(SavezCompressed, np.savez_compressed)
    savez_compressed(Path(npz_path), **prediction_arrays)
    print(f"predictions -> {npz_path}")

    clip = load_clip_arrays(manifest, config=data_config)
    scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
    if runtime.render_3d:
        renderer = SLCSSceneRenderer(
            figsize=(
                runtime.visualization.figure_width,
                runtime.visualization.figure_height,
            ),
            dpi=runtime.visualization.dpi,
        )
        gt_yaw = np.arctan2(clip.player_rotation[..., 1], clip.player_rotation[..., 0])
        video_path = renderer.render_video(
            SceneRenderInputs(
                player_position_m=result["player_position_meters"].numpy(),
                player_yaw_rad=result["player_yaw_radians"].numpy(),
                ball_position_m=result["ball_position_meters"].numpy(),
                gt_player_position_m=clip.player_position_norm * scale,
                gt_player_yaw_rad=gt_yaw.astype(np.float32),
                gt_ball_position_m=clip.ball_position_norm * scale,
                gt_player_valid=clip.player_label_valid,
                gt_ball_valid=clip.ball_label_valid,
            ),
            output_dir / "scene_3d.mp4",
            fps=clip.fps,
            frame_step=runtime.frame_step,
        )
        print(f"3D render  -> {video_path}")
    if runtime.render_overlay:
        overlay_path, missing = render_overlay_video(
            clip,
            manifest.camera_index(camera_id),
            player_position_m=result["player_position_meters"].numpy(),
            player_yaw_rad=result["player_yaw_radians"].numpy(),
            ball_position_m=result["ball_position_meters"].numpy(),
            output_path=output_dir / "overlay_2d.mp4",
            court_kp_indices=runtime.visualization.court_kp_indices,
            min_homography_points=runtime.visualization.homography_min_points,
            court_visibility_threshold=(
                runtime.visualization.court_visibility_threshold
            ),
        )
        print(f"2D overlay -> {overlay_path} (frames without homography: {missing})")


@hydra_main(
    config_path="../configs",
    config_name="predict_clip",
    version_base="1.3",
    validation_boundary="slcs.predict_clip",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for SLCS clip inference."""
    with torch.no_grad():
        run(config)


if __name__ == "__main__":
    main()
