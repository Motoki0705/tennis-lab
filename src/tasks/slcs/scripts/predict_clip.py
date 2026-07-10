"""
Run SLCS clip inference on one clip camera and render qualitative outputs.

Usage:
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=rec-a/clip_000
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=rec-a/clip_000 predict.camera_id=cam0
    python -m src.tasks.slcs.scripts.predict_clip predict.checkpoint=... predict.clip_id=... predict.render_3d=false

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/predict_clip.yaml`;
      dataset location comes from the shared `data` group.
    - Writes `predictions.npz` (full-timeline positions/yaw/uncertainty in
      meters and normalized units), an optional 3D+top-down comparison video
      and an optional 2D overlay video (observations + homography-based
      ground projections; requires the clip media).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from src.tasks.slcs.data.contract import CLIPS_DIR_NAME, ClipManifest, split_clip_id
from src.tasks.slcs.data.dataset import SLCSDataConfig, load_clip_arrays
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.visualization.overlay_2d import render_overlay_video
from src.tasks.slcs.visualization.renderer_3d import (
    SceneRenderInputs,
    SLCSSceneRenderer,
)
from src.utils.hydra import hydra_main
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def run(config: DictConfig) -> None:
    """Predict one clip camera and write artifacts."""
    predict_cfg = config.predict
    checkpoint = predict_cfg.get("checkpoint")
    clip_id = predict_cfg.get("clip_id")
    if not checkpoint or not clip_id:
        raise ValueError("predict.checkpoint and predict.clip_id are required.")
    recording_id, clip_name = split_clip_id(str(clip_id))
    clip_dir = Path(str(config.data.dataset_root)) / CLIPS_DIR_NAME / recording_id / clip_name

    data_config = SLCSDataConfig.from_config(config.data)
    manifest = ClipManifest.load(clip_dir)
    camera_id_cfg = predict_cfg.get("camera_id")
    camera_id = (
        str(camera_id_cfg) if camera_id_cfg is not None else manifest.camera_ids[0]
    )

    predictor = SLCSPredictor.load_from_checkpoint(
        str(checkpoint), device=str(predict_cfg.get("device", "cpu"))
    )
    result = predictor.predict_clip(
        clip_dir,
        camera_id,
        data_config=data_config,
        batch_size=int(predict_cfg.get("batch_size", 4)),
    )

    output_dir = Path(str(predict_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        **{key: value.numpy() for key, value in result.items()},
        clip_id=np.asarray(manifest.clip_id),
        camera_id=np.asarray(camera_id),
    )
    print(f"predictions -> {npz_path}")

    clip = load_clip_arrays(manifest, config=data_config)
    scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
    if bool(predict_cfg.get("render_3d", True)):
        renderer = SLCSSceneRenderer()
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
            frame_step=int(predict_cfg.get("frame_step", 1)),
        )
        print(f"3D render  -> {video_path}")
    if bool(predict_cfg.get("render_overlay", True)):
        overlay_path, missing = render_overlay_video(
            clip,
            manifest.camera_index(camera_id),
            player_position_m=result["player_position_meters"].numpy(),
            player_yaw_rad=result["player_yaw_radians"].numpy(),
            ball_position_m=result["ball_position_meters"].numpy(),
            output_path=output_dir / "overlay_2d.mp4",
        )
        print(f"2D overlay -> {overlay_path} (frames without homography: {missing})")


@hydra_main(config_path="../configs", config_name="predict_clip", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for SLCS clip inference."""
    with torch.no_grad():
        run(config)


if __name__ == "__main__":
    main()
