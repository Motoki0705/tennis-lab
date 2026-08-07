"""GVHMR demo: reconstruct SMPL meshes from a video and render them from the camera view.

Runs the full submodule chain (YOLO tracking -> ViTPose 2D pose -> HMR2 features
-> GVHMR SMPL-X regression -> SMPL vertex reconstruction) in the main ``.venv``
and writes an overlay video with the meshes rendered from the camera viewpoint.

Usage:
    python -m src.submodules.scripts.demo_gvhmr
    python -m src.submodules.scripts.demo_gvhmr video=samples/tennis_clip.mp4 num_tracks=2
    python -m src.submodules.scripts.demo_gvhmr max_frames=60 runtime.device=cpu

Notes:
    - Configuration is loaded from `src/submodules/configs/demo_gvhmr.yaml`.
    - Model checkpoints are read from `ckpt/` (symlinks to
      third_party/GVHMR/inputs/checkpoints).
    - GVHMR inference requires the licensed SMPL-X body model
      (`ckpt/body_models/smplx/SMPLX_NEUTRAL.npz`); download it from
      https://smpl-x.is.tue.mpg.de/.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import GvhmrDemoConfig
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "submodules.demo_gvhmr"

_PLAYER_COLORS: list[tuple[float, float, float]] = [
    (0.91, 0.44, 0.32),  # #E76F51
    (0.16, 0.62, 0.56),  # #2A9D8F
    (0.91, 0.77, 0.42),  # #E9C46A
    (0.37, 0.51, 0.67),  # #5E81AC
]


def _runtime_config(cfg: DictConfig) -> GvhmrDemoConfig:
    """Build the sole typed runtime contract from a composed Hydra job."""
    raw_config: object = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_config, Mapping):
        raise TypeError("GVHMR demo configuration must compose to a mapping.")
    return GvhmrDemoConfig.from_mapping(
        cast("Mapping[str, object]", raw_config),
        repository_root=PROJECT_ROOT,
    )


def _validate_boundary(cfg: DictConfig) -> None:
    """Validate the complete composed job before the demo performs any I/O."""
    _runtime_config(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


def _run(cfg: DictConfig) -> int:
    """Run the GVHMR demo end-to-end and write the camera-view overlay video."""
    import numpy as np
    import torch

    from src.submodules.models import (
        GvhmrMeshRecovery,
        GvhmrRequest,
        Hmr2FeatureExtractor,
        ImageFeatureRequest,
        Pose2DRequest,
        SmplVertexReconstructor,
        TrackRequest,
        ViTPosePose2D,
        YoloPersonTracker,
    )
    from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
    from src.utils.rendering.mesh_renderer import MeshRenderer, MeshStyle
    from src.utils.video.reader import OpenCVVideoFrameReader, probe_video_info
    from src.utils.video.writer import VideoWriter

    config = _runtime_config(cfg)

    video_path = config.video_path
    if not video_path.exists():
        LOGGER.error(f"Video not found: {video_path}")
        return 1

    faces = load_smpl_faces(config.assets.smpl_faces)
    output_path = config.output_path

    info = probe_video_info(video_path)
    max_frames = config.max_frames
    LOGGER.info(
        f"Input: {video_path} ({info.width}x{info.height}, {info.frame_count} frames)"
    )

    # 0. Fail fast: GVHMR needs the licensed SMPL-X body model; load it before
    #    spending minutes on tracking/preprocessing.
    mesh_model = GvhmrMeshRecovery(
        checkpoint=config.assets.gvhmr_checkpoint,
        body_models_dir=config.assets.body_models_dir,
        device=config.runtime.device,
        bundled_assets=config.assets.bundled,
    )
    mesh_model.load()

    # 1. Person tracking
    tracker = YoloPersonTracker(
        checkpoint=config.assets.yolo_checkpoint,
        device=config.runtime.device,
        confidence=config.runtime.tracking.yolo_confidence,
    )
    track_result = tracker.predict(
        TrackRequest(
            video_path=video_path,
            num_tracks=config.num_tracks,
            interactive=config.interactive_tracks,
        )
    )
    tracker.unload()
    LOGGER.info(f"Tracked persons: {track_result.track_ids}")

    # 2-4. Per-track: 2D pose, image features, GVHMR
    pose_model = ViTPosePose2D(
        checkpoint=config.assets.vitpose_checkpoint,
        device=config.runtime.device,
        flip_test=config.runtime.vitpose.flip_test,
        batch_size=config.runtime.vitpose.batch_size,
        head_config=config.runtime.vitpose.head,
    )
    feature_model = Hmr2FeatureExtractor(
        checkpoint=config.assets.hmr2_checkpoint,
        device=config.runtime.device,
        batch_size=config.runtime.hmr2.batch_size,
        mean_params_path=config.assets.bundled.hmr2_mean_params,
    )
    reconstructor = SmplVertexReconstructor(
        body_models_dir=config.assets.body_models_dir,
        device=config.runtime.device,
        bundled_assets=config.assets.bundled,
    )

    vertices_per_track: dict[int, torch.Tensor] = {}
    for track_id in track_result.track_ids:
        bbx_xys = track_result.bbx_xys(
            track_id,
            base_enlarge=config.runtime.tracking.bbox_enlarge,
        )
        if max_frames is not None:
            bbx_xys = bbx_xys[:max_frames]
        LOGGER.info(f"Track {track_id}: {len(bbx_xys)} frames")

        pose = pose_model.predict(Pose2DRequest(video_path=video_path, bbx_xys=bbx_xys))
        features = feature_model.predict(
            ImageFeatureRequest(video_path=video_path, bbx_xys=bbx_xys)
        )
        gvhmr = mesh_model.predict(
            GvhmrRequest(
                kp2d=pose.keypoints,
                bbx_xys=bbx_xys,
                f_imgseq=features.features,
                width=info.width,
                height=info.height,
                static_cam=config.runtime.static_cam,
            )
        )
        vertices_per_track[track_id] = reconstructor.reconstruct(
            gvhmr.smpl_params_incam
        )

    pose_model.unload()
    feature_model.unload()
    mesh_model.unload()

    # 5. Render camera-view overlay
    from src.submodules.vendor.gvhmr.utils.hmr_cam import estimate_K

    K_render = estimate_K(info.width, info.height).numpy()
    num_render_frames = min(v.shape[0] for v in vertices_per_track.values())
    renderer = MeshRenderer(faces, MeshStyle(alpha=config.mesh_alpha))
    LOGGER.info(f"Rendering {num_render_frames} frames -> {output_path}")
    with VideoWriter(output_path, fps=info.fps, crf=config.video_crf) as writer:
        reader = OpenCVVideoFrameReader(video_path, max_frames=num_render_frames)
        for packet in reader:
            frame = np.ascontiguousarray(packet.frame[..., ::-1])  # BGR -> RGB
            for idx, track_id in enumerate(track_result.track_ids):
                verts = vertices_per_track[track_id][packet.index].numpy()
                color = _PLAYER_COLORS[idx % len(_PLAYER_COLORS)]
                frame = renderer.render_overlay(frame, verts, K_render, color=color)
            writer.write_frame(frame)

    LOGGER.info(f"Done: {output_path}")
    return 0


main = hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="demo_gvhmr",
    validation_boundary=_BOUNDARY,
)(_run)


if __name__ == "__main__":
    raise SystemExit(main())
