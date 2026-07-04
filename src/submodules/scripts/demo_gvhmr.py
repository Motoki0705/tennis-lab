"""GVHMR demo: reconstruct SMPL meshes from a video and render them from the camera view.

Runs the full submodule chain (YOLO tracking -> ViTPose 2D pose -> HMR2 features
-> GVHMR SMPL-X regression -> SMPL vertex reconstruction) in the main ``.venv``
and writes an overlay video with the meshes rendered from the camera viewpoint.

Usage:
    python -m src.submodules.scripts.demo_gvhmr
    python -m src.submodules.scripts.demo_gvhmr video_path=data/samples/tennis_clip.mp4 num_tracks=2
    python -m src.submodules.scripts.demo_gvhmr max_frames=60 device=cpu

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
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)

_PLAYER_COLORS: list[tuple[float, float, float]] = [
    (0.91, 0.44, 0.32),  # #E76F51
    (0.16, 0.62, 0.56),  # #2A9D8F
    (0.91, 0.77, 0.42),  # #E9C46A
    (0.37, 0.51, 0.67),  # #5E81AC
]


@hydra_main(  # type: ignore[untyped-decorator]
    version_base="1.3", config_path="../configs", config_name="demo_gvhmr"
)
def main(cfg: DictConfig) -> int:
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

    video_path = Path(to_absolute_path(str(cfg.video_path)))
    if not video_path.exists():
        LOGGER.error(f"Video not found: {video_path}")
        return 1

    smpl_faces_path = Path(to_absolute_path(str(cfg.smpl_faces_path)))
    faces = load_smpl_faces(smpl_faces_path)

    output_name = cfg.output_name or f"{video_path.stem}_incam.mp4"
    output_path = Path(to_absolute_path(str(cfg.output_dir))) / str(output_name)

    info = probe_video_info(video_path)
    max_frames: int | None = cfg.max_frames
    LOGGER.info(
        f"Input: {video_path} ({info.width}x{info.height}, {info.frame_count} frames)"
    )

    # 0. Fail fast: GVHMR needs the licensed SMPL-X body model; load it before
    #    spending minutes on tracking/preprocessing.
    mesh_model = GvhmrMeshRecovery(
        checkpoint=to_absolute_path(str(cfg.checkpoints.gvhmr)), device=str(cfg.device)
    )
    mesh_model.load()

    # 1. Person tracking
    tracker = YoloPersonTracker(
        checkpoint=to_absolute_path(str(cfg.checkpoints.yolo)), device=str(cfg.device)
    )
    track_result = tracker.predict(
        TrackRequest(
            video_path=video_path,
            num_tracks=int(cfg.num_tracks),
            interactive=bool(cfg.interactive_tracks),
        )
    )
    tracker.unload()
    LOGGER.info(f"Tracked persons: {track_result.track_ids}")

    # 2-4. Per-track: 2D pose, image features, GVHMR
    pose_model = ViTPosePose2D(
        checkpoint=to_absolute_path(str(cfg.checkpoints.vitpose)), device=str(cfg.device)
    )
    feature_model = Hmr2FeatureExtractor(
        checkpoint=to_absolute_path(str(cfg.checkpoints.hmr2)), device=str(cfg.device)
    )
    reconstructor = SmplVertexReconstructor(device=str(cfg.device))

    vertices_per_track: dict[int, torch.Tensor] = {}
    for track_id in track_result.track_ids:
        bbx_xys = track_result.bbx_xys(track_id)
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
                static_cam=bool(cfg.static_cam),
            )
        )
        vertices_per_track[track_id] = reconstructor.reconstruct(gvhmr.smpl_params_incam)

    pose_model.unload()
    feature_model.unload()
    mesh_model.unload()

    # 5. Render camera-view overlay
    from src.submodules.vendor.gvhmr.utils.hmr_cam import estimate_K

    K_render = estimate_K(info.width, info.height).numpy()
    num_render_frames = min(v.shape[0] for v in vertices_per_track.values())
    renderer = MeshRenderer(faces, MeshStyle(alpha=float(cfg.mesh_alpha)))
    LOGGER.info(f"Rendering {num_render_frames} frames -> {output_path}")
    with VideoWriter(output_path, fps=info.fps, crf=int(cfg.video_crf)) as writer:
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


if __name__ == "__main__":
    raise SystemExit(main())
