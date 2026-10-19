"""Export one camera's view of a reconstructed scene as a Blender input bundle.

The bundle is plain files that Blender's bundled Python (with numpy) can read
without importing this repository::

    <output_dir>/
    ├── manifest.json          # schema, camera pose, frame timing, provenance
    ├── smpl_faces.npy         # (F, 3) int32 SMPL triangle indices
    ├── player_<p>.pc2         # (T, V, 3) court-frame vertices per player
    ├── player_valid.npy       # (P, T) bool after explicit short-gap filling
    ├── ball_positions.npy     # (T, 3) float32 court frame, NaN when invalid
    ├── ball_valid.npy         # (T,) bool after explicit short-gap filling
    └── plate/%05d.jpg         # decoded source frames of the chosen camera

Blender world coordinates are court coordinates (meters, XY ground, +Z up).
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.submodules.vendor.gvhmr.body_model import load_smpl_faces
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.visualization.blender.export.camera import (
    PinholeCamera,
    to_blender_camera,
)
from src.tennis_scene.visualization.blender.export.gaps import fill_short_gaps
from src.tennis_scene.visualization.blender.export.pointcache import write_pc2
from src.tennis_scene.visualization.rendering.smpl_court import (
    load_smpl_joint_regressor,
    smpl_vertices_to_court,
)
from src.utils.schema.court import (
    COURT_SKELETON,
    NET_HEIGHT_POST,
    STANDARD_COURT_CONFIG,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
    net_height_at_x,
)
from src.utils.video.encoding import encode_jpeg
from src.utils.video.reader import RandomAccessVideoReader, probe_video_info

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.schema import SceneResult

BUNDLE_SCHEMA = "tennis_scene_blender_bundle_v1"
PLATE_PATTERN = "plate/{frame:05d}.jpg"


@dataclass(frozen=True, slots=True)
class BlenderBundleConfig:
    """Explicit export choices; every presentation-side synthesis is bounded here."""

    camera_id: str
    player_max_gap_frames: int
    ball_max_gap_frames: int
    max_ball_reprojection_median_px: float
    sensor_width_mm: float
    plate_jpeg_quality: int

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty.")
        if self.player_max_gap_frames < 0 or self.ball_max_gap_frames < 0:
            raise ValueError("Gap limits must be >= 0.")
        for name in ("max_ball_reprojection_median_px", "sensor_width_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 1 <= self.plate_jpeg_quality <= 100:
            raise ValueError("plate_jpeg_quality must be in [1, 100].")


def ball_reprojection_errors_px(
    scene: SceneResult,
    ball_valid: NDArray[np.bool_],
    camera: PinholeCamera,
    camera_index: int,
) -> NDArray[np.float64]:
    """Pixel distance between projected 3D balls and the camera's 2D detections."""
    if scene.ball_3d is None or scene.ball_uv is None or scene.ball_vis is None:
        raise ValueError("Scene lacks ball_3d/ball_uv/ball_vis for camera validation.")
    usable = ball_valid & scene.ball_vis[camera_index]
    if not usable.any():
        raise ValueError("No frame has both a valid 3D ball and a 2D ball detection.")
    projected = camera.project(scene.ball_3d[usable])
    detected = scene.ball_uv[camera_index][usable] * [scene.width, scene.height]
    return np.asarray(np.linalg.norm(projected - detected, axis=-1), dtype=np.float64)


def court_geometry() -> dict[str, Any]:
    """Standard-court geometry in the scene's court frame (the camera fit's)."""
    keypoints = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy().astype(np.float64)
    ground = keypoints[:, 2] == 0.0
    post_bases = {15, 17}
    ground_lines = [
        [keypoints[a].tolist(), keypoints[b].tolist()]
        for a, b in COURT_SKELETON
        if ground[a] and ground[b] and not {a, b} & post_bases
    ]
    post_x = float(keypoints[17, 0])
    return {
        "ground_lines": ground_lines,
        "runoff_xy": [X_MIN, X_MAX, Y_MIN, Y_MAX],
        "net": {
            "post_x": post_x,
            "height_post": NET_HEIGHT_POST,
            "top": [
                [float(x), net_height_at_x(float(x))]
                for x in np.linspace(-post_x, post_x, 65)
            ],
        },
    }


def _write_plate(
    video_path: Path,
    output_dir: Path,
    *,
    frame_range: range,
    max_width: int | None,
    quality: int,
) -> None:
    """Write a contiguous source interval into ``output_dir/%05d.jpg``.

    Filenames retain source frame indices. ``max_width=None`` preserves the
    original resolution; an integer caps the width without upscaling. Callers
    own validation against scene/camera metadata and partial-bundle cleanup.
    """
    info = probe_video_info(video_path)
    if (
        frame_range.step != 1
        or not 0 <= frame_range.start < frame_range.stop <= info.frame_count
    ):
        raise ValueError("frame_range must be a non-empty contiguous video interval.")
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive or None.")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100].")
    output_dir.mkdir(parents=True)
    with RandomAccessVideoReader(video_path) as reader:
        for frame_index in frame_range:
            frame = reader.read(frame_index)
            if max_width is not None and frame.shape[1] > max_width:
                height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
                frame = cv2.resize(
                    frame, (max_width, height), interpolation=cv2.INTER_AREA
                )
            target = output_dir / f"{frame_index:05d}.jpg"
            target.write_bytes(encode_jpeg(frame, quality=quality))


def _validate_plate_video(video_path: Path, scene: SceneResult) -> None:
    info = probe_video_info(video_path)
    if not math.isclose(info.fps, scene.fps, rel_tol=1e-5):
        raise ValueError(f"Video FPS {info.fps} differs from scene FPS {scene.fps}.")
    if (info.width, info.height) != (scene.width, scene.height):
        raise ValueError(
            f"Video {video_path} is {info.width}x{info.height}, scene is "
            f"{scene.width}x{scene.height}."
        )
    if info.frame_count != scene.num_frames:
        raise ValueError(
            f"Video {video_path} has {info.frame_count} frames, "
            f"scene has {scene.num_frames}."
        )


def export_blender_bundle(
    *,
    scene_path: Path,
    smpl_faces_path: Path,
    smpl_joint_regressor_path: Path,
    output_dir: Path,
    config: BlenderBundleConfig,
) -> Path:
    """Write the bundle for ``config.camera_id`` and return its manifest path."""
    if output_dir.exists():
        raise FileExistsError(f"Bundle output already exists: {output_dir}")
    scene = load_scene_result(scene_path)
    if scene.schema_version != 2:
        raise ValueError("Blender export requires SceneResult v2 with validity masks.")
    player_valid_in = scene.player_smpl_valid
    ball_valid_in = scene.ball_3d_valid
    if player_valid_in is None or ball_valid_in is None:
        raise ValueError("Scene lacks player_smpl_valid/ball_3d_valid.")
    if scene.smpl_vertices_local is None or scene.smpl_global_orient is None:
        raise ValueError("Scene lacks SMPL vertices/global orientation.")
    if scene.ball_3d is None:
        raise ValueError("Scene lacks ball_3d.")

    camera_ids = scene.metadata.get("camera_ids")
    video_paths = scene.metadata.get("video_paths")
    if not isinstance(camera_ids, list) or not isinstance(video_paths, list):
        raise ValueError("Scene metadata must list camera_ids and video_paths.")
    if (
        len(camera_ids) != len(video_paths)
        or len(camera_ids) != scene.court_kp.shape[0]
    ):
        raise ValueError(
            "Scene camera_ids, video_paths and view axis differ in length."
        )
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Scene camera_ids must be unique.")
    if config.camera_id not in camera_ids:
        raise ValueError(
            f"Camera {config.camera_id!r} not in scene cameras {camera_ids}."
        )
    camera_index = camera_ids.index(config.camera_id)
    video_path = Path(video_paths[camera_index])
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Source video for {config.camera_id} not found: {video_path}"
        )

    camera = PinholeCamera.from_scene_metadata(
        scene.metadata, config.camera_id, width=scene.width, height=scene.height
    )
    reprojection = ball_reprojection_errors_px(
        scene, ball_valid_in, camera, camera_index
    )
    reprojection_median = float(np.median(reprojection))
    if reprojection_median > config.max_ball_reprojection_median_px:
        raise ValueError(
            f"Median ball reprojection error {reprojection_median:.2f}px on "
            f"{config.camera_id} exceeds {config.max_ball_reprojection_median_px}px; "
            "camera and 3D scene frames disagree."
        )

    faces = load_smpl_faces(smpl_faces_path).astype(np.int32)
    vertices = smpl_vertices_to_court(
        scene.smpl_vertices_local,
        scene.smpl_global_orient,
        scene.player_position,
        scene.player_yaw,
        load_smpl_joint_regressor(smpl_joint_regressor_path),
    )
    if faces.size == 0 or faces.min() < 0 or int(faces.max()) >= vertices.shape[2]:
        raise ValueError("SMPL faces reference vertices beyond the scene mesh.")
    if player_valid_in.shape != vertices.shape[:2]:
        raise ValueError(
            f"player_smpl_valid {player_valid_in.shape} does not match players "
            f"{vertices.shape[:2]}."
        )

    output_dir.mkdir(parents=True)
    try:
        players: list[dict[str, Any]] = []
        player_valid = np.zeros_like(player_valid_in)
        for index in range(vertices.shape[0]):
            filled = fill_short_gaps(
                vertices[index],
                player_valid_in[index],
                max_gap_frames=config.player_max_gap_frames,
            )
            player_valid[index] = filled.valid
            # PC2 must be finite: frames left invalid are written as zeros and
            # must be hidden by consumers through player_valid.npy.
            cache = np.where(filled.valid[:, None, None], filled.values, 0.0)
            pc2_name = f"player_{index}.pc2"
            write_pc2(output_dir / pc2_name, cache, start_frame=0.0)
            track_ids = scene.player_track_ids
            players.append(
                {
                    "index": index,
                    "track_id": None if track_ids is None else int(track_ids[index]),
                    "pc2": pc2_name,
                    "filled_frames": list(filled.filled_frames),
                    "invalid_frames": list(filled.unfilled_frames),
                }
            )
        np.save(output_dir / "player_valid.npy", player_valid)
        np.save(output_dir / "smpl_faces.npy", faces)

        ball = fill_short_gaps(
            scene.ball_3d,
            ball_valid_in,
            max_gap_frames=config.ball_max_gap_frames,
        )
        np.save(output_dir / "ball_positions.npy", ball.values)
        np.save(output_dir / "ball_valid.npy", ball.valid)

        _validate_plate_video(video_path, scene)
        _write_plate(
            video_path,
            output_dir / "plate",
            frame_range=range(scene.num_frames),
            max_width=None,
            quality=config.plate_jpeg_quality,
        )

        pose = to_blender_camera(camera, sensor_width_mm=config.sensor_width_mm)
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "source": {
                "scene": str(scene_path),
                "video": str(video_path),
                "camera_id": config.camera_id,
            },
            "num_frames": scene.num_frames,
            "fps": scene.fps,
            "width": scene.width,
            "height": scene.height,
            "camera": {
                "opencv": {
                    "K": camera.K.tolist(),
                    "R": camera.R.tolist(),
                    "t": camera.t.tolist(),
                },
                "blender": pose.to_dict(),
                "ball_reprojection_px": {
                    "median": reprojection_median,
                    "p90": float(np.percentile(reprojection, 90)),
                    "count": int(reprojection.size),
                },
            },
            "smpl_faces": "smpl_faces.npy",
            "players": players,
            "player_valid": "player_valid.npy",
            "player_max_gap_frames": config.player_max_gap_frames,
            "ball": {
                "positions": "ball_positions.npy",
                "valid": "ball_valid.npy",
                "max_gap_frames": config.ball_max_gap_frames,
                "filled_frames": list(ball.filled_frames),
                "invalid_frames": list(ball.unfilled_frames),
            },
            "court": court_geometry(),
            "plate": {
                "pattern": PLATE_PATTERN,
                "jpeg_quality": config.plate_jpeg_quality,
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest_path


__all__ = [
    "BUNDLE_SCHEMA",
    "BlenderBundleConfig",
    "ball_reprojection_errors_px",
    "export_blender_bundle",
]
