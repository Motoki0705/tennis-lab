"""CPU contact sheets comparing observed RGB landmarks with pseudo-3D teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.tasks.slcs.data.annotation import load_slcs_annotation, slcs_annotation_dir
from src.tennis_scene.dataset_pipeline.quality import project
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.reference_pipeline.observations import sha256
from src.tennis_scene.schema import SceneResult
from src.utils.io import save_json_atomic
from src.utils.schema.player import COCO17_SKELETON

GREEN = (40, 240, 40)
MAGENTA = (240, 40, 240)
PLAYER_COLORS = ((0, 220, 255), (255, 220, 0), (80, 140, 255))


def project_teacher(scene: SceneResult, view: int, frame: int) -> dict[str, np.ndarray]:
    """Project supported teacher labels only; never promote unweighted priors."""
    if scene.player_kp_3d is None or scene.ball_3d is None:
        raise ValueError("Review requires player_kp_3d and ball_3d")
    quality = scene.metadata["label_quality"]
    players = scene.player_position.shape[0]
    player_weight = np.asarray(quality["player_weight"], dtype=float)
    ball_weight = np.asarray(quality["ball_weight"], dtype=float)
    if player_weight.shape != (players, scene.num_frames) or ball_weight.shape != (scene.num_frames,):
        raise ValueError("Label weight shapes must match the scene timeline")
    if any(not np.isfinite(w).all() or ((w < 0) | (w > 1)).any() for w in (player_weight, ball_weight)):
        raise ValueError("Label weights must be finite in [0, 1]")
    camera = scene.metadata["reference"]["camera_fits"][view]
    pose, pose_valid = project(scene.player_kp_3d[:, frame], camera)
    ball, ball_valid = project(scene.ball_3d[frame:frame + 1], camera)
    return {
        "pose": pose,
        "pose_valid": pose_valid & (player_weight[:, frame, None] > 0),
        "ball": ball,
        "ball_valid": ball_valid & (ball_weight[frame:frame + 1] > 0),
        "player_weight": player_weight[:, frame],
        "ball_weight": ball_weight[frame:frame + 1],
    }


def _landmarks(image: np.ndarray, points: np.ndarray, valid: np.ndarray,
               color: tuple[int, int, int], *, skeleton: bool) -> None:
    valid = valid & np.isfinite(points).all(-1)
    h, w = image.shape[:2]
    valid &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    if skeleton:
        for a, b in COCO17_SKELETON:
            if valid[a] and valid[b]:
                cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 1, cv2.LINE_AA)
    for point in points[valid]:
        cv2.circle(image, tuple(point.astype(int)), 3 if skeleton else 6, color, 1, cv2.LINE_AA)


def render_review(clip_dir: Path, output_dir: Path, *, frames: list[int],
                  cameras: list[str] | None = None, panel_width: int = 640) -> tuple[Path, Path]:
    """Write a frame-row/camera-column sheet and an audited JSON sidecar."""
    manifest = ClipManifest.load(clip_dir)
    scene = load_slcs_annotation(manifest)
    selected = list(manifest.camera_ids) if cameras is None else cameras
    if not frames or len(set(frames)) != len(frames) or any(f < 0 or f >= scene.num_frames for f in frames):
        raise ValueError("Select unique clip frame indices within the timeline")
    if not selected or len(set(selected)) != len(selected) or panel_width < 320:
        raise ValueError("Select unique cameras and panel_width >= 320")
    reference = scene.metadata["reference"]
    if list(reference["camera_ids"]) != list(manifest.camera_ids) or len(reference["camera_fits"]) != len(manifest.camera_ids):
        raise ValueError("Reference camera order must exactly match the clip manifest")
    if scene.human_kp_2d is None or scene.human_kp_vis is None or scene.ball_uv is None or scene.ball_vis is None:
        raise ValueError("Review requires 2D pose/ball observations and visibility")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new review output directory: {output_dir}")
    panel_height = round(scene.height * panel_width / scene.width)
    rows: dict[int, list[np.ndarray]] = {f: [] for f in frames}
    panels: dict[tuple[int, str], dict[str, Any]] = {}
    for camera_id in selected:
        view = manifest.camera_index(camera_id)
        media = manifest.media_path(camera_id)
        capture = cv2.VideoCapture(str(media))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {media}")
            # Decode sequentially: codec seek approximations cannot shift the frame association.
            wanted = set(frames)
            for frame in range(max(frames) + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Video decode failed: {camera_id}, frame {frame}")
                if frame not in wanted:
                    continue
                if image.shape[:2] != (scene.height, scene.width):
                    raise ValueError("Decoded media size differs from annotation")
                teacher = project_teacher(scene, view, frame)
                scale = np.array([scene.width, scene.height])
                statuses = []
                for player in range(scene.player_position.shape[0]):
                    observed = scene.human_kp_2d[player, view, frame] * scale
                    visible = scene.human_kp_vis[player, view, frame] >= 0.3
                    _landmarks(image, observed, visible, GREEN, skeleton=True)
                    _landmarks(image, teacher["pose"][player], teacher["pose_valid"][player], MAGENTA, skeleton=True)
                    valid_points = observed[visible & np.isfinite(observed).all(-1)]
                    if len(valid_points):
                        anchor = tuple(valid_points.mean(0).astype(int))
                        cv2.putText(image, f"P{player}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLAYER_COLORS[player % len(PLAYER_COLORS)], 2)
                    weight = float(teacher["player_weight"][player])
                    statuses.append(f"P{player}: w={weight:.2f}" if weight > 0 else f"P{player}: UNSUPPORTED (hidden)")
                _landmarks(image, scene.ball_uv[view, frame:frame + 1] * scale,
                           scene.ball_vis[view, frame:frame + 1].astype(bool), GREEN, skeleton=False)
                _landmarks(image, teacher["ball"], teacher["ball_valid"], MAGENTA, skeleton=False)
                ball_weight = float(teacher["ball_weight"][0])
                statuses.append(f"ball: w={ball_weight:.2f}" if ball_weight > 0 else "ball: UNSUPPORTED (hidden)")
                header: np.ndarray = np.full((100, panel_width, 3), 22, np.uint8)
                lines = [f"{camera_id} | clip frame {frame} | {frame / scene.fps:.2f}s",
                         "2D: GREEN | pseudo-3D projection: MAGENTA",
                         " | ".join(statuses[:2]), statuses[-1]]
                for line, text in enumerate(lines):
                    cv2.putText(header, text, (8, 20 + line * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                rows[frame].append(np.vstack([header, cv2.resize(image, (panel_width, panel_height))]))
                panels[frame, camera_id] = {
                    "frame_idx": frame, "camera_id": camera_id, "camera_index": view,
                    "player_weights": teacher["player_weight"].tolist(), "ball_weight": ball_weight,
                    "observed_joints": (scene.human_kp_vis[:, view, frame] >= 0.3).sum(-1).tolist(),
                    "observed_ball": bool(scene.ball_vis[view, frame]),
                    "projectable_teacher_joints": teacher["pose_valid"].sum(-1).tolist(),
                    "projectable_teacher_ball": bool(teacher["ball_valid"][0]),
                    "media": str(media.resolve()),
                }
        finally:
            capture.release()
    sheet = np.vstack([np.hstack(rows[f]) for f in frames])
    output_dir.mkdir(parents=True, exist_ok=False)
    image_path, sidecar_path = output_dir / "contact_sheet.jpg", output_dir / "review.json"
    if not cv2.imwrite(str(image_path), sheet):
        raise OSError(f"Cannot write image: {image_path}")
    scene_path = slcs_annotation_dir(clip_dir) / "scene.npz"
    save_json_atomic({
        "schema_version": 1, "clip_id": manifest.clip_id, "video_id": manifest.video_id,
        "interpretation": "Pseudo teacher / approximate learned pinhole calibration; NOT measured 3D ground truth. Reprojection is observable consistency only.",
        "frame_reference": "zero-based absolute frame in clip camera video, not source recording offset",
        "layout": "rows follow frames; columns follow camera_ids", "frames": frames,
        "camera_ids": selected, "fps": scene.fps, "observation_visibility_threshold": 0.3,
        "unsupported_policy": "teacher weight <= 0 hidden and marked UNSUPPORTED; missing observations hidden",
        "player_id": "canonical scene player axis; yellow P0, cyan P1, orange P2 (palette repeats)",
        "scene": str(scene_path.resolve()), "scene_sha256": sha256(scene_path),
        "metadata_sha256": sha256(scene_path.with_suffix(".metadata.json")),
        "panels": [panels[f, c] for f in frames for c in selected],
    }, sidecar_path)
    return image_path.resolve(), sidecar_path.resolve()
