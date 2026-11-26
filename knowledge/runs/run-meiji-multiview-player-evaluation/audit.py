"""Quality diagnostics and native video overlays for the six-clip run."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from inputs import ClipInputs
from scipy.spatial.transform import Rotation

from src.tennis_scene.evaluation.player_pose.reference import (
    BoxQuality,
    Geometry,
    JointQuality,
    Quality,
    SmplRegressors,
    joint_boxes,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    describe,
)
from src.utils.schema.player import COCO17_SKELETON, COCO_KP_NAMES
from src.utils.video.reader import OpenCVVideoFrameReader, probe_video_info
from src.utils.video.writer import VideoWriter

MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20)


def overlap_and_size(inputs: ClipInputs, keypoints: np.ndarray) -> dict[str, Any]:
    boxes, valid = joint_boxes(
        keypoints[..., :2], keypoints[..., 2] > 0, inputs.size, margin=0.0, min_joints=2
    )
    overlapped = np.zeros(inputs.scene.num_frames, bool)
    for a, b in combinations(range(len(boxes)), 2):
        intersection = np.maximum(
            np.minimum(boxes[a, ..., 2:], boxes[b, ..., 2:])
            - np.maximum(boxes[a, ..., :2], boxes[b, ..., :2]),
            0,
        ).prod(-1)
        areas = (boxes[[a, b], ..., 2:] - boxes[[a, b], ..., :2]).prod(-1)
        overlapped |= (
            valid[a] & valid[b] & (intersection > 0.1 * (areas.sum(0) - intersection))
        ).any(0)
    heights = (boxes[..., 3] - boxes[..., 1])[valid]
    if not len(heights):
        raise ValueError(f"No observed player sizes in {inputs.clip_id}")
    scene = inputs.scene
    assert scene.player_smpl_valid is not None and scene.player_valid is not None
    all_valid = scene.player_smpl_valid.all(0)
    if int(all_valid.sum()) != inputs.record["players"]["all_four_placed_frames"]:
        raise ValueError(
            f"Placed body counts disagree with source pin for {inputs.clip_id}"
        )
    xy = scene.player_position[..., :2]
    adjacent_valid = scene.player_valid[:, 1:] & scene.player_valid[:, :-1]
    speed = np.linalg.norm(np.diff(xy, axis=1), axis=-1)[adjacent_valid] * scene.fps
    return {
        "all_four_placed_frames": int(all_valid.sum()),
        "all_four_placed_fraction": float(all_valid.mean()),
        "player_overlap_frames": int(overlapped.sum()),
        "player_overlap_fraction": float(overlapped.mean()),
        "observed_bbox_height_px": {
            "count": int(len(heights)),
            "p10_p50_p90": np.percentile(heights, [10, 50, 90]).tolist(),
        },
        "court_xy_speed_mps": {
            "median": float(np.median(speed)),
            "p90": float(np.percentile(speed, 90)),
        },
        "source_clip_duration_sec": scene.num_frames / scene.fps,
    }


def diagnostic_arrays(
    inputs: ClipInputs, geometry: Geometry, observation: np.ndarray
) -> dict[str, np.ndarray]:
    scene = inputs.scene
    assert scene.player_kp_3d is not None and scene.player_kp_3d_vis is not None
    triangle, tri_valid = scene.player_kp_3d, scene.player_kp_3d_vis
    tri_px = np.full(geometry.front.shape, np.nan)
    for view, camera in enumerate(inputs.cameras):
        uv, front = camera.project(triangle)
        valid = front & tri_valid & geometry.front[:, view]
        tri_px[:, view] = np.where(
            valid, np.linalg.norm(uv - geometry.pixels[:, view], axis=-1), np.nan
        )
    error = np.linalg.norm(geometry.pixels - observation[..., :2], axis=-1)
    return {
        "triangulation_error_m": np.where(
            tri_valid, np.linalg.norm(geometry.joints_m - triangle, axis=-1), np.nan
        ),
        "triangulation_error_px": tri_px,
        "observation_error_px": np.where(observation[..., 2] > 0, error, np.nan),
    }


def projection_check(
    inputs: ClipInputs,
    geometry: Geometry,
    observation: np.ndarray,
    regressors: SmplRegressors,
) -> dict[str, Any]:
    """Independent transform arithmetic and OpenCV check on frame 120 of clip_003."""
    if inputs.clip_id != "video_000/clip_003":
        raise ValueError("The transform check has a fixed source clip")
    frame = 120
    scene = inputs.scene
    assert (
        scene.smpl_vertices_local is not None and scene.smpl_global_orient is not None
    )
    assert scene.player_kp_3d is not None and scene.player_kp_3d_vis is not None
    vertices = scene.smpl_vertices_local[:, frame].astype(np.float64)
    translation = scene.player_position[:, frame]
    centered = vertices - regressors.root @ vertices
    compensation = Rotation.from_rotvec(scene.smpl_global_orient[:, frame]).as_matrix()
    yaw = Rotation.from_euler("z", scene.player_yaw[:, frame, None]).as_matrix()
    up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    composed = yaw @ up @ compensation.transpose(0, 2, 1)
    expected = (regressors.coco17 @ centered) @ composed.transpose(
        0, 2, 1
    ) + translation[:, None]
    joints = geometry.joints_m[:, frame]
    transform_error = float(np.max(np.abs(joints - expected)))
    if transform_error > 5e-6 or not np.isfinite(joints).all():
        raise ValueError(f"Scene mesh composition check failed: {transform_error}")
    swapped = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    face_order_checks = []
    per_camera = {}
    for view, camera in enumerate(inputs.cameras):
        pixels = geometry.pixels[:, view, frame]
        projected, _ = cv2.projectPoints(
            joints.astype(np.float64).reshape(-1, 3),
            cv2.Rodrigues(camera.rotation)[0],
            camera.translation,
            camera.intrinsic,
            None,
        )
        camera_error = float(np.max(np.abs(projected.reshape(pixels.shape) - pixels)))
        observed = observation[:, view, frame, :, 2] > 0
        delta = pixels - observation[:, view, frame, :, :2]
        ordinary = np.linalg.norm(delta, axis=-1)
        exchanged = np.linalg.norm(
            pixels[:, swapped] - observation[:, view, frame, :, :2], axis=-1
        )
        limbs = observed[:, 5:]
        face = observed[:, 1:5]
        median_ordinary = float(np.median(ordinary[:, 5:][limbs]))
        median_exchanged = float(np.median(exchanged[:, 5:][limbs]))
        if camera_error > 1e-6 or not median_ordinary < median_exchanged:
            raise ValueError(
                f"Projection or left/right check failed for {camera.camera_id}"
            )
        face_order_checks.append((ordinary[:, 1:5][face], exchanged[:, 1:5][face]))
        tri_uv, tri_front = camera.project(scene.player_kp_3d[:, frame])
        tri_valid = scene.player_kp_3d_vis[:, frame] & tri_front
        per_camera[camera.camera_id] = {
            "opencv_max_abs_px": camera_error,
            "vitpose_error_px": describe(ordinary[observed]),
            "vitpose_signed_median_xy_px": np.median(delta[observed], axis=0).tolist(),
            "triangulation_error_px": describe(
                np.linalg.norm(pixels - tri_uv, axis=-1)[tri_valid]
            ),
            "limb_error_median_px": median_ordinary,
            "swapped_limb_error_median_px": median_exchanged,
            "first_player_face": {
                name: {
                    "smpl_px": pixels[0, joint].tolist(),
                    "vitpose_px": observation[0, view, frame, joint, :2].tolist()
                    if observed[0, joint]
                    else None,
                }
                for joint, name in enumerate(COCO_KP_NAMES[:5])
            },
        }
    face_regular, face_swapped = [
        describe(np.concatenate([item[i] for item in face_order_checks]))
        for i in (0, 1)
    ]
    if not face_regular["median"] < face_swapped["median"]:
        raise ValueError("Exchanging eye and ear sides improved the fixed-frame audit")
    return {
        "clip_id": inputs.clip_id,
        "source_frame": frame,
        "root_regressor_max_offset_m": float(
            np.max(np.abs(regressors.root @ vertices))
        ),
        "transform_max_abs_m": transform_error,
        "coco_regressor_row_sums": regressors.coco17.sum(-1).tolist(),
        "face_order_error_px": face_regular,
        "swapped_face_order_error_px": face_swapped,
        "by_camera": per_camera,
    }


def compare_mesh_boxes(
    geometry: Geometry, size: tuple[int, int], minimum: int
) -> dict[str, np.ndarray]:
    results = {}
    for margin in MARGINS:
        boxes, valid = joint_boxes(
            geometry.pixels, geometry.inside, size, margin=margin, min_joints=minimum
        )
        mesh = geometry.mesh_boxes
        keep = valid & np.isfinite(mesh).all(-1)
        a, b = boxes[keep], mesh[keep]
        cross = np.maximum(
            np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0
        ).prod(-1)
        union = (a[:, 2:] - a[:, :2]).prod(-1) + (b[:, 2:] - b[:, :2]).prod(-1) - cross
        results[str(margin)] = cross / union
    return results


def quality_counts(inputs: ClipInputs, quality: Quality) -> list[dict[str, Any]]:
    rows = []
    frames = inputs.evaluation_frames
    assert inputs.scene.player_rejection_code is not None
    for p, player in enumerate(inputs.player_ids):
        inferred = inputs.inferred[p]
        raw_codes = inputs.scene.player_rejection_code[p, frames]
        for v, camera in enumerate(inputs.cameras):
            boxes = quality.box_reasons[p, v, frames]
            joints = quality.joint_reasons[p, v, frames]
            rows.append(
                {
                    "clip_id": inputs.clip_id,
                    "camera_id": camera.camera_id,
                    "player_id": int(player),
                    "source_frames": inputs.scene.num_frames,
                    "cohort_frames": len(frames),
                    "off_cohort_frames": inputs.scene.num_frames - len(frames),
                    "gvhmr_frames": int(inferred.sum()),
                    "box_excluded": int(np.count_nonzero(boxes)),
                    "joint_excluded": int(np.count_nonzero(joints)),
                    **{
                        f"box_{code.name.lower()}": int((boxes == code).sum())
                        for code in BoxQuality
                    },
                    **{
                        f"joint_{code.name.lower()}": int((joints == code).sum())
                        for code in JointQuality
                    },
                    **{
                        f"root_rejection_{code}": int((raw_codes == code).sum())
                        for code in np.unique(raw_codes)
                    },
                }
            )
    return rows


def smoke_predictions(
    inputs: ClipInputs, observed: np.ndarray, margin: float
) -> dict[FrameKey, Predictions]:
    """Pair observed poses with their own envelopes for an end-to-end scoring smoke check."""
    boxes, valid = joint_boxes(
        observed[..., :2],
        observed[..., 2] > 0,
        inputs.size,
        margin=margin,
        min_joints=2,
    )
    rows = {}
    for v, camera in enumerate(inputs.cameras):
        for frame in inputs.evaluation_frames:
            keep = valid[:, v, frame]
            poses = observed[keep, v, frame]
            rows[FrameKey(inputs.clip_id, camera.camera_id, int(frame))] = Predictions(
                boxes[keep, v, frame], poses[..., 2].max(-1), poses
            )
    return rows


def _skeleton(
    image: np.ndarray,
    xy: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    points = np.rint(np.where(valid[:, None], xy, 0)).astype(int).tolist()
    for a, b in COCO17_SKELETON:
        if valid[a] and valid[b]:
            cv2.line(image, points[a], points[b], color, 1, cv2.LINE_AA)
    for point, keep in zip(points, valid, strict=True):
        if keep:
            cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)


def _text(
    image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]
) -> None:
    for stroke, width in (((0, 0, 0), 4), (color, 1)):
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            stroke,
            width,
            cv2.LINE_AA,
        )


def render_overlay(
    inputs: ClipInputs,
    view: int,
    geometry: Geometry,
    quality: Quality,
    observed: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    camera = inputs.cameras[view].camera_id
    frames = inputs.evaluation_frames
    representative = (
        120 if inputs.clip_id.endswith("/clip_003") else int(frames[len(frames) // 2])
    )
    still = path.with_suffix(".png")
    expected = set(frames.tolist())
    rendered = set()
    pseudo_color, obs_color, box_color = (65, 240, 60), (230, 80, 240), (0, 195, 255)
    frame_count = 0
    with VideoWriter(path, fps=inputs.scene.fps / 2, crf=20) as writer:
        for packet in OpenCVVideoFrameReader(inputs.media[view]):
            frame = packet.index
            frame_count += 1
            if packet.original_size != inputs.size:
                raise ValueError(
                    f"Video resolution differs from scene: {inputs.media[view]}"
                )
            if frame not in expected:
                continue
            image = packet.frame
            for p, player in enumerate(inputs.player_ids):
                xy = observed[p, view, frame]
                _skeleton(image, xy[..., :2], xy[..., 2] > 0, obs_color, 4)
                _skeleton(
                    image,
                    geometry.pixels[p, view, frame],
                    geometry.inside[p, view, frame],
                    pseudo_color,
                    2,
                )
                box = quality.boxes[p, view, frame]
                if np.isfinite(box).all():
                    x0, y0, x1, y1 = np.rint(box).astype(int)
                    reason = BoxQuality(int(quality.box_reasons[p, view, frame]))
                    cv2.rectangle(
                        image,
                        (x0, y0),
                        (x1, y1),
                        box_color,
                        2 if reason == BoxQuality.ELIGIBLE else 1,
                        cv2.LINE_AA,
                    )
                    tag = (
                        f"P{player}"
                        if reason == BoxQuality.ELIGIBLE
                        else f"P{player} ({reason.name.lower()})"
                    )
                    _text(image, tag, (x0, max(18, y0 - 4)), box_color)
            _text(
                image,
                f"{inputs.clip_id}  {camera}  source frame {frame}",
                (20, 28),
                (255, 255, 255),
            )
            _text(image, "SMPL COCO17", (20, 55), pseudo_color)
            _text(image, "ViTPose observations", (195, 55), obs_color)
            _text(image, "SMPL pseudo-box", (445, 55), box_color)
            if frame == representative and not cv2.imwrite(str(still), image):
                raise OSError(f"Could not write representative frame {still}")
            writer.write_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rendered.add(frame)
    info = probe_video_info(path)
    if (
        frame_count != inputs.scene.num_frames
        or rendered != expected
        or not still.is_file()
        or (info.width, info.height, info.frame_count) != (*inputs.size, len(expected))
    ):
        raise ValueError(f"Overlay frame coverage does not match its source: {path}")
    return {
        "source_frame": representative,
        "video_frames": len(rendered),
        "fps": info.fps,
    }
