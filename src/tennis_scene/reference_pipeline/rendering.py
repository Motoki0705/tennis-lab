"""Synchronized source-observation and court-space prediction evaluation movie."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.reference_pipeline.observations import read_clip
from src.utils.schema.court import CourtConfig, court_keypoints_3d
from src.utils.schema.player import COCO17_SKELETON

BONES = COCO17_SKELETON
COLORS = [(70, 210, 255), (255, 180, 90)]
COURT_LINES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
]


def virtual_project(points: np.ndarray) -> np.ndarray:
    """Fixed perspective camera; preserve all predicted heights and locations."""
    eye = np.array([27.0, -35.0, 31.0])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points) - eye
    depth = relative @ forward
    result: np.ndarray = np.stack(
        [640 + 1200 * (relative @ right) / depth, 430 - 1200 * (relative @ up) / depth],
        -1,
    ).astype(np.int32)
    return result


def text(
    image: np.ndarray,
    value: str,
    at: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (230, 236, 241),
) -> None:
    cv2.putText(
        image, value, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def render(cfg: DictConfig, clip_dir: Path, output: Path) -> None:
    clip = read_clip(clip_dir)
    scene = load_scene_result(output / "scene.npz")
    if (
        scene.player_kp_3d is None
        or scene.ball_3d is None
        or scene.human_kp_2d is None
        or scene.human_kp_vis is None
        or scene.ball_uv is None
        or scene.ball_vis is None
    ):
        raise ValueError(
            "Evaluation requires joints, poses, and ball observations/predictions"
        )
    ball_meta = json.loads((output / "ball_import.metadata.json").read_text())
    camera_ids = clip["camera_ids"]
    if len(camera_ids) != 3:
        raise ValueError("This evaluation layout requires exactly three cameras")
    caps = [cv2.VideoCapture(str(clip_dir / p)) for p in clip["video_paths"]]
    step = int(cfg.sample_stride)
    movie = output / "evaluation.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1920x1080",
        "-r",
        str(scene.fps / step),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    court = court_keypoints_3d(CourtConfig(0.914, None)).numpy()
    projected = virtual_project(court)
    static: np.ndarray = np.full((1080, 1920, 3), (27, 23, 19), np.uint8)
    ground = virtual_project(
        np.array([[-8, -19, 0], [8, -19, 0], [8, 19, 0], [-8, 19, 0]], np.float32)
    )
    cv2.fillConvexPoly(static, ground, (75, 83, 43))
    cv2.fillConvexPoly(static, projected[[0, 1, 3, 2]], (85, 106, 52))
    for a, b in COURT_LINES:
        cv2.line(
            static,
            tuple(projected[a]),
            tuple(projected[b]),
            (210, 220, 205),
            2,
            cv2.LINE_AA,
        )
    for a, b in [(15, 16), (16, 18), (18, 17), (15, 17)]:
        cv2.line(
            static,
            tuple(projected[a]),
            tuple(projected[b]),
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
    text(static, "MEIJI / CLIP 000", (40, 50), 1.05)
    text(static, "PLCS + BLCS  |  Reference: cam0", (40, 86), 0.75)
    text(static, "Pose detail / front + side / fixed metric scale", (40, 820), 0.65)
    text(static, "DINO + BoT-SORT > ViTPose > PLCS", (580, 875), 0.65)
    text(static, "External ball observations > BLCS", (580, 910), 0.65)
    text(static, "CourtKP14 / cam0 reference / original speed", (580, 945), 0.65)
    text(static, "No 3D ground truth. No ground-height clamp.", (580, 980), 0.65)
    text(static, "Right: 2D observations; circles: ball input", (580, 1015), 0.65)
    written = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for frame_id in range(scene.num_frames):
                frames = []
                for cap in caps:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Video ended before frame {frame_id}")
                    frames.append(frame)
                if frame_id % step:
                    continue
                canvas = static.copy()
                for p, color in enumerate(COLORS):
                    joints = virtual_project(scene.player_kp_3d[p, frame_id])
                    for a, b in BONES:
                        cv2.line(
                            canvas,
                            tuple(joints[a]),
                            tuple(joints[b]),
                            color,
                            3,
                            cv2.LINE_AA,
                        )
                    for joint in joints:
                        cv2.circle(canvas, tuple(joint), 3, color, -1, cv2.LINE_AA)
                    root = virtual_project(
                        scene.player_position[p, frame_id : frame_id + 1]
                    )[0]
                    text(
                        canvas,
                        f"P{p + 1}",
                        (int(root[0]) + 8, int(root[1]) - 10),
                        0.65,
                        color,
                    )
                # Root-centred XY, but absolute Z: keep floating feet visible.
                # Orthographic front/side panels use the same 85 px per metre.
                for p, color in enumerate(COLORS):
                    origin_x = 40 + p * 270
                    local = scene.player_kp_3d[p, frame_id].copy()
                    local[:, :2] -= scene.player_position[p, frame_id, :2]
                    text(
                        canvas, f"P{p + 1}  front / side", (origin_x, 850), 0.55, color
                    )
                    cv2.line(
                        canvas,
                        (origin_x, 1045),
                        (origin_x + 245, 1045),
                        (160, 160, 160),
                        1,
                    )
                    for axis, center_x in [(0, origin_x + 60), (1, origin_x + 185)]:
                        detail = np.stack(
                            [center_x + 85 * local[:, axis], 1045 - 85 * local[:, 2]],
                            -1,
                        ).astype(np.int32)
                        for a, b in BONES:
                            cv2.line(
                                canvas,
                                tuple(detail[a]),
                                tuple(detail[b]),
                                color,
                                2,
                                cv2.LINE_AA,
                            )
                        for joint in detail:
                            cv2.circle(canvas, tuple(joint), 2, color, -1, cv2.LINE_AA)
                visible = bool(scene.ball_vis[:, frame_id].any())
                trail = virtual_project(
                    scene.ball_3d[max(0, frame_id - 30) : frame_id + 1]
                )
                if len(trail) > 1:
                    cv2.polylines(
                        canvas, [trail], False, (40, 200, 235), 2, cv2.LINE_AA
                    )
                cv2.circle(
                    canvas,
                    tuple(trail[-1]),
                    7,
                    (30, 245, 250) if visible else (140, 140, 140),
                    -1,
                    cv2.LINE_AA,
                )
                text(
                    canvas,
                    f"{frame_id / scene.fps:05.2f} s   frame {frame_id:04d}/{scene.num_frames - 1}",
                    (40, 126),
                    0.7,
                )
                text(
                    canvas,
                    f"Ball height: {scene.ball_3d[frame_id, 2]:.2f} m",
                    (40, 163),
                    0.7,
                )
                for v, (cam, source) in enumerate(zip(camera_ids, frames, strict=True)):
                    tile = cv2.resize(source, (640, 360))
                    for p, color in enumerate(COLORS):
                        points = (
                            scene.human_kp_2d[p, v, frame_id] * [640, 360]
                        ).astype(int)
                        vis = scene.human_kp_vis[p, v, frame_id] >= float(
                            cfg.pose_visibility_threshold
                        )
                        for a, b in BONES:
                            if vis[a] and vis[b]:
                                cv2.line(
                                    tile,
                                    tuple(points[a]),
                                    tuple(points[b]),
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )
                        if vis[0]:
                            text(
                                tile,
                                f"P{p + 1}",
                                tuple(points[0] - [0, 9]),
                                0.45,
                                color,
                            )
                    if scene.ball_vis[v, frame_id]:
                        ball = tuple(
                            (scene.ball_uv[v, frame_id] * [640, 360]).astype(int)
                        )
                        cv2.circle(tile, ball, 6, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.drawMarker(
                            tile, ball, (0, 255, 255), cv2.MARKER_CROSS, 14, 1
                        )
                    cv2.rectangle(tile, (0, 0), (640, 27), (20, 20, 20), -1)
                    text(
                        tile,
                        f"{cam}   ball: {ball_meta['status'][v][frame_id]}",
                        (12, 19),
                        0.5,
                    )
                    canvas[v * 360 : (v + 1) * 360, 1280:] = tile
                if encoder.stdin is None:
                    raise RuntimeError("Video encoder pipe missing")
                encoder.stdin.write(canvas.tobytes())
                written += 1
                if frame_id == 0:
                    cv2.imwrite(str(output / "evaluation_preview.jpg"), canvas)
        finally:
            for cap in caps:
                cap.release()
            if encoder.stdin is not None:
                encoder.stdin.close()
        code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code}")
    (output / "video_receipt.json").write_text(
        json.dumps(
            {
                "path": str(movie),
                "frames": written,
                "fps": scene.fps / step,
                "duration_seconds": written / (scene.fps / step),
            },
            indent=2,
        )
    )
    print(f"Saved {movie} ({written} frames)", flush=True)
