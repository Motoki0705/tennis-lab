"""Generate four-camera V4 shot panels with no access to hitter estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.analytics.hitter_v4 import (
    V4Benchmark,
    V4Clip,
    V4Shot,
    file_sha256,
)
from src.tennis_scene.visualization.blender.export.camera import PinholeCamera
from src.utils.schema.player import COCO17_SKELETON

CAMERA_IDS = ("cam0", "cam1", "cam2", "cam3")
COLORS = ((255, 210, 40), (45, 155, 255), (165, 80, 255), (110, 245, 75))
SCENE_FIELDS = (
    "player_position",
    "player_valid",
    "player_track_ids",
    "player_kp_3d",
    "player_kp_3d_vis",
    "ball_3d",
    "ball_3d_valid",
    "ball_uv",
    "ball_vis",
)
Image: TypeAlias = NDArray[np.uint8]


def _pixels(
    camera: PinholeCamera, xyz: NDArray[np.float32], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    in_front = (xyz @ camera.R[2] + camera.t[2]) > 0
    valid = mask & in_front & np.isfinite(xyz).all(axis=-1)
    xy = np.zeros((*xyz.shape[:-1], 2), dtype=np.int32)
    xy[valid] = np.round(camera.project(xyz[valid])).astype(np.int32)
    return xy, (
        valid
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < camera.width)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < camera.height)
    )


def _caption(
    image: Image, text: str, position: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA
    )


def _overlay(
    image: Image,
    camera: PinholeCamera,
    camera_index: int,
    frame: int,
    data: dict[str, NDArray[Any]],
) -> Image:
    overlay = image.copy()
    joints, joint_valid = _pixels(
        camera, data["player_kp_3d"][:, frame], data["player_kp_3d_vis"][:, frame]
    )
    roots, root_valid = _pixels(
        camera, data["player_position"][:, frame], data["player_valid"][:, frame]
    )
    for index, track_id in enumerate(data["player_track_ids"]):
        color = COLORS[index % len(COLORS)]
        for a, b in COCO17_SKELETON:
            if joint_valid[index, a] and joint_valid[index, b]:
                cv2.line(
                    overlay,
                    tuple(joints[index, a]),
                    tuple(joints[index, b]),
                    color,
                    1,
                    cv2.LINE_AA,
                )
        visible = joints[index, joint_valid[index]]
        if len(visible):
            x, y = map(int, visible[np.argmin(visible[:, 1])])
        elif root_valid[index]:
            x, y = map(int, roots[index])
        else:
            continue
        _caption(overlay, f"{index}/ID{track_id}", (x + 8, max(y - 9, 20)), color)
    ball, ball_valid = _pixels(
        camera,
        data["ball_3d"][frame : frame + 1],
        data["ball_3d_valid"][frame : frame + 1],
    )
    if ball_valid[0]:
        cv2.circle(overlay, tuple(ball[0]), 9, (255, 255, 255), 1, cv2.LINE_AA)
    if data["ball_vis"][camera_index, frame]:
        uv = tuple(np.round(data["ball_uv"][camera_index, frame]).astype(int))
        cv2.circle(overlay, uv, 5, (0, 240, 240), 1, cv2.LINE_AA)
    return overlay


def _crop_bounds(
    camera: PinholeCamera,
    camera_index: int,
    shot: V4Shot,
    data: dict[str, NDArray[Any]],
) -> tuple[int, int, int, int]:
    frame = shot.frame
    ball, valid = _pixels(
        camera,
        data["ball_3d"][frame : frame + 1],
        data["ball_3d_valid"][frame : frame + 1],
    )
    if valid[0]:
        x, y = map(int, ball[0])
    elif data["ball_vis"][camera_index, frame]:
        x, y = map(int, data["ball_uv"][camera_index, frame])
    else:
        x, y = camera.width // 2, camera.height // 2
    width, height = min(608, camera.width), min(360, camera.height)
    left = max(0, min(x - width // 2, camera.width - width))
    top = max(0, min(y - 145, camera.height - height))
    return left, top, left + width, top + height


def _read_frames(path: Path, frames: list[int]) -> dict[int, Image]:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open {path}")
        images: dict[int, Image] = {}
        next_frame = -1
        for frame in frames:
            if not 0 <= frame - next_frame <= 16:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise OSError(f"Could not seek to frame {frame} in {path}")
                next_frame = frame
            while next_frame < frame:
                if not capture.grab():
                    raise OSError(f"Could not seek to frame {frame} in {path}")
                next_frame += 1
            ok, image = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                raise OSError(f"Could not decode frame {frame} in {path}")
            images[frame] = image
            next_frame = frame + 1
        return images
    finally:
        capture.release()


def _shot_frames(shot: V4Shot, length: int) -> list[int]:
    width = max(8, shot.frame_uncertainty)
    return sorted(
        {max(0, shot.frame - width), shot.frame, min(length - 1, shot.frame + width)}
    )


def _clip_panels(clip: V4Clip, output: Path) -> dict[str, Any]:
    if set(clip.cameras) != set(CAMERA_IDS):
        raise ValueError(f"{clip.clip_id}: expected the four synchronized cameras")
    metadata = clip.metadata.read_json()
    with np.load(clip.scene.verify(), allow_pickle=False) as archive:
        shots = clip.shots(archive)
        width, height = int(archive["width"]), int(archive["height"])
        data = {field: archive[field] for field in SCENE_FIELDS}
    tracks = data["player_track_ids"].tolist()
    frames = {shot.source_event_id: _shot_frames(shot, clip.frames) for shot in shots}
    requested = sorted(
        {frame for event_frames in frames.values() for frame in event_frames}
    )
    wide: dict[str, list[Image]] = {shot.source_event_id: [] for shot in shots}
    close: dict[str, list[Image]] = {shot.source_event_id: [] for shot in shots}
    for camera_index, camera_id in enumerate(CAMERA_IDS):
        camera = PinholeCamera.from_scene_metadata(
            metadata, camera_id, width=width, height=height
        )
        images = _read_frames(clip.cameras[camera_id].verify(), requested)
        for shot in shots:
            event_id = shot.source_event_id
            left, top, right, bottom = _crop_bounds(camera, camera_index, shot, data)
            wide_row, close_row = [], []
            for frame in frames[event_id]:
                original = images[frame]
                if original.shape[:2] != (height, width):
                    raise ValueError(
                        f"{clip.clip_id}/{camera_id}: unexpected video dimensions"
                    )
                overlay = _overlay(original, camera, camera_index, frame, data)
                full = cv2.resize(overlay, (560, 315), interpolation=cv2.INTER_AREA)
                zoom = cv2.resize(
                    overlay[top:bottom, left:right],
                    (448, 265),
                    interpolation=cv2.INTER_AREA,
                )
                label = f"{camera_id}  f{frame}  ({frame - shot.frame:+d})"
                _caption(full, label, (10, 23), (255, 255, 255))
                _caption(zoom, label, (10, 23), (255, 255, 255))
                wide_row.append(full)
                close_row.append(zoom)
            wide[event_id].append(np.hstack(wide_row))
            close[event_id].append(np.hstack(close_row))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for shot in shots:
        event_id = shot.source_event_id
        panels = {}
        for role, images_by_event in (("wide", wide), ("close", close)):
            name = f"{event_id}-{role}.jpg"
            path = output / name
            if not cv2.imwrite(
                str(path),
                np.vstack(images_by_event[event_id]),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise OSError(f"Could not write {path}")
            panels[role] = {"file": name, "sha256": file_sha256(path)}
        rows.append(
            {
                "source_event_id": event_id,
                "frame": shot.frame,
                "subtype": shot.subtype,
                "review_frames": frames[event_id],
                "panels": panels,
            }
        )
    return {
        "clip_id": clip.clip_id,
        "split": clip.split,
        "camera_sha256": {name: ref.sha256 for name, ref in clip.cameras.items()},
        "scene_npz_sha256": clip.scene.sha256,
        "scene_metadata_sha256": clip.metadata.sha256,
        "v4_event_component_sha256": clip.event_component.sha256,
        "player_track_ids": tracks,
        "shots": rows,
    }


def render_audit(benchmark: V4Benchmark, output: Path, clip_ids: list[str]) -> None:
    cv2.setNumThreads(1)
    for clip_id in clip_ids:
        clip = benchmark.clips[clip_id]
        if clip.split not in ("val", "test"):
            raise ValueError(f"Visual audit is for val/test clips: {clip_id}")
        directory = output / clip_id
        document = _clip_panels(clip, directory)
        document["manifest_sha256"] = benchmark.manifest_sha256
        (directory / "panels.json").write_text(json.dumps(document, indent=2) + "\n")
        print(f"{clip_id}: rendered {len(document['shots'])} shots", flush=True)
