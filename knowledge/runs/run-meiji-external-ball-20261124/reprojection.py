"""Draw input pixels and projected 3D on representative synchronized frame crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.imports.external_ball_positions import (
    STATUSES,
    ExternalBallPositions,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

STATUS_COLORS = ((85, 229, 113), (248, 215, 66), (248, 151, 74), (101, 159, 250))
PROJECTED_COLOR = (255, 72, 222)
SIZE = (320, 180)


def text(canvas, value, origin, color=(231, 237, 244), scale=0.49):
    cv2.putText(
        canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def show_points(frame, uv, projected, status):
    height, width = frame.shape[:2]
    span = np.abs(projected - uv)
    crop_width = int(min(width, max(244, span[0] + 140, (span[1] + 88) * 16 / 9)))
    crop_height = int(min(height, round(crop_width * 9 / 16)))
    left, top = (
        int(np.clip(v, 0, maximum))
        for v, maximum in zip(
            (uv + projected - [crop_width, crop_height]) / 2,
            (width - crop_width, height - crop_height),
            strict=True,
        )
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(
        rgb[top : top + crop_height, left : left + crop_width],
        SIZE,
        interpolation=cv2.INTER_AREA,
    )
    a, b = (
        tuple(
            np.rint(
                (point - (left, top)) * (np.array(SIZE) / (crop_width, crop_height))
            ).astype(int)
        )
        for point in (uv, projected)
    )
    cv2.line(crop, a, b, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.circle(crop, a, 7, (24, 26, 32), 3, cv2.LINE_AA)
    cv2.circle(crop, a, 7, STATUS_COLORS[status], 2, cv2.LINE_AA)
    cv2.drawMarker(crop, b, PROJECTED_COLOR, cv2.MARKER_CROSS, 9, 2, cv2.LINE_AA)
    return crop


def make_figure(dataset: Path, video: str, clip: str, output: Path):
    root = dataset / "videos" / video / "clips" / clip / "annotations/tennis_scene"
    index = json.loads((root / "scene.json").read_text())
    store = ClipStore(root, index["source"], memory_entries=0)
    refs = store.references()
    positions = store.load(
        refs["external_ball_positions"], ArtifactCodec(ExternalBallPositions)
    )
    ball = store.load(
        refs["ball_triangulation"], ArtifactCodec(BallReconstructionResult)
    )
    geometry_node = (
        "camera_alignment"
        if "camera_alignment" in refs
        else "external_ball_camera_geometry"
    )
    geometry = store.load(
        refs[geometry_node], ArtifactCodec(CameraAlignmentOutput)
    ).geometry
    assert geometry is not None
    projected = np.stack(
        [camera.project(ball.trajectory.positions)[0] for camera in geometry.cameras]
    )
    in_image = ((projected >= 0) & (projected < positions.image_size)).all(axis=(0, 2))
    eligible = np.flatnonzero(ball.trajectory.inliers.all(axis=0) & in_image)
    per_frame_error = ball.trajectory.reprojection_px.mean(axis=0)
    ranked = eligible[np.argsort(per_frame_error[eligible])]
    frames = [int(ranked[int(q * (len(ranked) - 1))]) for q in (0.5, 0.95)]
    height = 540
    canvas = np.full(
        (height, len(geometry.cameras) * (SIZE[0] + 10) + 14, 3), (22, 29, 40), np.uint8
    )
    text(
        canvas,
        f"{video}/{clip}   |   File positions and 3D reprojections",
        (12, 29),
        scale=0.71,
    )
    labels = ["median camera mean", "95th percentile camera mean"]
    for row, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        y = 63 + row * 229
        text(
            canvas,
            f"Frame {frame}  |  {label} {per_frame_error[frame]:.1f} px",
            (12, y - 9),
        )
    for camera_index, video_row in enumerate(index["source"]["videos"]):
        x = 12 + camera_index * (SIZE[0] + 10)
        capture = cv2.VideoCapture(video_row["path"])
        try:
            for row, frame in enumerate(frames):
                y = 63 + row * 229
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise ValueError(
                        f"Cannot decode {clip}/{video_row['camera_id']} frame {frame}"
                    )
                status = int(positions.status[camera_index, frame])
                canvas[y : y + SIZE[1], x : x + SIZE[0]] = show_points(
                    image,
                    positions.uv_px[camera_index, frame],
                    projected[camera_index, frame],
                    status,
                )
                error = ball.trajectory.reprojection_px[camera_index, frame]
                text(
                    canvas,
                    f"{video_row['camera_id']}   {STATUSES[status]}   {error:.1f} px",
                    (x, y + SIZE[1] + 16),
                    STATUS_COLORS[status],
                )
        finally:
            capture.release()
    text(canvas, "Circle   External position (color by status)", (12, height - 15))
    text(canvas, "Cross   Projected 3D position", (396, height - 15), PROJECTED_COLOR)
    text(canvas, "Errors use original image pixels", (776, height - 15))
    path = output / f"reprojection-{clip}.png"
    Image.fromarray(canvas).quantize(colors=192).save(path, optimize=True)
    print(clip, frames, path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("clips", nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        make_figure(args.dataset, "video_000", clip, args.output)


if __name__ == "__main__":
    main()
