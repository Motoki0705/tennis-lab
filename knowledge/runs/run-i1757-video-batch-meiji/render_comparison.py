"""Render selected source frames and the saved scene observations on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack, closing
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd() / "knowledge/runs/run-i1757-video-batch-meiji"))

from benchmark import verify_dataset  # noqa: E402

from src.tennis_scene.archive import load_scene_result  # noqa: E402
from src.tennis_scene.generate_dataset.manifest import ClipManifest  # noqa: E402
from src.tennis_scene.pipeline.dataset_publication import (
    validate_dataset_result,  # noqa: E402
)
from src.tennis_scene.schema import SceneResult  # noqa: E402
from src.utils.geometry.triangulation import PointRejection  # noqa: E402
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH  # noqa: E402
from src.utils.schema.player import COCO17_SKELETON  # noqa: E402
from src.utils.video import OpenCVVideoFrameReader, VideoWriter  # noqa: E402

WIDTH, HEIGHT = 360, 202
COLORS = [(245, 180, 45), (90, 180, 255), (125, 230, 110), (225, 120, 245)]
BALL = (36, 232, 255)


def text(image, value, x, y, color=(245, 245, 245)):
    cv2.putText(
        image,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA
    )


def camera_overlay(image, scene: SceneResult, camera: str, frame: int):
    view = scene.metadata["camera_ids"].index(camera)
    panel = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    for points, confidence, color in zip(
        scene.human_kp_2d[:, view, frame],
        scene.human_kp_vis[:, view, frame],
        COLORS,
        strict=True,
    ):
        coords = np.rint(points * [WIDTH, HEIGHT]).astype(int)
        for a, b in COCO17_SKELETON:
            if confidence[a] > 0 and confidence[b] > 0:
                cv2.line(
                    panel, tuple(coords[a]), tuple(coords[b]), color, 1, cv2.LINE_AA
                )
    if scene.ball_vis[view, frame]:
        center = tuple(
            np.rint(scene.ball_uv[view, frame] * [WIDTH, HEIGHT]).astype(int)
        )
        cv2.circle(panel, center, 8, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.circle(panel, center, 8, BALL, 2, cv2.LINE_AA)
    text(panel, camera, 8, 20)
    return panel


def court_view(scene: SceneResult, frame: int):
    view = np.full((108, 194, 3), (40, 64, 40), dtype=np.uint8)

    def project(point):
        return round(97 + 6 * point[1]), round(58 + 6 * point[0])

    def bounds(y):
        return project((-HALF_DOUBLES_WIDTH, y)), project((HALF_DOUBLES_WIDTH, y))

    start, stop = bounds(0)
    cv2.rectangle(
        view, bounds(-HALF_LENGTH)[0], bounds(HALF_LENGTH)[1], (145, 145, 145), 1
    )
    cv2.line(view, start, stop, (145, 145, 145), 1)
    for player, color in enumerate(COLORS):
        points = scene.player_kp_3d[player, frame]
        valid = scene.player_kp_3d_vis[player, frame]
        for a, b in COCO17_SKELETON:
            if valid[a] and valid[b]:
                cv2.line(
                    view, project(points[a]), project(points[b]), color, 1, cv2.LINE_AA
                )
        if scene.player_valid[player, frame]:
            cv2.circle(
                view, project(scene.player_position[player, frame]), 2, color, -1
            )
    if scene.ball_3d_valid[frame]:
        cv2.circle(view, project(scene.ball_3d[frame]), 3, BALL, -1)
    text(view, "Court XY", 6, 17)
    return view


def condition_panel(
    images, scene: SceneResult, cameras, frame: int, label: str, selected: bool
):
    panel = np.full((2 * HEIGHT + 148, 2 * WIDTH, 3), 28, dtype=np.uint8)
    text(
        panel,
        f"{label}  |  source frame {frame}",
        10,
        22,
        BALL if selected else (245, 245, 245),
    )
    for view, (image, camera) in enumerate(zip(images, cameras, strict=True)):
        row, column = divmod(view, 2)
        y, x = 32 + row * HEIGHT, column * WIDTH
        panel[y : y + HEIGHT, x : x + WIDTH] = camera_overlay(
            image, scene, camera, frame
        )
    if scene.ball_3d_valid[frame]:
        x, y, z = scene.ball_3d[frame]
        ball = f"Ball 3D (m)  {x:+.3f}, {y:+.3f}, {z:+.3f}"
    else:
        reason = PointRejection(scene.ball_rejection_code[frame]).name
        ball = f"Ball 3D missing  {reason}"
    text(
        panel,
        f"{ball}  |  {scene.ball_vis[:, frame].sum()} 2D views",
        10,
        2 * HEIGHT + 56,
        BALL,
    )
    text(
        panel,
        f"Players  {scene.player_valid[:, frame].sum()}/4 valid roots, "
        f"{scene.player_kp_3d_vis[:, frame].sum()}/68 valid 3D joints",
        10,
        2 * HEIGHT + 78,
    )
    panel[2 * HEIGHT + 36 : 2 * HEIGHT + 144, 518:712] = court_view(scene, frame)
    return panel


def write_video(path, clip, record, scenes, labels, targets):
    frames = {
        frame
        for center in targets
        for frame in range(max(0, center - 5), min(clip.num_frames, center + 6))
    }
    if not frames or any(not 0 <= frame < clip.num_frames for frame in targets):
        raise ValueError("Select frames from the clip's source timeline")
    for scene in scenes:
        validate_dataset_result(scene, record)
        if scene.schema_version != 2 or len(scene.player_position) != 4:
            raise ValueError("Expected a published v2 doubles scene")
        if set(scene.metadata["camera_ids"]) != set(clip.camera_ids):
            raise ValueError("Archive and source camera IDs differ")
    partial = path.with_suffix(".partial.mp4")
    try:
        with ExitStack() as stack:
            readers = [
                stack.enter_context(
                    closing(
                        iter(
                            OpenCVVideoFrameReader(
                                clip.media_path(camera), max_frames=max(frames) + 1
                            )
                        )
                    )
                )
                for camera in clip.camera_ids
            ]
            writer = stack.enter_context(VideoWriter(partial, fps=clip.fps / 3, crf=19))
            frame = -1
            for packets in zip(*readers, strict=True):
                frame = packets[0].index
                if any(packet.index != frame for packet in packets):
                    raise ValueError("Camera frame indices disagree")
                if frame not in frames:
                    continue
                images = [packet.frame for packet in packets]
                panels = [
                    condition_panel(
                        images, scene, clip.camera_ids, frame, label, frame in targets
                    )
                    for scene, label in zip(scenes, labels, strict=True)
                ]
                writer.write_frame(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB))
            if frame != max(frames):
                raise ValueError("Source video ended before the final selected frame")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path, required=True, help="JSON from report.py compare"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Frozen three-clip input"
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument(
        "--clip", required=True, help="Full clip ID, e.g. video_000/clip_027"
    )
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    comparison = json.loads(args.comparison.read_text())
    if not comparison["same_input_pins"]:
        raise ValueError("CPU review requires matching pinned input")
    row = next(row for row in comparison["clips"] if row["clip_id"] == args.clip)
    if not row["both_scenes_validated"]:
        raise ValueError("Both published scenes must have passed reload validation")
    record = verify_dataset(args.dataset.resolve()).clips[args.clip]
    clip = ClipManifest.load(args.dataset / record.path)
    scenes = [load_scene_result(item["scene"]["archive"]) for item in row["evidence"]]
    write_video(
        args.output,
        clip,
        record,
        scenes,
        (args.reference_label, args.candidate_label),
        set(args.frames),
    )


if __name__ == "__main__":
    main()
