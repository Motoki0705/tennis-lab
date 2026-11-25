"""Render synchronized SMPL-reference and model poses on the same source frames."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from fixed_inputs import load_cohort, read_video, row_keys
from numpy.typing import NDArray
from replay_accuracy import read_condition, row_columns

from src.tennis_scene.evaluation.player_pose.bundle import load_manifest
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, ReferenceFrame
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.writer import VideoWriter

COLORS = ((42, 204, 255), (125, 223, 83), (248, 166, 88), (250, 121, 207))
PANEL_WIDTH, PANEL_HEIGHT = 640, 360
CAPTION_HEIGHT = 32


def draw_panel(
    frame: NDArray[np.uint8],
    boxes: NDArray[np.floating],
    player_ids: NDArray[np.integer],
    xy: NDArray[np.floating],
    mask: NDArray[np.bool_],
    region: tuple[int, int, int, int],
    caption: str,
) -> NDArray[np.uint8]:
    left, top, right, bottom = region
    crop = frame[top:bottom, left:right]
    scale = min(PANEL_WIDTH / crop.shape[1], PANEL_HEIGHT / crop.shape[0])
    width = min(PANEL_WIDTH, max(1, round(crop.shape[1] * scale)))
    height = min(PANEL_HEIGHT, max(1, round(crop.shape[0] * scale)))
    offset = np.array(
        [(PANEL_WIDTH - width) // 2, (PANEL_HEIGHT - height) // 2 + CAPTION_HEIGHT]
    )
    x, y = (int(value) for value in offset)
    panel: NDArray[np.uint8] = np.full(
        (PANEL_HEIGHT + CAPTION_HEIGHT, PANEL_WIDTH, 3), 18, np.uint8
    )
    panel[y : y + height, x : x + width] = cv2.resize(
        crop, (width, height), interpolation=cv2.INTER_LINEAR
    )
    origin = np.array([left, top])
    scale_xy = np.array([width / crop.shape[1], height / crop.shape[0]])
    for player, box, coords, valid in zip(player_ids, boxes, xy, mask, strict=True):
        color = COLORS[int(player) % len(COLORS)]
        border = np.rint((box.reshape(2, 2) - origin) * scale_xy + offset)
        a, b = (
            np.clip(border, offset, offset + [width - 1, height - 1])
            .astype(int)
            .tolist()
        )
        cv2.rectangle(panel, a, b, color, 1)
        cv2.putText(
            panel,
            str(player),
            (a[0], min(b[1], a[1] + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        inside = valid & (coords >= origin).all(-1) & (coords < [right, bottom]).all(-1)
        points = (
            np.rint(
                (np.where(inside[:, None], coords, origin) - origin) * scale_xy + offset
            )
            .astype(int)
            .tolist()
        )
        for u, v in COCO17_SKELETON:
            if inside[u] and inside[v]:
                cv2.line(panel, points[u], points[v], color, 2, cv2.LINE_AA)
        for point, keep in zip(points, inside, strict=True):
            if keep:
                cv2.circle(panel, point, 3, color, -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        caption,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return panel


def frame_slices(
    references: dict[FrameKey, ReferenceFrame],
) -> dict[FrameKey, slice]:
    start = 0
    result = {}
    for key in sorted(references):
        end = start + len(references[key].player_ids)
        result[key] = slice(start, end)
        start = end
    return result


def _sheet(
    image: NDArray[np.uint8],
    key: FrameKey,
    bounds: slice,
    reference: ReferenceFrame,
    cases: dict[str, NDArray[np.float32]],
    player_id: int | None,
) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    keep: NDArray[np.bool_] = np.ones(len(reference.player_ids), bool)
    region = (0, 0, width, height)
    if player_id is not None:
        keep = reference.player_ids == player_id
        if not keep.any():
            raise ValueError(f"Player {player_id} has no fixed box in {key}")
        x0, y0, x1, y1 = reference.boxes_xyxy[keep][0]
        margin = max(x1 - x0, y1 - y0) * 0.5
        region = (
            max(0, math.floor(x0 - margin)),
            max(0, math.floor(y0 - margin)),
            min(width, math.ceil(x1 + margin)),
            min(height, math.ceil(y1 + margin)),
        )
    panels = [
        ("SMPL projection", reference.keypoints_xy[keep], reference.joint_valid[keep])
    ]
    for label, points in cases.items():
        joints = points[bounds][keep]
        panels.append((label, joints[..., :2], np.isfinite(joints).all(-1)))
    clip = key.clip_id.rsplit("/", 1)[-1]
    tiles = [
        draw_panel(
            image,
            reference.boxes_xyxy[keep],
            reference.player_ids[keep],
            xy,
            mask,
            region,
            f"{label}  {clip}/{key.camera_id}  frame {key.source_frame}",
        )
        for label, xy, mask in panels
    ]
    columns = min(len(tiles), 3)
    rows = math.ceil(len(tiles) / columns)
    grid = np.full((rows * tiles[0].shape[0], columns * PANEL_WIDTH, 3), 18, np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        grid[
            y * tile.shape[0] : (y + 1) * tile.shape[0],
            x * PANEL_WIDTH : (x + 1) * PANEL_WIDTH,
        ] = tile
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--player", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    suffix = args.output.suffix
    if suffix not in (".png", ".mp4") or (
        (args.frame is not None) != (suffix == ".png")
    ):
        parser.error(
            "Supply a source --frame for PNG, or an MP4 path for the evaluation timeline"
        )
    if args.player is not None and args.frame is None:
        parser.error("Supply a source --frame for the single-player crop")
    if args.output.exists():
        raise FileExistsError(args.output)

    bundle_id, references, videos = load_cohort(
        args.bundle, args.bundle_sha256, args.media_root
    )
    (video,) = (
        video
        for video in videos
        if (video.clip_id, video.camera_id) == (args.clip, args.camera)
    )
    columns = row_columns(references, row_keys(references))
    slices = frame_slices(references)
    cases = {}
    for job in args.jobs:
        (model, backend), batch, _, _ = read_condition(
            args.queue, job, (bundle_id, args.bundle_sha256), columns
        )
        label = f"{model} {backend}"
        if label in cases:
            raise ValueError(f"Repeated overlay panel: {label}")
        cases[label] = batch.keypoints_xyc

    if args.frame is not None:
        if args.frame not in video.frames:
            raise ValueError(f"Frame {args.frame} is outside the fixed cohort")
        video = replace(video, frames=(args.frame,))
    frames = (
        _sheet(image, key, slices[key], references[key], cases, args.player)
        for key, image in read_video(video)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        (sheet,) = frames
        ok, pixels = cv2.imencode(".png", sheet)
        if not ok:
            raise RuntimeError("OpenCV did not encode the synchronized sheet")
        with args.output.open("xb") as handle:
            handle.write(pixels.tobytes())
    else:
        steps = set(np.diff(video.frames).tolist())
        if len(steps) != 1:
            raise ValueError("The video requires a uniformly sampled source timeline")
        step = steps.pop()
        manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
        (clip,) = (
            clip for clip in manifest["clips"] if clip["clip_id"] == video.clip_id
        )
        with TemporaryDirectory(dir=args.output.parent) as staging:
            path = Path(staging) / args.output.name
            with VideoWriter(path, fps=clip["fps"] / step, crf=20) as writer:
                for sheet in frames:
                    writer.write_frame(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
            args.output.hardlink_to(path)


if __name__ == "__main__":
    main()
