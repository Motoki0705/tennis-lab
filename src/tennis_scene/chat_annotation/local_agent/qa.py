"""Orchestrator QA for ball annotations: metrics, suspicious-frame sheets, review videos.

  qa.py metrics <annotation.json> --manifest M          # JSON metrics + flagged frames
  qa.py task <task_id> [--sheets]                        # metrics for a campaign task's final attempt
  qa.py video <annotation.json> --manifest M --out X.mp4 [--scale 0.5]

Metrics are heuristics for choosing what to look at; they never accept or reject by themselves.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.chat_annotation.runtime.contracts import (
    BallAnnotation,
    ClipManifest,
)
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from .common import iter_frames, load_annotation, load_manifest, locate_video
from .configuration import paths
from .path_contracts import campaign_resolver, validate_command_paths

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)


STATUS_BGR = {
    "visible": (0, 255, 255),
    "occluded": (0, 140, 255),
    "interpolated": (255, 0, 255),
}


def seconds(manifest: ClipManifest, index: int) -> float:
    return float(manifest.frames[index].clip_pts * Fraction(manifest.time_base))


def ball_metrics(annotation: BallAnnotation, manifest: ClipManifest) -> dict[str, Any]:
    report = validate_annotation(annotation, manifest)
    n = len(annotation.frames)
    status_count: dict[str, int] = {}
    positions: dict[str, dict[int, tuple[float, float]]] = {}
    unresolved_frames = []
    multi = []
    for frame in annotation.frames:
        if len(frame.balls) > 1:
            multi.append(frame.frame_index)
        for ball in frame.balls:
            status_count[ball.status] = status_count.get(ball.status, 0) + 1
            if ball.status == "unresolved":
                unresolved_frames.append(frame.frame_index)
            if ball.center_px is not None:
                positions.setdefault(ball.track_id, {})[frame.frame_index] = tuple(
                    ball.center_px
                )
    breaks = {f.frame_index for f in annotation.frames if f.interpolation_break}
    speed_flags, accel_flags, isolated = [], [], []
    for track, points in positions.items():
        keys = sorted(points)
        for a, b in zip(keys, keys[1:], strict=False):
            if b - a != 1:
                continue
            dt = seconds(manifest, b) - seconds(manifest, a)
            d = math.dist(points[a], points[b])
            if dt > 0 and d / dt > 3500:
                speed_flags.append(
                    {"track": track, "frames": [a, b], "px_per_s": round(d / dt)}
                )
        for a, b, c in zip(keys, keys[1:], keys[2:], strict=False):
            if (
                not (b - a == 1 and c - b == 1)
                or b in breaks
                or a in breaks
                or c in breaks
            ):
                continue
            ax = points[a][0] - 2 * points[b][0] + points[c][0]
            ay = points[a][1] - 2 * points[b][1] + points[c][1]
            if math.hypot(ax, ay) > 25:
                accel_flags.append(
                    {
                        "track": track,
                        "frame": b,
                        "second_diff_px": round(math.hypot(ax, ay), 1),
                    }
                )
        for k in keys:
            if k - 1 not in points and k + 1 not in points:
                isolated.append({"track": track, "frame": k})
    with_ball = sum(1 for f in annotation.frames if f.balls)
    return {
        "status_field": annotation.status,
        "validation_status": report.status,
        "errors": report.errors[:5],
        "frames": n,
        "reviewed": report.reviewed_frames,
        "frames_with_ball": with_ball,
        "status_count": status_count,
        "unresolved_ratio_of_ball_frames": len(unresolved_frames) / with_ball
        if with_ball
        else 0.0,
        "tracks": {t: [min(p), max(p), len(p)] for t, p in positions.items()},
        "break_frames": len(breaks),
        "frames_with_notes": sum(1 for f in annotation.frames if f.notes),
        "clip_issues": annotation.issues[:5],
        "multi_ball_frames": multi[:20],
        "speed_flags": speed_flags[:30],
        "accel_flags": accel_flags[:30],
        "isolated_points": isolated[:30],
    }


def review_video(
    annotation: BallAnnotation, manifest: ClipManifest, out: Path, scale: float
) -> None:
    """Human review render: ring + label + 12-frame trail; nominal-rate MP4 for viewing only."""
    video = locate_video(manifest)
    rows = {f.frame_index: f for f in annotation.frames}
    trail: list[tuple[int, tuple[float, float]]] = []
    width = int(round(manifest.width * scale / 2) * 2)
    height = int(round(manifest.height * scale / 2) * 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(manifest.nominal_fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for index, image in iter_frames(video, manifest, 0, len(manifest.frames)):
            row = rows[index]
            canvas = image.copy()
            for ball in row.balls:
                if ball.center_px is None:
                    continue
                trail.append((index, (ball.center_px[0], ball.center_px[1])))
            trail = [(i, p) for i, p in trail if index - i < 12]
            for _i, p in trail:
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, cv2.LINE_AA
                )
            for ball in row.balls:
                if ball.center_px is None:
                    continue
                c = (int(round(ball.center_px[0])), int(round(ball.center_px[1])))
                color = STATUS_BGR.get(ball.status, (255, 255, 255))
                cv2.circle(canvas, c, 14, color, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"{ball.track_id} {ball.status}",
                    (c[0] + 18, c[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            state = (
                "UNREVIEWED"
                if not row.reviewed
                else ("NOTE: " + row.notes[:70] if row.notes else "reviewed")
            )
            states = (
                ",".join(f"{b.track_id}:{b.status}" for b in row.balls)
                or "no active ball"
            )
            text = f"frame {index}/{len(manifest.frames) - 1}  {states}  {'BREAK ' if row.interpolation_break else ''}{state}"
            cv2.rectangle(canvas, (0, 0), (manifest.width, 48), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                text,
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
            frame = av.VideoFrame.from_ndarray(small, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def flagged_sheet(
    annotation_path: Path, manifest: ClipManifest, frames: list[int], out: Path
) -> None:
    video = locate_video(manifest)
    annotation = load_annotation(annotation_path)
    rows = {f.frame_index: f for f in annotation.frames}
    wanted = sorted(set(frames))
    tiles = []
    if not wanted:
        return
    lookup = set(wanted)
    for index, image in iter_frames(video, manifest, wanted[0], wanted[-1] + 1):
        if index not in lookup:
            continue
        balls = [b for b in rows[index].balls if b.center_px is not None]
        cx, cy = (
            balls[0].center_px if balls else (manifest.width / 2, manifest.height / 2)
        )
        x0 = int(max(0, min(manifest.width - 160, cx - 80)))
        y0 = int(max(0, min(manifest.height - 160, cy - 80)))
        crop = cv2.resize(
            image[y0 : y0 + 160, x0 : x0 + 160],
            (320, 320),
            interpolation=cv2.INTER_CUBIC,
        )
        for b in balls:
            u, v = int((b.center_px[0] - x0) * 2), int((b.center_px[1] - y0) * 2)
            cv2.circle(
                crop,
                (u, v),
                16,
                STATUS_BGR.get(b.status, (255, 255, 255)),
                1,
                cv2.LINE_AA,
            )
        label: NDArray[np.uint8] = np.zeros((18, 320, 3), dtype=np.uint8)
        text = f"f{index} " + ",".join(
            f"{b.track_id}:{b.status[:3]}" for b in rows[index].balls
        )
        cv2.putText(
            label,
            text[:40],
            (3, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        tiles.append(np.vstack([label, crop]))
    cols = 5
    rows_n = math.ceil(len(tiles) / cols)
    sheet: NDArray[np.uint8] = np.full(
        (rows_n * 342, cols * 324, 3), 40, dtype=np.uint8
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 342 : r * 342 + 338, c * 324 : c * 324 + 320] = tile
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def task_paths(task_id: str) -> tuple[Path, Path]:
    state = json.loads(
        (paths().campaign_dir / "state.json").read_text(encoding="utf-8")
    )
    task = state["tasks"][task_id]
    for record in reversed(task["attempts"]):
        path = Path(record["dir"]) / f"annotation_{task['clip_id']}.json"
        if path.exists():
            return path, Path(task["manifest"])
    raise FileNotFoundError(f"no annotation for {task_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("metrics")
    p.add_argument("annotation", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("task")
    p.add_argument("task_id")
    p.add_argument(
        "--sheets", action="store_true", help="flagged + random sample crop sheets"
    )
    p.add_argument("--video", action="store_true")
    p = sub.add_parser("video")
    p.add_argument("annotation", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args(argv)
    PATH_BOUNDARY.validate({"campaign": paths().campaign_dir}, resolver=campaign_resolver(paths()))
    validate_command_paths()
    if args.command == "task":
        annotation_path, manifest_path = task_paths(args.task_id)
    else:
        checked = validate_command_paths(annotation=args.annotation, manifest=args.manifest)
        annotation_path, manifest_path = checked['annotation'], checked['manifest']
    if args.command == 'video':
        args.out = validate_command_paths(output=args.out)['output']
    manifest = load_manifest(manifest_path)
    annotation = load_annotation(annotation_path)
    if not isinstance(annotation, BallAnnotation):
        raise SystemExit("qa.py handles ball annotations only")
    if args.command == "video":
        review_video(annotation, manifest, args.out, args.scale)
        print(args.out)
        return 0
    metrics = ball_metrics(annotation, manifest)
    if args.command == "task":
        qa_dir = paths().campaign_dir / "qa" / args.task_id
        if args.sheets:
            flagged = (
                [f["frames"][1] for f in metrics["speed_flags"]]
                + [f["frame"] for f in metrics["accel_flags"]]
                + [f["frame"] for f in metrics["isolated_points"]]
            )
            flagged_sheet(
                annotation_path, manifest, flagged[:40], qa_dir / "flagged.jpg"
            )
            visible = [
                f.frame_index
                for f in annotation.frames
                if any(b.center_px is not None for b in f.balls)
            ]
            random.seed(0)
            sample = (
                sorted(random.sample(visible, min(30, len(visible)))) if visible else []
            )
            flagged_sheet(annotation_path, manifest, sample, qa_dir / "sample.jpg")
            metrics["sheets"] = [
                str(qa_dir / "flagged.jpg"),
                str(qa_dir / "sample.jpg"),
            ]
        if args.video:
            out = qa_dir / "review.mp4"
            review_video(annotation, manifest, out, 0.5)
            metrics["video"] = str(out)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0
