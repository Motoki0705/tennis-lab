"""Render the annotation faithfully and package exactly two deliverables."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import (
    Annotation,
    ClipManifest,
    ValidationReport,
    annotation_clip_id,
    read_json,
)
from .media import check_clip, decode_range, encode_video
from .validation import validate_annotation

BALL_COLORS = {
    "visible": (0, 255, 255),
    "occluded": (0, 140, 255),
    "interpolated": (255, 0, 255),
}


def _line(
    image: NDArray[np.uint8],
    a: list[float],
    b: list[float],
    color: tuple[int, int, int],
    dashed: bool,
) -> None:
    first = tuple(int(max(-1_000_000, min(1_000_000, x))) for x in a)
    last = tuple(int(max(-1_000_000, min(1_000_000, x))) for x in b)
    height, width = image.shape[:2]
    accepted, start, stop = cv2.clipLine((0, 0, width, height), first, last)
    if not accepted:
        return
    if not dashed:
        cv2.line(image, start, stop, color, 1, cv2.LINE_AA)
        return
    distance = float(np.linalg.norm(np.asarray(stop) - start))
    steps = max(1, int(distance / 7))
    for index in range(0, steps, 2):
        p = tuple(
            round(start[j] + (stop[j] - start[j]) * index / steps) for j in range(2)
        )
        q = tuple(
            round(start[j] + (stop[j] - start[j]) * min(index + 1, steps) / steps)
            for j in range(2)
        )
        cv2.line(image, p, q, color, 1, cv2.LINE_AA)


def render_overlay(
    video: Path, manifest: ClipManifest, annotation: Annotation, output: Path
) -> None:
    timeline = check_clip(video, manifest)
    footer_height = 80

    def decorated() -> Iterator[tuple[av.VideoFrame, int, int]]:
        for index, decoded in enumerate(
            decode_range(video, timeline, 0, len(timeline.pts))
        ):
            image: NDArray[np.uint8] = decoded.to_ndarray(format="bgr24").astype(
                np.uint8, copy=False
            )
            row = annotation.frames[index]
            for player in row.players:
                if player.bbox_xyxy is None:
                    continue
                x1, y1, x2, y2 = player.bbox_xyxy
                color = (255, 255, 0)
                corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
                    _line(image, a, b, color, player.bbox_source == "inferred")
                position = (
                    round(max(0, min(manifest.width - 1, x1))),
                    round(max(14, min(manifest.height - 1, y1))),
                )
                cv2.putText(
                    image,
                    player.track_id,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
            for ball in row.balls:
                if ball.center_px is None:
                    continue
                center = tuple(map(round, ball.center_px))
                color = BALL_COLORS[ball.status]
                cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 13, 1)
                cv2.putText(
                    image,
                    f"{ball.track_id}:{ball.status}",
                    (center[0] + 7, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            mapped = manifest.frames[index]
            timestamp = float(mapped.clip_pts * Fraction(manifest.time_base))
            unresolved = (
                bool(row.notes)
                or any(p.bbox_xyxy is None for p in row.players)
                or any(
                    b.center_px is None and b.status != "out_of_frame"
                    for b in row.balls
                )
            )
            state = (
                "UNREVIEWED"
                if not row.reviewed
                else ("REVIEWED / UNRESOLVED" if unresolved else "REVIEWED")
            )
            canvas: NDArray[np.uint8] = np.zeros(
                (manifest.height + footer_height, manifest.width, 3), dtype=np.uint8
            )
            canvas[: manifest.height] = image
            labels = (
                f"frame {index}/{len(manifest.frames) - 1} | {timestamp:.3f}s | {annotation.status} | {state}",
                "Player: solid=observed dashed=inferred | Ball: yellow=visible",
                "Ball: orange=occluded magenta=interpolated | null is not drawn",
            )
            for number, text in enumerate(labels):
                cv2.putText(
                    canvas,
                    text,
                    (6, manifest.height + 20 + 23 * number),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            yield (
                av.VideoFrame.from_ndarray(canvas, format="bgr24"),
                mapped.clip_pts,
                mapped.duration_pts,
            )

    encode_video(
        output,
        width=manifest.width,
        height=manifest.height + footer_height,
        time_base=Fraction(manifest.time_base),
        rate=Fraction(manifest.nominal_fps),
        frames=decorated(),
        crf=18,
        preset="medium",
    )


def finalize(
    video: Path, manifest_path: Path, annotation_path: Path, output: Path
) -> tuple[Path, ValidationReport]:
    manifest = ClipManifest.model_validate(read_json(manifest_path))
    annotation = Annotation.model_validate(read_json(annotation_path))
    report = validate_annotation(annotation, manifest)
    if report.errors:
        raise ValueError("annotation validation failed: " + "; ".join(report.errors))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    clip_id = annotation_clip_id(manifest)
    annotation_name = f"annotation_{clip_id}.json"
    overlay_name = f"overlay_{clip_id}.mp4"
    archive_name = f"annotation_{clip_id}.zip"
    # Publish only after both deliverables and the ZIP have succeeded.
    with tempfile.TemporaryDirectory(
        prefix=".annotation-", dir=output.parent
    ) as staging:
        temporary = Path(staging)
        render_overlay(video, manifest, annotation, temporary / overlay_name)
        shutil.copyfile(annotation_path, temporary / annotation_name)
        with zipfile.ZipFile(
            temporary / archive_name, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for name in (overlay_name, annotation_name):
                bundle.write(temporary / name, name)
        temporary.rename(output)
    return output / archive_name, report
