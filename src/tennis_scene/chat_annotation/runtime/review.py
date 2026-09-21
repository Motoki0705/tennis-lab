"""Deterministic qualitative review artifacts and the fixed final response."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import (
    Annotation,
    ClipManifest,
    ValidationReport,
    read_json,
    sha256_file,
    write_json,
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


def render_review(
    video: Path,
    manifest: ClipManifest,
    annotation: Annotation,
    definition: dict[str, Any],
    output: Path,
) -> None:
    timeline = check_clip(video, manifest)
    frames = {frame.frame_index: frame for frame in annotation.frames}
    samples = {sample.frame_index: sample for sample in annotation.court_samples}
    target = [f.frame_index for f in manifest.frames if f.is_target]
    flagged = [
        f.frame_index
        for f in annotation.frames
        if any(
            ball.status in {"occluded", "interpolated"} or ball.center_px is None
            for ball in f.balls
        )
        or any(person.kind == "unknown" for person in f.people)
    ]
    uniform = [
        target[int(i)] for i in np.linspace(0, len(target) - 1, min(6, len(target)))
    ]
    selected = sorted(set(uniform + flagged[:6]))
    sheets: dict[int, NDArray[np.uint8]] = {}
    footer_height = 120

    def decorated() -> Iterator[tuple[av.VideoFrame, int, int]]:
        for index, decoded in enumerate(
            decode_range(video, timeline, 0, len(timeline.pts))
        ):
            image: NDArray[np.uint8] = decoded.to_ndarray(format="bgr24").astype(
                np.uint8, copy=False
            )
            row = frames.get(index)
            if row is not None:
                for region in row.ignore_regions:
                    ix1, iy1, ix2, iy2 = map(round, region.bbox_xyxy)
                    cv2.rectangle(image, (ix1, iy1), (ix2, iy2), (128, 128, 128), 1)
                    cv2.putText(
                        image,
                        "IGNORE crowd",
                        (ix1, max(12, iy1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (180, 180, 180),
                        1,
                    )
                for person in row.people:
                    if person.bbox_xyxy is None:
                        continue
                    x1, y1, x2, y2 = person.bbox_xyxy
                    color = (
                        (255, 255, 0)
                        if person.kind == "player"
                        else (0, 165, 255)
                        if person.kind == "non_player"
                        else (180, 180, 180)
                    )
                    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
                        _line(image, a, b, color, person.bbox_source == "inferred")
                    text = f"{person.track_id}:{person.non_player_role or person.kind}"
                    position = (
                        round(max(0, min(manifest.width - 1, x1))),
                        round(max(14, min(manifest.height - 1, y1))),
                    )
                    cv2.putText(
                        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                    )
                for ball in row.balls:
                    if ball.center_px is None or ball.status is None:
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
                if row.court_reference_frame is not None:
                    sample = samples[row.court_reference_frame]
                    reused = sample.frame_index != index
                    for a, b in definition["skeleton"]:
                        first, last = sample.points[a], sample.points[b]
                        if first.point_px is not None and last.point_px is not None:
                            derived = (
                                reused
                                or first.source != "observed"
                                or last.source != "observed"
                            )
                            _line(
                                image,
                                first.point_px,
                                last.point_px,
                                (80, 220, 80),
                                derived,
                            )
                    for point in sample.points:
                        if (
                            point.point_px is not None
                            and 0 <= point.point_px[0] < manifest.width
                            and 0 <= point.point_px[1] < manifest.height
                        ):
                            cv2.putText(
                                image,
                                str(point.index),
                                tuple(map(round, point.point_px)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (80, 220, 80),
                                1,
                            )
            canvas = cv2.copyMakeBorder(
                image, 0, footer_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            ).astype(np.uint8, copy=False)
            mapped = manifest.frames[index]
            seconds = (mapped.source_pts - manifest.source_start_pts) * Fraction(
                manifest.time_base
            )
            caption = f"{'TARGET' if mapped.is_target else 'CONTEXT ONLY'} frame={index} source={mapped.source_frame_index} t={float(seconds):.6f}s"
            coverage = (
                f"review P/B/C={row.people_review}/{row.balls_review}/{row.court_review}"
                if row
                else "reference interval; no target annotations"
            )
            for line, text in enumerate(
                [
                    caption,
                    f"court={annotation.court_mode}; {coverage}",
                    "ball: visible=yellow, occluded=orange, interpolated=magenta",
                    "bbox: player=cyan, non-player=orange; dashed=derived/reused",
                    "gray=unknown/ignore; null coordinates are not plotted",
                ]
            ):
                cv2.putText(
                    canvas,
                    text,
                    (5, manifest.height + 19 + 24 * line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (255, 255, 255),
                    1,
                )
            if index in selected:
                sheets[index] = canvas.copy()
            yield (
                av.VideoFrame.from_ndarray(canvas, format="bgr24"),
                mapped.clip_pts,
                mapped.duration_pts,
            )

    encode_video(
        output / "overlay.mp4",
        width=manifest.width,
        height=manifest.height + footer_height,
        time_base=Fraction(manifest.time_base),
        rate=Fraction(manifest.nominal_fps),
        frames=decorated(),
        crf=18,
        preset="medium",
    )
    tile_width = min(960, max(480, manifest.width))
    tile_height = (
        round((manifest.height + footer_height) * tile_width / manifest.width) + 26
    )
    sheet: NDArray[np.uint8] = np.zeros(
        (((len(selected) + 2) // 3) * tile_height, 3 * tile_width, 3), dtype=np.uint8
    )
    for cell, index in enumerate(selected):
        resized = cv2.resize(
            sheets[index], (tile_width, tile_height - 26), interpolation=cv2.INTER_AREA
        )
        x, y = cell % 3 * tile_width, cell // 3 * tile_height
        sheet[y : y + tile_height - 26, x : x + tile_width] = resized
        cv2.putText(
            sheet,
            f"SAMPLE frame {index} / source {manifest.frames[index].source_frame_index}",
            (x + 5, y + tile_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    if not cv2.imwrite(
        str(output / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise OSError("could not write contact sheet")
    write_json(
        output / "review_manifest.json",
        {
            "contact_sheet_frames": selected,
            "contact_sheet_is_sampled": True,
            "overlay_frames": len(manifest.frames),
            "court_reuse_does_not_claim_current_visibility": True,
        },
    )


def final_response(
    manifest: ClipManifest, report: ValidationReport, artifact: str
) -> str:
    target = [frame for frame in manifest.frames if frame.is_target]
    start = (target[0].source_pts - manifest.source_start_pts) * Fraction(
        manifest.time_base
    )
    stop = (
        target[-1].source_pts + target[-1].duration_pts - manifest.source_start_pts
    ) * Fraction(manifest.time_base)
    source = manifest.source.url or f"local:{manifest.source.filename}"
    lines = (
        f"状態: {report.status}",
        f"入力: {manifest.filename}",
        f"元動画: {source} / {float(start):.6f}–{float(stop):.6f}秒",
        f"処理: {report.reviewed_frames}/{report.target_frames}フレーム、要確認{len(report.errors) + len(report.issues)}件",
        f"成果物: {artifact}",
    )
    return "\n".join(" ".join(line.splitlines()) for line in lines) + "\n"


def finalize(
    video: Path,
    manifest_path: Path,
    annotation_path: Path,
    kit_root: Path,
    output: Path,
    definition: dict[str, Any],
    download_base: str | None,
) -> tuple[Path, ValidationReport]:
    manifest = ClipManifest.model_validate(read_json(manifest_path))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "clip_manifest.json")
    shutil.copyfile(annotation_path, output / "annotations.json")
    shutil.copyfile(kit_root / "kit_manifest.json", output / "kit_manifest.json")
    annotation: Annotation | None = None
    try:
        check_clip(video, manifest)
        annotation = Annotation.model_validate(read_json(annotation_path))
        report = validate_annotation(
            annotation, manifest, sha256_file(manifest_path), definition
        )
        if not report.errors:
            render_review(video, manifest, annotation, definition, output)
    except Exception as error:
        report = ValidationReport(
            status="failed",
            reviewed_frames=0,
            target_frames=sum(frame.is_target for frame in manifest.frames),
            errors=[f"{type(error).__name__}: {error}"],
            issues=[],
        )
    write_json(output / "validation_report.json", report.model_dump(mode="json"))
    write_json(
        output / "provenance.json",
        {
            "source": manifest.source.model_dump(mode="json"),
            "clip_id": manifest.clip_id,
            "input_manifest_sha256": sha256_file(manifest_path),
            "kit_id": manifest.kit_id,
            "kit_version": manifest.kit_version,
            "teacher": annotation.teacher if annotation else None,
            "annotation_is_human_verified": False,
        },
    )
    (output / "issues.txt").write_text(
        "\n".join(report.errors + report.issues) + "\n", encoding="utf-8"
    )
    archive = (
        output / f"{manifest.source.source_id}__{manifest.clip_id}__annotations.zip"
    )
    link = (
        f"{download_base.rstrip('/')}/{archive.name}"
        if download_base
        else f"sandbox:{archive.resolve().as_posix()}"
    )
    response = final_response(manifest, report, f"[{archive.name}]({link})")
    (output / "FINAL_RESPONSE.txt").write_text(response, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(output.iterdir()):
            if file != archive and file.is_file():
                bundle.write(file, file.name)
    return archive, report
