"""Export saved player proposals as a CPU video without touching campaign state."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    Document,
    FrameMap,
    FrameRange,
    MediaFingerprint,
)
from src.tasks.base.annotation.media import check_clip, decode_clip, encode_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import file_digest
from src.tasks.base.annotation.predictions import PredictionReceipt

from .contracts import PlayerPredictions
from .inference import PlayerPredictionAdapter


class VideoPreview(Document):
    prediction: ArtifactRef
    context: PredictionReceipt
    video: MediaFingerprint
    width: int
    height: int
    time_base: str
    frames: tuple[FrameMap, ...]
    target_range: FrameRange
    reason: str
    renderer: str = "player-prediction-video.v1"


def export_prediction_video(
    layout: AnnotationLayout,
    prediction: ArtifactRef,
    output: Path,
    *,
    reason: str,
) -> VideoPreview:
    """Write preview.mp4 and provenance.json in a new, separate scratch directory.

    Check receipt, configuration, proposals and native media; model weight files
    are referenced by their frozen hashes, never loaded. No ledger is opened.
    """
    if not reason.strip():
        raise ValueError("preview requires a presentation reason")
    if not output.is_absolute() or output.resolve() != output:
        raise ValueError("preview output must be an absolute nonsymlink path")
    if any(
        output.is_relative_to(root) or root.is_relative_to(output)
        for root in (
            layout.root,
            layout.repository / "data",
            layout.repository / "ckpt",
            layout.repository / "outputs",
        )
    ):
        raise ValueError("preview output must be separate from production artifacts")
    if output.exists():
        raise FileExistsError(output)
    receipt = read_document(layout, prediction, PredictionReceipt)
    context = receipt.context
    manifest = read_document(layout, context.manifest, ClipManifest)
    if (
        context.task != "player"
        or context.model.task != "player"
        or context.clip_id != manifest.clip_id
        or receipt.prediction.task != "player"
        or receipt.prediction.request_id != context.request_id
    ):
        raise ValueError("player preview receipt does not match its frozen inputs")
    PlayerPredictionAdapter().validate(
        layout, context, manifest, receipt.prediction.artifact
    )
    verify_reference(layout, context.model.configuration)
    proposals = read_document(layout, receipt.prediction.artifact, PlayerPredictions)
    media = verify_reference(layout, manifest.media)
    timeline = check_clip(media, manifest)

    def frames() -> Iterator[tuple[av.VideoFrame, int, int]]:
        for index, frame in enumerate(
            decode_clip(media, manifest, 0, len(manifest.frames))
        ):
            bgr = frame.to_ndarray(format="bgr24")
            for item in proposals.rows[index].observations:
                box = item.box
                x1, y1, x2, y2 = (round(v) for v in (box.x1, box.y1, box.x2, box.y2))
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    bgr,
                    f"{item.candidate_index}: {item.confidence:.3f}",
                    (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            mapped = manifest.frames[index]
            scope = (
                "target"
                if manifest.target_range.start <= index < manifest.target_range.stop
                else "context"
            )
            cv2.putText(
                bgr,
                f"UNREVIEWED f{index} source={mapped.source_frame_index} "
                f"pts={mapped.source_pts} {scope} boxes={len(proposals.rows[index].observations)}",
                (4, manifest.height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            rendered = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            yield rendered, mapped.clip_pts, mapped.duration_pts

    output.mkdir(parents=True)
    try:
        video = output / "preview.mp4"
        encode_video(
            video,
            width=manifest.width,
            height=manifest.height,
            time_base=Fraction(manifest.time_base),
            rate=timeline.rate,
            frames=frames(),
            crf=18,
            preset="fast",
        )
        result = VideoPreview(
            prediction=prediction,
            context=receipt,
            video=MediaFingerprint(
                digest=file_digest(video), size_bytes=video.stat().st_size
            ),
            width=manifest.width,
            height=manifest.height,
            time_base=manifest.time_base,
            frames=manifest.frames,
            target_range=manifest.target_range,
            reason=reason,
        )
        (output / "provenance.json").write_bytes(result.canonical_bytes() + b"\n")
        return result
    except BaseException:
        shutil.rmtree(output)
        raise
