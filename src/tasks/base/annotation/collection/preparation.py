"""Explicit reviewed media/target windows to real, timestamp-preserving clips."""

from __future__ import annotations

import os
import tempfile
from bisect import bisect_left, bisect_right
from fractions import Fraction
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from ..contracts import (
    ArtifactRef,
    ClipManifest,
    FrameMap,
    FrameRange,
    SourceIdentity,
    content_digest,
)
from ..media import Timeline, check_clip, decode_range, encode_video, probe_video
from ..paths import AnnotationLayout
from ..plan import file_digest
from .contracts import (
    PreparationConfig,
    PreparationReceipt,
    SelectionReceipt,
    TimeWindow,
)


def window_ranges(
    timeline: Timeline, window: TimeWindow
) -> tuple[FrameRange, FrameRange]:
    boundaries = [timeline.boundary(i) for i in range(len(timeline.pts) + 1)]
    if Fraction(str(window.stop)) > boundaries[-1]:
        raise ValueError("selected window extends beyond the real source")

    def quantize(start: float, stop: float) -> FrameRange:
        first = bisect_left(boundaries, Fraction(str(start)))
        last = bisect_right(boundaries, Fraction(str(stop))) - 1
        if first >= last:
            raise ValueError("selected interval contains no complete real frame")
        return FrameRange(start=first, stop=last)

    media = quantize(window.start, window.stop)
    target = quantize(window.target_start, window.target_stop)
    if not media.start <= target.start < target.stop <= media.stop:
        raise ValueError("quantized target is outside its media window")
    return media, FrameRange(
        start=target.start - media.start, stop=target.stop - media.start
    )


def preparation_identity(
    source: SourceIdentity,
    configuration: ArtifactRef,
    selection: ArtifactRef,
    window: TimeWindow,
) -> str:
    identity: str = content_digest(
        [
            source.sha256,
            configuration.model_dump(),
            selection.model_dump(),
            window.model_dump(),
        ]
    )
    return identity


def prepare_clip(
    layout: AnnotationLayout,
    source: SourceIdentity,
    timeline: Timeline,
    window: TimeWindow,
    configuration: ArtifactRef,
    config: PreparationConfig,
    selection: ArtifactRef,
) -> ArtifactRef:
    media_range, target_range = window_ranges(timeline, window)
    if media_range.stop - media_range.start < config.minimum_clip_frames:
        raise ValueError(
            "selected window has too few real frames; padding is not permitted"
        )
    clip_id = content_digest([source.canonical_id, media_range.model_dump()])
    digest = preparation_identity(source, configuration, selection, window)
    destination = layout.path("clips", clip_id)
    with artifact_lock(layout, "clip-" + clip_id):
        if destination.exists():
            reference = annotation_reference(layout, destination / "manifest.json")
            manifest = read_document(layout, reference, ClipManifest)
            if manifest.preparation_digest != digest or manifest.source != source:
                raise ValueError(
                    "existing clip differs from the immutable preparation inputs"
                )
            if (
                manifest.source_range != media_range
                or manifest.target_range != target_range
            ):
                raise ValueError(
                    "existing clip interval differs from the selected window"
                )
            check_clip(verify_reference(layout, manifest.media), manifest)
            return reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="building-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary)
            video = staging / "media.mp4"
            start, stop = media_range.start, media_range.stop
            source_path = verify_reference(layout, source.media)
            encode_video(
                video,
                width=timeline.width,
                height=timeline.height,
                time_base=timeline.time_base,
                rate=timeline.rate,
                frames=(
                    (
                        frame,
                        timeline.pts[index] - timeline.pts[start],
                        timeline.durations[index],
                    )
                    for index, frame in enumerate(
                        decode_range(source_path, timeline, start, stop), start
                    )
                ),
                crf=config.crf,
                preset=config.preset,
            )
            media = ArtifactRef(
                role="annotation",
                path=(destination / "media.mp4").relative_to(layout.root).as_posix(),
                digest=file_digest(video),
                size_bytes=video.stat().st_size,
            )
            manifest = ClipManifest(
                source=source,
                media=media,
                preparation_digest=digest,
                width=timeline.width,
                height=timeline.height,
                time_base=str(timeline.time_base),
                nominal_fps=str(timeline.rate),
                source_range=media_range,
                target_range=target_range,
                frames=tuple(
                    FrameMap(
                        frame_index=index - start,
                        source_frame_index=index,
                        source_pts=timeline.pts[index],
                        clip_pts=timeline.pts[index] - timeline.pts[start],
                        duration_pts=timeline.durations[index],
                    )
                    for index in range(start, stop)
                ),
            )
            check_clip(video, manifest)
            manifest_path = staging / "manifest.json"
            manifest_path.write_bytes(manifest.canonical_bytes() + b"\n")
            for path in (video, manifest_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            sync_directory(staging)
            os.rename(staging, destination)
            sync_directory(destination.parent)
        return annotation_reference(layout, destination / "manifest.json")


def prepare_source(
    layout: AnnotationLayout,
    source: SourceIdentity,
    configuration: ArtifactRef,
    config: PreparationConfig,
    selection_ref: ArtifactRef,
) -> ArtifactRef:
    selection = read_document(layout, selection_ref, SelectionReceipt)
    if (
        selection.source_id != source.canonical_id
        or selection.source_digest != source.media.digest
    ):
        raise ValueError("selection belongs to another source or encoding")
    if not selection.windows:
        raise ValueError("excluded source has no windows to prepare")
    timeline = probe_video(verify_reference(layout, source.media))
    clips = tuple(
        prepare_clip(
            layout, source, timeline, window, configuration, config, selection_ref
        )
        for window in selection.windows
    )
    receipt = PreparationReceipt(
        source_id=source.canonical_id,
        source_digest=source.media.digest,
        selection=selection_ref,
        configuration=configuration,
        clips=clips,
    )
    path = layout.path(
        "sources", source.canonical_id, configuration.digest, "prepared.json"
    )
    return write_document(layout, path, receipt)
