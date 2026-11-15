"""Verify retained v2 -> v3 conversion and independently registered native images."""

from __future__ import annotations

from fractions import Fraction

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    read_annotation,
)
from src.tasks.ball_detection.annotation.video.inputs import validate_video_input
from src.tasks.ball_detection.annotation.video.migration import (
    MigrationRecord,
    convert_v2_annotation,
)
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import (
    ClipManifest,
    SourceProvenance,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout

from .adopted import PublicationClip
from .contracts import MigrationPublishRequest
from .videos import compile_video_annotation


def load_migrated_annotations(
    layout: AnnotationLayout, request: MigrationPublishRequest
) -> tuple[PublicationClip, ...]:
    results = []
    for saved in request.annotations:
        expected = read_document(layout, saved.expected_input, VideoAnnotationInput)
        record = read_document(layout, saved.migration, MigrationRecord)
        original = verify_reference(layout, saved.original)
        annotation, regenerated = convert_v2_annotation(original, expected_input=expected, evidence=record.evidence)
        stored = read_annotation(verify_reference(layout, saved.annotation), expected_input=expected, require_complete=True)
        if regenerated != record or annotation != stored:
            raise ValueError("migration record/annotation differs from retained v2 conversion")
        # Imported inputs have no annotation worker/ledger. Validate their pinned
        # original bytes and actual decoded timeline here instead.
        validate_video_input(verify_reference(layout, saved.source_media), expected)
        manifest = read_document(layout, saved.manifest, ClipManifest)
        source = expected.source
        if (
            not isinstance(manifest.source, SourceProvenance)
            or manifest.jpeg is None
            or (manifest.source.provider, manifest.source.channel_id, manifest.source.video_id)
            != ("local", content_digest(source.recording_id), source.sha256)
            or (manifest.source.media.digest, manifest.source.media.size_bytes)
            != (source.media_sha256, source.size_bytes)
            or manifest.preparation_digest != expected.sha256
            or (manifest.width, manifest.height, Fraction(manifest.time_base), Fraction(manifest.nominal_fps))
            != (source.width, source.height, Fraction(source.time_base), Fraction(source.nominal_fps))
            or manifest.source_range != expected.frame_range
            or (manifest.target_range.start, manifest.target_range.stop) != (0, len(expected.frames))
            or tuple((r.source_frame_index, r.source_pts) for r in manifest.frames)
            != tuple((r.frame_index, r.pts) for r in expected.frames)
        ):
            raise ValueError("migration image manifest mixes a different camera/input/frame/PTS")
        results.append(compile_video_annotation(
            layout, saved.annotation, expected, saved.manifest,
            provenance=dict(
                input=saved.expected_input.model_dump(mode="json"),
                source_media=saved.source_media.model_dump(mode="json"),
                manifest=saved.manifest.model_dump(mode="json"),
                original_v2=saved.original.model_dump(mode="json"),
                migration=saved.migration.model_dump(mode="json"),
            ),
        ))
    return tuple(sorted(results, key=lambda a: a.spec.clip_id))
