"""Processed chat-annotation ball labels (``tennis_chat_ball_annotation.v1``) -> :class:`ClipSpec`.

Status mapping:

* ``visible`` -> ``observed``; ``interpolated`` -> ``interpolated``.
* ``occluded`` -> ``occlusion_estimated`` when a centre is given, otherwise
  ``unresolved``; ``occluded`` is set in both cases.
* ``out_of_frame`` -> ``out_of_frame``; ``unresolved`` -> ``unresolved``.
* A frame with ``reviewed=false`` is stored with ``annotated=false``.
* ``interpolation_break`` (hit, bounce, cut or play boundary) -> ``segment_break``.
  The format has no typed hit/bounce events.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    ClipSpec,
    SourceFrame,
    VideoFrames,
)
from src.tennis_scene.chat_annotation.layout import published_video_path
from src.tennis_scene.chat_annotation.manifests import (
    load_prepared_manifests,
    require_regular_path,
)
from src.tennis_scene.chat_annotation.runtime.contracts import (
    Ball,
    BallAnnotation,
    ClipManifest,
    parse_annotation,
    read_json,
    sha256_file,
)
from src.tennis_scene.chat_annotation.runtime.validation import validate_annotation

PROCESSED_BALL_DIR = Path("annotated/processed/ball")


@dataclass(frozen=True, slots=True)
class ChatAnnotationSourceConfig:
    annotation_root: Path
    allowed_statuses: frozenset[str]


def _instance(ball: Ball) -> BallInstance:
    xy = None if ball.center_px is None else (float(ball.center_px[0]), float(ball.center_px[1]))
    if ball.status == "visible":
        return BallInstance(ball.track_id, "observed", xy, False)
    if ball.status == "occluded":
        return BallInstance(ball.track_id, "unresolved" if xy is None else "occlusion_estimated", xy, True)
    return BallInstance(ball.track_id, ball.status, xy, False)


def to_clip_spec(
    annotation: BallAnnotation, manifest: ClipManifest, annotation_path: Path, video: Path
) -> ClipSpec:
    labels = [
        SourceFrame(
            pts=manifest.frames[frame.frame_index].clip_pts,
            annotated=frame.reviewed,
            is_target=manifest.frames[frame.frame_index].is_target,
            segment_break=frame.interpolation_break,
            event="unlabeled",
            balls=tuple(_instance(ball) for ball in frame.balls),
        )
        for frame in annotation.frames
    ]
    time_base = Fraction(manifest.time_base)
    return ClipSpec(
        clip_id=f"chat_annotation/{annotation.clip_id}",
        source="chat_annotation",
        group_id=manifest.source.source_id,
        camera_id=None,
        width=manifest.width,
        height=manifest.height,
        time_base=time_base,
        fps=Fraction(manifest.nominal_fps),
        has_events=False,
        annotation_path=annotation_path,
        annotation_sha256=sha256_file(annotation_path),
        media_sha256=manifest.sha256,
        media=VideoFrames(video, tuple(frame.clip_pts for frame in manifest.frames), time_base),
        labels=ClipLabels.from_frames(labels, width=manifest.width, height=manifest.height),
        provenance={
            "source_title": manifest.source.title,
            "annotation_status": annotation.status,
            "annotation_issues": list(annotation.issues),
            "video_path": str(video),
        },
    )


def collect_chat_annotation(config: ChatAnnotationSourceConfig) -> list[ClipSpec]:
    """Every processed ball annotation, validated against its prepared clip.

    Any schema, identity, coverage or coordinate error aborts the build, as in
    ``player_detection``: an unusable processed annotation is a data problem to
    resolve in the annotation workflow, never something to skip.
    """
    root = config.annotation_root.resolve()
    manifests = load_prepared_manifests(root)
    directory = root / PROCESSED_BALL_DIR
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No processed ball annotations in {directory}")
    specs: list[ClipSpec] = []
    for path in paths:
        require_regular_path(path, root)
        annotation = parse_annotation(read_json(path))
        if not isinstance(annotation, BallAnnotation):
            raise ValueError(f"{path}: expected a ball annotation schema")
        if annotation.clip_id != path.stem:
            raise ValueError(f"{path}: clip_id {annotation.clip_id!r} differs from file name")
        manifest = manifests.get(annotation.clip_id)
        if manifest is None:
            raise ValueError(f"{path}: no prepared clip manifest for {annotation.clip_id}")
        report = validate_annotation(annotation, manifest)
        if report.errors:
            raise ValueError(f"{path}: invalid annotation: {'; '.join(report.errors)}")
        if annotation.status not in config.allowed_statuses:
            raise ValueError(f"{path}: status {annotation.status!r} is not in {sorted(config.allowed_statuses)}")
        video = published_video_path(root, manifest)
        require_regular_path(video, root)
        if not video.is_file():
            raise FileNotFoundError(f"Prepared clip video is missing: {video}")
        if sha256_file(video) != manifest.sha256:
            raise ValueError(f"{video}: sha256 differs from its clip manifest")
        specs.append(to_clip_spec(annotation, manifest, path, video))
    return specs
