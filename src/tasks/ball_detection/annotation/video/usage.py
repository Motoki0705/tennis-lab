"""Shared usage decisions, with no observed-only gate or inference weights."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import VideoBallAnnotation, VideoBallFrame


def is_processed(frame: VideoBallFrame) -> bool:
    return frame.state != "unprocessed"


def is_unprocessed(frame: VideoBallFrame) -> bool:
    return frame.state == "unprocessed"


def is_final_unknown(frame: VideoBallFrame) -> bool:
    return frame.state == "unknown"


def is_position_usable(frame: VideoBallFrame) -> bool:
    return frame.state == "in_frame" and frame.position is not None


def is_position_unavailable(frame: VideoBallFrame) -> bool:
    """Processed without usable coordinates; unprocessed is a separate category."""
    return is_processed(frame) and not is_position_usable(frame)


def is_direct_observation(frame: VideoBallFrame) -> bool:
    return (
        is_position_usable(frame)
        and frame.position is not None
        and frame.position.provenance.kind == "observed"
    )


def is_estimated(frame: VideoBallFrame) -> bool:
    return (
        is_position_usable(frame)
        and frame.position is not None
        and frame.position.provenance.kind != "observed"
    )


def is_active_ball_negative_candidate(
    frame: VideoBallFrame, *, target_semantics: str
) -> bool:
    """Candidate only for an active-ball teacher, never an all-ball negative."""
    if target_semantics != "active_ball":
        raise ValueError(
            "negative candidates require explicit active_ball target_semantics"
        )
    return frame.state in {"out_of_frame", "no_active_ball"}


@dataclass(frozen=True)
class UsageMasks:
    """Tuples align with annotation.frames, including all unprocessed rows."""

    processed: tuple[bool, ...]
    unprocessed: tuple[bool, ...]
    final_unknown: tuple[bool, ...]
    position_usable: tuple[bool, ...]
    position_unavailable: tuple[bool, ...]
    direct_observation: tuple[bool, ...]
    estimated: tuple[bool, ...]
    active_ball_negative_candidate: tuple[bool, ...]


def usage_masks(annotation: VideoBallAnnotation) -> UsageMasks:
    rows = annotation.frames
    return UsageMasks(
        processed=tuple(map(is_processed, rows)),
        unprocessed=tuple(map(is_unprocessed, rows)),
        final_unknown=tuple(map(is_final_unknown, rows)),
        position_usable=tuple(map(is_position_usable, rows)),
        position_unavailable=tuple(map(is_position_unavailable, rows)),
        direct_observation=tuple(map(is_direct_observation, rows)),
        estimated=tuple(map(is_estimated, rows)),
        active_ball_negative_candidate=tuple(
            is_active_ball_negative_candidate(
                row, target_semantics=annotation.target.target_semantics
            )
            for row in rows
        ),
    )


@dataclass(frozen=True)
class AnnotationSummary:
    total: int
    processed: int
    unprocessed: int
    final_unknown: int
    position_usable: int
    position_unavailable: int
    direct_observation: int
    estimated: int
    active_ball_negative_candidate: int
    in_frame: int
    out_of_frame: int
    no_active_ball: int
    occluded_with_position: int
    occluded_without_position: int

    @property
    def complete(self) -> bool:
        """Completion allows final unknown, but requires every requested row."""
        return self.unprocessed == 0


def summarize_annotation(annotation: VideoBallAnnotation) -> AnnotationSummary:
    """Count one camera/scope; 81/25 acceptance uses unavailable/final_unknown."""
    masks = usage_masks(annotation)
    return AnnotationSummary(
        total=len(annotation.frames),
        processed=sum(masks.processed),
        unprocessed=sum(masks.unprocessed),
        final_unknown=sum(masks.final_unknown),
        position_usable=sum(masks.position_usable),
        position_unavailable=sum(masks.position_unavailable),
        direct_observation=sum(masks.direct_observation),
        estimated=sum(masks.estimated),
        active_ball_negative_candidate=sum(masks.active_ball_negative_candidate),
        in_frame=sum(f.state == "in_frame" for f in annotation.frames),
        out_of_frame=sum(f.state == "out_of_frame" for f in annotation.frames),
        no_active_ball=sum(f.state == "no_active_ball" for f in annotation.frames),
        occluded_with_position=sum(
            f.occlusion == "occluded" and f.position is not None
            for f in annotation.frames
        ),
        occluded_without_position=sum(
            f.occlusion == "occluded" and f.position is None for f in annotation.frames
        ),
    )
