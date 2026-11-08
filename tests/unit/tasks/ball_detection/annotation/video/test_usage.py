"""The masks and counts below are hand-counted from the ten-row state table."""

from dataclasses import asdict

import pytest

from src.tasks.ball_detection.annotation.video import (
    VideoBallAnnotation,
    is_active_ball_negative_candidate,
    is_direct_observation,
    is_estimated,
    is_final_unknown,
    is_position_unavailable,
    is_position_usable,
    is_processed,
    is_unprocessed,
    summarize_annotation,
    usage_masks,
)


def test_masks_separate_completion_positions_origins_and_negatives(
    annotation: VideoBallAnnotation,
) -> None:
    masks = usage_masks(annotation)
    # Lists name absolute source indices, not zero-based positions in the scope.
    expected_indices = {
        "processed": [41, 42, 43, 44, 45, 46, 47, 48, 49],
        "unprocessed": [50],
        "final_unknown": [49],
        "position_usable": [41, 42, 43, 44, 45],
        "position_unavailable": [46, 47, 48, 49],
        "direct_observation": [41],
        "estimated": [42, 43, 44, 45],
        "active_ball_negative_candidate": [47, 48],
    }
    for name, mask in asdict(masks).items():
        assert [
            row.frame_index
            for row, keep in zip(annotation.frames, mask, strict=True)
            if keep
        ] == expected_indices[name]
    predicates = {
        "processed": is_processed,
        "unprocessed": is_unprocessed,
        "final_unknown": is_final_unknown,
        "position_usable": is_position_usable,
        "position_unavailable": is_position_unavailable,
        "direct_observation": is_direct_observation,
        "estimated": is_estimated,
        "active_ball_negative_candidate": lambda row: is_active_ball_negative_candidate(
            row, target_semantics="active_ball"
        ),
    }
    for name, predicate in predicates.items():
        assert [
            row.frame_index for row in annotation.frames if predicate(row)
        ] == expected_indices[name]
    with pytest.raises(ValueError, match="explicit active_ball"):
        is_active_ball_negative_candidate(
            annotation.frames[7], target_semantics="all_balls"
        )


def test_summary_counts_unprocessed_separately_and_allows_final_unknown(
    annotation: VideoBallAnnotation,
) -> None:
    summary = summarize_annotation(annotation)
    assert asdict(summary) == {
        "total": 10,
        "processed": 9,
        "unprocessed": 1,
        "final_unknown": 1,
        "position_usable": 5,
        "position_unavailable": 4,
        "direct_observation": 1,
        "estimated": 4,
        "active_ball_negative_candidate": 2,
        "in_frame": 6,
        "out_of_frame": 1,
        "no_active_ball": 1,
        "occluded_with_position": 2,
        "occluded_without_position": 1,
    }
    assert not summary.complete
    data = annotation.model_dump_json()
    complete = VideoBallAnnotation.model_validate_json(
        data.replace('"state":"unprocessed"', '"state":"unknown"')
    )
    completed_summary = summarize_annotation(complete)
    assert completed_summary.complete
    assert (
        completed_summary.processed,
        completed_summary.unprocessed,
        completed_summary.final_unknown,
        completed_summary.position_unavailable,
    ) == (10, 0, 2, 5)
