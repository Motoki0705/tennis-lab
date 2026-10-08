import pytest
import torch

from src.tasks.ball_detection.training.coordinate_evaluation import (
    CoordinateMetrics,
    selection_score,
)


def batch(clip: str, start: int, step: int, common: bool = True) -> dict:
    return dict(clip_id=[clip], start=[start], frame_step=[step], common_evaluation=[common],
                source=["tracknet" if common else "chat_annotation"],
                frame_indices=(start + torch.arange(32) * step)[None],
                position_valid=torch.ones(1, 32, dtype=torch.bool))


def test_overlap_ownership_is_order_independent_and_uses_actual_frame_indices() -> None:
    first, second = batch("clip", 0, 1), batch("clip", 16, 1)
    a, b = CoordinateMetrics((1,)), CoordinateMetrics((1,))
    for metrics, entries in ((a, [(first, 1.), (second, 3.)]), (b, [(second, 3.), (first, 1.)])):
        for value, error in entries:
            metrics.add(value, torch.full((1, 32), error))
    assert a.report() == b.report()
    summary = a.report()["scopes"]["common"]["by_frame_step"]["1"]
    assert summary["observed_frames"] == 48
    assert summary["mean_error_px"] == 2.
    assert summary["median_error_px"] == 2.


def test_each_fps_is_counted_separately_and_common_scope_excludes_other_clips() -> None:
    metrics = CoordinateMetrics((1, 2, 4))
    for step, error in ((1, 1.), (2, 2.), (4, 3.)):
        metrics.add(batch("shared", 0, step), torch.full((1, 32), error))
        metrics.add(batch("extra", 0, step, False), torch.full((1, 32), 9.))
    report = metrics.report()
    common, full = report["scopes"]["common"], report["scopes"]["full"]
    assert common["frame_fps_pairs"] == 96 and common["unique_observed_frames"] == 64
    assert full["frame_fps_pairs"] == 192 and full["unique_observed_frames"] == 128
    assert selection_score(report, "common") == 2.
    assert selection_score(report, "full") == 5.5
    assert not common["by_source"]["chat_annotation"]["by_frame_step"]["1"]["available"]


def test_empty_scope_never_falls_back_and_conflicting_membership_is_rejected() -> None:
    metrics = CoordinateMetrics((1,))
    metrics.add(batch("extra", 0, 1, False), torch.zeros(1, 32))
    with pytest.raises(ValueError, match="empty configured FPS"):
        selection_score(metrics.report(), "common")
    with pytest.raises(ValueError, match="inconsistent"):
        metrics.add(batch("extra", 0, 1, True), torch.zeros(1, 32))


def test_missing_fps_cannot_be_omitted_from_selection_average() -> None:
    metrics = CoordinateMetrics((1, 2, 4))
    metrics.add(batch("clip", 0, 1), torch.ones(1, 32))
    with pytest.raises(ValueError, match="empty configured FPS"):
        selection_score(metrics.report(), "full")
