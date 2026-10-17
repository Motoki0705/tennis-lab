import pytest
import torch

from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.ball_detection.evaluation.hybrid import HybridMetrics


def batch(start: int = 0, step: int = 1, *, absent: bool = False, source: str = "fixture") -> dict:
    positive = torch.zeros(1, 32, dtype=torch.bool)
    if not absent:
        positive[:, :16] = True
    valid = torch.ones_like(positive)
    valid[:, -1] = False
    return dict(clip_id=[source], start=[start], frame_step=[step], frame_indices=torch.arange(32)[None] + start,
                common_evaluation=[True], source=[source], position_valid=positive, heatmap_valid=valid)


def test_presence_and_position_use_same_owner_and_report_source_supports() -> None:
    metrics = HybridMetrics((1, 2), presence_threshold=.7)
    data = batch()
    error = torch.full((1, 32), 5.)
    scores = torch.full((1, 32), .5) # 16 misses, no false positives
    losses = torch.ones(1, 32)
    metrics.add(data, error, scores, losses, losses)
    # Worse duplicate must not change either metric.
    metrics.add(data, error * 9, scores + .4, losses * 3, losses * 3)
    metrics.add(batch(step=2), error * 3, scores + .4, losses, losses)
    metrics.add(batch(absent=True, source="negative_only"), error, scores + .4, losses, losses)
    report = metrics.report()
    full = report["scopes"]["full"]
    assert full["macro_mean_error_px"] == 10.
    assert full["by_frame_step"]["1"]["p95_error_px"] == 5.
    first, second = (full["by_frame_step"][str(step)]["presence"] for step in (1, 2))
    assert (first["false_positive"], first["false_negative"]) == (31, 16)
    assert (second["false_positive"], second["false_negative"]) == (15, 0)
    negative = full["by_source"]["negative_only"]
    assert negative["by_frame_step"]["1"]["presence"]["false_positive_rate"] == 1.
    assert negative["macro_false_positive_rate"] is None # FPS 2 has no negatives
    assert negative["by_frame_step"]["1"]["mean_error_px"] is None


def test_all_absent_evaluation_is_defined_but_cannot_select_a_coordinate_best() -> None:
    metrics = HybridMetrics((1,), presence_threshold=.5)
    values = torch.zeros(1, 32)
    metrics.add(batch(absent=True), values, values + .8, values, values)
    report = metrics.report()
    result = report["scopes"]["full"]["by_frame_step"]["1"]
    assert result["observed_frames"] == 0 and result["p95_error_px"] is None
    assert result["presence"]["false_positive"] == 31
    assert result["presence"]["false_negative_rate"] is None
    with pytest.raises(ValueError, match="empty configured FPS"):
        selection_score(report, "full")
