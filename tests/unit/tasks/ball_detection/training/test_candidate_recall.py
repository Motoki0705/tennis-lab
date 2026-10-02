"""Native-grid validation recall, source units and unique-frame aggregation."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.tasks.ball_detection.training.candidate_recall import (
    ValidationCandidateRecall,
    candidate_log_values,
)


def settings() -> dict[str, Any]:
    return dict(max_candidates=8, nms_kernel=5, patch_size=5, subpixel_refine=True, radius_source_px=20.0)


def batch(*, frames: int = 1, start: int = 0) -> dict[str, Any]:
    return {
        "original_size": torch.tensor([[201.0, 101.0]]),
        "source": ["meiji"],
        "candidate_reference": {
            "xy": torch.tensor([[[60.0, 50.0]] * frames]),
            "observed": torch.ones(1, frames, dtype=torch.bool),
            "frame_id": torch.arange(start, start + frames)[None],
            "window_start": torch.tensor([start]),
            "source_scale": torch.tensor([0.5]),
            "namespace": ["store"],
            "camera": ["cam0"],
        },
    }


def maps(frames: int = 1) -> torch.Tensor:
    values = torch.zeros(1, frames, 11, 21)
    values[:, :, 5, 5] = 0.1   # correct weak candidate, exactly 20 source px away
    values[:, :, 5, 15] = 0.9  # higher-scored wrong candidate
    return values


def test_weak_candidates_inclusive_source_distance_and_ranking() -> None:
    metric = ValidationCandidateRecall(settings())
    inputs = batch(frames=3)
    inputs["candidate_reference"]["xy"][0, 1, 0] += 0.01
    inputs["candidate_reference"]["observed"][0, 2] = False
    inputs["candidate_reference"]["xy"][0, 2] = torch.nan
    metric.update(maps(3), inputs)
    report = metric.compute()
    for group in ("", "meiji", "meiji/cam0"):
        assert report[group]["frames"] == 3
        assert report[group]["observed"] == 2
        assert report[group]["recall_at_k"] == 0.5
        assert report[group]["recall_at_1"] == 0
        assert report[group]["not_in_candidates_rate"] == 0.5
        assert report[group]["wrong_ranked_above_true_rate"] == 0.5
        assert report[group]["wrong_strictly_higher_score_rate"] == 0.5
    assert candidate_log_values(report)["val/meiji/candidate_recall_at_8_20px"] == 0.5


def test_uniform_heatmap_has_no_fabricated_candidate_at_zero() -> None:
    metric = ValidationCandidateRecall(settings())
    inputs = batch()
    inputs["candidate_reference"]["xy"].zero_()
    metric.update(torch.zeros_like(maps()), inputs)
    assert metric.compute()[""]["recalled_at_k"] == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_overlaps_select_nearest_centre_then_earlier_window(reverse: bool) -> None:
    metric = ValidationCandidateRecall(settings())
    # Windows [0..3] and [2..5]; frame 2 belongs to the earlier window,
    # frame 3 to the later one. Candidate outcomes deliberately disagree.
    updates = [(maps(4), batch(frames=4)), (torch.zeros_like(maps(4)), batch(frames=4, start=2))]
    for heatmaps, inputs in updates[:: -1 if reverse else 1]:
        metric.update(heatmaps, inputs)
    report = metric.compute()[""]
    assert report["frames"] == report["observed"] == 6
    assert report["recalled_at_k"] == 3
    assert report["recall_at_k"] == 0.5
    # Exact centre-distance tie between windows starting at 0 and 1
    # for frame 2 of a four-frame window. Earlier start wins.
    metric.reset()
    metric.update(torch.zeros_like(maps(4)), batch(frames=4, start=1))
    metric.update(maps(4), batch(frames=4))
    assert metric.frames[("store", 2)].counts.recalled_at_k == 1


def test_batch_partition_and_distributed_repeats_do_not_weight_the_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = ValidationCandidateRecall(settings()), ValidationCandidateRecall(settings())
    left.update(maps(), batch())
    right.update(torch.zeros_like(maps(3)), batch(frames=3, start=1))
    # The first window can be repeated by a padded DistributedSampler.
    right.update(maps(), batch())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(parts: list[Any], _: Any) -> None:
        parts[:] = [left.frames, right.frames]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert left.compute()[""]["observed"] == 4
    assert left.compute()[""]["recall_at_k"] == 0.25
    assert right.compute() == left.compute()


def test_source_camera_and_namespace_are_not_conflated() -> None:
    metric = ValidationCandidateRecall(settings())
    first, second = batch(), batch()
    second["source"] = ["chat_annotation"]
    second["candidate_reference"]["namespace"] = ["web"]
    second["candidate_reference"]["camera"] = [""]
    metric.update(maps(), first)
    metric.update(torch.zeros_like(maps()), second)
    report = metric.compute()
    assert report[""]["observed"] == 2
    assert report["meiji/cam0"]["recall_at_k"] == 1
    assert report["chat_annotation"]["recall_at_k"] == 0
    assert set(report) == {"", "meiji", "meiji/cam0", "chat_annotation"}


def test_empty_denominator_is_explicit_and_epoch_reset_drops_old_frames() -> None:
    metric = ValidationCandidateRecall(settings())
    metric.update(maps(), batch())
    metric.reset()
    with pytest.raises(ValueError, match="single observed"):
        metric.compute()
    report = metric.compute(require_observed=False)
    assert report[""]["recall_at_k"] is None
    assert "val/candidate_recall_at_8_20px" not in candidate_log_values(report)
    metric.update(torch.zeros_like(maps()), batch())
    assert metric.compute()[""]["recall_at_k"] == 0


@pytest.mark.parametrize("corruption", ["missing", "scale", "identity", "mask", "nan"])
def test_invalid_reference_never_falls_back_to_stored_pixels(corruption: str) -> None:
    inputs = batch()
    reference = inputs["candidate_reference"]
    if corruption == "missing":
        del reference["source_scale"]
    elif corruption == "scale":
        reference["source_scale"].zero_()
    elif corruption == "identity":
        reference["frame_id"] -= 1
    elif corruption == "mask":
        reference["observed"] = reference["observed"].float()
    else:
        reference["xy"][:] = torch.nan
    with pytest.raises((KeyError, ValueError)):
        ValidationCandidateRecall(settings()).update(maps(), inputs)


def test_duplicate_identity_cannot_change_target_or_outcome() -> None:
    metric = ValidationCandidateRecall(settings())
    metric.update(maps(), batch())
    with pytest.raises(ValueError, match="inconsistent candidate"):
        metric.update(torch.zeros_like(maps()), batch())
    inputs = batch()
    inputs["candidate_reference"]["xy"] += 1
    with pytest.raises(ValueError, match="identity changed"):
        metric.update(maps(), inputs)
