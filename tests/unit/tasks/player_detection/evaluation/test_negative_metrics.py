import pytest
import torch

from src.tasks.player_detection.configuration import DetectionEvaluationConfig
from src.tasks.player_detection.evaluation.metrics import (
    PlayerDetectionMetrics,
    match_detections,
)
from src.tasks.player_detection.models.dino_detector import FrameDetections


def test_detection_on_confirmed_absence_is_a_false_positive_and_reduces_ap() -> None:
    boxes = torch.tensor([[1., 1., 10., 10.]])
    empty = torch.empty(0, 4)
    config = DetectionEvaluationConfig(score_threshold=.3, iou_threshold=.5, max_detections=10)
    positive = FrameDetections(boxes, torch.tensor([.5]))
    false_positive = FrameDetections(boxes, torch.tensor([.99]))
    reference = PlayerDetectionMetrics(config)
    reference.update([positive], [boxes])
    metrics = PlayerDetectionMetrics(config)
    metrics.update([positive, false_positive], [boxes, empty])
    headline, diagnostic = metrics.compute()
    assert headline["map"] < reference.compute()[0]["map"]
    assert headline["precision"] == .5 and headline["recall"] == 1.
    assert diagnostic["false_positives"] == diagnostic["negative_images"] == 1.
    assert diagnostic["negative_false_positives"] == diagnostic["negative_images_with_false_positives"] == 1.
    assert diagnostic["target_instances"] == 1. and diagnostic["images"] == 2.
    assert match_detections(boxes, torch.tensor([.99]), empty, iou_threshold=.5) == (0, 1, 0)


@pytest.mark.parametrize("predicted", [False, True])
def test_negative_only_validation_cannot_return_map_sentinel_as_selection_score(predicted: bool) -> None:
    metrics = PlayerDetectionMetrics(DetectionEvaluationConfig(score_threshold=.3, iou_threshold=.5, max_detections=10))
    boxes = torch.tensor([[1., 1., 10., 10.]]) if predicted else torch.empty(0, 4)
    metrics.update([FrameDetections(boxes, torch.ones(len(boxes)))], [torch.empty(0, 4)])
    with pytest.raises(ValueError, match="negative-only evaluation"):
        metrics.compute()
