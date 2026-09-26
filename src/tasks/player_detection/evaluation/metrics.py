"""Player detection metrics: COCO box AP plus the deployed operating point."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from src.tasks.player_detection.configuration import DetectionEvaluationConfig
from src.tasks.player_detection.models.dino_detector import FrameDetections

HEADLINE_METRICS = ("map", "map_50", "map_75", "precision", "recall", "f1")


def match_detections(
    boxes: torch.Tensor, scores: torch.Tensor, targets: torch.Tensor, *, iou_threshold: float
) -> tuple[int, int, int]:
    """Greedy score-ordered one-to-one matching; returns (TP, FP, FN)."""
    if boxes.shape[0] == 0:
        return 0, 0, int(targets.shape[0])
    if targets.shape[0] == 0:
        return 0, int(boxes.shape[0]), 0
    iou = box_iou(boxes[scores.argsort(descending=True)], targets)
    taken = torch.zeros(targets.shape[0], dtype=torch.bool)
    true_positive = 0
    for row in iou:
        candidates = row.masked_fill(taken, -1.0)
        best = int(candidates.argmax())
        if float(candidates[best]) >= iou_threshold:
            taken[best] = True
            true_positive += 1
    return true_positive, int(boxes.shape[0]) - true_positive, int(targets.shape[0]) - true_positive


@dataclass
class PlayerDetectionMetrics:
    """Accumulates one evaluation pass (CPU tensors, original pixels)."""

    config: DetectionEvaluationConfig
    _map: MeanAveragePrecision = field(init=False)
    _counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
        self._counts = np.zeros(4, dtype=np.int64)  # TP, FP, FN, images

    def update(self, detections: list[FrameDetections], targets: list[torch.Tensor]) -> None:
        if len(detections) != len(targets):
            raise ValueError("Detections and targets must describe the same images")
        self._map.update(
            [
                {"boxes": d.boxes_xyxy, "scores": d.scores, "labels": torch.zeros_like(d.scores, dtype=torch.int64)}
                for d in detections
            ],
            [
                {"boxes": t.float().cpu(), "labels": torch.zeros(t.shape[0], dtype=torch.int64)}
                for t in targets
            ],
        )
        for detection, target in zip(detections, targets, strict=True):
            keep = detection.scores >= self.config.score_threshold
            self._counts[:3] += match_detections(
                detection.boxes_xyxy[keep],
                detection.scores[keep],
                target.float().cpu(),
                iou_threshold=self.config.iou_threshold,
            )
            self._counts[3] += 1

    def compute(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(headline, diagnostics)`` with disjoint keys."""
        if self._counts[3] == 0:
            raise RuntimeError("No evaluation images were accumulated")
        coco = {key: float(value) for key, value in self._map.compute().items() if value.numel() == 1}
        true_positive, false_positive, false_negative, images = (int(v) for v in self._counts)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        headline = {
            "map": coco["map"],
            "map_50": coco["map_50"],
            "map_75": coco["map_75"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        diagnostics = {
            **{f"coco_{key}": value for key, value in coco.items() if key not in {"map", "map_50", "map_75", "classes"}},
            "true_positives": float(true_positive),
            "false_positives": float(false_positive),
            "false_negatives": float(false_negative),
            "images": float(images),
            "false_positives_per_image": false_positive / images,
            "score_threshold": self.config.score_threshold,
            "iou_threshold": self.config.iou_threshold,
        }
        return headline, diagnostics
