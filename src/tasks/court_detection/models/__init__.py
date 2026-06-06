"""Court detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.court_detection.models.dinov3_detr import DINOv3DETR
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "CourtHierarchicalModel",
    "DINOv3DETR",
    "build_court_detection_model",
]


def build_court_detection_model(config: DictConfig) -> nn.Module:
    """Build a court detection model from config.

    The number of output channels comes from ``config.model.num_classes``.
    A mismatch against the selected data task is rejected early.
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    num_classes = int(model_cfg.get("num_classes", 7))
    task = str(data_cfg.get("task", "seg"))
    expected_num_classes = num_classes
    if task == "seg":
        expected_num_classes = int(data_cfg.get("num_classes", 7))
    elif task == "kp":
        expected_num_classes = int(data_cfg.get("num_keypoints", 14))
    elif task == "line":
        expected_num_classes = int(data_cfg.get("num_classes", 1))

    if num_classes != expected_num_classes:
        raise ValueError(
            "Model/data configuration mismatch: "
            f"task={task!r} expects model.num_classes={expected_num_classes}, "
            f"but got {num_classes}.",
        )

    model_name = str(model_cfg.get("name", "court_hierarchical"))
    if model_name in {"court_dinov3_detr_seg", "dinov3_detr"}:
        if task != "seg":
            raise ValueError(
                "DINOv3DETR currently supports only semantic segmentation, "
                f"got task={task!r}."
            )
        return DINOv3DETR.from_config(config)
    return CourtHierarchicalModel.from_config(config)
