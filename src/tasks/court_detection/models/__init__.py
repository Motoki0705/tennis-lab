"""Court detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.court_detection.models.court_fpn import CourtFPN
from src.tasks.court_detection.models.court_unet import CourtUNet

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["CourtFPN", "CourtUNet", "build_court_detection_model"]


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

    model_name = str(model_cfg.get("name", "court_fpn"))
    if model_name.startswith("court_unet"):
        return CourtUNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=num_classes,
        )
    return CourtFPN.from_config(config)
