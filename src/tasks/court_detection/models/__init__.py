"""Court detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "CourtHierarchicalModel",
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

    encoder_name = str((model_cfg.get("encoder", {}) or {}).get("name", "default"))
    decoder_name = str((model_cfg.get("decoder", {}) or {}).get("name", "fpn"))
    if (encoder_name, decoder_name) == ("dinov3", "dpt") and task != "seg":
        raise ValueError(
            "DINOv3+DPT currently supports only semantic segmentation, "
            f"got task={task!r}."
        )
    return CourtHierarchicalModel.from_config(config)
