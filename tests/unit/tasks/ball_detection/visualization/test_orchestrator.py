"""Runtime-contract tests for ball visualization orchestration."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

from src.tasks.ball_detection.scripts import visualize as visualize_script
from src.tasks.ball_detection.visualization import orchestrator
from src.tasks.ball_detection.visualization.orchestrator import build_runtime_config
from src.utils.device import DeviceSelectionError
from src.utils.paths import PROJECT_ROOT


def _visualization_config(*, device: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="visualize", overrides=[f"run.device={device}"])


def test_runtime_config_keeps_resolved_torch_device() -> None:
    config = _visualization_config(device="cpu")
    assert isinstance(config.data.image_size, ListConfig)
    assert isinstance(config.data.augmentation.normalize_imagenet.mean, ListConfig)
    assert isinstance(config.visualization.draw.gt_color_rgb, ListConfig)

    runtime = build_runtime_config(config)

    assert isinstance(runtime.device, torch.device)
    assert runtime.device == torch.device("cpu")
    assert runtime.image_size_hw == (288, 512)
    assert runtime.imagenet_mean == (0.485, 0.456, 0.406)
    assert runtime.draw.gt_color_rgb == (255, 96, 96)
    assert runtime.layout.background_rgb == (18, 18, 18)


def test_runtime_config_rejects_unavailable_cuda_before_predictor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constructions: list[object] = []
    monkeypatch.setattr(
        orchestrator.BallDetectionPredictor,
        "load_from_checkpoint",
        lambda *_args, **_kwargs: constructions.append(object()),
    )
    raw_main = cast(
        Callable[[DictConfig], int],
        inspect.unwrap(visualize_script.main),
    )

    with pytest.raises(DeviceSelectionError, match="CUDA is unavailable"):
        raw_main(_visualization_config(device="cuda"))

    assert constructions == []
