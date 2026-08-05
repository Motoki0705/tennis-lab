"""Semantic constraints for the DINO SSL execution boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tasks.ball_detection.configuration import validate_youtube_boundary
from src.utils.configuration import ConfigurationError
from src.utils.paths import PROJECT_ROOT


def _config() -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="prepare_dinov3_ssl_images")


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("workflow.discovery.queries", []),
        ("workflow.discovery.queries", [" "]),
        ("workflow.discovery.max_results_per_query", 0),
        ("workflow.discovery.min_duration_sec", -1),
        ("workflow.discovery.max_duration_sec", -1),
        ("workflow.discovery.min_duration_sec", 4000),
        ("workflow.processing.max_new_videos", 0),
        ("workflow.storage.max_root_gb", 0),
        ("workflow.frames.frames_per_video", 0),
        ("workflow.frames.output_ext", "gif"),
        ("workflow.frames.jpeg_quality", 0),
        ("workflow.frames.jpeg_quality", 101),
        ("workflow.gate.backend", "legacy"),
        ("workflow.gate.vllm.base_url", " "),
        ("workflow.gate.vllm.model", ""),
        ("workflow.gate.vllm.timeout_sec", 0),
        ("workflow.gate.vllm.max_tokens", 0),
        ("workflow.gate.vllm.accept_labels", []),
        ("workflow.gate.vllm.prompt", ""),
        ("workflow.gate.vllm.server.command", []),
        ("workflow.gate.vllm.server.health_url", ""),
        ("workflow.gate.vllm.server.startup_timeout_sec", 0),
        ("workflow.gate.vllm.server.poll_interval_sec", 0),
        ("workflow.gate.vllm.server.request_timeout_sec", 0),
        ("workflow.gate.vllm.server.shutdown_timeout_sec", 0),
    ],
)
def test_dino_ssl_rejects_invalid_semantic_boundary_values(
    path: str,
    invalid: object,
) -> None:
    config = deepcopy(_config())
    with open_dict(config):
        OmegaConf.update(config, path, invalid, merge=False)

    with pytest.raises(ConfigurationError):
        validate_youtube_boundary(config)
