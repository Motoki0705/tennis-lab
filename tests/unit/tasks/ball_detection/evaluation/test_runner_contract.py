"""Single-authority Ball evaluation manifest execution contract."""

from __future__ import annotations

import inspect

import pytest
from omegaconf import OmegaConf

from src.tasks.ball_detection.configuration import validate_manifest_boundary
from src.tasks.ball_detection.evaluation.runner import EvaluationPipeline
from src.utils.configuration import ConfigurationError


def test_pipeline_requires_manifest_and_evaluator_without_overrides() -> None:
    signature = inspect.signature(EvaluationPipeline)

    assert tuple(signature.parameters) == ("manifest", "evaluator")
    assert signature.parameters["evaluator"].default is inspect.Parameter.empty


@pytest.mark.parametrize("old_key", ["output_dir", "resume", "fail_fast", "device"])
def test_manifest_cli_rejects_old_nullable_override_keys(old_key: str) -> None:
    config = OmegaConf.create(
        {
            "paths": {
                "project_root": ".",
                "data_root": "data",
                "checkpoint_root": "ckpt",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            "manifest_path": "manifest.yaml",
            old_key: None,
        }
    )

    with pytest.raises(ConfigurationError):
        validate_manifest_boundary(config)
