"""Hydra composition coverage for mixed-source Court training."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.training.runner_mixed import (
    resolve_mixed_training_config,
)
from src.utils.configuration import SemanticConfigurationError

pytestmark = pytest.mark.integration

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def test_train_mixed_config_reuses_two_sources_with_canonical_kp_scope() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=["run.output_dir=court_detection/mixed-source/config-test"],
        )

    standard, mixed = resolve_mixed_training_config(config)

    assert standard.data.source.kind == "synthetic_court"
    assert standard.data.source.keypoint_court_scope == "target_court"
    assert set(mixed.sources) == {
        "synthetic_court",
        "tennis_court_detector",
    }
    assert dict(mixed.train_batch_counts) == {
        "synthetic_court": 4,
        "tennis_court_detector": 4,
    }
    tennis = mixed.sources["tennis_court_detector"]
    assert isinstance(tennis, TennisCourtDetectorSourceConfig)
    assert tennis.excluded_sample_ids == ("QszoUKyCOHo_600",)


def test_pose_overrides_keep_consistency_disabled_and_synthetic_only() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "data/augmentation=pose_safe",
                "loss.pose.enabled=true",
                "loss.pose.translation_weight=1.0",
                "loss.pose.rotation_weight=1.0",
                "loss.pose.focal_weight=1.0",
                "loss.consistency.enabled=false",
                "run.output_dir=court_detection/mixed-source/dense-pose-test",
            ],
        )

    standard, mixed = resolve_mixed_training_config(config)
    synthetic = mixed.sources["synthetic_court"]

    assert standard.loss.pose.enabled
    assert not standard.loss.consistency.enabled
    assert synthetic.kind == "synthetic_court"
    assert synthetic.keypoint_court_scope == "target_court"


def test_mixed_kp_config_rejects_noncanonical_synthetic_scope() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "data.source.keypoint_court_scope=all_courts",
                "run.output_dir=court_detection/mixed-source/invalid-scope-test",
            ],
        )

    with pytest.raises(SemanticConfigurationError, match="Mixed KP training"):
        resolve_mixed_training_config(config)


def test_train_mixed_requires_explicit_variant_output_dir() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_mixed")

    with pytest.raises(
        SemanticConfigurationError,
        match="explicit variant-specific run.output_dir",
    ):
        resolve_mixed_training_config(config)


@pytest.mark.parametrize(
    "output_dir",
    [
        "court_detection/mixed-source/dense-only",
        "court_detection/mixed-source/dense-pose",
    ],
)
def test_train_mixed_preserves_explicit_variant_output_dir(
    output_dir: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[f"run.output_dir={output_dir}"],
        )

    standard, _ = resolve_mixed_training_config(config)

    assert standard.run.output_dir == output_dir
