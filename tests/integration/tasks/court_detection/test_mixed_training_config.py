"""Hydra composition coverage for mixed-source Court training."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    TennisCourtDetectorSourceConfig,
)
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
    assert standard.data.source.court_scope == "target_court"
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


def test_pose_preset_is_default_for_mixed_training_and_synthetic_only() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "data/augmentation=pose_safe",
                "run.output_dir=court_detection/mixed-source/dense-pose-test",
            ],
        )

    standard, mixed = resolve_mixed_training_config(config)
    synthetic = mixed.sources["synthetic_court"]
    runtime = CourtTrainingConfig.from_config(standard)

    assert runtime.loss.pose.enabled
    assert (
        runtime.loss.pose.translation_weight,
        runtime.loss.pose.rotation_weight,
        runtime.loss.pose.focal_weight,
    ) == (1.0, 1.0, 1.0)
    assert runtime.loss.dense_weights == {"kp": 1.0, "seg": 1.0, "line": 1.0}
    assert not runtime.loss.consistency.enabled
    assert synthetic.kind == "synthetic_court"
    assert synthetic.court_scope == "target_court"


def test_pose_lora_training_selects_best_checkpoint_by_direct_pose_loss() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "training=pose_lora",
                "run.output_dir=court_detection/mixed-source/pose-lora-test",
            ],
        )

    standard, _ = resolve_mixed_training_config(config)
    CourtTrainingConfig.from_config(standard)

    assert standard.training.checkpoint.monitor == "val/loss_direct_pose"
    assert standard.training.checkpoint.save_last is True
    assert standard.model.encoder.lora.enabled is True


def test_mixed_kp_config_rejects_noncanonical_synthetic_scope() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_mixed",
            overrides=[
                "data.source.court_scope=all_courts",
                "run.output_dir=court_detection/mixed-source/invalid-scope-test",
            ],
        )

    with pytest.raises(
        SemanticConfigurationError,
        match="single-court SEG/LINE targets require",
    ):
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
