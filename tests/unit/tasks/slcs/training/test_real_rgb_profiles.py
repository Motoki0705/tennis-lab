"""Real-RGB ablation profiles keep the comparison budget and data contract fixed."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import src.utils.hydra  # noqa: F401 -- register the public path/run resolvers
from src.utils.paths import PROJECT_ROOT


def _compose_profile(name: str, *overrides: str) -> DictConfig:
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/tasks/slcs/configs"),
    ):
        return compose(config_name=name, overrides=list(overrides))


def test_full_no_smooth_profile_matches_completed_training_overrides() -> None:
    output = "run.output_dir=slcs/train/config_fixture/fixed"
    original = _compose_profile(
        "train_real_rgb", "loss.ball_position_smoothness_weight=0.0", output
    )
    profile = _compose_profile("train_real_rgb_no_ball_smooth", output)
    assert OmegaConf.to_container(profile, resolve=True) == OmegaConf.to_container(
        original, resolve=True
    )
    assert profile.data.dataset_root == "slcs/real_rgb_v1"
    assert profile.training.trainer.max_epochs == 60
    assert profile.training.warmup_steps == 200


def test_gap48_changes_only_burst_length_not_labels_model_or_training_budget() -> None:
    output = "run.output_dir=slcs/train/config_fixture/fixed"
    baseline = _compose_profile("train_real_rgb_no_ball_smooth", output)
    candidate = _compose_profile("train_real_rgb_gap48", output)
    assert baseline.data.augmentation.burst_max_frames == 24
    assert candidate.data.augmentation.burst_max_frames == 48
    assert candidate.data.augmentation.enabled is True
    baseline.data.augmentation.burst_max_frames = 48
    assert OmegaConf.to_container(candidate, resolve=True) == OmegaConf.to_container(
        baseline, resolve=True
    )


@pytest.mark.parametrize(
    "profile,experiment",
    [
        ("train_real_rgb_no_ball_smooth", "real_rgb_no_ball_smooth"),
        ("train_real_rgb_gap48", "real_rgb_gap48"),
    ],
)
def test_ablation_output_identity_is_separate_and_stable(
    profile: str, experiment: str
) -> None:
    config = _compose_profile(profile)
    output = config.run.output_dir
    assert Path(output).parts[:3] == ("slcs", "train", experiment)
    assert len(Path(output).parts) == 4
    assert config.run.output_dir == output
