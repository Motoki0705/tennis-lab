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


def test_velocity_changes_only_loss_and_disables_automatic_test() -> None:
    output = "run.output_dir=slcs/train/config_fixture/fixed"
    baseline = _compose_profile("train_real_rgb_no_ball_smooth", output)
    candidate = _compose_profile("train_real_rgb_velocity", output)
    assert candidate.data.augmentation.burst_max_frames == 24
    assert candidate.loss.ball_position_smoothness_weight == 0.0
    assert candidate.loss.ball_velocity_weight == 0.0011117380640846516
    assert candidate.loss.ball_velocity_scale_mps == 11.259468485469933
    assert candidate.training.trainer.max_epochs == 60
    assert candidate.run.test_after_fit is False
    baseline.loss.ball_velocity_weight = candidate.loss.ball_velocity_weight
    baseline.loss.ball_velocity_scale_mps = candidate.loss.ball_velocity_scale_mps
    baseline.run.test_after_fit = False
    assert OmegaConf.to_container(candidate, resolve=True) == OmegaConf.to_container(
        baseline, resolve=True
    )


@pytest.mark.parametrize(
    "profile,experiment",
    [
        ("train_real_rgb_no_ball_smooth", "real_rgb_no_ball_smooth"),
        ("train_real_rgb_gap48", "real_rgb_gap48"),
        ("train_real_rgb_velocity", "real_rgb_velocity"),
        ("train_real_rgb_missing_ball_court", "real_rgb_missing_ball_court"),
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


def test_missing_ball_court_changes_only_architecture_and_automatic_test() -> None:
    output = "run.output_dir=slcs/train/config_fixture/fixed"
    baseline = _compose_profile("train_real_rgb_no_ball_smooth", output)
    candidate = _compose_profile("train_real_rgb_missing_ball_court", output)
    assert baseline.model.missing_ball_court_context is False
    assert candidate.model.missing_ball_court_context is True
    assert candidate.run.seed == 42
    assert candidate.data.augmentation.burst_max_frames == 24
    assert candidate.training.trainer.max_epochs == 60
    assert candidate.run.test_after_fit is False
    baseline.model.missing_ball_court_context = True
    baseline.run.test_after_fit = False
    assert OmegaConf.to_container(candidate, resolve=True) == OmegaConf.to_container(
        baseline, resolve=True
    )
