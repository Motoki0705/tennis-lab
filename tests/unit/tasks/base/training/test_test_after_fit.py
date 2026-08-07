"""Unit tests for validation-only controlled training runs."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.tasks.base.training.runner import BaseTrainingRunner


def _config(*, fast_dev_run: bool, test_after_fit: bool) -> DictConfig:
    return OmegaConf.create(
        {
            "run": {
                "fast_dev_run": fast_dev_run,
                "test_after_fit": test_after_fit,
            }
        }
    )


def test_normal_training_runs_test_phase() -> None:
    assert not BaseTrainingRunner().skip_test(
        _config(fast_dev_run=False, test_after_fit=True)
    )


def test_controlled_training_can_skip_final_test() -> None:
    assert BaseTrainingRunner().skip_test(
        _config(fast_dev_run=False, test_after_fit=False)
    )


def test_fast_dev_run_still_skips_followup_test() -> None:
    assert BaseTrainingRunner().skip_test(
        _config(fast_dev_run=True, test_after_fit=True)
    )
