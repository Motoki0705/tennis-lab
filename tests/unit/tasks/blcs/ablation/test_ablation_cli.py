"""Experiment entry points validate paths/arguments before run side effects."""

import sys
from argparse import Namespace

import pytest

from src.tasks.ball_detection.scripts import (
    train_blcs_ablation,
)


def test_training_boundary_preserves_the_fixed_run_arguments(tmp_path, monkeypatch):
    received: list[Namespace] = []
    monkeypatch.setattr(train_blcs_ablation, "run_experiment", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_blcs_ablation",
            "--arm",
            "rollout_velocity",
            "--seed",
            "43",
            "--steps",
            "4000",
            "--data-root",
            str(tmp_path),
            "--output",
            "blcs/i1044/boundary-test",
        ],
    )
    train_blcs_ablation.main()
    assert len(received) == 1
    args = received[0]
    assert (args.arm, args.seed, args.steps) == ("rollout_velocity", 43, 4000)
    assert args.data_root == tmp_path
    assert args.output == "blcs/i1044/boundary-test"
    assert not (args.smoke or args.chunked or args.gan)


def test_training_boundary_rejects_output_escape_before_runner(tmp_path, monkeypatch):
    received: list[Namespace] = []
    monkeypatch.setattr(train_blcs_ablation, "run_experiment", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_blcs_ablation",
            "--arm",
            "position",
            "--data-root",
            str(tmp_path),
            "--output",
            "../escaped",
        ],
    )
    with pytest.raises(ValueError):
        train_blcs_ablation.main()
    assert received == []
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    ("arm", "weights"),
    [
        ("position", (0.0, 0.0, 0.0, 0.0)),
        ("reprojection", (0.1, 0.0, 0.0, 0.0)),
        ("velocity", (0.1, 0.1, 0.0, 0.0)),
        ("rollout_position", (0.1, 0.1, 1.0, 0.0)),
        ("rollout_velocity", (0.1, 0.1, 1.0, 0.1)),
        ("parameters", (0.1, 0.1, 1.0, 0.1)),
    ],
)
def test_progressive_losses_are_explicit_and_keep_model_capacity_matched(
    tmp_path, arm, weights
):
    from src.tasks.blcs.ablation.progressive_losses import experiment_config

    config = experiment_config(
        data_root=tmp_path,
        scene_dir="blcs/declared-snapshot",
        output="blcs/ablation/test",
        arm=arm,
        seed=43,
    )
    assert (
        tuple(
            config.loss[key]
            for key in (
                "reprojection_weight",
                "velocity_weight",
                "rollout_position_weight",
                "rollout_velocity_weight",
            )
        )
        == weights
    )
    assert config.model.predict_velocity is True
    assert config.model.physics_head is (arm == "parameters")
    assert config.loss.parameters_weight == (0.01 if arm == "parameters" else 0)
    assert config.data.scene_dir == "blcs/declared-snapshot"
    assert config.run.test_after_fit is False
