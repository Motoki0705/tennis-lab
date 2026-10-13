"""Experiment entry points validate paths/arguments before run side effects."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from src.tasks.blcs.experiments.scripts import issue1044, repeat_selected


def test_training_boundary_preserves_the_fixed_run_arguments(tmp_path, monkeypatch):
    received: list[Namespace] = []
    monkeypatch.setattr(issue1044, "run_experiment", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1044",
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
    issue1044.main()
    assert len(received) == 1
    args = received[0]
    assert (args.arm, args.seed, args.steps) == ("rollout_velocity", 43, 4000)
    assert args.data_root == tmp_path
    assert args.output == "blcs/i1044/boundary-test"
    assert not (args.smoke or args.chunked or args.gan)


def test_training_boundary_rejects_output_escape_before_runner(tmp_path, monkeypatch):
    received: list[Namespace] = []
    monkeypatch.setattr(issue1044, "run_experiment", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1044",
            "--arm",
            "position",
            "--data-root",
            str(tmp_path),
            "--output",
            "../escaped",
        ],
    )
    with pytest.raises(ValueError):
        issue1044.main()
    assert received == []
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("arguments", [[], ["--select-only", "--seed", "43"]])
def test_repeat_rejects_incomplete_or_conflicting_mode_before_selection(
    arguments, monkeypatch
):
    received: list[Path] = []
    monkeypatch.setattr(repeat_selected, "select_additional_arm", received.append)
    monkeypatch.setattr(sys, "argv", ["repeat_selected", *arguments])
    with pytest.raises(SystemExit, match="2"):
        repeat_selected.main()
    assert received == []
