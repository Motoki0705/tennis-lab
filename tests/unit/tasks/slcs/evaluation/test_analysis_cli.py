"""Executable analysis boundaries preserve arguments and fail before side effects."""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

MODULES = (
    "calibrate_ball_velocity",
    "compare_ball_anchors",
    "compare_ball_transitions",
    "compare_conditions",
    "evaluate_run",
    "report_validation",
    "render_pr_clip",
)


@pytest.mark.parametrize("name", MODULES)
def test_cpu_module_help(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"src.tasks.slcs.scripts.{name}", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout


@pytest.mark.parametrize("name", ("compare_ball_anchors", "compare_ball_transitions"))
def test_saved_comparison_cli_routes_explicit_inputs(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"src.tasks.slcs.scripts.{name}")
    calls: list[dict[str, Any]] = []
    function = (
        "save_ball_anchor_comparison"
        if name == "compare_ball_anchors"
        else "save_ball_transition_comparison"
    )
    monkeypatch.setattr(module, function, lambda **kwargs: calls.append(kwargs))
    argv = [
        name,
        "--baseline",
        str(tmp_path / "baseline"),
        "--candidate",
        str(tmp_path / "candidate"),
        "--output",
        str(tmp_path / "report.json"),
    ]
    if name == "compare_ball_transitions":
        argv.extend(["--fast-speed-mps", "12.5"])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert calls[0]["baseline"] == tmp_path / "baseline"
    assert calls[0]["output"] == tmp_path / "report.json"
    if name == "compare_ball_transitions":
        assert calls[0]["fast_speed_mps"] == 12.5
    argv[2] = "relative"
    with pytest.raises(ValueError, match="absolute"):
        module.main()
    assert len(calls) == 1


def test_calibration_cli_preserves_train_only_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.scripts import calibrate_ball_velocity as cli

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli, "calibrate_training_run", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--output-root",
            str(tmp_path),
            "--training-run",
            "slcs/train/a/b",
            "--output",
            "slcs/analyze/a/b",
            "--velocity-scale-mps",
            "4.5",
            "--gradient-ratio",
            "0.2",
            "--seed",
            "7",
            "--batch-size",
            "3",
        ],
    )
    cli.main()
    assert calls == [
        dict(
            output_root=tmp_path,
            training_run="slcs/train/a/b",
            output="slcs/analyze/a/b",
            device="cpu",
            batch_size=3,
            seed=7,
            velocity_scale_mps=4.5,
            gradient_ratio=0.2,
        )
    ]


def test_report_cli_keeps_all_training_labels_and_rejects_duplicate_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.scripts import report_validation as cli

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "generate_report", lambda **kwargs: calls.append(kwargs))
    argv = [
        "report",
        "--evaluation",
        f"A={tmp_path}/eval",
        "--training",
        f"A={tmp_path}/train",
        "--output-root",
        str(tmp_path),
        "--output",
        "slcs/visualize/a/b",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls[0]["evaluations"] == {"A": tmp_path / "eval"}
    assert calls[0]["training"] == {"A": tmp_path / "train"}
    argv.extend(["--evaluation", f"A={tmp_path}/other"])
    with pytest.raises(ValueError, match="unique"):
        cli.main()
    assert len(calls) == 1


def test_render_cli_passes_typed_request_and_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.slcs.scripts import render_pr_clip as cli
    from src.tasks.slcs.visualization.pr_clip import RenderRequest

    calls: list[tuple[RenderRequest, tuple[str, ...]]] = []

    def render(request: RenderRequest, *, command_line: tuple[str, ...]) -> Path:
        calls.append((request, command_line))
        return tmp_path

    monkeypatch.setattr(cli, "render", render)
    argv = [
        "render",
        "--overlay",
        str(tmp_path / "rgb.mp4"),
        "--scene",
        str(tmp_path / "3d.mp4"),
        "--output-root",
        str(tmp_path),
        "--experiment",
        "fixture",
        "--run-id",
        "one",
        "--label",
        "Fixture",
        "--model",
        "baseline",
        "--clip-id",
        "clip",
        "--camera-id",
        "cam0",
        "--checkpoint-sha256",
        "a" * 64,
        "--epoch",
        "56",
        "--start",
        "0",
        "--end",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls[0][0].epoch == 56
    assert calls[0][0].fps == 10
    assert calls[0][1] == tuple(argv)
