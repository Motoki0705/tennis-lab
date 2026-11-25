"""Check the EdgeCrafter queue entry points without loading model weights."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

RUN = (
    Path(__file__).resolve().parents[3]
    / "knowledge/runs/run-meiji-player-model-comparison"
)


@pytest.fixture
def command_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\n"
        """import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ["COMMAND_LOG"]).open("a", encoding="utf-8") as log:
    log.write(json.dumps(arguments) + "\\n")
phase = arguments[3]
if os.environ.get("MISSING_PHASE") == phase:
    sys.exit(0)
if phase in {"export", "build"}:
    directory = Path(arguments[arguments.index("--engine") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    receipt = "export.json" if phase == "export" else "engine.json"
    (directory / receipt).write_text("{}", encoding="utf-8")
    Path(arguments[arguments.index("--receipt") + 1]).write_text("{}", encoding="utf-8")
if phase == "evaluate":
    if os.environ.get("FAILED_BACKEND") == arguments[arguments.index("--backend") + 1]:
        sys.exit(9)
    directory = Path(arguments[arguments.index("--output") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.jsonl"):
        (directory / name).write_text("{}", encoding="utf-8")
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    calls_path = tmp_path / "commands.jsonl"
    monkeypatch.setenv("COMMAND_LOG", str(calls_path))
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    return calls_path


def _calls(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("model", "backends", "phases"),
    [
        (
            "ecpose-l-o365",
            ["pt16", "trt16"],
            ["evaluate", "export", "build", "evaluate"],
        ),
        ("ecdet-l-coco", ["trt16"], ["export", "build", "evaluate"]),
    ],
)
def test_queue_entrypoint_only_requests_new_half_precision_accuracy(
    tmp_path: Path,
    command_log: Path,
    model: str,
    backends: list[str],
    phases: list[str],
) -> None:
    subprocess.run(
        ["bash", str(RUN / "edgecrafter_run.sh"), model, "a" * 64],
        cwd=tmp_path,
        check=True,
        timeout=10,
    )
    calls = _calls(command_log)
    assert [call[3] for call in calls] == phases
    evaluations = [call for call in calls if call[3] == "evaluate"]
    assert [call[call.index("--backend") + 1] for call in evaluations] == backends
    for call in evaluations:
        assert call[call.index("--parity") + 1] == "not_measured"
        assert ("--pose-box-margin" in call) == model.startswith("ecpose-")
    assert evaluations[-1][evaluations[-1].index("--output") + 1].endswith(
        "/trt16-trace2"
    )


@pytest.mark.parametrize(
    ("variable", "value", "expected_error", "successful_output"),
    [
        (
            "FAILED_BACKEND",
            "pt16",
            "pt16 infeasible. Phase accuracy exited 9",
            "trt16-trace2",
        ),
        (
            "MISSING_PHASE",
            "build",
            "trt16 infeasible. Phase engine_build did not produce",
            "pt16",
        ),
    ],
)
def test_accuracy_reports_failed_condition_without_losing_other_output(
    tmp_path: Path,
    command_log: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    expected_error: str,
    successful_output: str,
) -> None:
    monkeypatch.setenv(variable, value)
    result = subprocess.run(
        ["bash", str(RUN / "edgecrafter_run.sh"), "ecpose-l-o365", "a" * 64],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert f"ecpose-l-o365 {expected_error}" in result.stderr
    output = tmp_path / "outputs/player-model-comparison/ecpose-l-o365"
    assert (output / successful_output / "predictions.jsonl").is_file()
    if value == "build":
        assert not (output / "trt16-trace2/predictions.jsonl").exists()
        assert [call[3] for call in _calls(command_log)] == [
            "evaluate",
            "export",
            "build",
        ]


@pytest.mark.parametrize(
    ("backend_args", "backends"), [((), ("pt16", "trt16")), (("pt16",), ("pt16",))]
)
def test_speed_runs_selected_half_precision_backends_at_both_pinned_batch_sizes(
    tmp_path: Path,
    command_log: Path,
    backend_args: tuple[str, ...],
    backends: tuple[str, ...],
) -> None:
    model = "ecpose-x-o365"
    output = Path("outputs/player-model-comparison") / model
    engine = tmp_path / output / "trt16-trace2/engine"
    if "trt16" in backends:
        engine.mkdir(parents=True)
        for name in ("engine.plan", "engine.json"):
            (engine / name).write_text("fixture", encoding="utf-8")
    subprocess.run(
        [
            "bash",
            str(RUN / "edgecrafter_speed_run.sh"),
            model,
            "fixed-inputs",
            "b" * 40,
            *backend_args,
        ],
        cwd=tmp_path,
        check=True,
        timeout=10,
    )
    args = [
        dict(zip(call[3::2], call[4::2], strict=True)) for call in _calls(command_log)
    ]
    assert [(row["--backend"], row["--batch-size"]) for row in args] == [
        (backend, batch) for backend in backends for batch in ("1", "2")
    ]
    for row in args:
        assert row["--harness-commit"] == "b" * 40
        assert row["--parity"] == "not_measured"
        assert row["--pose-box-margin"] == "0.10"
        assert Path(row["--output"]) == (
            output / "speed-bbbbbbbbb-threads4-1" / row["--backend"]
        )
        if row["--backend"] == "trt16":
            assert tmp_path / row["--engine"] == engine
        else:
            assert "--engine" not in row


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (["--backend", "pt32", "--parity", "not_measured"], "invalid choice: 'pt32'"),
        (["--backend", "pt16"], "the following arguments are required: --parity"),
    ],
)
def test_accuracy_cli_requires_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    error: str,
) -> None:
    script = str(RUN / "edgecrafter.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script,
            "evaluate",
            "--model",
            "ecpose-l-o365",
            "--manifest-sha256",
            "a" * 64,
            "--output",
            "unused",
            *options,
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(script, run_name="__main__")
    assert result.value.code == 2
    assert error in capsys.readouterr().err
