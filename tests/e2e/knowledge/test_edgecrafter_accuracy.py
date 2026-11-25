"""Check the EdgeCrafter accuracy entry points without loading model weights."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

RUN = (
    Path(__file__).resolve().parents[3]
    / "knowledge/runs/run-meiji-player-model-comparison"
)


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
    tmp_path: Path, model: str, backends: list[str], phases: list[str]
) -> None:
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
if phase in {"export", "build"}:
    directory = Path(arguments[arguments.index("--engine") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    receipt = "export.json" if phase == "export" else "engine.json"
    (directory / receipt).touch()
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    calls_path = tmp_path / "commands.jsonl"
    subprocess.run(
        ["bash", str(RUN / "edgecrafter_run.sh"), model, "a" * 64],
        cwd=tmp_path,
        env={**os.environ, "COMMAND_LOG": str(calls_path), "TENNIS_REPRO_DIR": ""},
        check=True,
        timeout=10,
    )
    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
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
