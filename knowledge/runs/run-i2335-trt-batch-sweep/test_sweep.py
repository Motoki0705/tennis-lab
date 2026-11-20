"""A failed TensorRT build leaves the explicit PyTorch measurements intact."""

import json
from pathlib import Path
from types import SimpleNamespace

import sweep


def test_conversion_failure_is_nonzero_without_running_a_replacement_case(
    tmp_path, monkeypatch
):
    executed = []

    def run(command, *, stdout, stderr, check):
        script = Path(command[2]).name
        condition = command[command.index("--condition") + 1]
        executed.append((script, condition))
        if script == "convert.py":
            print("TensorRT ONNX parse failed", file=stdout)
            return SimpleNamespace(returncode=9)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-regression")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    root = tmp_path / "run"

    assert sweep.run_sweep("clipreid", tmp_path, root) == 1
    assert executed == [
        ("benchmark.py", condition)
        for condition in ("0", "1")
        for _ in sweep.CLIP_BATCHES
    ] + [("convert.py", "2")]
    stages = json.loads((root / "sweep.json").read_text())["stages"]
    assert stages["build"]["exit_code"] == 9
    assert stages["build"]["verified"] is False
    assert "TensorRT ONNX parse failed" in (root / stages["build"]["log"]).read_text()
