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
        script = next(Path(value).name for value in command if value.endswith(".py"))
        condition = command[command.index("--condition") + 1]
        executed.append((script, condition))
        if script == "convert.py":
            if condition in sweep.CLIP_RECIPES:
                assert (
                    command[command.index("--ptq-recipe") + 1]
                    == sweep.CLIP_RECIPES[condition]
                )
            print("TensorRT ONNX parse failed", file=stdout)
            return SimpleNamespace(returncode=9)
        assert command[command.index("--profiles") + 1] == "nsys"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    monkeypatch.setattr(sweep, "verify_inputs", lambda _: ({"tensors": {}}, []))
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-regression")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    root = tmp_path / "run"

    assert sweep.run_sweep("clipreid", tmp_path, root) == 1
    assert executed == [
        ("benchmark.py", condition)
        for condition in ("0", "1")
        for _ in sweep.CLIP_BATCHES
    ] + [("convert.py", condition) for condition in ("2", "3a", "3b")]
    stages = json.loads((root / "sweep.json").read_text())["stages"]
    assert stages["build-2"]["exit_code"] == 9
    assert stages["build-2"]["verified"] is False
    assert "TensorRT ONNX parse failed" in (root / stages["build-2"]["log"]).read_text()
