"""A failed TensorRT build leaves the explicit PyTorch measurements intact."""

import json
import signal
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


def test_gvhmr_retry_passes_zero_builder_optimization_level(tmp_path):
    command = sweep.build_command(
        "gvhmr",
        "2",
        tmp_path / "inputs",
        tmp_path / "engine-2",
        fixed_length=177,
        builder_optimization_level=0,
    )
    assert command[command.index("--fixed-length") + 1] == "177"
    assert command[command.index("--builder-optimization-level") + 1] == "0"
    assert command[
        command.index("--batch-profile") + 1 : command.index("--batch-profile") + 3
    ] == ["4", "4"]


def test_fixed_gvhmr_build_records_failed_shape_and_exits(tmp_path, monkeypatch):
    root = tmp_path / "run"
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-static-build")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    monkeypatch.setattr(
        sweep.sys,
        "argv",
        [
            "sweep.py",
            "--model",
            "gvhmr",
            "--inputs",
            str(tmp_path),
            "--output",
            str(root),
            "--fixed-length",
            "177",
            "--build-fixed-batch",
            "2",
        ],
    )
    commands = []

    def run(command, *, stdout, stderr, check):
        commands.append(command)
        print("Fatal Python error: Segmentation fault", file=stdout)
        return SimpleNamespace(returncode=-signal.SIGSEGV)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    assert sweep.main() == 1
    (command,) = commands
    assert Path(command[4]).name == "build_gvhmr_static.py"
    assert "--batch-profile" not in command
    assert command[-4:] == ["--length", "177", "--batch-size", "2"]
    report = json.loads((root / "sweep.json").read_text())
    assert report["fixed_shape"] == {"length": 177, "batch_size": 2}
    (stage,) = report["stages"].values()
    assert stage["exit_code"] == -signal.SIGSEGV
    assert stage["verified"] is False
    assert "Segmentation fault" in (root / stage["log"]).read_text()
