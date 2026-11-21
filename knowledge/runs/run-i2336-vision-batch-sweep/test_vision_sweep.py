"""CPU coverage for the build and measurement dependency in the sweep CLI."""

from types import SimpleNamespace

import pytest
import sweep_vision
from benchmark import write_json
from vision_inputs import VITPOSE, read


@pytest.mark.parametrize("failure", ["build", "range"])
def test_failed_plan_never_reaches_measurement(tmp_path, monkeypatch, failure):
    root = tmp_path / "out"
    data = SimpleNamespace(
        model=VITPOSE,
        identity={"model": VITPOSE},
        input_shape=(3, 256, 192),
        multiplier=2,
        directory=tmp_path / "inputs",
    )
    args = SimpleNamespace(
        output=root, reference=None, batch=[16], condition=["2"], reuse_conversion=None
    )
    monkeypatch.setenv("TENNIS_RUN_ID", "cpu-fixture")

    def build(_data, _condition, _root, report, _args):
        success = failure != "build"
        report["stages"]["build-2"] = {
            "exit_code": 0 if success else 1,
            "verified": success,
        }
        write_json(root / "sweep.json", report)
        return root / "engine-2" if success else None

    def stage(_root, report, name, command, *, rejected_shape=None):
        assert "--mode" in command and command[command.index("--mode") + 1] == "reject"
        assert rejected_shape in {"pixels=(0, 3, 256, 192)", "pixels=(66, 3, 256, 192)"}
        success = name == "reject-c2-b0"
        report["stages"][name] = {"exit_code": 1 if success else 0, "verified": success}
        write_json(root / "sweep.json", report)
        return success

    monkeypatch.setattr(sweep_vision, "run_stage", stage)
    monkeypatch.setattr(sweep_vision, "build_conversion", build)
    monkeypatch.setattr(
        sweep_vision,
        "load_conversion",
        lambda *_: (
            None,
            {
                "input_profile": {
                    "pixels": [[2, 3, 256, 192], [32, 3, 256, 192], [64, 3, 256, 192]]
                }
            },
            None,
        ),
    )

    assert sweep_vision.run_sweep(data, args) == 1
    stages = read(root / "sweep.json")["stages"]
    assert set(stages) == (
        {"build-2"}
        if failure == "build"
        else {"build-2", "reject-c2-b0", "reject-c2-b33"}
    )
    assert not (root / "c2-b16").exists()
