"""CPU coverage for the build and measurement dependency in the sweep CLI."""

from types import SimpleNamespace

import numpy as np
import pytest
import sweep_vision
from benchmark import write_json
from convert_vit_batch import simplify_hmr2_token_output
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


def test_hmr2_ptq_export_lowers_only_the_singleton_token_squeeze():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, reference

    source = "/hmr/smpl_head/transformer/transformer/Add_17_output_0"

    def model(axis):
        when_true = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["axis"],
                    value=helper.make_tensor("axis", TensorProto.INT64, [1], [axis]),
                ),
                helper.make_node("Squeeze", [source, "axis"], ["squeezed"]),
            ],
            "squeeze",
            [],
            [
                helper.make_tensor_value_info(
                    "squeezed", TensorProto.FLOAT, ["batch", 1024]
                )
            ],
        )
        when_false = helper.make_graph(
            [helper.make_node("Identity", [source], ["unchanged"])],
            "identity",
            [],
            [
                helper.make_tensor_value_info(
                    "unchanged", TensorProto.FLOAT, ["batch", 1, 1024]
                )
            ],
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["one_token"],
                    ["features"],
                    name="/hmr/smpl_head/If",
                    then_branch=when_true,
                    else_branch=when_false,
                )
            ],
            "head",
            [
                helper.make_tensor_value_info(
                    source, TensorProto.FLOAT, ["batch", 1, 1024]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "features", TensorProto.FLOAT, ["batch", 1024]
                )
            ],
            initializer=[helper.make_tensor("one_token", TensorProto.BOOL, [], [True])],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

    with pytest.raises(ValueError, match="singleton token axis"):
        simplify_hmr2_token_output(model(0))
    exported = model(1)
    original = reference.ReferenceEvaluator(model(1))
    simplify_hmr2_token_output(exported)
    onnx.checker.check_model(exported)
    assert all(node.op_type != "If" for node in exported.graph.node)
    lowered = reference.ReferenceEvaluator(exported)
    for batch in (1, 16):
        token = np.arange(batch * 1024, dtype=np.float32).reshape(batch, 1, 1024)
        expected = original.run(None, {source: token})[0]
        np.testing.assert_array_equal(lowered.run(None, {source: token})[0], expected)
