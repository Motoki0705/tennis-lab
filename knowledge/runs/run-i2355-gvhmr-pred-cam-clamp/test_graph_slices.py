from pathlib import Path

import numpy as np
import onnx
import pytest
from graph_slices import infer_tensor_specs, slice_graph
from onnx import TensorProto, helper, numpy_helper


def write_branching_graph(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "bias"], ["boundary"], name="before"),
            helper.make_node("Mul", ["boundary", "scale"], ["hidden"], name="kept"),
            helper.make_node("Relu", ["hidden"], ["result"], name="output"),
            helper.make_node("Neg", ["x"], ["unused"], name="unused"),
        ],
        "branching",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 3])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT16, [2, 3])],
        initializer=[
            numpy_helper.from_array(np.ones((3,), dtype=np.float16), name="bias"),
            numpy_helper.from_array(np.full((3,), 2, dtype=np.float16), name="scale"),
        ],
        value_info=[
            helper.make_tensor_value_info("boundary", TensorProto.FLOAT16, [2, 3]),
            helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [2, 3]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_slice_graph_stops_at_promoted_input(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    target = tmp_path / "slice.onnx"
    write_branching_graph(source)

    result = slice_graph(
        source,
        target,
        output_name="result",
        output_type=TensorProto.FLOAT16,
        output_shape=(2, 3),
        promoted_inputs={"boundary": (TensorProto.FLOAT16, (2, 3))},
    )

    model = onnx.load(target)
    assert result["nodes"] == 2
    assert [node.name for node in model.graph.node] == ["kept", "output"]
    assert [value.name for value in model.graph.input] == ["boundary"]
    assert [value.name for value in model.graph.output] == ["result"]
    assert [value.name for value in model.graph.initializer] == ["scale"]


def test_infer_tensor_specs_returns_selected_static_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    write_branching_graph(source)

    specs = infer_tensor_specs(source, ["boundary", "result", "boundary"])

    assert specs == {
        "boundary": (TensorProto.FLOAT16, (2, 3)),
        "result": (TensorProto.FLOAT16, (2, 3)),
    }


def test_slice_graph_rejects_unknown_promoted_input(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    write_branching_graph(source)

    with pytest.raises(ValueError, match="Unknown promoted inputs"):
        slice_graph(
            source,
            tmp_path / "slice.onnx",
            output_name="result",
            output_type=TensorProto.FLOAT16,
            output_shape=(2, 3),
            promoted_inputs={"missing": (TensorProto.FLOAT16, (2, 3))},
        )


def test_slice_graph_omits_unconsumed_promoted_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    target = tmp_path / "slice.onnx"
    write_branching_graph(source)

    result = slice_graph(
        source,
        target,
        output_name="hidden",
        output_type=TensorProto.FLOAT16,
        output_shape=(2, 3),
        promoted_inputs={
            "boundary": (TensorProto.FLOAT16, (2, 3)),
            "unused": (TensorProto.FLOAT16, (2, 3)),
        },
    )

    model = onnx.load(target)
    assert [value.name for value in model.graph.input] == ["boundary"]
    assert result["promoted_inputs"] == ["boundary"]
    assert result["unused_promoted_inputs"] == ["unused"]
