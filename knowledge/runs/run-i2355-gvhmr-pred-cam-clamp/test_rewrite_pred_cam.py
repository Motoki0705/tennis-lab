"""CPU contracts for the exact pred_cam graph rewrite."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from graph_slices import slice_graph
from minimal_repro import compare_minimal, export_minimal
from rewrite_pred_cam import rewrite_pred_cam_clamp


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_rewrite_preserves_the_inplace_clamp_and_removes_scatternd(
    tmp_path: Path, dtype: str
):
    source = tmp_path / f"{dtype}-scatter.onnx"
    target = tmp_path / f"{dtype}-concat.onnx"
    raw_cam = export_minimal(source, batch=2, frames=5, dtype=dtype)

    report = rewrite_pred_cam_clamp(source, target)

    assert report["scatternd_before"] == 1
    assert report["scatternd_after"] == 0
    assert report["replacement"] == ["Slice", "Clip", "Slice", "Concat"]
    assert all(node.op_type != "ScatterND" for node in onnx.load(target).graph.node)
    parity = compare_minimal(source, target, raw_cam)
    assert parity["source_rewrite_max_abs"] == 0


def test_rewrite_rejects_a_graph_without_the_exact_exporter_pattern(tmp_path: Path):
    source = tmp_path / "identity.onnx"
    target = tmp_path / "rewritten.onnx"
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["raw_cam"], ["pred_cam"])],
        "not-the-pinned-clamp",
        [
            onnx.helper.make_tensor_value_info(
                "raw_cam", onnx.TensorProto.FLOAT, [2, 5, 3]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "pred_cam", onnx.TensorProto.FLOAT, [2, 5, 3]
            )
        ],
    )
    onnx.save(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 19)]),
        source,
    )

    with pytest.raises(ValueError, match="exactly one ScatterND"):
        rewrite_pred_cam_clamp(source, target)

    assert not target.exists()


def test_graph_slice_can_promote_an_internal_tensor_to_a_checked_input(tmp_path: Path):
    source = tmp_path / "source.onnx"
    target = tmp_path / "head.onnx"
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3])
    one = onnx.numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="one")
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["x", "one"], ["middle"]),
            onnx.helper.make_node("Mul", ["middle", "middle"], ["y"]),
        ],
        "promoted-input",
        [x],
        [y],
        [one],
    )
    onnx.save(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 19)]),
        source,
    )

    report = slice_graph(
        source,
        target,
        output_name="y",
        output_type=onnx.TensorProto.FLOAT,
        output_shape=(2, 3),
        promoted_inputs={"middle": (onnx.TensorProto.FLOAT, (2, 3))},
    )

    model = onnx.load(target)
    assert report["inputs"] == ["middle"]
    assert [node.op_type for node in model.graph.node] == ["Mul"]
    assert [value.name for value in model.graph.input] == ["middle"]
