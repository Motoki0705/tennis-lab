"""CPU contracts for pinning attention operators to FP32."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from minimal_repro import run_onnx
from onnx import TensorProto, helper
from pin_precision import pin_fp32

PREFIX = "/network/blocks.0/attn"


def write_attention_tail(path: Path) -> None:
    nodes = [
        helper.make_node(
            "Softmax", ["scores"], ["probabilities"], name=f"{PREFIX}/Softmax"
        ),
        helper.make_node(
            "MatMul",
            ["probabilities", "value"],
            ["weighted"],
            name=f"{PREFIX}/MatMul_1",
        ),
        helper.make_node("Relu", ["weighted"], ["result"], name="after"),
    ]
    graph = helper.make_graph(
        nodes,
        "attention-tail",
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT16, [2, 4, 4]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT16, [2, 4, 3]),
        ],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT16, [2, 4, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_pinned_region_computes_in_fp32_and_returns_fp16(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    write_attention_tail(source)
    rng = np.random.default_rng(0)
    feeds = {
        "scores": rng.normal(size=(2, 4, 4)).astype(np.float16),
        "value": rng.normal(size=(2, 4, 3)).astype(np.float16),
    }

    report = pin_fp32(source, target, suffixes=("Softmax", "MatMul_1"), blocks=1)

    model = onnx.load(target)
    assert report["pinned_operators"] == 2
    assert report["casts_to_fp32"] == 2
    assert report["casts_to_fp16"] == 1
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        item.name: item.type.tensor_type.elem_type for item in inferred.graph.value_info
    }
    assert types["probabilities__fp32"] == TensorProto.FLOAT
    assert types["weighted__fp32"] == TensorProto.FLOAT
    assert types["weighted"] == TensorProto.FLOAT16
    scores = feeds["scores"].astype(np.float32)
    exp = np.exp(scores - scores.max(-1, keepdims=True))
    reference = (exp / exp.sum(-1, keepdims=True)) @ feeds["value"].astype(np.float32)
    expected = np.maximum(reference, 0).astype(np.float16)
    np.testing.assert_array_equal(run_onnx(target, feeds)["result"], expected)


def test_pin_requires_every_block_operator(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    write_attention_tail(source)

    with pytest.raises(ValueError, match="Region operators not found"):
        pin_fp32(source, tmp_path / "target.onnx", suffixes=("Softmax",), blocks=2)
