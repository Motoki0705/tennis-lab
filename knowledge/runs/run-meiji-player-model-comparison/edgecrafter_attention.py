"""Precision rewrite for the ECDet decoder self-attention score graph."""

from __future__ import annotations

from importlib import import_module
from typing import Any

FP32_ATTENTION_CONDITION = "trt16-fp32attn"
FP32_ATTENTION_SETTING = "fp32_decoder_attention_softmax"
SELF_ATTENTION_SOFTMAX = tuple(
    f"/model/decoder/decoder/layers.{index}/self_attn/Softmax" for index in range(4)
)


class AttentionRewriteUnavailable(ValueError):
    """The source does not have the diagnosed FP16 attention region."""


def promote_attention_scores(model: Any, softmax_names: tuple[str, ...]) -> Any:
    """Compute QK and Softmax in FP32 and retain the original FP16 interface."""
    onnx = import_module("onnx")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    by_name = {node.name: node for node in graph.node}
    missing = set(softmax_names) - by_name.keys()
    if missing:
        raise AttentionRewriteUnavailable(
            f"Missing decoder Softmax nodes: {sorted(missing)}"
        )
    producers = {value: node for node in graph.node for value in node.output}
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.output, *graph.value_info)
    }
    half, single = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    replacements: dict[str, list[Any]] = {}
    removed = set()
    for name in softmax_names:
        softmax = by_name[name]
        (score,) = softmax.input
        (probabilities,) = softmax.output
        matmul = producers.get(score)
        if (
            softmax.op_type != "Softmax"
            or matmul is None
            or matmul.op_type != "MatMul"
            or [node.name for node in graph.node if score in node.input] != [name]
            or score in {value.name for value in graph.output}
            or any(
                value_types.get(value) != half
                for value in (*matmul.input, score, probabilities)
            )
        ):
            raise AttentionRewriteUnavailable(
                f"Expected an isolated FP16 MatMul-to-Softmax region at {name}"
            )
        fp32_inputs = [
            f"{name}_fp32_input_{index}" for index in range(len(matmul.input))
        ]
        replacements[matmul.name] = [
            *(
                onnx.helper.make_node("Cast", [original], [cast], name=cast, to=single)
                for original, cast in zip(matmul.input, fp32_inputs, strict=True)
            ),
            onnx.helper.make_node(
                "MatMul", fp32_inputs, [f"{score}_fp32"], name=matmul.name
            ),
        ]
        widened = onnx.NodeProto()
        widened.CopyFrom(softmax)
        widened.input[0] = f"{score}_fp32"
        widened.output[0] = f"{probabilities}_fp32"
        replacements[name] = [
            widened,
            onnx.helper.make_node(
                "Cast",
                [widened.output[0]],
                [probabilities],
                name=f"{name}_restore_fp16",
                to=half,
            ),
        ]
        removed.add(score)
    nodes = [
        replacement
        for node in graph.node
        for replacement in replacements.get(node.name, [node])
    ]
    infos = [value for value in graph.value_info if value.name not in removed]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    graph.value_info.extend(infos)
    onnx.checker.check_model(inferred, full_check=True)
    return onnx.shape_inference.infer_shapes(inferred, strict_mode=True)
