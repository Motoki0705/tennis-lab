"""Rewrite the released GVHMR attention masks for TensorRT FP16."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np

ATTENTION_MASK_ENCODING = "additive-finite"
ATTENTION_MASK_FILL = -32768.


def _mask_input(where: Any, producers: dict[str, Any], initializers: dict[str, Any], onnx: Any) -> str:
    cast = producers.get(where.input[0])
    if cast is None or cast.op_type != "Cast" or not any(
        attr.name == "to" and attr.i == onnx.TensorProto.BOOL for attr in cast.attribute
    ):
        raise ValueError(f"Expected a boolean mask Cast at {where.name}")
    expand = producers.get(cast.input[0])
    reshape = producers.get(expand.input[0]) if expand is not None and expand.op_type == "Expand" else None
    if reshape is None or reshape.op_type != "Reshape":
        raise ValueError(f"Expected a broadcast attention mask at {where.name}")

    fill_name = where.input[1]
    fill = initializers.get(fill_name)
    if fill is None:
        constant = producers.get(fill_name)
        if constant is not None and constant.op_type == "Constant":
            fill = next((attr.t for attr in constant.attribute if attr.name == "value"), None)
    if fill is None:
        raise ValueError(f"Expected a constant mask fill at {where.name}")
    value = onnx.numpy_helper.to_array(fill)
    if value.dtype != np.float16 or value.shape != () or not np.isneginf(value):
        raise ValueError(f"Expected a scalar FP16 -inf fill at {where.name}")
    return str(reshape.output[0])


def rewrite_attention_masks(path: Path, *, blocks: int) -> None:
    """Replace the two Where lowerings immediately upstream of each attention Softmax.

    Both masks retain their Reshape and broadcast in Where. Full-length inputs
    always leave an unmasked key in each query row.
    """
    onnx = importlib.import_module("onnx")
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    producers = {value: node for node in graph.node for value in node.output}
    initializers = {item.name: item for item in graph.initializer}
    softmaxes = {
        node.name: node for node in graph.node
        if node.op_type == "Softmax" and re.fullmatch(r"/network/blocks\.\d+/attn/Softmax", node.name)
    }
    expected = {f"/network/blocks.{index}/attn/Softmax" for index in range(blocks)}
    if softmaxes.keys() != expected or not expected:
        raise ValueError("GVHMR attention Softmax layout changed")

    masks: dict[str, str] = {}
    for name, softmax in softmaxes.items():
        scope = name.removesuffix("Softmax")
        score = softmax.input[0]
        for _ in range(2):
            where = producers.get(score)
            if where is None or where.op_type != "Where" or not where.name.startswith(scope):
                raise ValueError(f"Expected two mask Where nodes before {name}")
            masks[where.name] = _mask_input(where, producers, initializers, onnx)
            score = where.input[2]
        scaled_scores = producers.get(score)
        if scaled_scores is None or scaled_scores.op_type != "Div" or not scaled_scores.name.startswith(scope):
            raise ValueError(f"Expected scaled attention scores before {name}")

    finite_fill, zero = "__attention_mask_fill", "__attention_mask_zero"
    for name, value in ((finite_fill, ATTENTION_MASK_FILL), (zero, 0.)):
        graph.initializer.append(onnx.numpy_helper.from_array(np.array(value, dtype=np.float16), name=name))
    nodes = []
    for node in graph.node:
        if node.name in masks:
            bias = f"{node.output[0]}_finite_bias"
            nodes.append(onnx.helper.make_node(
                "Where", [masks[node.name], finite_fill, zero], [bias], name=f"{node.name}/Bias",
            ))
            nodes.append(onnx.helper.make_node("Add", [node.input[2], bias], list(node.output), name=node.name))
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save_model(model, path)
