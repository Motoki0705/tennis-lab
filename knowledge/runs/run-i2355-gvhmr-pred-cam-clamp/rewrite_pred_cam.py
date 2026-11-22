"""Replace the exporter-generated in-place pred_cam clamp with Slice/Concat."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_map(graph):
    return {name: node for node in graph.node for name in node.output}


def _constant_array(model, name: str) -> np.ndarray:
    import onnx

    initializer = next(
        (item for item in model.graph.initializer if item.name == name), None
    )
    if initializer is not None:
        return onnx.numpy_helper.to_array(initializer)
    producer = _producer_map(model.graph).get(name)
    if producer is None or producer.op_type != "Constant":
        raise ValueError(f"Expected a constant tensor for {name!r}")
    value = next(
        (attribute.t for attribute in producer.attribute if attribute.name == "value"),
        None,
    )
    if value is None:
        raise ValueError(f"Constant {producer.name!r} has no tensor value")
    return onnx.numpy_helper.to_array(value)


def _attribute_int(node, name: str, default: int | None = None) -> int:
    attribute = next((item for item in node.attribute if item.name == name), None)
    if attribute is None:
        if default is None:
            raise ValueError(f"{node.name!r} has no {name!r} attribute")
        return default
    return int(attribute.i)


def _output_type(model, name: str) -> tuple[int, tuple[int | None, ...]]:
    value = next((item for item in model.graph.output if item.name == name), None)
    if value is None:
        raise ValueError(f"{name!r} is not a graph output")
    tensor = value.type.tensor_type
    shape = tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in tensor.shape.dim
    )
    return int(tensor.elem_type), shape


def _validate_clamp_scatter(model):
    import onnx

    candidates = [
        node
        for node in model.graph.node
        if node.op_type == "ScatterND" and list(node.output) == ["pred_cam"]
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one ScatterND that produces pred_cam, "
            f"found {len(candidates)}"
        )
    scatter = candidates[0]
    if scatter.attribute:
        reduction = next(
            (item.s.decode() for item in scatter.attribute if item.name == "reduction"),
            "none",
        )
        if reduction != "none":
            raise ValueError(f"Unexpected ScatterND reduction {reduction!r}")
    if len(scatter.input) != 3:
        raise ValueError("pred_cam ScatterND must have data, indices, and updates")

    producer = _producer_map(model.graph)
    base, _indices, updates = scatter.input
    reshape = producer.get(updates)
    expand = (
        producer.get(reshape.input[0])
        if reshape and reshape.op_type == "Reshape"
        else None
    )
    clip = (
        producer.get(expand.input[0]) if expand and expand.op_type == "Expand" else None
    )
    gather = producer.get(clip.input[0]) if clip and clip.op_type == "Clip" else None
    if gather is None or gather.op_type != "Gather" or gather.input[0] != base:
        raise ValueError("ScatterND updates are not the clamp of pred_cam[..., 0]")
    if _attribute_int(gather, "axis", 0) != 2:
        raise ValueError("pred_cam clamp Gather must use axis 2")
    gather_index = _constant_array(model, gather.input[1])
    if gather_index.shape != () or int(gather_index) != 0:
        raise ValueError("pred_cam clamp Gather must select scalar index 0")
    if len(clip.input) < 2 or not clip.input[1]:
        raise ValueError("pred_cam clamp Clip has no lower bound")
    lower = _constant_array(model, clip.input[1])
    if lower.shape != () or float(lower) != 0.25:
        raise ValueError(f"Unexpected pred_cam lower bound {lower!r}")
    if len(clip.input) >= 3 and clip.input[2]:
        raise ValueError("pred_cam clamp unexpectedly has an upper bound")

    elem_type, shape = _output_type(model, "pred_cam")
    if elem_type not in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}:
        raise ValueError(f"Unexpected pred_cam element type {elem_type}")
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Unexpected pred_cam shape {shape}")
    return scatter, base, elem_type, shape


def _prune_unreachable(model) -> int:
    graph = model.graph
    required = {value.name for value in graph.output}
    retained = []
    for node in reversed(graph.node):
        if required.intersection(node.output):
            retained.append(node)
            required.update(name for name in node.input if name)
    retained.reverse()
    removed = len(graph.node) - len(retained)
    del graph.node[:]
    graph.node.extend(retained)
    initializers = [item for item in graph.initializer if item.name in required]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    value_info = [item for item in graph.value_info if item.name in required]
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    return removed


def rewrite_pred_cam_clamp(source: Path, target: Path) -> dict[str, object]:
    """Rewrite only the exact in-place clamp lowering used by the pinned export."""
    import onnx
    from onnx import helper, numpy_helper

    source = source.resolve(strict=True)
    target = target.resolve() if target.exists() else target.absolute()
    if source == target:
        raise ValueError("Source and target ONNX paths must differ")
    model = onnx.load(source, load_external_data=True)
    scatter, base, elem_type, shape = _validate_clamp_scatter(model)
    original_nodes = len(model.graph.node)
    original_scatter = sum(node.op_type == "ScatterND" for node in model.graph.node)

    prefix = "__pred_cam_clamp_concat"
    existing = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    } | {item.name for item in model.graph.initializer}
    names = {
        key: f"{prefix}_{key}"
        for key in ("starts0", "ends1", "starts1", "ends3", "axes", "steps", "min")
    }
    if existing.intersection(names.values()):
        raise ValueError("The pred_cam rewrite names already exist")
    float_dtype = np.float16 if elem_type == onnx.TensorProto.FLOAT16 else np.float32
    tensors = {
        names["starts0"]: np.array([0], dtype=np.int64),
        names["ends1"]: np.array([1], dtype=np.int64),
        names["starts1"]: np.array([1], dtype=np.int64),
        names["ends3"]: np.array([3], dtype=np.int64),
        names["axes"]: np.array([2], dtype=np.int64),
        names["steps"]: np.array([1], dtype=np.int64),
        names["min"]: np.array(0.25, dtype=float_dtype),
    }
    model.graph.initializer.extend(
        numpy_helper.from_array(value, name=name) for name, value in tensors.items()
    )
    first = f"{prefix}_first"
    clipped = f"{prefix}_first_clipped"
    rest = f"{prefix}_rest"
    replacement = [
        helper.make_node(
            "Slice",
            [base, names["starts0"], names["ends1"], names["axes"], names["steps"]],
            [first],
            name=f"{prefix}/SliceFirst",
        ),
        helper.make_node(
            "Clip",
            [first, names["min"], ""],
            [clipped],
            name=f"{prefix}/ClipFirst",
        ),
        helper.make_node(
            "Slice",
            [base, names["starts1"], names["ends3"], names["axes"], names["steps"]],
            [rest],
            name=f"{prefix}/SliceRest",
        ),
        helper.make_node(
            "Concat",
            [clipped, rest],
            ["pred_cam"],
            name=f"{prefix}/Concat",
            axis=2,
        ),
    ]
    index = list(model.graph.node).index(scatter)
    nodes = list(model.graph.node)
    nodes[index : index + 1] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    pruned = _prune_unreachable(model)
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target, save_as_external_data=False)
    rewritten = onnx.load(target, load_external_data=False)
    rewritten_scatter = sum(
        node.op_type == "ScatterND" for node in rewritten.graph.node
    )
    if rewritten_scatter:
        raise RuntimeError("The rewritten graph still contains ScatterND")
    return {
        "source": str(source),
        "target": str(target),
        "source_sha256": sha256_file(source),
        "target_sha256": sha256_file(target),
        "pred_cam_dtype": "fp16" if elem_type == onnx.TensorProto.FLOAT16 else "fp32",
        "pred_cam_shape": list(shape),
        "source_nodes": original_nodes,
        "target_nodes": len(rewritten.graph.node),
        "pruned_nodes": pruned,
        "scatternd_before": original_scatter,
        "scatternd_after": rewritten_scatter,
        "replacement": ["Slice", "Clip", "Slice", "Concat"],
    }
