"""Exact rewrites of the exported GVHMR attention-mask lowering.

The legacy exporter lowers ``scores.masked_fill(mask, -inf)`` to
``Where(Cast(Expand(Reshape(mask), shape)), -inf, scores)``.  ``Where``
broadcasts its condition, so the explicit ``Expand`` (and the bool-to-bool
``Cast``) is redundant.  ``drop_mask_expand`` feeds the reshaped mask to
``Where`` directly; ``additive_mask`` additionally replaces the full-size
``Where`` with ``scores + Where(mask, -inf, 0)``.  Both rewrites only accept
the exact pattern and fail on anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rewrite_pred_cam import (
    _constant_array,
    _producer_map,
    _prune_unreachable,
    sha256_file,
)

MODES = ("drop-expand", "additive")


def _mask_where_nodes(model):
    """Return (where, reshaped_mask, scores) for every exported masked_fill."""
    import onnx

    producer = _producer_map(model.graph)
    matches = []
    for node in model.graph.node:
        if node.op_type != "Where":
            continue
        cast = producer.get(node.input[0])
        if cast is None or cast.op_type != "Cast":
            continue
        expand = producer.get(cast.input[0])
        if expand is None or expand.op_type != "Expand":
            continue
        reshape = producer.get(expand.input[0])
        if reshape is None or reshape.op_type != "Reshape":
            continue
        (to,) = [item.i for item in cast.attribute if item.name == "to"]
        if to != onnx.TensorProto.BOOL:
            raise ValueError(f"{cast.name!r} does not cast the mask to bool")
        fill = _constant_array(model, node.input[1])
        if fill.shape != () or not np.isneginf(fill):
            raise ValueError(f"{node.name!r} does not fill with -inf")
        matches.append((node, reshape.output[0], node.input[2], fill.dtype))
    return matches


def _fold_constant(model, name: str) -> np.ndarray:
    """Evaluate a tensor that depends only on Constant nodes and initializers."""
    import onnx
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    producer = _producer_map(model.graph)
    initializers = {item.name: item for item in model.graph.initializer}
    graph_inputs = {item.name for item in model.graph.input}
    needed, stack, used = [], [name], set()
    while stack:
        value = stack.pop()
        if value in initializers or not value:
            continue
        if value in graph_inputs:
            raise ValueError(f"{name!r} depends on graph input {value!r}")
        node = producer[value]
        if id(node) in used:
            continue
        used.add(id(node))
        needed.append(node)
        stack.extend(node.input)
    order = {id(node): index for index, node in enumerate(model.graph.node)}
    needed.sort(key=lambda node: order[id(node)])
    constants = [
        initializers[item]
        for node in needed
        for item in node.input
        if item in initializers
    ]
    graph = helper.make_graph(
        needed,
        "fold",
        [],
        [helper.make_tensor_value_info(name, onnx.TensorProto.UNDEFINED, None)],
        initializer=constants,
    )
    folded = helper.make_model(graph, opset_imports=model.opset_import)
    (value,) = ReferenceEvaluator(folded).run(None, {})
    return np.asarray(value)


def _shapes(model) -> dict[str, tuple[int, ...]]:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    values = (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    return {
        value.name: tuple(
            int(dim.dim_value) for dim in value.type.tensor_type.shape.dim
        )
        for value in values
    }


def rewrite_attention_mask(
    source: Path, target: Path, *, mode: str, expected: int
) -> dict[str, object]:
    import onnx
    from onnx import helper, numpy_helper

    if mode not in MODES:
        raise ValueError(f"Unknown attention-mask rewrite {mode!r}")
    source = source.resolve(strict=True)
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(source, load_external_data=True)
    before = _shapes(model)
    matches = _mask_where_nodes(model)
    if len(matches) != expected:
        raise ValueError(
            f"Expected {expected} masked_fill lowerings, found {len(matches)}"
        )
    for index, (where, mask, scores, dtype) in enumerate(matches):
        mask_shape, scores_shape = before[mask], before[scores]
        expand = _producer_map(model.graph)[
            _producer_map(model.graph)[where.input[0]].input[0]
        ]
        expand_shape = tuple(int(v) for v in _fold_constant(model, expand.input[1]))
        expanded = np.broadcast_shapes(mask_shape, expand_shape)
        if len(mask_shape) != len(scores_shape) or any(
            dim not in {1, full}
            for shape in (mask_shape, expanded)
            for dim, full in zip(shape, scores_shape, strict=True)
        ):
            raise ValueError(
                f"{where.name!r} mask {mask_shape} / expanded {expanded} "
                f"is not a broadcast of scores {scores_shape}"
            )
        if mode == "drop-expand":
            where.input[0] = mask
            continue
        prefix = f"__additive_mask_{index}"
        zero = f"{prefix}_zero"
        bias = f"{prefix}_bias"
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros((), dtype=dtype), name=zero)
        )
        position = list(model.graph.node).index(where)
        model.graph.node.insert(
            position,
            helper.make_node(
                "Where", [mask, where.input[1], zero], [bias], name=f"{prefix}/Bias"
            ),
        )
        output = where.output[0]
        where.CopyFrom(
            helper.make_node("Add", [scores, bias], [output], name=f"{prefix}/Add")
        )
    pruned = _prune_unreachable(model)
    onnx.checker.check_model(model)
    after = _shapes(model)
    for name in (value.name for value in model.graph.output):
        if after[name] != before[name]:
            raise RuntimeError(f"Rewrite changed output {name} shape")
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target, save_as_external_data=False)
    remaining = len(_mask_where_nodes(onnx.load(target)))
    if remaining:
        raise RuntimeError(f"{remaining} masked_fill lowerings remain")
    return {
        "source": str(source),
        "target": str(target),
        "source_sha256": sha256_file(source),
        "target_sha256": sha256_file(target),
        "mode": mode,
        "rewritten_masks": len(matches),
        "pruned_nodes": pruned,
        "target_nodes": len(model.graph.node),
    }


def remove_masks_for_diagnosis(
    source: Path, target: Path, *, fills: tuple[str, ...], expected: int
) -> dict[str, object]:
    """Bypass selected masked_fill nodes.  Not equivalent; diagnosis only.

    ``fills`` names the ``Where`` node suffixes (for example ``Where_1`` for the
    local window mask and ``Where_3`` for the padding mask) whose output is
    replaced by their unmasked scores input.
    """
    import onnx

    source = source.resolve(strict=True)
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(source, load_external_data=True)
    bypassed = {
        where.output[0]: scores
        for where, _mask, scores, _dtype in _mask_where_nodes(model)
        if where.name.rsplit("/", 1)[-1] in fills
    }
    if len(bypassed) != expected:
        raise ValueError(f"Expected {expected} masks to bypass, found {len(bypassed)}")

    def resolve(name: str) -> str:
        while name in bypassed:
            name = bypassed[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in bypassed:
                node.input[index] = resolve(name)
    pruned = _prune_unreachable(model)
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target, save_as_external_data=False)
    return {
        "source": str(source),
        "target": str(target),
        "target_sha256": sha256_file(target),
        "mode": "diagnostic-remove-" + "-".join(fills),
        "equivalent": False,
        "bypassed_masks": len(bypassed),
        "pruned_nodes": pruned,
    }
