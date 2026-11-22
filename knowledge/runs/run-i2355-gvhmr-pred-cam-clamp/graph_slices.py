"""Create checked ONNX subgraphs for TensorRT build isolation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _value_info(name: str, elem_type: int, shape: tuple[int | str, ...]):
    from onnx import helper

    return helper.make_tensor_value_info(name, elem_type, list(shape))


def infer_tensor_specs(
    source: Path, names: Iterable[str]
) -> dict[str, tuple[int, tuple[int | str, ...]]]:
    """Return inferred ONNX dtype/shape pairs for selected static graph values."""
    import onnx

    requested = tuple(dict.fromkeys(names))
    model = onnx.load(source.resolve(strict=True), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    values = {
        value.name: value
        for collection in (
            inferred.graph.input,
            inferred.graph.value_info,
            inferred.graph.output,
        )
        for value in collection
    }
    missing = set(requested) - values.keys()
    if missing:
        raise ValueError(f"No inferred tensor metadata for: {sorted(missing)}")

    result = {}
    for name in requested:
        tensor = values[name].type.tensor_type
        shape: list[int | str] = []
        for axis, dimension in enumerate(tensor.shape.dim):
            if dimension.HasField("dim_value"):
                shape.append(int(dimension.dim_value))
            elif dimension.dim_param:
                shape.append(dimension.dim_param)
            else:
                raise ValueError(f"Tensor {name!r} has unknown dimension {axis}")
        result[name] = (int(tensor.elem_type), tuple(shape))
    return result


def slice_graph(
    source: Path,
    target: Path,
    *,
    output_name: str,
    output_type: int,
    output_shape: tuple[int | str, ...],
    promoted_inputs: dict[str, tuple[int, tuple[int | str, ...]]] | None = None,
) -> dict[str, object]:
    """Retain only dependencies of one output, stopping at promoted inputs.

    Promoted inputs that the retained nodes never consume are not added to the
    sliced graph; they are listed in ``unused_promoted_inputs`` instead.
    """
    import onnx

    source = source.resolve(strict=True)
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(source, load_external_data=True)
    graph = model.graph
    producers = {name: node for node in graph.node for name in node.output}
    known_values = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {value.name for value in graph.value_info}
        | set(producers)
    )
    if output_name not in known_values:
        raise ValueError(f"Unknown slice output {output_name!r}")
    promoted_inputs = promoted_inputs or {}
    if unknown := promoted_inputs.keys() - known_values:
        raise ValueError(f"Unknown promoted inputs: {sorted(unknown)}")

    original_inputs = {value.name: value for value in graph.input}
    stop_values = set(original_inputs) | set(promoted_inputs)
    required_values = {output_name}
    required_nodes = set()
    stack = [output_name]
    while stack:
        value = stack.pop()
        if value in stop_values:
            continue
        node = producers.get(value)
        if node is None:
            continue
        identity = id(node)
        if identity in required_nodes:
            continue
        required_nodes.add(identity)
        for item in node.input:
            if item:
                required_values.add(item)
                stack.append(item)

    nodes = [node for node in graph.node if id(node) in required_nodes]
    required_values.update(name for node in nodes for name in node.output)
    initializers = [item for item in graph.initializer if item.name in required_values]
    used_original_inputs = {
        name for node in nodes for name in node.input if name in original_inputs
    }
    consumed = {name for node in nodes for name in node.input}
    unused_promoted = sorted(
        name
        for name in promoted_inputs
        if name not in consumed and name != output_name
    )
    inputs = [
        original_inputs[name]
        for name in original_inputs
        if name in used_original_inputs and name not in promoted_inputs
    ]
    inputs.extend(
        _value_info(name, elem_type, shape)
        for name, (elem_type, shape) in promoted_inputs.items()
        if name not in unused_promoted
    )
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.input[:]
    graph.input.extend(inputs)
    del graph.output[:]
    graph.output.extend([_value_info(output_name, output_type, output_shape)])
    retained_info = [
        value
        for value in graph.value_info
        if value.name in required_values
        and value.name not in {item.name for item in inputs}
        and value.name != output_name
    ]
    del graph.value_info[:]
    graph.value_info.extend(retained_info)
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target, save_as_external_data=False)
    return {
        "source": str(source),
        "target": str(target),
        "output": output_name,
        "inputs": [value.name for value in inputs],
        "nodes": len(nodes),
        "initializers": len(initializers),
        "promoted_inputs": sorted(set(promoted_inputs) - set(unused_promoted)),
        "unused_promoted_inputs": unused_promoted,
    }
