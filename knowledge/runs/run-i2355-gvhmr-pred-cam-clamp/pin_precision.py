"""Pin selected attention operators of an FP16 ONNX graph to FP32 compute.

For every transformer block, the operators named by ``suffixes`` (for example
``Softmax`` and ``MatMul_1`` under ``/network/blocks.N/attn/``) form a region.
FP16 tensors entering the region are cast to FP32, the region computes in FP32,
and region outputs used elsewhere are cast back to FP16.  With a strongly typed
TensorRT network this fixes those layers' precision explicitly.  The rewrite
changes rounding, so it is checked against the FP32 reference, not bit parity.
"""

from __future__ import annotations

from pathlib import Path

from rewrite_pred_cam import _producer_map, sha256_file

PASS_THROUGH_TYPES = {"BOOL", "INT64", "INT32"}


def _elem_types(model) -> dict[str, int]:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    values = (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    types = {value.name: value.type.tensor_type.elem_type for value in values}
    types.update({item.name: item.data_type for item in model.graph.initializer})
    return types


def region_names(blocks: int, suffixes: tuple[str, ...]) -> set[str]:
    return {
        f"/network/blocks.{block}/attn/{suffix}"
        for block in range(blocks)
        for suffix in suffixes
    }


def pin_fp32(
    source: Path, target: Path, *, suffixes: tuple[str, ...], blocks: int
) -> dict[str, object]:
    import onnx
    from onnx import TensorProto, helper

    source = source.resolve(strict=True)
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(source, load_external_data=True)
    names = region_names(blocks, suffixes)
    region = [node for node in model.graph.node if node.name in names]
    if {node.name for node in region} != names:
        missing = sorted(names - {node.name for node in region})
        raise ValueError(f"Region operators not found: {missing[:4]}")
    types = _elem_types(model)
    region_ids = {id(node) for node in region}
    produced_inside = {name for node in region for name in node.output}
    graph_outputs = {value.name for value in model.graph.output}
    used_outside = {
        name
        for node in model.graph.node
        if id(node) not in region_ids
        for name in node.input
    } | graph_outputs

    def fp32(name: str) -> str:
        return f"{name}__fp32"

    for node in region:
        for name in node.output:
            if types.get(name) != TensorProto.FLOAT16:
                raise ValueError(f"Region output {name!r} is not FP16")
    entering: set[str] = set()
    for node in region:
        for name in node.input:
            if not name or name in produced_inside:
                continue
            elem = types.get(name)
            if elem == TensorProto.FLOAT16:
                entering.add(name)
            elif TensorProto.DataType.Name(elem) not in PASS_THROUGH_TYPES:
                raise ValueError(f"Unexpected region input type for {name!r}")

    producer = _producer_map(model.graph)
    nodes = []
    emitted: set[str] = set()
    casts_in = casts_out = 0
    for node in model.graph.node:
        if id(node) in region_ids:
            for name in node.input:
                if name in entering and name not in emitted:
                    nodes.append(
                        helper.make_node(
                            "Cast",
                            [name],
                            [fp32(name)],
                            name=f"{name}__cast_fp32",
                            to=TensorProto.FLOAT,
                        )
                    )
                    emitted.add(name)
                    casts_in += 1
            renamed = onnx.NodeProto()
            renamed.CopyFrom(node)
            for index, name in enumerate(renamed.input):
                if name in entering or name in produced_inside:
                    renamed.input[index] = fp32(name)
            for index, name in enumerate(renamed.output):
                renamed.output[index] = fp32(name)
            nodes.append(renamed)
            for name in node.output:
                if name in used_outside:
                    nodes.append(
                        helper.make_node(
                            "Cast",
                            [fp32(name)],
                            [name],
                            name=f"{name}__cast_fp16",
                            to=TensorProto.FLOAT16,
                        )
                    )
                    casts_out += 1
        else:
            nodes.append(node)
    del producer
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    value_info = [
        item for item in model.graph.value_info if item.name not in produced_inside
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)
    checked = _elem_types(model)
    for name in produced_inside:
        if checked.get(fp32(name)) != TensorProto.FLOAT:
            raise RuntimeError(f"{name!r} is not computed in FP32 after pinning")
        if name in used_outside and checked.get(name) != TensorProto.FLOAT16:
            raise RuntimeError(f"{name!r} is not returned to FP16")
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target, save_as_external_data=False)
    return {
        "source": str(source),
        "target": str(target),
        "target_sha256": sha256_file(target),
        "mode": "pin-fp32-" + "-".join(suffixes),
        "pinned_operators": len(region),
        "casts_to_fp32": casts_in,
        "casts_to_fp16": casts_out,
    }
