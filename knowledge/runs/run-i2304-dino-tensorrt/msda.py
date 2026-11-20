"""Select an explicit export for the pinned DINO bilinear sampling core."""

from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import torch
from convert import ONNX_LAYER
from fixed_inputs import REPOSITORY

from src.submodules.models.dino.architecture import load_dino_build_function

PLUGIN = "MultiscaleDeformableAttnPlugin_TRT"
PLUGIN_VERSION = "2"


def sampling_names(graph, method, count):
    op_type = PLUGIN if method == "plugin" else "GridSample"
    nodes = [node for node in graph.node if node.op_type == op_type]
    names = {node.name for node in nodes}
    if len(nodes) != count or len(names) != count or "" in names:
        raise ValueError(f"Expected {count} uniquely named ONNX {op_type} operations")
    return names


def sampling_layers(details, names):
    layers = []
    seen = set()
    for layer in details["Layers"]:
        matches = names.intersection(ONNX_LAYER.findall(layer.get("Metadata", "")))
        if matches:
            layers.append(
                {
                    "name": layer["Name"],
                    "type": layer["LayerType"],
                    "operations": sorted(matches),
                    "inputs": [tensor["Format/Datatype"] for tensor in layer["Inputs"]],
                }
            )
            seen.update(matches)
    if missing := names - seen:
        raise RuntimeError(
            f"Missing TensorRT layer evidence for ONNX sampling operations: {sorted(missing)}"
        )
    return layers


def upstream_modules():
    prefix = load_dino_build_function(REPOSITORY.resolve()).__module__.removesuffix(
        ".dino.dino"
    )
    return (
        import_module(f"{prefix}.dino.ops.modules.ms_deform_attn"),
        import_module(f"{prefix}.dino.ops.functions.ms_deform_attn_func"),
    )


def grid_core(reference, value, geometry, locations, weights):
    # The exporter is specialized to one frozen H/W. Do not reuse this graph
    # for a different bucket, including shapes with the same number of pixels.
    shapes = [(height, width) for height, width in geometry.tolist()]
    return reference(value, shapes, locations, weights)


class PluginCore(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value,
        geometry,
        starts,
        locations,
        weights,
        batch,
        queries,
        heads,
        channels,
        fp16,
    ):
        _, upstream = upstream_modules()
        flat = grid_core(
            upstream.ms_deform_attn_core_pytorch, value, geometry, locations, weights
        )
        return flat.reshape(batch, queries, heads, channels)

    @staticmethod
    def symbolic(
        graph,
        value,
        geometry,
        starts,
        locations,
        weights,
        batch,
        queries,
        heads,
        channels,
        fp16,
    ):
        dtype = torch.float16 if fp16 else torch.float32
        return graph.op(
            f"trt::{PLUGIN}",
            value,
            geometry,
            starts,
            locations,
            weights,
            plugin_version_s=PLUGIN_VERSION,
        ).setType(
            value.type().with_dtype(dtype).with_sizes([batch, queries, heads, channels])
        )


def plugin_core(value, geometry, starts, locations, weights):
    # Record the observed static bucket while tracing. The legacy symbolic
    # pass can discard upstream shape annotations before translating this op.
    batch, queries = int(value.shape[0]), int(locations.shape[1])
    heads, channels = int(value.shape[2]), int(value.shape[3])
    if value.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"Unsupported TensorRT MSDA dtype: {value.dtype}")
    return PluginCore.apply(
        value,
        geometry.to(torch.int32),
        starts.to(torch.int32),
        locations.to(value.dtype),
        weights.to(value.dtype),
        batch,
        queries,
        heads,
        channels,
        value.dtype == torch.float16,
    ).flatten(2)


@contextmanager
def export_sampling(method):
    layer, upstream = upstream_modules()

    def apply(value, geometry, starts, locations, weights, _step):
        if method == "grid":
            return grid_core(
                upstream.ms_deform_attn_core_pytorch,
                value,
                geometry,
                locations,
                weights,
            )
        if method == "plugin":
            return plugin_core(value, geometry, starts, locations, weights)
        raise ValueError(f"Unknown MSDA export: {method}")

    with patch.object(layer, "MSDeformAttnFunction", SimpleNamespace(apply=apply)):
        yield


def export_onnx(model, tensors, path, *, inputs, outputs):
    # Swin and deformable-attention geometry use fixed shapes in the legacy trace.
    with torch.no_grad():
        torch.onnx.export(
            model,
            tensors,
            str(path),
            input_names=inputs,
            output_names=outputs,
            opset_version=19,
            dynamo=False,
            external_data=True,
        )
