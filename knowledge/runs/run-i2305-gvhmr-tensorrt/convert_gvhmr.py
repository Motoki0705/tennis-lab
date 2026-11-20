"""Export and build one explicitly typed, dynamic SIGA24 denoiser."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from benchmark import write_json
from convert import TRT_PRECISIONS, build_engine, require_queue
from inputs import sha256_file
from locked_inputs import HERE, checked_segments
from model import INPUT_WIDTHS, OUTPUT_WIDTHS, PLAN, dynamic_attention, load_network
from torch.onnx import symbolic_helper, symbolic_opset9, symbolic_opset11

MAJOR_TYPES = {"MatMul", "Gemm", "Conv"}
OPSET = 19
BLOCK_SCOPE = re.compile(r"/(blocks\.\d+)/")


@symbolic_helper.parse_args("s", "v", "is")
def export_attention_matmul(graph, equation, operands, _path=None):
    left, right = symbolic_helper._unpack_list(operands)
    if equation == "abcd,abed->abce":
        right = graph.op("Transpose", right, perm_i=[0, 1, 3, 2])
    elif equation != "abcd,abde->abce":
        raise ValueError(f"Unknown GVHMR attention equation: {equation}")
    return graph.op("MatMul", left, right)


@symbolic_helper.parse_args("v", "v", "v", "b")
def export_index_put(graph, data, indices, value, accumulate=False):
    """Keep the coordinate axis when lowering obs[~visible] = 0."""
    keys = (
        symbolic_helper._unpack_list(indices)
        if symbolic_helper._is_packed_list(indices)
        else [indices]
    )
    rank = symbolic_helper._get_tensor_rank
    if (
        not accumulate
        and len(keys) == 1
        and symbolic_helper._is_bool(keys[0])
        and rank(data) == 4
        and rank(keys[0]) == 3
        and rank(value) == 0
    ):
        mask = symbolic_helper._unsqueeze_helper(graph, keys[0], [-1])
        return symbolic_opset9.masked_fill(graph, data, mask, value)
    return symbolic_opset11.index_put(graph, data, indices, value, accumulate)


def network_args(arrays, dtype):
    return tuple(
        torch.from_numpy(arrays[name]).to(dtype=dtype)
        if name != "length"
        else torch.from_numpy(arrays[name])
        for name in INPUT_WIDTHS
    )


def export_network(model, samples, output, dtype, opt):
    import onnx

    dynamic_axes = {
        name: {1: "frames"}
        for name in (*INPUT_WIDTHS, *OUTPUT_WIDTHS)
        if name not in {"length", "length_out"}
    }
    lowerings = {
        "aten::einsum": export_attention_matmul,
        "aten::index_put": export_index_put,
    }
    for operator, lowering in lowerings.items():
        torch.onnx.register_custom_op_symbolic(operator, lowering, OPSET)
    try:
        # The short (117) and both local-attention sequence lengths come from real segments.
        for frames, arrays in samples.items():
            args = network_args(arrays, dtype)
            native = model(*args)
            with dynamic_attention(model):
                converted = model(*args)
            for name, a, b in zip(OUTPUT_WIDTHS, native, converted, strict=True):
                torch.testing.assert_close(
                    a, b, rtol=0, atol=0, msg=f"L={frames} {name}"
                )
        with dynamic_attention(model):
            torch.onnx.export(
                model,
                network_args(samples[opt], dtype),
                str(output),
                input_names=list(INPUT_WIDTHS),
                output_names=list(OUTPUT_WIDTHS),
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
                external_data=False,
            )
    finally:
        for operator in lowerings:
            torch.onnx.unregister_custom_op_symbolic(operator, OPSET)
    onnx.checker.check_model(str(output))
    model = onnx.load(output)
    for value in (*model.graph.input, *model.graph.output):
        if (
            value.name not in {"length", "length_out"}
            and value.type.tensor_type.shape.dim[1].dim_param != "frames"
        ):
            raise ValueError(f"Export lost the shared sequence axis for {value.name}")
    return major_operations(model, output)


def major_operations(model, source):
    nodes = [node for node in model.graph.node if node.op_type in MAJOR_TYPES]
    if len(nodes) != 110 or len({node.name for node in nodes}) != 110:
        raise ValueError(
            f"Expected 110 named GVHMR compute nodes in {source}, got {len(nodes)}"
        )
    block_counts = Counter(
        match.group(1) for node in nodes if (match := BLOCK_SCOPE.search(node.name))
    )
    if block_counts != {f"blocks.{i}": 8 for i in range(12)}:
        raise ValueError(
            f"Unexpected GVHMR block compute inventory: {dict(block_counts)}"
        )
    return {node.name: node.op_type for node in nodes}


def quantize_graph(graph, target, mode, stages, expected, settings):
    import onnx
    from modelopt.onnx.quantization.quantize import quantize
    from onnxruntime.quantization import CalibrationDataReader

    class SegmentCalibration(CalibrationDataReader):
        def __init__(self):
            self.sequences = [
                {name: arrays[name] for name in INPUT_WIDTHS} for arrays in stages
            ]
            self.rewind()

        def get_next(self):
            return next(self.iterator, None)

        def rewind(self):
            self.iterator = iter(self.sequences)

        def __len__(self):
            return len(self.sequences)

    quantize(
        str(graph),
        quantize_mode=mode,
        calibration_data_reader=SegmentCalibration(),
        nodes_to_quantize=[name for name, dtype in expected.items() if dtype == mode],
        op_types_to_quantize=["MatMul", "Gemm"],
        calibration_method=settings["calibration_method"],
        calibration_eps=settings["calibration_eps"],
        high_precision_dtype=settings["high_precision_dtype"],
        use_external_data_format=False,
        output_path=str(target),
    )
    onnx.checker.check_model(str(target))
    quantized = onnx.load(target)
    actual = major_operations(quantized, target)
    if expected.keys() != actual.keys():
        raise ValueError("Quantization changed the major ONNX operator names")
    producers = {out: node for node in quantized.graph.node for out in node.output}
    for node in quantized.graph.node:
        if node.name not in expected:
            continue
        dq_operands = sum(
            value in producers and producers[value].op_type == "DequantizeLinear"
            for value in node.input[:2]
        )
        required = 2 if expected[node.name] == mode else 0
        if dq_operands != required:
            raise ValueError(
                f"{node.name} requires {required} Q/DQ operands, found {dq_operands}"
            )
    return {
        "mode": mode,
        "sequences": len(stages),
        "frames": sum(int(stage["length"][0]) for stage in stages),
        "Q": sum(node.op_type == "QuantizeLinear" for node in quantized.graph.node),
        "DQ": sum(node.op_type == "DequantizeLinear" for node in quantized.graph.node),
        **settings,
    }


@torch.inference_mode()
def convert(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    segments, checkpoint = checked_segments(args.inputs, args.checkpoints)
    profile = json.loads((HERE / "shape-profile.json").read_text())
    settings = json.loads((HERE / "ptq.json").read_text())
    representatives = {}
    calibration = []
    for row, arrays in segments:
        representatives.setdefault(row["length"], arrays)
        if row["split"] == "calibration":
            calibration.append(arrays)
    if (
        representatives.keys() != set(profile["profile_lengths"].values())
        or not calibration
    ):
        raise ValueError("The shape profile does not describe the tensor lock")
    mode = TRT_PRECISIONS[args.condition]
    graph = args.output / "export.onnx"
    dtype = torch.float16 if args.condition == "2" else torch.float32
    network = load_network(checkpoint).to(dtype=dtype)
    operations = export_network(
        network, representatives, graph, dtype, profile["profile_lengths"]["opt"]
    )
    del network
    expected = {
        name: mode if BLOCK_SCOPE.search(name) else "fp16" for name in operations
    }
    write_json(args.output / "expected-precision.json", expected)
    quantization = None
    if args.condition != "2":
        target = args.output / "quantized.onnx"
        quantization = quantize_graph(
            graph, target, mode, calibration, expected, settings
        )
        graph = target
    evidence = build_engine(
        graph,
        args.output / PLAN,
        required_precision=mode,
        expected_operations=expected,
        profile_shapes=profile["inputs"],
    )
    result = {
        "condition": args.condition,
        "dynamic_profile": profile,
        "tensors_sha256": sha256_file(HERE / "tensor-lock.json"),
        "graph_sha256": sha256_file(graph),
        "engine_sha256": sha256_file(args.output / PLAN),
        "compute_precision": evidence,
        "quantization": quantization,
        "native_mask_adapter_parity": sorted(representatives),
    }
    write_json(args.output / "conversion.json", result)
    print(f"GVHMR conversion passed: {evidence}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=TRT_PRECISIONS, required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_queue()
    convert(args)


if __name__ == "__main__":
    main()
