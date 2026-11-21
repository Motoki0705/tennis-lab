"""Try one fully static, FP16 TensorRT engine for a locked GVHMR batch."""

import argparse
from pathlib import Path

import onnx
import torch
from benchmark import write_json
from convert import build_checked_plan, require_queue, verify_compute_precision
from convert_gvhmr import export_network
from inputs import sha256_file
from locked_inputs import HERE, checked_segments
from model import PLAN, input_shapes, load_network, pack_sequences
from sweep import GVHMR_BATCHES, GVHMR_LENGTHS


@torch.inference_mode()
def build(args):
    torch.set_num_threads(4)
    segments, checkpoint = checked_segments(args.inputs, args.checkpoints)
    selected = [
        (row, arrays)
        for row, arrays in segments
        if row["split"] == "eval" and row["length"] == args.length
    ][: args.batch_size]
    if len(selected) != args.batch_size:
        raise ValueError("The locked evaluation sequences cannot fill the fixed batch")
    args.output.mkdir(parents=True)
    inputs = pack_sequences([arrays for _, arrays in selected])
    dtype = torch.float16
    network = load_network(checkpoint).to(dtype=dtype)
    graph = args.output / "export.onnx"
    operations = export_network(
        network, {args.length: inputs}, graph, dtype, args.length, dynamic_length=False
    )
    del network
    expected_shapes = input_shapes(args.length, args.batch_size)
    exported = onnx.load(graph)
    actual_shapes = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in exported.graph.input
    }
    del exported
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"Expected fixed GVHMR inputs {expected_shapes}, got {actual_shapes}"
        )
    print(f"GVHMR static ONNX inputs: {actual_shapes}", flush=True)
    expected = dict.fromkeys(operations, "fp16")
    expected_file = args.output / "expected-precision.json"
    write_json(expected_file, expected)
    plan = args.output / PLAN
    evidence = build_checked_plan(
        graph,
        plan,
        allow_tf32=False,
        verbose=True,
        verify=lambda details: verify_compute_precision(
            details, "fp16", expected_operations=expected
        ),
    )
    write_json(
        args.output / "conversion.json",
        {
            "condition": "2",
            "fixed_shape": {
                "batch_size": args.batch_size,
                "length": args.length,
                "inputs": expected_shapes,
            },
            "export_segments": [row["id"] for row, _ in selected],
            "tensors_sha256": sha256_file(HERE / "tensor-lock.json"),
            "graph_sha256": sha256_file(graph),
            "engine_sha256": sha256_file(plan),
            "expected_precision_sha256": sha256_file(expected_file),
            "compute_precision": evidence,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", required=True, type=int, choices=GVHMR_LENGTHS)
    parser.add_argument(
        "--batch-size", required=True, type=int, choices=GVHMR_BATCHES[1:]
    )
    args = parser.parse_args()
    require_queue()
    build(args)


if __name__ == "__main__":
    main()
