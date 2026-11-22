"""Type and build one fixed DINO TensorRT plan. Unsupported stages fail closed."""

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from batching import batch_profile_points, require_dynamic_axis
from benchmark import write_json
from benchmark_dino import CHECKPOINT_SHA256, verify_inputs
from convert import (
    TRT_PRECISIONS,
    WORKSPACE_BYTES,
    build_checked_plan,
    require_queue,
    verify_compute_precision,
    verify_qdq_scope,
)
from fixed_inputs import CHECKPOINT, load_sample, rows_for, sha256_file
from measure_dino import INPUT_NAME
from msda import PLUGIN, sampling_layers, sampling_names

COMPUTE_OPS = {"Conv", "MatMul", "Gemm"}
MIN_BUILD_FREE_GIB = 11
BOX_OUTPUT = re.compile(
    r"/(?:enc_out_bbox_embed|bbox_embed\.0)/layers\.2(?:_\d+)?/MatMul$"
)
DECODER_ATTENTION_BMM = re.compile(r"/decoder/layers\.\d+/self_attn/MatMul_[34]$")
SWIN_ATTENTION_BMM = re.compile(
    r"/model/backbone/backbone\.0/layers\.\d+/blocks\.(\d+)/attn/MatMul(?:_1)?$"
)
# The first decoder V projection takes repeated learned tgt embeddings and a
# constant weight. The verified B=1 graph has no activation input to this GEMM.
CONSTANT_OPERATIONS = frozenset(
    {"/model/transformer/decoder/layers.0/self_attn/MatMul_2"}
)


def calibration_frames(directory, rows):
    from onnxruntime.quantization import CalibrationDataReader

    class CalibrationFrames(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_first(self):
            return {INPUT_NAME: load_sample(directory, rows[0])}

        def get_next(self):
            row = next(self.pending, None)
            if row is None:
                return None
            self.read += 1
            return {INPUT_NAME: load_sample(directory, row)}

        def rewind(self):
            self.read = 0
            self.pending = iter(rows)

        def __len__(self):
            return len(rows)

    return CalibrationFrames()


def precision_requirements(graph, condition):
    if condition not in TRT_PRECISIONS:
        raise ValueError(f"Unsupported TensorRT condition: {condition}")
    precision = TRT_PRECISIONS[condition]
    nodes = [node for node in graph.node if node.op_type in COMPUTE_OPS]
    if (
        not nodes
        or any(not node.name for node in nodes)
        or len({node.name for node in nodes}) != len(nodes)
    ):
        raise ValueError("No uniquely named DINO Conv/Gemm/MatMul operations")
    required = dict.fromkeys((node.name for node in nodes), precision)
    for node in nodes:
        swin = SWIN_ATTENTION_BMM.fullmatch(node.name)
        shifted_window = swin is not None and int(swin[1]) % 2 == 1
        if (
            BOX_OUTPUT.search(node.name)
            or (condition == "3a" and DECODER_ATTENTION_BMM.search(node.name))
            or (condition == "3b" and (node.op_type == "Conv" or shifted_window))
        ):
            required[node.name] = "fp16"
    return required


def verify_qdq(graph, expected, precision):
    verify_qdq_scope(
        graph, {name for name, dtype in expected.items() if dtype == precision}
    )


def load_manifest(directory, stem):
    from export_dino import graph_files

    path = directory / f"{stem}.onnx"
    receipt = json.loads((directory / f"{stem}.json").read_text())
    if graph_files(path) != receipt["files"]:
        raise ValueError(f"ONNX artifact digest changed: {path}")
    return path, receipt


def copy_calibration_graph(source, files, directory):
    """Copy ONNX and external weights before ModelOpt's in-place shape inference."""
    directory.mkdir()
    for name in files:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, target)
    return directory / source.name


def bind_initializer_weights(graph):
    """Expose constant decoder weights hidden behind Identity edges to ModelOpt."""
    initializers = {tensor.name for tensor in graph.initializer}
    aliases = {}
    changed = []
    for node in graph.node:
        if node.op_type == "Identity" and not node.domain:
            origin = aliases.get(node.input[0], node.input[0])
            if origin in initializers:
                aliases[node.output[0]] = origin
        elif node.op_type in {"Gemm", "MatMul"} and node.input[1] in aliases:
            node.input[1] = aliases[node.input[1]]
            changed.append(node.name)
    return changed


def prepare(args, lock, export, source):
    import onnx
    from export_dino import graph_files

    target = args.output / "typed.onnx"
    primary = onnx.load(source, load_external_data=False).graph
    required = precision_requirements(primary, args.condition)
    del primary
    quantization = None
    precision = TRT_PRECISIONS[args.condition]
    if args.condition == "2":
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel

        model = convert_float_to_float16(onnx.load(source), keep_io_types=True)
        OnnxModel.graph_topological_sort(model.graph)
        onnx.save_model(
            model, target, save_as_external_data=True, location="weights.bin"
        )
        del model
    else:
        if export["method"] != "grid":
            raise ValueError(
                "ModelOpt CPU calibration requires the standard ONNX sampling graph"
            )
        from modelopt.onnx.quantization.quantize import quantize

        rows = rows_for(lock, args.bucket, "calibration")
        reader = calibration_frames(args.inputs, rows)
        selection = [
            rf"^{re.escape(name)}$"
            for name, dtype in required.items()
            if dtype == precision
        ]
        calibration_source = copy_calibration_graph(
            source, export["files"], args.output / "modelopt-input"
        )
        model = onnx.load(calibration_source, load_external_data=False)
        rebound = bind_initializer_weights(model.graph)
        onnx.save_model(model, calibration_source)
        del model
        print(
            f"Resolved initializer aliases for {len(rebound)} matrix multiplies",
            flush=True,
        )
        calibration_eps = ["cpu"]
        quantize(
            str(calibration_source),
            quantize_mode=precision,
            calibration_data_reader=reader,
            calibration_method="max",
            calibration_eps=calibration_eps,
            high_precision_dtype="fp16",
            nodes_to_quantize=selection,
            use_external_data_format=True,
            output_path=str(target),
        )
        if calibration_eps != ["cpu"]:
            raise RuntimeError(
                f"ModelOpt changed the calibration providers: {calibration_eps}"
            )
        if reader.read != len(rows):
            raise RuntimeError(
                f"PTQ did not consume all fixed frames: {reader.read}/{len(rows)}"
            )
        graph = onnx.load(target, load_external_data=False).graph
        verify_qdq(graph, required, precision)
        quantization = {
            "mode": precision,
            "settings_origin": 2297,
            "calibration_method": "max",
            "calibration_providers": calibration_eps,
            "calibration_frames": len(rows),
            "rebound_weight_operands": rebound,
            "qdq": dict(
                Counter(
                    node.op_type
                    for node in graph.node
                    if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
                )
            ),
            "high_precision_dtype": "fp16",
        }
    onnx.checker.check_model(str(target), full_check=True)
    write_json(
        args.output / "typed.json",
        {
            **export,
            "export_files": export["files"],
            "files": graph_files(target),
            "condition": args.condition,
            "precision_requirements": required,
            "quantization": quantization,
            "typed_queue_job": os.environ["TENNIS_RUN_ID"],
        },
    )


def verify_layers(details, precision, requirements, sampling, method):
    evidence = verify_compute_precision(
        details,
        precision,
        expected_operations=requirements,
        constant_operations=CONSTANT_OPERATIONS,
    )
    if unobserved := evidence.get("unobserved_constant_onnx_operations"):
        raise RuntimeError(f"Missing DINO constant tactic evidence: {unobserved}")
    layers = sampling_layers(details, sampling, method=method, precision=precision)
    return {
        **evidence,
        "sampling_operation": PLUGIN if method == "plugin" else "GridSample",
        "count": len(sampling),
        "sampling_execution_layer_types": dict(
            Counter(layer["type"] for layer in layers)
        ),
        "sampling_operand_formats": dict(
            Counter(dtype for layer in layers for dtype in layer["inputs"])
        ),
        "sampling_output_formats": dict(
            Counter(dtype for layer in layers for dtype in layer["outputs"])
        ),
    }


def build(
    args,
    source,
    typed,
    graph,
    *,
    batch_profile=None,
    builder_optimization_level=None,
    workspace_bytes=WORKSPACE_BYTES,
):
    method = typed["method"]
    precision = TRT_PRECISIONS[args.condition]
    if args.condition in {"3a", "3b"}:
        verify_qdq(graph, typed["precision_requirements"], precision)
    plugin = method == "plugin"
    sampling = sampling_names(graph, method, 12 if plugin else 48)

    profile = None
    if batch_profile is not None:
        for value in (*graph.input, *graph.output):
            require_dynamic_axis(value, 0, "batch")
        profile = {
            INPUT_NAME: [
                (size, *typed["input_shape"][1:])
                for size in batch_profile_points(*batch_profile)
            ]
        }
    plan = args.output / "dino.plan"
    performance = build_checked_plan(
        source,
        plan,
        verify=lambda details: verify_layers(
            details, precision, typed["precision_requirements"], sampling, method
        ),
        init_plugins=plugin,
        min_free_gib=MIN_BUILD_FREE_GIB,
        allow_tf32=False,
        profile_shapes=profile,
        builder_optimization_level=builder_optimization_level,
        workspace_bytes=workspace_bytes,
    )
    write_json(
        args.output / "conversion.json",
        {
            **typed,
            "plan_sha256": sha256_file(plan),
            "workspace_bytes": workspace_bytes,
            "compute_precision": performance,
            "queue_job": os.environ["TENNIS_RUN_ID"],
            **(
                {"input_profile": profile, "sampling_nodes": sorted(sampling)}
                if profile is not None
                else {}
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build"))
    parser.add_argument("--condition", required=True, choices=TRT_PRECISIONS)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=Path)
    parser.add_argument("--batch-profile", nargs=2, type=int, metavar=("OPT", "MAX"))
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6))
    parser.add_argument(
        "--workspace-mib", type=int, default=WORKSPACE_BYTES // (1024**2)
    )
    args = parser.parse_args()
    if args.stage == "prepare" and (
        args.batch_profile is not None
        or args.builder_optimization_level is not None
        or args.workspace_mib != WORKSPACE_BYTES // (1024**2)
    ):
        parser.error("Profile and optimization level belong to the build stage")
    if args.workspace_mib <= 0:
        parser.error("Workspace limit must be positive")
    require_queue()
    stage = "export" if args.stage == "prepare" else "typed"
    source, receipt = load_manifest(args.source, stage)
    lock = verify_inputs(args.inputs, args.bucket, args.checkpoint)
    if (
        receipt["input_shape"] != lock["buckets"][args.bucket]["profile"]["opt"]
        or receipt["tensor_set_sha256"] != lock["tensor_set_sha256"]
        or receipt["checkpoint_sha256"] != CHECKPOINT_SHA256
        or (stage == "typed" and receipt["condition"] != args.condition)
        or (
            stage == "typed"
            and bool(receipt.get("dynamic_batch")) != bool(args.batch_profile)
        )
    ):
        raise ValueError(
            "ONNX artifact does not match the specified model, inputs or condition"
        )
    args.output.mkdir(parents=True)
    if args.stage == "prepare":
        prepare(args, lock, receipt, source)
    else:
        import onnx

        build(
            args,
            source,
            receipt,
            onnx.load(source, load_external_data=False).graph,
            batch_profile=args.batch_profile,
            builder_optimization_level=args.builder_optimization_level,
            workspace_bytes=args.workspace_mib * 1024**2,
        )


if __name__ == "__main__":
    main()
