"""Export the fixed CLIP model and build one explicitly typed TensorRT engine."""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

WORKSPACE_BYTES = 512 * 1024**2
PLAN = "clipreid.plan"
CONDITIONS = ("0", "1", "2", "3a", "3b")
TRT_PRECISIONS = {"2": "fp16", "3a": "int8", "3b": "fp8"}

OPERAND_CODE = {"fp16": "f16", "int8": "i8", "fp8": "e4m3"}
TENSOR_DTYPE = {"fp16": r"FP16|Float16|Half", "int8": "Int8", "fp8": "FP8"}
COMPUTE_TYPES = {
    "gemm",
    "matmul",
    "correlation",
    "conv",
    "convolution",
    "matrixmultiply",
    "fullyconnected",
}
ONNX_LAYER = re.compile(r"\[ONNX Layer: ([^\]]+)\]")
# TensorRT 10.14 xmma tactics list the operand codes before accumulation/epilogue.
TACTIC_OPERANDS = re.compile(
    r"_(?:gemm|gemvx)_(?:interleaved_|indexed_|wo_smem_)*"
    r"(f16|i8|e4m3|f32|tf32|bf16)(?:f16|i8|e4m3|f32|tf32|bf16|i32)?_"
    r"(f16|i8|e4m3|f32|tf32|bf16)(?:f16|i8|e4m3|f32|tf32|bf16|i32)?_"
)


def require_queue():
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("Run GPU workloads through the shared training queue.")


def clipreid_precision_requirements(condition):
    """Pin the compute ops in the CLIP ViT-B/16 export, including fused attention."""
    precision = TRT_PRECISIONS[condition]
    return {
        f"/visual/transformer/resblocks/resblocks.{block}/{name}": precision
        for block in range(12)
        for name in (
            "attn/MatMul",
            "attn/MatMul_1",
            "attn/MatMul_2",
            "attn/Gemm",
            "mlp/c_fc/MatMul",
            "mlp/c_proj/MatMul",
        )
    } | {
        "/visual/conv1/Conv": "int8" if condition == "3a" else "fp16",
        "/visual/MatMul": "fp16",
    }


def inspect_engine(engine, context=None):
    import tensorrt as trt

    inspector = engine.create_engine_inspector()
    if context is not None:
        inspector.execution_context = context
    return json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))


def verify_compute_precision(details, required_precision, *, expected_operations=None):
    """Require operand evidence for each compute layer and cover the pinned ONNX ops."""
    precisions = Counter()
    seen = set()
    for layer in details["Layers"]:
        kind = layer["LayerType"].lower()
        tactic = layer.get("TacticName", "")
        operands = TACTIC_OPERANDS.search(tactic)
        fused_attention = kind == "kgen" and tactic.startswith("_gemm_mha_v2_")
        if kind not in COMPUTE_TYPES and not operands and not fused_attention:
            continue

        name = layer["Name"]
        operations = set(ONNX_LAYER.findall(layer.get("Metadata", "")))
        if expected_operations is None:
            precision = required_precision
        else:
            operations &= expected_operations.keys()
            targets = {expected_operations[op] for op in operations}
            if len(targets) != 1:
                raise RuntimeError(
                    f"Compute layer {name} has no unique precision requirement"
                )
            (precision,) = targets

        inputs = [tensor["Format/Datatype"] for tensor in layer.get("Inputs", [])]
        pattern = re.compile(rf"\b(?:{TENSOR_DTYPE[precision]})\b", re.IGNORECASE)
        if fused_attention:
            matches = len(inputs) == 3 and all(
                pattern.search(dtype) for dtype in inputs
            )
        else:
            code = OPERAND_CODE[precision]
            matches = (
                operands is not None
                and operands.groups() == (code, code)
                and bool(inputs and pattern.search(inputs[0]))
            )
        if not matches:
            raise RuntimeError(
                f"Compute layer {name} expected {precision} operands, "
                f"got inputs {inputs} and tactic {tactic!r}"
            )
        precisions[precision] += 1
        seen.update(operations)

    if not precisions[required_precision]:
        raise RuntimeError(f"No compute layers verified at {required_precision}")
    evidence = {"engine_layers_by_precision": dict(precisions)}
    if expected_operations is not None:
        missing = expected_operations.keys() - seen
        if missing:
            raise RuntimeError(
                f"Missing compute evidence for ONNX operations: {sorted(missing)}"
            )
        evidence["onnx_operations_by_precision"] = dict(
            Counter(expected_operations.values())
        )
    return evidence


def set_profile_shapes(profile, profile_shapes):
    for name, shapes in profile_shapes.items():
        expected = tuple(tuple(shape) for shape in shapes)
        profile.set_shape(name, *expected)
        actual = tuple(tuple(shape) for shape in profile.get_shape(name))
        if actual != expected:
            raise ValueError(
                f"TensorRT profile {name}={actual} differs from {expected}"
            )


def build_engine(
    onnx_file,
    plan,
    *,
    required_precision,
    expected_operations=None,
    profile_shapes=None,
):
    import tensorrt as trt
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("The pinned TensorRT build requires one CC 12.0 CUDA GPU")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB free before TensorRT engine build")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_file}\n{errors}")
    if profile_shapes is not None:
        names = {network.get_input(i).name for i in range(network.num_inputs)}
        if names != profile_shapes.keys():
            raise ValueError(f"Optimization profile inputs do not match ONNX: {names}")
        profile = builder.create_optimization_profile()
        set_profile_shapes(profile, profile_shapes)
        if config.add_optimization_profile(profile) != 0:
            raise RuntimeError("TensorRT did not accept the first optimization profile")
        config.clear_flag(trt.BuilderFlag.TF32)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_file}")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the built engine")
    if profile_shapes is None:
        shapes_by_label = {"": None}
        context = None
    else:
        shapes_by_label = {
            f".{label}": {name: shapes[i] for name, shapes in profile_shapes.items()}
            for i, label in enumerate(("min", "opt", "max"))
        }
        context = engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        if context is None:
            raise RuntimeError("Could not create engine inspection context")
    details_by_label = {}
    for label, shapes in shapes_by_label.items():
        if shapes is not None:
            for name, shape in shapes.items():
                if not context.set_input_shape(name, shape):
                    raise ValueError(
                        f"TensorRT rejected inspection shape {name}={shape}"
                    )
        details = inspect_engine(engine, context)
        plan.with_suffix(f"{label}.layers.json").write_text(
            json.dumps(details, indent=2) + "\n"
        )
        details_by_label[label] = details
    evidence = {
        label.removeprefix("."): verify_compute_precision(
            details, required_precision, expected_operations=expected_operations
        )
        for label, details in details_by_label.items()
    }
    with plan.open("xb") as stream:
        stream.write(memoryview(engine_bytes))
    return evidence[""] if profile_shapes is None else evidence


def convert(condition, checkpoint, calibration, output):
    import onnx
    import torch
    from clipreid import BATCH_SHAPE, INPUT_NAME, OUTPUT_NAME, VisualFeatures

    torch.set_num_threads(4)
    dtype = torch.float16 if condition == "2" else torch.float32
    model = VisualFeatures(checkpoint).to(dtype=dtype).eval()
    graph = output / "export.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (torch.zeros(BATCH_SHAPE, dtype=dtype),),
            str(graph),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=19,
            dynamo=False,
            external_data=False,
        )
    onnx.checker.check_model(str(graph))
    del model
    mode = TRT_PRECISIONS[condition]
    quantization = None
    if condition in {"3a", "3b"}:
        from modelopt.onnx.quantization.quantize import quantize

        if calibration is None:
            raise ValueError("PTQ requires the fixed calibration tensors")
        data = np.load(calibration, allow_pickle=False)
        if (
            data.ndim != 4
            or data.shape[1:] != BATCH_SHAPE[1:]
            or data.dtype != np.float32
        ):
            raise ValueError("Unexpected calibration tensor dtype or shape")
        target = output / "quantized.onnx"
        quantize(
            str(graph),
            quantize_mode=mode,
            calibration_data={INPUT_NAME: data},
            calibration_method="max",
            calibration_eps=["cpu"],
            high_precision_dtype="fp16",
            use_external_data_format=False,
            output_path=str(target),
        )
        graph = target
        quantized = onnx.load(graph)
        quantize_nodes = sum(
            node.op_type == "QuantizeLinear" for node in quantized.graph.node
        )
        dequantize_nodes = sum(
            node.op_type == "DequantizeLinear" for node in quantized.graph.node
        )
        if not quantize_nodes or not dequantize_nodes:
            raise RuntimeError(f"{mode} PTQ produced no explicit Q/DQ pairs")
        quantization = {
            "mode": mode,
            "crops": len(data),
            "Q": quantize_nodes,
            "DQ": dequantize_nodes,
        }
    evidence = build_engine(
        graph,
        output / PLAN,
        required_precision=mode,
        expected_operations=clipreid_precision_requirements(condition),
    )
    print(f"Engine compute precision: {evidence}", flush=True)
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "workspace_bytes": WORKSPACE_BYTES,
                "compute_precision": evidence,
                "quantization": quantization,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("2", "3a", "3b"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require_queue()
    convert(args.condition, args.checkpoint, args.calibration, args.output)


if __name__ == "__main__":
    main()
