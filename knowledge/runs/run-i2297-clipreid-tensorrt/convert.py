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


def constant_tensors(graph):
    """Identify initializer data propagated through weight layout and Q/DQ."""
    constants = {item.name for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" or (
            node.op_type
            in {
                "Cast",
                "Transpose",
                "Identity",
                "Reshape",
                "QuantizeLinear",
                "DequantizeLinear",
            }
            and all(name in constants for name in node.input if name)
        ):
            constants.update(node.output)
    return constants


def linear_names(graph):
    """Find MatMul/Gemm with a constant second operand, even after Q/DQ insertion."""
    constants = constant_tensors(graph)
    return {
        node.name
        for node in graph.node
        if node.op_type in {"MatMul", "Gemm"} and node.input[1] in constants
    }


def quantized_matrix_ops(graph):
    """Compute ops with explicit DQ on both multiplicands."""
    dequantized = {
        result
        for node in graph.node
        if node.op_type == "DequantizeLinear"
        for result in node.output
    }
    return {
        node.name
        for node in graph.node
        if node.op_type in {"MatMul", "Gemm", "Conv", "ConvTranspose"}
        and set(node.input[:2]) <= dequantized
    }


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


def verify_compute_precision(
    details, required_precision, *, expected_operations=None, constant_operations=()
):
    """Require operand evidence for each compute layer and cover the pinned ONNX ops."""
    precisions = Counter()
    seen = set()
    for layer in details["Layers"]:
        kind = layer["LayerType"].lower()
        tactic = layer.get("TacticName") or ""
        operands = TACTIC_OPERANDS.search(tactic)
        metadata = set(ONNX_LAYER.findall(layer.get("Metadata") or ""))
        operations = metadata
        if expected_operations is not None:
            operations = metadata & expected_operations.keys()
        fused_attention = kind == "kgen" and tactic.startswith(
            ("_gemm_mha_v2_", "_gemv_mha_v1_")
        )
        singleton_matmul = (
            kind == "kgen"
            and bool(operations)
            and tactic.startswith("__myl_ReshMoveMul_")
        )
        if (
            kind not in COMPUTE_TYPES
            and not operands
            and not (fused_attention or singleton_matmul)
        ):
            continue

        name = layer["Name"]
        if expected_operations is None:
            precision = required_precision
        else:
            targets = {expected_operations[op] for op in operations}
            if len(targets) != 1:
                raise RuntimeError(
                    f"Compute layer {name} has no unique precision requirement"
                )
            (precision,) = targets

        inputs = [tensor["Format/Datatype"] for tensor in layer.get("Inputs", [])]
        pattern = re.compile(rf"\b(?:{TENSOR_DTYPE[precision]})\b", re.IGNORECASE)
        if fused_attention:
            # GVHMR fuses the local and padding masks between K and V.
            qkv = (
                [inputs[0], inputs[1], inputs[4]]
                if len(inputs) == 5 and inputs[2:4] == ["Bool", "Bool"]
                else inputs
            )
            matches = len(qkv) == 3 and all(pattern.search(dtype) for dtype in qkv)
        elif singleton_matmul:
            # HMR2 self-attention has one token. TensorRT lowers its value
            # MatMul to one scalar multiply per head and channel.
            shapes = sorted(tuple(t["Dimensions"]) for t in layer["Inputs"])
            matches = (
                precision == "fp16"
                and len(operations) == 1
                and shapes == [(1, 8, 1, 1), (8, 1, 64)]
                and all(pattern.search(dtype) for dtype in inputs)
            )
        else:
            code = OPERAND_CODE[precision]
            # The FP16 residual is the first binding of this INT8 GEMM epilogue.
            # Require both INT8 tactic operands and an explicit fused Q/DQ pair.
            int8_residual = (
                precision == "int8"
                and kind == "fusion"
                and name.startswith("__myl_FcCastAddMulAdd_")
                and tactic.endswith("_by_fusion_tactic")
                and inputs == ["Half", "Int8"]
                and len(operations) == 1
                and any(
                    node.endswith("_QuantizeLinear")
                    and node.removesuffix("_QuantizeLinear") + "_DequantizeLinear"
                    in metadata
                    for node in metadata
                )
            )
            matches = (
                operands is not None
                and operands.groups() == (code, code)
                and (bool(inputs and pattern.search(inputs[0])) or int8_residual)
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
        unobserved = expected_operations.keys() - seen
        missing = unobserved - set(constant_operations)
        if missing:
            raise RuntimeError(
                f"Missing compute evidence for ONNX operations: {sorted(missing)}"
            )
        evidence["onnx_operations_by_precision"] = dict(
            Counter(expected_operations[op] for op in seen)
        )
        if unobserved:
            evidence["unobserved_constant_onnx_operations"] = sorted(unobserved)
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
    constant_operations=(),
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
    context_shapes = {}
    for label, shapes in shapes_by_label.items():
        if shapes is not None:
            for name, shape in shapes.items():
                if not context.set_input_shape(name, shape):
                    raise ValueError(
                        f"TensorRT rejected inspection shape {name}={shape}"
                    )
            actual_shapes = {
                name: list(context.get_tensor_shape(name)) for name in shapes
            }
            expected_shapes = {name: list(shape) for name, shape in shapes.items()}
            if actual_shapes != expected_shapes:
                raise ValueError(
                    f"TensorRT inspection context returned {actual_shapes}, expected {expected_shapes}"
                )
            context_shapes[label] = actual_shapes
        details = inspect_engine(engine, context)
        plan.with_suffix(f"{label}.layers.json").write_text(
            json.dumps(details, indent=2) + "\n"
        )
        details_by_label[label] = details
    evidence = {
        label.removeprefix("."): verify_compute_precision(
            details,
            required_precision,
            expected_operations=expected_operations,
            constant_operations=constant_operations,
        )
        for label, details in details_by_label.items()
    }
    for label, shapes in context_shapes.items():
        evidence[label.removeprefix(".")]["bound_input_shapes"] = shapes
    with plan.open("xb") as stream:
        stream.write(memoryview(engine_bytes))
    return evidence[""] if profile_shapes is None else evidence


def modelopt_ptq(
    graph,
    target,
    calibration,
    mode,
    *,
    input_name,
    method="max",
    nodes=None,
    external_data=False,
    disable_mha_qdq=False,
):
    import onnx
    from modelopt.onnx.quantization.quantize import quantize

    if nodes is not None and not nodes:
        raise ValueError("Explicit PTQ selection is empty")
    if method not in {"max", "entropy"}:
        raise ValueError(f"Unsupported ModelOpt calibration method: {method}")
    quantize(
        str(graph),
        quantize_mode=mode,
        calibration_data={input_name: calibration},
        calibration_method=method,
        calibration_eps=["cpu"],
        nodes_to_quantize=None
        if nodes is None
        else [rf"^{re.escape(name)}$" for name in nodes],
        high_precision_dtype="fp16",
        disable_mha_qdq=disable_mha_qdq,
        use_external_data_format=external_data,
        output_path=str(target),
    )
    model = onnx.load(target, load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    if not ops["QuantizeLinear"] or not ops["DequantizeLinear"]:
        raise RuntimeError(f"{mode} PTQ produced no explicit Q/DQ pairs")
    matrix = quantized_matrix_ops(model.graph)
    if nodes is not None and matrix != set(nodes):
        raise RuntimeError(
            f"{mode} Q/DQ compute scope differs from request: "
            f"missing {sorted(set(nodes) - matrix)}, unexpected {sorted(matrix - set(nodes))}"
        )
    return {
        "mode": mode,
        "method": method,
        "crops": len(calibration),
        "Q": ops["QuantizeLinear"],
        "DQ": ops["DequantizeLinear"],
        "qdq_matrix_operations": len(matrix),
    }


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
        quantization = modelopt_ptq(graph, target, data, mode, input_name=INPUT_NAME)
        graph = target
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


def convert_adapter(adapter, condition, checkpoint, calibration, output, recipe):
    mode = TRT_PRECISIONS[condition]
    exported = output / "export.onnx"
    adapter.export(checkpoint, exported, fp16=condition == "2")
    graph = exported
    quantization = None
    selected = set()
    if mode in {"int8", "fp8"}:
        if calibration is None:
            raise ValueError("PTQ requires fixed calibration tensors")
        target = output / "quantized.onnx"
        data = np.load(calibration, allow_pickle=False)
        if (
            data.dtype != np.float32
            or data.shape[1:] != adapter.batch_shape[1:]
            or not len(data)
        ):
            raise ValueError("Calibration tensor has an unexpected dtype or shape")
        quantization, selected = adapter.quantize(graph, target, data, mode, recipe)
        graph = target
    requirements = adapter.compute_requirements(exported, mode, selected)
    plan = output / f"{adapter.model}.plan"
    plan.with_suffix(".precision.json").write_text(
        json.dumps(requirements, indent=2) + "\n"
    )
    evidence = build_engine(graph, plan, required_precision=mode, **requirements)
    receipt = {
        "condition": condition,
        "workspace_bytes": WORKSPACE_BYTES,
        "quantization": quantization,
        "compute_precision": evidence,
    }
    (output / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Engine compute precision: {evidence}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("2", "3a", "3b"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--ptq-recipe")
    args = parser.parse_args()
    require_queue()
    if args.adapter:
        from benchmark import load_adapter

        convert_adapter(
            load_adapter(args.adapter, args.model),
            args.condition,
            args.checkpoint,
            args.calibration,
            args.output,
            args.ptq_recipe,
        )
    else:
        convert(args.condition, args.checkpoint, args.calibration, args.output)


if __name__ == "__main__":
    main()
