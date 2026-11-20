"""Qualify CUDA and TensorRT with executed FP16/INT8/FP8 matrix products."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from convert import build_engine, require_queue

SHAPE = (512, 512)


def matrix_graph(path, dtype):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx"]),
            helper.make_node("DequantizeLinear", ["qx", "scale", "zero"], ["dx"]),
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["qw"]),
            helper.make_node("DequantizeLinear", ["qw", "scale", "zero"], ["dw"]),
            helper.make_node("MatMul", ["dx", "dw"], ["y"]),
        ],
        "explicit_qdq_matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, SHAPE)],
        [
            numpy_helper.from_array(np.full(SHAPE, 0.5, dtype=np.float16), "weight"),
            numpy_helper.from_array(np.asarray(0.125, dtype=np.float16), "scale"),
            helper.make_tensor("zero", dtype, [], bytes([0]), raw=True),
        ],
    )
    # TRT 10.14 accepts ONNX opset 19 and these values match that IR.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=9
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def execute_plan(plan):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    if engine is None:
        raise RuntimeError("Could not load Q/DQ probe engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create Q/DQ probe context")
    x = torch.full(SHAPE, 0.25, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    if not context.set_tensor_address(
        "x", x.data_ptr()
    ) or not context.set_tensor_address("y", y.data_ptr()):
        raise RuntimeError("Could not bind Q/DQ probe tensors")
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("TensorRT Q/DQ smoke inference failed")
    torch.cuda.synchronize()
    if not bool((y == 64).all()):
        raise ValueError(
            f"Incorrect Q/DQ probe output: {y.min().item()}, {y.max().item()}"
        )
    return y[0, 0].item()


def probe_matmul(profile=False):
    import torch

    a = torch.ones(SHAPE, device="cuda", dtype=torch.float16)
    b = torch.ones_like(a)
    c = torch.empty_like(a)
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    if profile:
        from profiles import PROFILE_RANGE

        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range(PROFILE_RANGE):
            torch.mm(a, b, out=c)
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    if c[0, 0].item() != 512:
        raise ValueError("CUDA FP16 smoke output is incorrect")
    return c[0, 0].item()


def probe(output):
    from importlib.metadata import version

    import onnx
    import onnxruntime as ort
    import tensorrt as trt
    import torch

    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("Expected the pinned PyTorch 2.13.0 CUDA 13.0 environment")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Expected CC 12.0")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    result = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "modelopt": version("nvidia-modelopt"),
        "builder_fast_fp16": builder.platform_has_fast_fp16,
        "builder_fast_int8": builder.platform_has_fast_int8,
        "torch_fp16_matmul_output": probe_matmul(),
        "probes": {},
    }
    del builder
    binary = Path(sys.prefix) / "bin/trtexec"
    version_probe = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    output_log = version_probe.stdout + version_probe.stderr
    (output / "trtexec.log").write_text(output_log)
    result["trtexec"] = {
        "exit_code": version_probe.returncode,
        "lines": [
            line
            for line in output_log.splitlines()
            if "TensorRT" in line or "Unknown option" in line
        ][:5],
    }
    for mode, dtype in [
        ("int8", onnx.TensorProto.INT8),
        ("fp8", onnx.TensorProto.FLOAT8E4M3FN),
    ]:
        graph = output / f"{mode}.onnx"
        matrix_graph(graph, dtype)
        plan = graph.with_suffix(".plan")
        layers = build_engine(graph, plan, required_precision=mode)
        result["probes"][mode] = {
            "output": execute_plan(plan),
            "precision_layers": layers,
        }
    text = json.dumps(result, indent=2) + "\n"
    (output / "environment.json").write_text(text)
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--profile-matmul", action="store_true")
    modes.add_argument("--invalid-onnx", action="store_true")
    args = parser.parse_args()
    if args.output is None and not args.profile_matmul:
        parser.error("--output is required")
    require_queue()
    if args.profile_matmul:
        probe_matmul(profile=True)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.invalid_onnx:
            bad_graph = args.output / "bad.onnx"
            bad_graph.write_bytes(b"intentional malformed ONNX")
            build_engine(bad_graph, args.output / "bad.plan", required_precision="fp16")
        else:
            probe(args.output)


if __name__ == "__main__":
    main()
