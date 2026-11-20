"""Export the fixed CLIP model and build one explicitly typed TensorRT engine."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

WORKSPACE_BYTES = 512 * 1024**2
PLAN = "clipreid.plan"
CONDITIONS = ("0", "1", "2", "3a", "3b")


def require_queue():
    if not os.environ.get("TENNIS_RUN_ID") or not os.environ.get("TENNIS_REPRO_DIR"):
        raise RuntimeError("Run GPU workloads through the shared training queue.")


def precision_layers(details, precision):
    spelling = {"fp16": r"FP16|Float16|Half", "int8": "Int8", "fp8": "FP8"}[precision]
    pattern = re.compile(rf"\b(?:{spelling})\b", re.IGNORECASE)
    return [
        layer["Name"]
        for layer in details["Layers"]
        if any(
            pattern.search(tensor.get("Format/Datatype", ""))
            for key in ("Inputs", "Outputs")
            for tensor in layer.get(key, [])
        )
    ]


def build_engine(onnx_file, plan, *, required_precision):
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
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_file}")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the built engine")
    inspector = engine.create_engine_inspector()
    details = json.loads(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    )
    output = plan.with_suffix(".layers.json")
    output.write_text(json.dumps(details, indent=2) + "\n")
    layers = precision_layers(details, required_precision)
    if not layers:
        raise RuntimeError(
            f"No engine layer exposes {required_precision} tensors. See {output}"
        )
    with plan.open("xb") as stream:
        stream.write(memoryview(engine_bytes))
    return layers


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
    mode = {"2": "fp16", "3a": "int8", "3b": "fp8"}[condition]
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
    layers = build_engine(graph, output / PLAN, required_precision=mode)
    print(f"Engine exposes {mode} tensors in {len(layers)} layers", flush=True)
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "workspace_bytes": WORKSPACE_BYTES,
                "required_precision_layers": layers,
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
