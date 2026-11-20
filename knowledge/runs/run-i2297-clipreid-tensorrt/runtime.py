"""Measure synchronous warm inference in a fresh process, or capture one NVTX range."""

import argparse
import json
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from clipreid import (
    BATCH_SHAPE,
    FEATURE_DIMENSION,
    INPUT_NAME,
    OUTPUT_NAME,
    VisualFeatures,
)
from convert import (
    CONDITIONS,
    PLAN,
    TRT_PRECISIONS,
    clipreid_precision_requirements,
    inspect_engine,
    require_queue,
    verify_compute_precision,
)
from profiles import PROFILE_RANGE

WARMUP = 30
BLOCKS = 3
REPEATS_PER_BLOCK = 100


class TorchRunner:
    def __init__(self, condition, checkpoint, adapter=None):
        autocast = adapter is not None and adapter.autocast and condition == "1"
        dtype = torch.float32 if condition == "0" or autocast else torch.float16
        model = (
            VisualFeatures(checkpoint)
            if adapter is None
            else adapter.torch_model(checkpoint)
        )
        self.model = model.to(dtype=dtype).eval().cuda()
        shape = BATCH_SHAPE if adapter is None else adapter.engine_shape
        self.pixels = torch.empty(shape, device="cuda", dtype=dtype)
        self.autocast = autocast

    def run(self):
        with (
            torch.autocast("cuda", dtype=torch.float16)
            if self.autocast
            else nullcontext()
        ):
            return self.model(self.pixels)


class TensorRTContext:
    def __init__(self, plan, *, omit_layer_nvtx=False):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine load failed: {plan}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        if omit_layer_nvtx:
            self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
            if self.context.nvtx_verbosity != trt.ProfilingVerbosity.NONE:
                raise RuntimeError("Could not disable TensorRT layer NVTX ranges")
        self.stream = torch.cuda.current_stream()

    def require_bindings(self, expected):
        actual = {
            name: self.engine.get_tensor_mode(name)
            for name in (
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            )
        }
        if expected != actual:
            raise ValueError(f"Unexpected engine bindings: {actual}")

    def set_input_shape(self, name, shape):
        if not self.context.set_input_shape(name, shape):
            raise ValueError(f"TensorRT rejected input shape {name}={shape}")

    def bind(self, name, expected_shape, dtypes):
        shape = tuple(self.context.get_tensor_shape(name))
        if shape != expected_shape:
            raise ValueError(f"Unexpected engine shape for {name}: {shape}")
        dtype = self.engine.get_tensor_dtype(name)
        buffer = torch.empty(shape, device="cuda", dtype=dtypes[dtype])
        if not self.context.set_tensor_address(name, buffer.data_ptr()):
            raise RuntimeError(f"Could not bind TensorRT tensor: {name}")
        return buffer

    def execute(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")


class TensorRTRunner(TensorRTContext):
    def __init__(self, plan, *, omit_layer_nvtx=False, adapter=None):
        import tensorrt as trt

        super().__init__(plan, omit_layer_nvtx=omit_layer_nvtx)
        self.require_bindings(
            {
                INPUT_NAME: trt.TensorIOMode.INPUT,
                OUTPUT_NAME: trt.TensorIOMode.OUTPUT,
            }
        )
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        input_shape = BATCH_SHAPE if adapter is None else adapter.engine_shape
        output_shape = (
            (1, FEATURE_DIMENSION) if adapter is None else adapter.engine_output_shape
        )
        self.pixels = self.bind(INPUT_NAME, input_shape, dtypes)
        self.features = self.bind(OUTPUT_NAME, output_shape, dtypes)

    def run(self):
        self.execute()
        return self.features


class DeviceMemory:
    """NVML whole-device snapshots, collected on a synchronized, shared-queue GPU."""

    def __init__(self):
        import pynvml

        self.api = pynvml
        self.samples = {}
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() != 1:
            raise RuntimeError("This run requires one NVML GPU")
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.uuid = pynvml.nvmlDeviceGetUUID(self.gpu)
        self.driver = pynvml.nvmlSystemGetDriverVersion()

    def sample(self, stage):
        memory = self.api.nvmlDeviceGetMemoryInfo(self.gpu)
        self.samples[stage] = int(memory.used)
        return int(memory.free)

    def result(self):
        return {
            "nvml_device_peak_sampled_bytes": max(self.samples.values()),
            "nvml_device_peak_minus_idle_bytes": max(self.samples.values())
            - self.samples["idle"],
            "nvml_device_snapshots_bytes": self.samples,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def close(self):
        self.api.nvmlShutdown()


def load_sample(runner, sample):
    runner.pixels.copy_(torch.from_numpy(sample[None]))
    torch.cuda.synchronize()


def warmup(runner, data, *, load=load_sample):
    for i in range(WARMUP):
        load(runner, data[i % len(data)])
        runner.run()
    torch.cuda.synchronize()


def profile_one(runner, sample, *, load=load_sample):
    load(runner, sample)
    cudart = torch.cuda.cudart()
    if int(cudart.cudaProfilerStart()) != 0:
        raise RuntimeError("cudaProfilerStart failed")
    try:
        with torch.cuda.nvtx.range(PROFILE_RANGE):
            runner.run()
            torch.cuda.synchronize()
    finally:
        if int(cudart.cudaProfilerStop()) != 0:
            raise RuntimeError("cudaProfilerStop failed")


def torch_dispatch_precisions(infer, expected_dtype):
    from torch.utils._python_dispatch import TorchDispatchMode

    # The Python dispatch layer can see either the composite op or its
    # lowering, depending on the PyTorch release and execution path.
    compute = {
        "linear",
        "matmul",
        "addmm",
        "mm",
        "bmm",
        "conv2d",
        "conv_transpose2d",
        "convolution",
        "_convolution",
    }
    watched = compute | {
        "layer_norm",
        "native_layer_norm",
        "softmax",
        "_softmax",
        "gelu",
        "add",
    }
    seen = defaultdict(Counter)

    class Dtypes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name.removeprefix("aten::")
            result = func(*args, **(kwargs or {}))
            if name in watched:
                tensors = (result,) if isinstance(result, torch.Tensor) else result
                seen[name][str(tensors[0].dtype)] += 1
            return result

    with Dtypes():
        infer()
    expected = str(expected_dtype)
    measured = {name for name in seen if name in compute}
    if not measured:
        raise RuntimeError(
            f"PyTorch precision probe observed no matrix or convolution ops: {dict(seen)}"
        )
    if any(set(seen[op]) != {expected} for op in measured):
        raise RuntimeError(
            f"PyTorch compute did not execute at {expected}: {dict(seen)}"
        )
    return dict(seen)


def stats(samples_ms, batch=1):
    p25, p50, p75, p95 = np.quantile(samples_ms, [0.25, 0.5, 0.75, 0.95])
    return {
        "median_ms": float(p50),
        "p95_ms": float(p95),
        "iqr_ms": float(p75 - p25),
        "mad_ms": float(np.median(np.abs(np.asarray(samples_ms) - p50))),
        "images_per_second": len(samples_ms) * batch * 1000 / sum(samples_ms),
    }


def measure_latency(runner, data, memory, *, load=load_sample, summarize=stats):
    memory.sample("after_warmup")
    blocks = []
    for block in range(BLOCKS):
        elapsed = []
        for i in range(REPEATS_PER_BLOCK):
            load(runner, data[(block * REPEATS_PER_BLOCK + i) % len(data)])
            start = time.perf_counter_ns()
            runner.run()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        blocks.append(elapsed)
        memory.sample(f"after_timed_block_{block + 1}")

    return {
        **summarize([value for block in blocks for value in block]),
        "warmup": WARMUP,
        "blocks": [summarize(block) for block in blocks],
        "latency_samples_ms": blocks,
    }


def measure(runner, data, memory, output_shape=(FEATURE_DIMENSION,)):
    batch = runner.pixels.shape[0]
    latency = measure_latency(
        runner, data, memory, summarize=lambda samples: stats(samples, batch)
    )
    features = []
    for sample in data:
        load_sample(runner, sample)
        result = runner.run()
        torch.cuda.synchronize()
        features.append(result[0].float().cpu().numpy())
    embedding = np.stack(features)
    if (
        embedding.shape != (len(data), *output_shape)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid measured model output: {embedding.shape}")
    memory.sample("after_accuracy")
    return {
        **latency,
        **memory.result(),
        "batch_shape": tuple(runner.pixels.shape),
        "input_dtype": str(runner.pixels.dtype),
        "driver": memory.driver,
        "device_uuid": memory.uuid,
        "device": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }, embedding


@torch.inference_mode()
def run(args):
    from benchmark import load_adapter

    torch.set_num_threads(4)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    adapter = load_adapter(args.adapter, args.model) if args.adapter else None
    shape = BATCH_SHAPE if adapter is None else adapter.batch_shape
    data = np.load(args.eval, allow_pickle=False)
    if (
        data.ndim != 4
        or data.shape[1:] != shape[1:]
        or data.dtype != np.float32
        or not len(data)
    ):
        raise ValueError("Unexpected evaluation tensor dtype or shape")
    if args.profile_index is not None and not 0 <= args.profile_index < len(data):
        raise ValueError("Representative input index is outside evaluation tensors")
    memory = None if args.profile_index is not None else DeviceMemory()
    try:
        if memory is not None and memory.sample("idle") < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB free before inference")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise RuntimeError("The pinned run requires one CC 12.0 CUDA GPU")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        trt_settings = {}
        if args.condition in TRT_PRECISIONS:
            plan = args.plan or args.output / PLAN
            runner = TensorRTRunner(
                plan,
                omit_layer_nvtx=memory is not None or args.omit_trt_layer_nvtx,
                adapter=adapter,
            )
            if memory is not None:
                expected = (
                    clipreid_precision_requirements(args.condition)
                    if adapter is None
                    else json.loads(plan.with_suffix(".precision.json").read_text())
                )
                trt_settings = {
                    "trt_nvtx_verbosity": runner.context.nvtx_verbosity.name,
                    "trt_compute_precision": verify_compute_precision(
                        inspect_engine(runner.engine),
                        TRT_PRECISIONS[args.condition],
                        expected_operations=expected,
                    ),
                }
        else:
            runner = TorchRunner(args.condition, args.checkpoint, adapter)
        if adapter is not None:
            runner = adapter.wrap_runner(runner)
        torch.cuda.synchronize()
        if memory is not None:
            memory.sample("after_setup")
        warmup(runner, data)
        if args.profile_index is not None:
            profile_one(runner, data[args.profile_index])
            return
        output_shape = (FEATURE_DIMENSION,) if adapter is None else adapter.output_shape
        result, features = measure(runner, data, memory, output_shape)
        if adapter is not None:
            result.update(adapter.measurement_settings(args.condition))
            if args.condition in {"0", "1"}:
                load_sample(runner, data[0])
                expected_dtype = (
                    torch.float32 if args.condition == "0" else torch.float16
                )
                result["torch_first_eval_crop_op_output_dtypes"] = (
                    torch_dispatch_precisions(runner.run, expected_dtype)
                )
                torch.cuda.synchronize()
        np.save(args.output / "features.npy", features, allow_pickle=False)
        (args.output / "measurement.json").write_text(
            json.dumps({**result, **trt_settings}, indent=2, allow_nan=False) + "\n"
        )
    finally:
        if memory is not None:
            memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile-index", type=int)
    parser.add_argument("--omit-trt-layer-nvtx", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
