"""Measure synchronous warm inference in a fresh process, or capture one NVTX range."""

import argparse
import json
import time
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
from convert import CONDITIONS, PLAN, require_queue
from profiles import PROFILE_RANGE

WARMUP = 30
BLOCKS = 3
REPEATS_PER_BLOCK = 100


class TorchRunner:
    def __init__(self, condition, checkpoint):
        dtype = torch.float32 if condition == "0" else torch.float16
        self.model = VisualFeatures(checkpoint).to(dtype=dtype).eval().cuda()
        self.pixels = torch.empty(BATCH_SHAPE, device="cuda", dtype=dtype)

    def run(self):
        return self.model(self.pixels)


class TensorRTRunner:
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
        expected = {
            INPUT_NAME: trt.TensorIOMode.INPUT,
            OUTPUT_NAME: trt.TensorIOMode.OUTPUT,
        }
        actual = {
            name: self.engine.get_tensor_mode(name)
            for name in (
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            )
        }
        if expected != actual:
            raise ValueError(f"Unexpected engine bindings: {actual}")
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.pixels = self.bind(INPUT_NAME, BATCH_SHAPE, dtypes)
        self.features = self.bind(OUTPUT_NAME, (1, FEATURE_DIMENSION), dtypes)
        self.stream = torch.cuda.current_stream()

    def bind(self, name, expected_shape, dtypes):
        shape = tuple(self.context.get_tensor_shape(name))
        if shape != expected_shape:
            raise ValueError(f"Unexpected engine shape for {name}: {shape}")
        dtype = self.engine.get_tensor_dtype(name)
        buffer = torch.empty(shape, device="cuda", dtype=dtypes[dtype])
        if not self.context.set_tensor_address(name, buffer.data_ptr()):
            raise RuntimeError(f"Could not bind TensorRT tensor: {name}")
        return buffer

    def run(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
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


def warmup(runner, data):
    for i in range(WARMUP):
        load_sample(runner, data[i % len(data)])
        runner.run()
    torch.cuda.synchronize()


def profile_one(runner, sample):
    load_sample(runner, sample)
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


def stats(samples_ms):
    p25, p50, p75, p95 = np.quantile(samples_ms, [0.25, 0.5, 0.75, 0.95])
    return {
        "median_ms": float(p50),
        "p95_ms": float(p95),
        "iqr_ms": float(p75 - p25),
        "mad_ms": float(np.median(np.abs(np.asarray(samples_ms) - p50))),
        "images_per_second": len(samples_ms) * BATCH_SHAPE[0] * 1000 / sum(samples_ms),
    }


def measure(runner, data, memory):
    memory.sample("after_warmup")
    blocks = []
    for block in range(BLOCKS):
        elapsed = []
        for i in range(REPEATS_PER_BLOCK):
            load_sample(runner, data[(block * REPEATS_PER_BLOCK + i) % len(data)])
            start = time.perf_counter_ns()
            runner.run()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        blocks.append(elapsed)
        memory.sample(f"after_timed_block_{block + 1}")

    features = []
    for sample in data:
        load_sample(runner, sample)
        result = runner.run()
        torch.cuda.synchronize()
        features.append(result[0].float().cpu().numpy())
    embedding = np.stack(features)
    if (
        embedding.shape != (len(data), FEATURE_DIMENSION)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError("Invalid CLIP features")
    memory.sample("after_accuracy")
    return {
        **stats([value for block in blocks for value in block]),
        **memory.result(),
        "warmup": WARMUP,
        "blocks": [stats(block) for block in blocks],
        "latency_samples_ms": blocks,
        "batch_shape": BATCH_SHAPE,
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
    torch.set_num_threads(4)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    data = np.load(args.eval, allow_pickle=False)
    if (
        data.ndim != 4
        or data.shape[1:] != BATCH_SHAPE[1:]
        or data.dtype != np.float32
        or not len(data)
    ):
        raise ValueError("Unexpected evaluation tensor dtype or shape")
    memory = None if args.profile_index is not None else DeviceMemory()
    try:
        if memory is not None and memory.sample("idle") < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB free before CLIP inference")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise RuntimeError("The pinned CLIP run requires one CC 12.0 CUDA GPU")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        runner = (
            TorchRunner(args.condition, args.checkpoint)
            if args.condition in {"0", "1"}
            else TensorRTRunner(
                args.output / PLAN, omit_layer_nvtx=args.omit_trt_layer_nvtx
            )
        )
        torch.cuda.synchronize()
        if memory is not None:
            memory.sample("after_setup")
        warmup(runner, data)
        if args.profile_index is not None:
            profile_one(runner, data[args.profile_index])
            return
        result, features = measure(runner, data, memory)
        np.save(args.output / "features.npy", features, allow_pickle=False)
        (args.output / "measurement.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
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
    parser.add_argument("--profile-index", type=int)
    parser.add_argument("--omit-trt-layer-nvtx", action="store_true")
    args = parser.parse_args()
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
