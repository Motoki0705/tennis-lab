"""Measure synchronous warm inference in a fresh process, or capture one NVTX range."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from batching import batch_slices, require_engine_profile
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
    def __init__(self, condition, checkpoint, *, batch_size=1):
        self.dtype = torch.float32 if condition == "0" else torch.float16
        self.model = VisualFeatures(checkpoint).to(dtype=self.dtype).eval().cuda()
        self.set_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        self.pixels = torch.empty(
            (batch_size, *BATCH_SHAPE[1:]), device="cuda", dtype=self.dtype
        )

    def run(self):
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
    def __init__(
        self, plan, *, omit_layer_nvtx=False, batch_size=1, input_profile=None
    ):
        import tensorrt as trt

        super().__init__(plan, omit_layer_nvtx=omit_layer_nvtx)
        self.require_bindings(
            {
                INPUT_NAME: trt.TensorIOMode.INPUT,
                OUTPUT_NAME: trt.TensorIOMode.OUTPUT,
            }
        )
        self.dynamic_batch = input_profile is not None
        if self.dynamic_batch:
            require_engine_profile(self.engine, input_profile)
            if tuple(self.engine.get_tensor_shape(INPUT_NAME)) != (
                -1,
                *BATCH_SHAPE[1:],
            ):
                raise ValueError("CLIP engine lacks a dynamic input batch dimension")
        self.dtypes = {trt.float16: torch.float16}
        if not self.dynamic_batch:
            self.dtypes[trt.float32] = torch.float32
        self.set_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        shape = (batch_size, *BATCH_SHAPE[1:])
        if self.dynamic_batch:
            self.set_input_shape(INPUT_NAME, shape)
            if unresolved := self.context.infer_shapes():
                raise ValueError(f"Unresolved TensorRT shape inputs: {unresolved}")
        self.pixels = self.bind(INPUT_NAME, shape, self.dtypes)
        self.features = self.bind(
            OUTPUT_NAME, (batch_size, FEATURE_DIMENSION), self.dtypes
        )

    def run(self):
        self.execute()
        return self.features

    def compute_precision(self, condition):
        return verify_compute_precision(
            inspect_engine(self.engine, self.context if self.dynamic_batch else None),
            TRT_PRECISIONS[condition],
            expected_operations=clipreid_precision_requirements(condition),
        )


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


def load_batch(runner, batch):
    if runner.pixels.shape != batch.shape:
        raise ValueError(
            f"Batch shape {batch.shape} differs from {runner.pixels.shape}"
        )
    runner.pixels.copy_(torch.from_numpy(batch))
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


def stats(samples_ms, batch=1):
    p25, p50, p75, p95 = np.quantile(samples_ms, [0.25, 0.5, 0.75, 0.95])
    return {
        "median_ms": float(p50),
        "p95_ms": float(p95),
        "iqr_ms": float(p75 - p25),
        "mad_ms": float(np.median(np.abs(np.asarray(samples_ms) - p50))),
        "images_per_second": len(samples_ms) * batch * 1000 / sum(samples_ms),
    }


def measure_latency(
    runner, data, memory, *, load=load_sample, summarize=stats, stage_prefix=""
):
    memory.sample(f"{stage_prefix}after_warmup")
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
        memory.sample(f"{stage_prefix}after_timed_block_{block + 1}")

    return {
        **summarize([value for block in blocks for value in block]),
        "warmup": WARMUP,
        "blocks": [summarize(block) for block in blocks],
        "latency_samples_ms": blocks,
    }


def collect_features(runner, batches):
    features = []
    for batch in batches:
        load_batch(runner, batch)
        result = runner.run()
        torch.cuda.synchronize()
        value = result.float().cpu().numpy()
        if (
            value.shape != (len(batch), FEATURE_DIMENSION)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid CLIP features")
        features.append(value)
    return features


def measure(runner, batches, memory, *, check_precision=None):
    batch_shape = tuple(runner.pixels.shape)
    tail = batches[-1:] if len(batches[-1]) != batch_shape[0] else []
    full = batches[:-1] if tail else batches
    warmup(runner, full, load=load_batch)
    latency = measure_latency(
        runner,
        full,
        memory,
        load=load_batch,
        summarize=partial(stats, batch=batch_shape[0]),
    )
    features = collect_features(runner, full)
    if tail:
        tail_size = len(tail[0])
        runner.set_batch_size(tail_size)
        warmup(runner, tail, load=load_batch)
        latency["tail_batch"] = {
            "size": tail_size,
            **measure_latency(
                runner,
                tail,
                memory,
                load=load_batch,
                summarize=partial(stats, batch=tail_size),
                stage_prefix="tail_",
            ),
        }
        if check_precision is not None:
            latency["tail_batch"]["trt_compute_precision"] = check_precision()
        features.extend(collect_features(runner, tail))
    memory.sample("after_accuracy")
    return {
        **latency,
        **memory.result(),
        "batch_shape": batch_shape,
        "input_dtype": str(runner.pixels.dtype),
        "driver": memory.driver,
        "device_uuid": memory.uuid,
        "device": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }, np.concatenate(features, axis=0)


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
    if not (args.probe and args.dynamic_batch):
        batch_slices(len(data), args.batch_size)
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
        trt_settings = {}
        check_precision = None
        if args.condition in TRT_PRECISIONS:
            plan = args.plan or args.output / PLAN
            input_profile = None
            if args.dynamic_batch:
                certificate = json.loads((plan.parent / "conversion.json").read_text())
                input_profile = certificate["input_profile"]
            runner = TensorRTRunner(
                plan,
                omit_layer_nvtx=(
                    memory is not None or args.omit_trt_layer_nvtx or args.dynamic_batch
                ),
                batch_size=args.batch_size,
                input_profile=input_profile,
            )
            if memory is not None:
                check_precision = partial(runner.compute_precision, args.condition)
                trt_settings = {
                    "trt_nvtx_verbosity": runner.context.nvtx_verbosity.name,
                    "trt_compute_precision": check_precision(),
                }
        else:
            runner = TorchRunner(
                args.condition, args.checkpoint, batch_size=args.batch_size
            )
        torch.cuda.synchronize()
        batches = [data[rows] for rows in batch_slices(len(data), args.batch_size)]
        if args.probe:
            collect_features(runner, batches[:1])
            print(f"CLIP executed B={args.batch_size}", flush=True)
            return
        if memory is not None:
            memory.sample("after_setup")
        if args.profile_index is not None:
            full = [batch for batch in batches if len(batch) == args.batch_size]
            index = args.profile_index // args.batch_size
            if not 0 <= index < len(full):
                raise ValueError("The CLIP profiling crop must belong to a full batch")
            warmup(runner, full, load=load_batch)
            profile_one(runner, full[index], load=load_batch)
            return
        result, features = measure(
            runner, batches, memory, check_precision=check_precision
        )
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--profile-index", type=int)
    parser.add_argument("--omit-trt-layer-nvtx", action="store_true")
    args = parser.parse_args()
    if args.dynamic_batch and args.condition != "2":
        parser.error("--dynamic-batch requires condition 2")
    if args.batch_size < 1 and not (args.probe and args.dynamic_batch):
        parser.error("--batch-size must be positive")
    if args.probe and args.profile_index is not None:
        parser.error("--probe and --profile-index are mutually exclusive")
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
