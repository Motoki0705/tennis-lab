"""Measure or profile one locked GVHMR shape in a fresh CUDA process."""

import argparse
import json
import os
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from batching import batch_slices, require_engine_profile
from benchmark import write_json
from convert import (
    CONDITIONS,
    TRT_PRECISIONS,
    inspect_engine,
    require_queue,
    verify_compute_precision,
)
from inputs import sha256_file
from locked_inputs import HERE, checked_segments, sweep_profile
from model import (
    OUTPUT_WIDTHS,
    PLAN,
    input_shapes,
    load_network,
    output_shapes,
    pack_sequences,
)
from runtime import (
    DeviceMemory,
    TensorRTContext,
    measure_latency,
    profile_one,
    stats,
    warmup,
)


def read_conversion(directory, condition):
    certificate = json.loads((directory / "conversion.json").read_text())
    plan = directory / PLAN
    profile = certificate["dynamic_profile"]
    batches = profile.get("profile_batches")
    if batches:
        expected_profile = sweep_profile(batches["opt"], batches["max"])
        if condition != "2" or certificate["expected_precision_sha256"] != sha256_file(
            directory / "expected-precision.json"
        ):
            raise ValueError(
                "GVHMR batch precision requirements differ from conversion"
            )
    else:
        expected_profile = json.loads((HERE / "shape-profile.json").read_text())
    if (
        certificate["condition"] != condition
        or certificate["tensors_sha256"] != sha256_file(HERE / "tensor-lock.json")
        or profile != expected_profile
        or certificate["engine_sha256"] != sha256_file(plan)
        or set(certificate["compute_precision"]) != {"min", "opt", "max"}
    ):
        raise ValueError(
            "TensorRT plan or its conversion receipt does not match this run"
        )
    return certificate


class PytorchRunner:
    def __init__(self, condition, checkpoint, frames, *, batch_size=1):
        dtype = torch.float32 if condition == "0" else torch.float16
        self.model = load_network(checkpoint).to(device="cuda", dtype=dtype)
        self.buffers = {
            key: torch.empty(
                shape, device="cuda", dtype=torch.int64 if key == "length" else dtype
            )
            for key, shape in input_shapes(frames, batch_size).items()
        }

    def run(self):
        return self.model(*self.buffers.values())


class TensorRTSequenceRunner(TensorRTContext):
    def __init__(self, plan, frames, *, batch_size=1, input_profile=None):
        import tensorrt as trt

        super().__init__(plan, omit_layer_nvtx=True)
        shapes = input_shapes(frames, batch_size)
        self.require_bindings(
            {
                **dict.fromkeys(shapes, trt.TensorIOMode.INPUT),
                **dict.fromkeys(OUTPUT_WIDTHS, trt.TensorIOMode.OUTPUT),
            }
        )
        if self.engine.num_optimization_profiles != 1:
            raise ValueError(
                "The GVHMR engine must have exactly one optimization profile"
            )
        leading_dim = 1
        if input_profile is not None:
            require_engine_profile(self.engine, input_profile)
            leading_dim = -1
        for name, shape in shapes.items():
            expected = (
                (leading_dim,) if name == "length" else (leading_dim, -1, *shape[2:])
            )
            if tuple(self.engine.get_tensor_shape(name)) != expected:
                raise ValueError(
                    f"Unexpected GVHMR engine shape for {name}, expected {expected}"
                )
            self.set_input_shape(name, shape)
        dtypes = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bool: torch.bool,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
        }
        self.buffers = {
            name: self.bind(name, shape, dtypes) for name, shape in shapes.items()
        }
        unresolved = self.context.infer_shapes()
        if unresolved:
            raise ValueError(f"Unresolved TensorRT shape inputs: {unresolved}")
        self.outputs = tuple(
            self.bind(name, shape, dtypes)
            for name, shape in output_shapes(frames, batch_size).items()
        )
        if (
            self.buffers["length"].dtype != torch.int64
            or self.outputs[-2].dtype != torch.bool
        ):
            raise ValueError("TensorRT did not preserve the GVHMR length/mask types")
        if input_profile is not None and (
            any(
                tensor.dtype != torch.float16
                for name, tensor in self.buffers.items()
                if name != "length"
            )
            or any(tensor.dtype != torch.float16 for tensor in self.outputs[:3])
        ):
            raise ValueError(
                "The batched GVHMR engine requires FP16 floating point bindings"
            )

    def run(self):
        self.execute()
        return self.outputs


def load_sample(runner, sample):
    for name, buffer in runner.buffers.items():
        if buffer.shape != sample[name].shape:
            raise ValueError(f"Batch shape for {name} differs from the runtime binding")
        buffer.copy_(sample[name])
    torch.cuda.synchronize()


def seq_stats(times, *, frames, batch_size=1):
    result = stats(times, batch=batch_size)
    result["sequences_per_second"] = result.pop("images_per_second")
    result["frames_per_second"] = result["sequences_per_second"] * frames
    return result


def host_sample(runner, arrays):
    return {
        name: torch.from_numpy(arrays[name]).to(dtype=buffer.dtype)
        for name, buffer in runner.buffers.items()
    }


def collect_outputs(runner, data, frames, *, batch_size=1):
    outputs = {name: [] for name in OUTPUT_WIDTHS}
    for sample in data:
        load_sample(runner, sample)
        result = runner.run()
        torch.cuda.synchronize()
        for (name, shape), tensor in zip(
            output_shapes(frames, batch_size).items(), result, strict=True
        ):
            value = tensor.cpu().numpy().copy()
            if value.shape != shape:
                raise ValueError(
                    f"Unexpected GVHMR output shape for {name}: {value.shape}"
                )
            outputs[name].append(value)
    return {name: np.concatenate(items, axis=0) for name, items in outputs.items()}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    if args.mode == "measure" and args.output is None:
        raise ValueError("Measurement requires an output directory")
    if args.condition in TRT_PRECISIONS:
        if args.conversion is None:
            raise ValueError("TensorRT requires a verified conversion directory")
        conversion = read_conversion(args.conversion, args.condition)
    else:
        if args.conversion is not None:
            raise ValueError("PyTorch does not use a TensorRT plan")
        conversion = None
    dynamic_batch = (
        conversion is not None and "profile_batches" in conversion["dynamic_profile"]
    )
    if conversion is not None and args.batch_size != 1 and not dynamic_batch:
        raise ValueError("A batched GVHMR measurement requires a dynamic batch engine")
    segments, checkpoint = checked_segments(args.inputs, args.checkpoints)
    profile = json.loads((HERE / "shape-profile.json").read_text())
    frames = args.length
    selected = [
        (row, data)
        for row, data in segments
        if row["length"] == frames and (row["split"] == "eval" or args.mode == "probe")
    ]
    if args.mode != "probe" and not selected:
        raise ValueError(f"No evaluation segments at L={frames}")
    if args.mode != "probe" and len(selected) % args.batch_size:
        raise ValueError("The fixed GVHMR length bucket must fill every batch")
    memory = DeviceMemory() if args.mode != "profile" else None
    try:
        if memory is not None and memory.sample("idle") < 6 * 1024**3:
            raise RuntimeError("Less than 6 GiB free before GVHMR inference")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise RuntimeError(
                "The pinned GVHMR comparison requires one CC 12.0 CUDA GPU"
            )
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        runner = (
            PytorchRunner(
                args.condition, checkpoint, frames, batch_size=args.batch_size
            )
            if conversion is None
            else TensorRTSequenceRunner(
                args.conversion / PLAN,
                frames,
                batch_size=args.batch_size,
                input_profile=conversion["dynamic_profile"]["inputs"]
                if dynamic_batch
                else None,
            )
        )
        if not selected:
            raise ValueError(f"GVHMR has no locked segment at L={frames}")
        rows = [row for row, _ in selected]
        batches = batch_slices(len(rows), args.batch_size)
        if args.mode == "probe":
            batches = batches[:1]
        data = [
            host_sample(
                runner, pack_sequences([arrays for _, arrays in selected[group]])
            )
            for group in batches
        ]
        if args.mode == "probe":
            result = collect_outputs(runner, data, frames, batch_size=args.batch_size)
            if (
                result["length_out"].tolist() != [frames] * args.batch_size
                or not result["mask_out"].all()
                or any(
                    not np.isfinite(result[name]).all()
                    for name in ("pred_x", "pred_cam", "static_conf_logits")
                )
            ):
                raise ValueError(f"Invalid probe output at L={frames}")
            print(f"Engine executed L={frames}; all {frames} frames valid", flush=True)
            print(f"GVHMR executed B={args.batch_size}, L={frames}", flush=True)
            return
        trt_settings = {}
        if dynamic_batch and args.mode == "measure":
            expected = json.loads(
                (args.conversion / "expected-precision.json").read_text()
            )
            trt_settings["trt_compute_precision"] = verify_compute_precision(
                inspect_engine(runner.engine, runner.context),
                "fp16",
                expected_operations=expected,
            )
        torch.cuda.synchronize()
        if memory is not None:
            memory.sample("after_setup")
        warmup(runner, data, load=load_sample)
        profile_id = profile["representatives"][str(frames)]
        index = [row["id"] for row in rows].index(profile_id) // args.batch_size
        if args.mode == "profile":
            profile_one(runner, data[index], load=load_sample)
            return
        timings = measure_latency(
            runner,
            data,
            memory,
            load=load_sample,
            summarize=partial(seq_stats, frames=frames, batch_size=args.batch_size),
        )
        predictions = collect_outputs(runner, data, frames, batch_size=args.batch_size)
        memory.sample("after_accuracy")
        args.output.mkdir(parents=True, exist_ok=True)
        prediction_file = args.output / "raw-outputs.npz"
        np.savez(prediction_file, **predictions)
        write_json(
            args.output / "measurement.json",
            {
                "condition": args.condition,
                "length": frames,
                "batch_size": args.batch_size,
                "segments": [row["id"] for row in rows],
                "profile_representative": profile_id,
                "profile_batch_segments": [row["id"] for row in rows[batches[index]]],
                "tensor_lock_sha256": sha256_file(HERE / "tensor-lock.json"),
                "engine_sha256": conversion["engine_sha256"] if conversion else None,
                "raw_outputs_sha256": sha256_file(prediction_file),
                "input_dtypes": {
                    name: str(tensor.dtype) for name, tensor in runner.buffers.items()
                },
                "output_dtypes": {
                    name: str(tensor.dtype) for name, tensor in predictions.items()
                },
                "trt_nvtx_verbosity": runner.context.nvtx_verbosity.name
                if conversion
                else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "tensorrt": version("tensorrt-cu13") if conversion else None,
                "device": torch.cuda.get_device_name(),
                "device_uuid": memory.uuid,
                "driver": memory.driver,
                "cc": torch.cuda.get_device_capability(),
                "queue_run_id": os.environ["TENNIS_RUN_ID"],
                **trt_settings,
                **timings,
                **memory.result(),
            },
        )
    finally:
        if memory is not None:
            memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument(
        "--mode", choices=("measure", "profile", "probe"), default="measure"
    )
    args = parser.parse_args()
    if args.batch_size < 1 and not (args.mode == "probe" and args.condition == "2"):
        parser.error("--batch-size must be positive")
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
