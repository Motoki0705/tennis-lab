"""Measure or profile one requested batch in a fresh CUDA process."""

import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from batching import batch_slices, require_engine_profile
from benchmark import write_json
from convert import (
    TRT_PRECISIONS,
    inspect_engine,
    require_queue,
    verify_compute_precision,
)
from convert_dino import verify_layers
from measure_dino import (
    INPUT_NAME,
    TORCH_MEMORY_FRACTION,
    TorchDino,
    decode_batch,
    setup_cuda,
)
from runtime import (
    DeviceMemory,
    TensorRTContext,
    TorchRunner,
    load_batch,
    measure,
    measure_latency,
    profile_one,
    stats,
    torch_dispatch_precisions,
    warmup,
)
from vision_inputs import CONDITIONS, DINO, MODELS, PinnedInputs, load_conversion, read

RESERVE_BYTES = 3 * 1024**3


class ReservedMemory(DeviceMemory):
    def sample(self, stage):
        free = super().sample(stage)
        if stage != "idle" and free < RESERVE_BYTES:
            raise MemoryError(
                f"GPU memory reserve at {stage}: free={free}, reserve={RESERVE_BYTES}"
            )
        return free


class BatchPlan(TensorRTContext):
    def __init__(self, data, plan, conversion):
        import tensorrt as trt

        self.data = data
        self.conversion = conversion
        self.dino = data.model == DINO
        super().__init__(
            plan,
            omit_layer_nvtx=True,
            init_plugins=conversion.get("method") == "plugin",
            reserve_bytes=RESERVE_BYTES,
        )
        self.require_bindings(
            {INPUT_NAME: trt.TensorIOMode.INPUT}
            | dict.fromkeys(data.output_shapes, trt.TensorIOMode.OUTPUT)
        )
        require_engine_profile(self.engine, conversion["input_profile"])
        if tuple(self.engine.get_tensor_shape(INPUT_NAME)) != (-1, *data.input_shape):
            raise ValueError(
                "The TensorRT plan does not have the expected dynamic input axis"
            )
        self.dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        required_io = trt.float32 if self.dino else trt.float16
        if conversion["condition"] == "2" and any(
            self.engine.get_tensor_dtype(name) != required_io
            for name in (INPUT_NAME, *data.output_shapes)
        ):
            raise ValueError(
                "FP16 plan bindings differ from the declared model I/O types"
            )

    def batch_shape(self, batch_size):
        return (batch_size * self.data.multiplier, *self.data.input_shape)

    def set_batch_size(self, batch_size):
        shape = self.batch_shape(batch_size)
        self.set_input_shape(INPUT_NAME, shape)
        if unresolved := self.context.infer_shapes():
            raise ValueError(f"Unresolved TensorRT shapes: {unresolved}")
        self.pixels = self.bind(INPUT_NAME, shape, self.dtypes)
        self.outputs = {
            name: self.bind(name, (shape[0], *suffix), self.dtypes)
            for name, suffix in self.data.output_shapes.items()
        }

    def run(self):
        self.execute()
        return self.outputs if self.dino else self.outputs["features"]

    def precision(self, output, manifest):
        details = inspect_engine(self.engine, self.context)
        write_json(output, details)
        mode = TRT_PRECISIONS[self.conversion["condition"]]
        if self.dino:
            return verify_layers(
                details,
                mode,
                self.conversion["precision_requirements"],
                set(self.conversion["sampling_nodes"]),
                self.conversion["method"],
            )
        return verify_compute_precision(details, mode, **read(manifest))


def dino_precision(runner, condition):
    dtype = torch.float32 if condition == "0" else torch.float16
    matrices = {
        "backbone.0.patch_embed.proj": dtype,
        "backbone.0.layers.0.blocks.0.attn.qkv": dtype,
    }
    sampling = {
        name: torch.float32
        for name, module in runner.model.named_modules()
        if type(module).__name__ == "MSDeformAttn"
    }
    if len(sampling) != 12:
        raise ValueError(f"Expected 12 DINO MSDA blocks, got {len(sampling)}")
    observed = {}

    def collect(name, _module, _args, result):
        observed[name] = result.dtype

    expected = matrices | sampling
    hooks = [
        runner.model.get_submodule(name).register_forward_hook(partial(collect, name))
        for name in expected
    ]
    try:
        runner.run()
        torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    if observed != expected:
        raise RuntimeError(f"DINO dtype probe expected {expected}, got {observed}")
    return {name: str(dtype) for name, dtype in observed.items()}


def measure_dino_batches(runner, batches, data, memory, output, batch_size):
    if any(len(batch) != batch_size for batch in batches):
        raise ValueError("The pinned DINO bucket must divide into complete batches")
    latency = measure_latency(
        runner,
        batches,
        memory,
        load=load_batch,
        summarize=partial(stats, batch=batch_size),
    )
    detections = []
    for batch, span in zip(
        batches, batch_slices(len(data.rows), batch_size), strict=True
    ):
        load_batch(runner, batch)
        detections.extend(decode_batch(runner.run(), data.rows[span]))
    memory.sample("after_accuracy")
    write_json(output / data.prediction_name, detections)
    return {**latency, **memory.result(), "batch_shape": list(runner.pixels.shape)}


def run(args):
    with torch.no_grad() if args.model == DINO else torch.inference_mode():
        run_cuda(args)


def run_cuda(args):
    data = PinnedInputs(args.model, args.inputs)
    dino = data.model == DINO
    batch = args.batch_size
    memory = ReservedMemory()
    try:
        setup_cuda(memory)
        engine, settings, check_tail = None, {}, None
        if args.conversion:
            plan, conversion, _ = load_conversion(data, args.condition, args.conversion)
            if dino:
                torch.cuda.set_stream(torch.cuda.Stream())
            engine = BatchPlan(data, plan, conversion)
            if args.mode == "reject":
                engine.set_input_shape(INPUT_NAME, engine.batch_shape(batch))
                raise RuntimeError(f"TensorRT accepted out-of-range batch B={batch}")
            engine.set_batch_size(batch)
            runner = engine if dino else data.adapter.wrap_runner(engine)
            settings = {
                "backend": "TensorRT",
                "engine_batch_shape": engine.batch_shape(batch),
                "trt_nvtx_verbosity": engine.context.nvtx_verbosity.name,
                "trt_device_memory_size_bytes": engine.engine.device_memory_size_v2,
                "output_dtypes": {
                    name: str(value.dtype) for name, value in engine.outputs.items()
                },
            }
        elif dino:
            runner = TorchDino(
                args.condition, data.checkpoint, (batch, *data.input_shape)
            )
            settings = {
                "backend": "PyTorch eager",
                "autocast_dtype": str(runner.detector._autocast_dtype),
            }
        else:
            runner = data.adapter.wrap_runner(
                TorchRunner(
                    args.condition, data.checkpoint, data.adapter, batch_size=batch
                )
            )
            settings = data.adapter.measurement_settings(args.condition)
            settings["engine_batch_shape"] = (
                batch * data.multiplier,
                *data.input_shape,
            )
            settings["backend"] = "PyTorch eager"
        torch.cuda.synchronize()
        memory.sample("after_setup")

        batches = data.batches(batch)
        full = [items for items in batches if len(items) == batch]
        representative = full[data.profile_index // batch]
        if args.mode == "profile":
            warmup(runner, full, load=load_batch)
            memory.sample("after_profile_warmup")
            profile_one(runner, representative, load=load_batch)
            return
        args.output.mkdir(exist_ok=True)
        if engine is not None:
            manifest = args.conversion / "precision.json"
            settings["trt_compute_precision"] = engine.precision(
                args.output / "layers.json", manifest
            )
            check_tail = partial(
                engine.precision, args.output / "tail.layers.json", manifest
            )
        if dino:
            warmup(runner, full, load=load_batch)
            result = measure_dino_batches(
                runner, batches, data, memory, args.output, batch
            )
        else:
            result, predictions = measure(
                runner,
                batches,
                memory,
                data.adapter.output_shape,
                check_precision=check_tail,
            )
            np.save(args.output / data.prediction_name, predictions, allow_pickle=False)
        if engine is None:
            runner.set_batch_size(batch)
            load_batch(runner, representative)
            settings["torch_compute_dtypes"] = (
                dino_precision(runner, args.condition)
                if dino
                else torch_dispatch_precisions(
                    runner.run,
                    torch.float32 if args.condition == "0" else torch.float16,
                )
            )
        write_json(
            args.output / "measurement.json",
            {
                **result,
                **settings,
                "queue_job": os.environ["TENNIS_RUN_ID"],
                "input_dtype": str(runner.pixels.dtype),
                "device": torch.cuda.get_device_name(),
                "device_uuid": memory.uuid,
                "driver": memory.driver,
                "cc": torch.cuda.get_device_capability(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "min_free_gpu_bytes": RESERVE_BYTES,
                "torch_allocator_fraction": TORCH_MEMORY_FRACTION,
            },
        )
    finally:
        memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument(
        "--mode", choices=("measure", "profile", "reject"), required=True
    )
    args = parser.parse_args()
    if args.condition not in CONDITIONS[args.model]:
        parser.error("Condition unavailable for this model")
    if (args.conversion is not None) != args.condition.startswith(("2", "3")):
        parser.error("TensorRT conditions require an explicit conversion")
    if args.mode == "reject" and args.conversion is None:
        parser.error("Rejection probe requires TensorRT")
    require_queue()
    run(args)


if __name__ == "__main__":
    main()
