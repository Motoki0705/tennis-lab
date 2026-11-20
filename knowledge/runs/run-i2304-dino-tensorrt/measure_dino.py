"""Measure one fixed DINO shape using the shared timing and profiling routines."""

import argparse
import os
from pathlib import Path

import torch
from benchmark import write_json
from convert import CONDITIONS, TRT_PRECISIONS, require_queue
from fixed_inputs import (
    CHECKPOINT,
    CONFIDENCE,
    LOCK,
    MAX_LONG_SIDE,
    REPOSITORY,
    SHORT_SIDE,
    load_sample,
    rows_for,
)
from runtime import (
    DeviceMemory,
    TensorRTContext,
    measure_latency,
    profile_one,
    warmup,
)
from runtime import (
    load_sample as upload,
)

from src.submodules.models.dino.architecture import DINO_NUM_CLASSES, DINO_NUM_QUERIES
from src.submodules.models.dino.person_detector import (
    DinoPersonDetector,
    decode_person_detections,
)

INPUT_NAME = "pixels"
LOGITS = "pred_logits"
BOXES = "pred_boxes"
OUTPUT_SHAPES = {
    LOGITS: (1, DINO_NUM_QUERIES, DINO_NUM_CLASSES),
    BOXES: (1, DINO_NUM_QUERIES, 4),
}
MIN_FREE_GIB = 12


class EvaluationInputs:
    def __init__(self, directory, rows):
        self.directory = directory
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_sample(self.directory, self.rows[index])[0]


class TorchDino:
    def __init__(self, condition, checkpoint, shape):
        self.detector = DinoPersonDetector(
            checkpoint.resolve(),
            REPOSITORY.resolve(),
            device="cuda",
            confidence=CONFIDENCE,
            short_side=SHORT_SIDE,
            max_long_side=MAX_LONG_SIDE,
            precision="fp32" if condition == "0" else "fp16",
        )
        self.detector.load()
        if self.detector._model is None:
            raise RuntimeError("DINO detector model failed to load")
        self.model = self.detector._model
        self.pixels = torch.empty(shape, device="cuda", dtype=torch.float32)

    def run(self):
        dtype = self.detector._autocast_dtype
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            return self.model(self.pixels)


class DinoTRTContext(TensorRTContext):
    def __init__(self, plan, *, input_shapes, output_shapes, init_plugins=False):
        import tensorrt as trt

        super().__init__(plan, omit_layer_nvtx=True, init_plugins=init_plugins)
        self.require_bindings(
            dict.fromkeys(input_shapes, trt.TensorIOMode.INPUT)
            | dict.fromkeys(output_shapes, trt.TensorIOMode.OUTPUT)
        )
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.inputs = {
            name: self.bind(name, shape, dtypes) for name, shape in input_shapes.items()
        }
        self.outputs = {
            name: self.bind(name, shape, dtypes)
            for name, shape in output_shapes.items()
        }


class TensorRTDino(DinoTRTContext):
    def __init__(self, plan, shape):
        super().__init__(
            plan,
            input_shapes={INPUT_NAME: shape},
            output_shapes=OUTPUT_SHAPES,
            init_plugins=True,
        )
        self.pixels = self.inputs[INPUT_NAME]

    def run(self):
        self.execute()
        return self.outputs


def setup_cuda(memory):
    torch.set_num_threads(4)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    if memory.sample("idle") < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(f"Less than {MIN_FREE_GIB} GiB free before DINO inference")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("The fixed DINO run requires one CC 12.0 CUDA GPU")
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(0.60)
    torch.cuda.reset_peak_memory_stats()


def collect_decoded(runner, data, output):
    results = []
    for i, row in enumerate(data.rows):
        upload(runner, data[i])
        raw = runner.run()
        outputs = {key: raw[key].float() for key in OUTPUT_SHAPES}
        if not all(torch.isfinite(tensor).all() for tensor in outputs.values()):
            raise ValueError(f"Nonfinite DINO model outputs: {row['file']}")
        (detection,) = decode_person_detections(
            outputs, image_sizes=[tuple(row["original_hw"])], confidence=CONFIDENCE
        )
        results.append(
            {
                "file": row["file"],
                "boxes": detection.boxes_xyxy.tolist(),
                "scores": detection.scores.tolist(),
            }
        )
    write_json(output / "detections.json", results)


@torch.no_grad()
def measure(args):
    import json

    lock = json.loads(LOCK.read_text())
    rows = rows_for(lock, args.bucket, "eval")
    data = EvaluationInputs(args.inputs, rows)
    shape = tuple(rows[0]["shape"])
    memory = DeviceMemory()
    try:
        setup_cuda(memory)
        if args.condition in TRT_PRECISIONS:
            torch.cuda.set_stream(torch.cuda.Stream())
            runner = TensorRTDino(args.plan, shape)
            settings = {
                "backend": "TensorRT",
                "trt_nvtx_verbosity": runner.context.nvtx_verbosity.name,
                "output_dtypes": {
                    key: str(value.dtype) for key, value in runner.outputs.items()
                },
                "cuda_graph": False,
                "cuda_stream": "non-default",
            }
        else:
            runner = TorchDino(args.condition, args.checkpoint, shape)
            settings = {
                "backend": "PyTorch eager",
                "autocast_dtype": str(runner.detector._autocast_dtype),
                "deformable_attention_dtype": "torch.float32",
                "tf32": False,
            }
        torch.cuda.synchronize()
        memory.sample("after_setup")
        warmup(runner, data)
        if args.profile:
            rep = lock["buckets"][args.bucket]["representative"]
            index = next(i for i, row in enumerate(rows) if row["file"] == rep)
            profile_one(runner, data[index])
            return
        latency = measure_latency(runner, data, memory)
        collect_decoded(runner, data, args.output)
        memory.sample("after_accuracy")
        result = {
            **latency,
            **memory.result(),
            **settings,
            "batch_shape": shape,
            "input_dtype": str(runner.pixels.dtype),
            "driver": memory.driver,
            "device_uuid": memory.uuid,
            "device": torch.cuda.get_device_name(),
            "cc": torch.cuda.get_device_capability(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "queue_job": os.environ["TENNIS_RUN_ID"],
        }
        write_json(args.output / "measurement.json", result)
    finally:
        memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (args.condition in TRT_PRECISIONS) != (args.plan is not None):
        parser.error("Specify --plan only for TensorRT conditions")
    require_queue()
    measure(args)


if __name__ == "__main__":
    main()
