"""Run the pinned RF-DETR-L detector on the Meiji player evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, TypeVar, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import JsonValue
from torchvision.transforms import functional as tvf

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    ConditionUnavailable,
    Conversion,
    ConversionResult,
    Failure,
    ParityStatus,
    Stage,
    Status,
    benchmark,
    measure_conversion,
)
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs, TimingSchedule
from src.utils.io import load_json, write_json_atomic, write_jsonl
from src.utils.models.tensorrt_artifacts import (
    PLAN_FILE,
    EngineBundle,
    InputProfile,
    TensorSpec,
)
from src.utils.models.tensorrt_build import build_bundle, export_bundle
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.video.reader import RandomAccessVideoReader

CORE_VERSION = "1.12.1"
CORE_REVISION = "f29e2a8bbb7cdcb8920bf74a335062f88a35046d"
MODEL_ID = "rfdetr-l"
RESOLUTION = 704
LARGE_SHA256 = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
PERSON_ID = 1
NUM_QUERIES = 300
COCO_LOGIT_SLOTS = 91
SCORE_MIN = 0.25
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCHEMA = "player_model_predictions_v1"
WORKSPACE_BYTES = 2 * 1024**3
SPEED_BASELINE = "knowledge.runs.run-person-model-speed-baseline"
HARNESS_MERGE_COMMIT = "59faa32a28d2f54e6492abdba2429846fa0397e3"
PARITY_NOT_EVALUATED_REASON = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"
Image: TypeAlias = NDArray[np.uint8]
Tensors: TypeAlias = dict[str, torch.Tensor]
_Value = TypeVar("_Value")


@dataclass(frozen=True)
class Model:
    weights: Path
    sha256: ClassVar[str] = LARGE_SHA256

    @classmethod
    def open(cls, weights: Path, sha256: str | None = None) -> Model:
        if sha256 is not None and sha256 != cls.sha256:
            raise ValueError("RF-DETR-L requires the official 2026 COCO checkpoint")
        if dual_sha256(weights) != cls.sha256:
            raise ValueError(
                f"Missing or mismatched pinned checkpoint SHA-256: {weights}"
            )
        return cls(weights.resolve())

    def open_upstream(self, device: torch.device) -> Any:
        if metadata.version("rfdetr") != CORE_VERSION:
            raise ValueError(f"Expected rfdetr {CORE_VERSION}")
        model = import_module("rfdetr").RFDETRLarge(
            pretrain_weights=str(self.weights), device=str(device)
        )
        labels = import_module("rfdetr.utilities.class_names").prediction_labels(model)
        if (
            labels.class_id_to_name.get(PERSON_ID) != "person"
            or model.model.model.class_embed.out_features != COCO_LOGIT_SLOTS
            or model.model.resolution != RESOLUTION
            or model.model.postprocess.num_select != NUM_QUERIES
        ):
            raise ValueError(
                f"Unexpected COCO head or input configuration for {MODEL_ID}"
            )
        return model

    def settings(self, backend: Backend) -> dict[str, JsonValue]:
        return {
            "rfdetr_version": CORE_VERSION,
            "rfdetr_commit": CORE_REVISION,
            "license": "Apache-2.0",
            "resolution": [RESOLUTION, RESOLUTION],
            "resize": "RGB, float32 bilinear, antialias=False, stretch to square",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "queries": NUM_QUERIES,
            "global_top_k_query_class_pairs": NUM_QUERIES,
            "coco_person_label": PERSON_ID,
            "min_person_score_inclusive": SCORE_MIN,
            "nms": False,
            "tf32": False,
            "forward": {
                Backend.PT16: "PyTorch eager, CUDA FP16 autocast",
                Backend.TRT16: "Strongly typed TensorRT, upstream FP16 ONNX cast, FP32 bindings",
            }[backend],
        }

    def export_settings(self, batch_size: int) -> dict[str, JsonValue]:
        return {
            **self.settings(Backend.TRT16),
            "opset": 17,
            "static_batch_size": batch_size,
        }

    def input_shape(self, batch_size: int) -> tuple[int, ...]:
        return batch_size, 3, RESOLUTION, RESOLUTION


def prepare_images(
    images: Sequence[Image], resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    sizes_hw = []
    for image in images:
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
        ):
            raise ValueError("RF-DETR requires nonempty BGR uint8 images")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = tvf.to_tensor(rgb)
        resized = tvf.resize(chw, [resolution, resolution], antialias=False)
        tensors.append(tvf.normalize(resized, IMAGENET_MEAN, IMAGENET_STD))
        sizes_hw.append(image.shape[:2])
    return torch.stack(tensors), torch.tensor(sizes_hw, dtype=torch.float32)


def person_predictions(rows: NDArray[np.float32]) -> Predictions:
    """Select COCO person rows after upstream top-k and native pixel projection."""
    boxes, scores, labels = rows[:, :4], rows[:, 4], rows[:, 5]
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite RF-DETR detection rows")
    valid_box = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    keep = (labels == PERSON_ID) & (scores >= SCORE_MIN) & valid_box
    return Predictions(boxes[keep], scores[keep])


class Detector:
    def __init__(
        self,
        model: Model,
        backend: Backend,
        device: torch.device,
        engine: EngineBundle | None,
    ) -> None:
        self.backend = backend
        self.device = device
        postprocess = import_module("rfdetr.models.postprocess").PostProcess
        self.postprocess = postprocess(num_select=NUM_QUERIES)
        if backend is Backend.TRT16:
            if engine is None:
                raise ValueError("TRT16 needs an engine bundle")
            if metadata.version("rfdetr") != CORE_VERSION:
                raise ValueError(f"Expected rfdetr {CORE_VERSION}")
            self.runner = TensorRTRunner(engine, device)
        elif backend is Backend.PT16:
            self.network: torch.nn.Module = (
                model.open_upstream(device).model.model.float().eval().to(device)
            )
        else:
            raise ValueError("RF-DETR measurements require PT16 or TRT16")

    def transfer(
        self, data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, sizes_hw = data
        return images.to(self.device), sizes_hw.to(self.device)

    def forward(self, images: torch.Tensor) -> Tensors:
        if self.backend is Backend.TRT16:
            result = self.runner({"input": images})
            return {"pred_boxes": result["dets"], "pred_logits": result["labels"]}
        with torch.autocast(self.device.type, dtype=torch.float16):
            return cast(Tensors, self.network(images))

    def decode(
        self, outputs: Tensors, sizes_hw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boxes, logits = outputs["pred_boxes"].float(), outputs["pred_logits"].float()
        count = len(sizes_hw)
        if boxes.shape != (count, NUM_QUERIES, 4) or logits.shape != (
            count,
            NUM_QUERIES,
            COCO_LOGIT_SLOTS,
        ):
            raise ValueError(
                f"Unexpected RF-DETR raw shapes: {boxes.shape}, {logits.shape}"
            )
        finite = torch.isfinite(boxes).all() & torch.isfinite(logits).all()
        decoded = self.postprocess(
            {"pred_boxes": boxes, "pred_logits": logits}, sizes_hw
        )
        rows = torch.stack(
            [
                torch.cat(
                    [row["boxes"], row["scores"][:, None], row["labels"][:, None]],
                    dim=1,
                )
                for row in decoded
            ]
        )
        return rows, finite

    @staticmethod
    def finish(decoded: tuple[torch.Tensor, torch.Tensor]) -> list[Predictions]:
        rows, finite = decoded
        if not finite.item():
            raise ConditionUnavailable(
                Status.UNAVAILABLE, "Nonfinite RF-DETR raw model output"
            )
        return [person_predictions(row) for row in rows.cpu().numpy()]

    def predict(self, images: Sequence[Image]) -> list[Predictions]:
        with torch.no_grad():
            inputs, sizes_hw = self.transfer(prepare_images(images, RESOLUTION))
            return self.finish(self.decode(self.forward(inputs), sizes_hw))

    def timed(self, images: Sequence[Image], stages: BatchStages) -> list[Predictions]:
        prepared = stages.stage(
            Stage.PREPROCESS,
            lambda: prepare_images(images, RESOLUTION),
        )
        inputs, sizes_hw = stages.stage(Stage.H2D, lambda: self.transfer(prepared))
        outputs = stages.measure_forward(self.backend, lambda: self.forward(inputs))
        decoded = stages.stage(
            Stage.POSTPROCESS, lambda: self.decode(outputs, sizes_hw)
        )
        predictions: list[Predictions] = stages.stage(
            Stage.D2H, lambda: self.finish(decoded)
        )
        return predictions


def load_engine(model: Model, path: Path, batch_size: int) -> EngineBundle:
    bundle = EngineBundle.read(path)
    settings = model.export_settings(batch_size)
    # The already measured L plan carries two empty fields in its export manifest.
    previous_settings = settings | {
        "rfdetr_plus_version": None,
        "rfdetr_plus_commit": None,
    }
    if bundle.export.settings == previous_settings:
        settings = previous_settings
    return bundle.for_model(
        model=MODEL_ID,
        settings=settings,
        sources={"model": model.sha256},
        inputs={
            "input": TensorSpec(dtype="float32", shape=model.input_shape(batch_size))
        },
        outputs={
            "dets": TensorSpec(dtype="float32", shape=(batch_size, NUM_QUERIES, 4)),
            "labels": TensorSpec(
                dtype="float32", shape=(batch_size, NUM_QUERIES, COCO_LOGIT_SLOTS)
            ),
        },
    )


def benchmark_adapter(
    model: Model,
    backend: Backend,
    device: torch.device,
    batch_size: int,
    engine_path: Path,
) -> BenchmarkAdapter[Sequence[Image], list[Predictions]]:
    if backend is Backend.TRT16:
        quiet_nvtx = import_module(f"{SPEED_BASELINE}.speed_models").quiet_nvtx

    def load() -> Callable[[Sequence[Image], BatchStages], list[Predictions]]:
        def create() -> Detector:
            if backend is Backend.TRT16:
                engine = load_engine(model, engine_path, batch_size)
                detector = Detector(model, backend, device, engine)
                quiet_nvtx(detector.runner)
                return detector
            return Detector(model, backend, device, None)

        detector = benchmark_call(create)

        def infer(images: Sequence[Image], stages: BatchStages) -> list[Predictions]:
            return benchmark_call(lambda: detector.timed(images, stages))

        return infer

    return BenchmarkAdapter(
        backend,
        load,
        {"model": model.weights},
        {"plan": engine_path / PLAN_FILE} if backend is Backend.TRT16 else {},
    )


def benchmark_call(operation: Callable[[], _Value]) -> _Value:
    try:
        return operation()
    except (ConditionUnavailable, torch.OutOfMemoryError):
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ConditionUnavailable(
            Status.UNAVAILABLE, f"{type(error).__name__}: {error}"
        ) from error


def speed_batches(
    input_dir: Path, batch_size: int
) -> tuple[PinnedBenchmarkInputs, TimingSchedule, list[Batch[Sequence[Image]]]]:
    fixed = import_module(f"{SPEED_BASELINE}.speed_inputs").InputSet(input_dir)
    pins: PinnedBenchmarkInputs = fixed.pins
    schedule = pins.schedule("frames", batch_size)
    batches: list[Batch[Sequence[Image]]] = [
        Batch(ids, tuple(fixed.frame(key) for key in ids), processed_items=len(ids))
        for ids in schedule.batch_ids
    ]
    return pins, schedule, batches


def speed_conversion(directory: Path) -> tuple[ConversionResult, ...]:
    results = []
    for phase in Conversion:
        path = directory / f"{phase.value}.json"
        if not path.is_file():
            continue
        row = load_json(path)
        if row["phase"] != phase.value:
            raise ValueError(f"Wrong conversion phase in {path}")
        failure = row["failure"]
        results.append(
            ConversionResult(
                phase,
                row["wall_ms"],
                None
                if failure is None
                else Failure(
                    Status(failure["status"]), failure["phase"], failure["message"]
                ),
            )
        )
    return tuple(results)


def measure_speed(
    model: Model,
    backend: Backend,
    device: torch.device,
    destination: Path,
    engine: Path,
    input_dir: Path,
    batch_size: int,
) -> Status:
    # Match the host thread settings used by the fixed-input DINO benchmark.
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pins, schedule, batches = speed_batches(input_dir, batch_size)
    conversions = speed_conversion(engine.parent) if backend is Backend.TRT16 else ()
    case = BenchmarkCase(
        MODEL_ID,
        backend,
        batch_size,
        "image",
        pins.manifest_sha256,
        {
            **model.settings(backend),
            "dataset_id": pins.manifest["id"],
            "timing_protocol_sha256": pins.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "harness_merge_commit": HARNESS_MERGE_COMMIT,
            "input_decode": "predecoded BGR",
            "host_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "cuda_stream": "non-default",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    adapter = benchmark_adapter(model, backend, device, batch_size, engine)
    report = benchmark(
        case,
        batches,
        {backend: adapter},
        device=device,
        dedicated_stream=True,
        protocol=schedule.protocol,
        conversion=conversions,
        parity_not_evaluated_reason=PARITY_NOT_EVALUATED_REASON,
    )
    path = destination / "report.json"
    write_json_atomic(path, report.to_dict())
    print(f"{MODEL_ID} {backend} B={batch_size} {report.status} {path}", flush=True)
    if report.failure is not None:
        print(f"{report.failure.phase}: {report.failure.message}", file=sys.stderr)
    return report.status


def export_onnx(model: Model, target: Path, batch_size: int) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        stage = Path(work)
        upstream = model.open_upstream(torch.device("cpu"))
        fp32 = upstream.export(
            output_dir=str(stage),
            batch_size=batch_size,
            shape=model.input_shape(batch_size)[2:],
            opset_version=17,
            verbose=False,
        )
        cast_graph = import_module("rfdetr.export._tensorrt.exporter").fp16_source_graph

        def write_graph(path: Path) -> None:
            with cast_graph(str(fp32), dynamic_batch=False) as graph:
                Path(graph).rename(path)

        export_bundle(
            stage / "bundle",
            model=MODEL_ID,
            settings=model.export_settings(batch_size),
            sources={"model": model.weights},
            export=write_graph,
        )
        (stage / "bundle").rename(target)


def header(benchmark_id: str, manifest_sha: str, backend: Backend) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "manifest_sha256": manifest_sha,
        "model_id": MODEL_ID,
        "backend": backend.value.lower(),
        "has_pose": False,
    }


def write_predictions(
    path: Path, identity: dict[str, Any], rows: Mapping[FrameKey, Predictions]
) -> None:
    with tempfile.TemporaryDirectory(dir=path.parent) as work:
        destination = Path(work) / path.name
        write_jsonl(
            destination,
            [
                identity,
                *(
                    {
                        **key._asdict(),
                        "boxes_xyxy": value.boxes_xyxy.tolist(),
                        "scores": value.scores.tolist(),
                        "keypoints_xyc": None,
                    }
                    for key, value in rows.items()
                ),
            ],
        )
        destination.replace(path)


def evaluate(
    model: Model,
    backend: Backend,
    device: torch.device,
    output: Path,
    engine_path: Path,
    evaluation_bundle: Path,
    manifest_sha: str,
    source: Path,
) -> None:
    cohort = load_manifest(evaluation_bundle, expected_sha256=manifest_sha)
    references = load_reference_frames(evaluation_bundle, expected_sha256=manifest_sha)
    source_sha = cohort["source_benchmark"]["manifest_sha256"]
    pinned_file(source, {"path": "manifest.json", "sha256": source_sha})
    benchmark_id = cohort["benchmark_id"]
    engine = load_engine(model, engine_path, 1) if backend is Backend.TRT16 else None
    detector = Detector(model, backend, device, engine)
    # The evaluation CLI is a separate process. Match the harness's precision controls.
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    predictions = {}
    for clip in cohort["clips"]:
        if [row["camera_id"] for row in clip["source_videos"]] != clip["camera_ids"]:
            raise ValueError(f"Unexpected camera ordering: {clip['clip_id']}")
        for media in clip["source_videos"]:
            with RandomAccessVideoReader(pinned_file(source, media)) as reader:
                for frame in clip["evaluation_frames"]:
                    key = FrameKey(clip["clip_id"], media["camera_id"], frame)
                    image = reader.read(frame)
                    if image.shape != (clip["height"], clip["width"], 3):
                        raise ValueError(f"Unexpected native frame dimensions: {key}")
                    predictions[key] = detector.predict((image,))[0]
            print(f"Predicted {clip['clip_id']} {media['camera_id']}", flush=True)
    if predictions.keys() != references.keys():
        raise ValueError("Predictions lack the exact set of evaluation frames")
    identity = header(benchmark_id, manifest_sha, backend)
    raw = output / "predictions.jsonl"
    write_predictions(raw, identity, predictions)
    write_json_atomic(
        output / "accuracy.json",
        {
            **identity,
            "weights_sha256": model.sha256,
            "engine_sha256": None if engine is None else engine.engine.plan_sha256,
            "source_manifest_sha256": source_sha,
            "predictions_sha256": dual_sha256(raw),
            "settings": model.settings(backend),
            "batch_size": 1,
            "environment": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "torchvision": metadata.version("torchvision"),
                "opencv": cv2.__version__,
                "tensorrt": None
                if engine is None
                else engine.engine.environment.tensorrt_version,
            },
            "parity_vs_pt32": None,
            "parity_status": ParityStatus.NOT_EVALUATED,
            "parity_not_evaluated_reason": PARITY_NOT_EVALUATED_REASON,
            "pseudo_gt": score_players(references, predictions),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "build"):
        commands.add_parser(command).add_argument("--batch-size", type=int, default=1)
    accuracy = commands.add_parser("evaluate")
    accuracy.add_argument("--evaluation-bundle", type=Path, required=True)
    accuracy.add_argument("--manifest-sha256", required=True)
    accuracy.add_argument("--source", type=Path, required=True)
    speed = commands.add_parser("speed")
    speed.add_argument("--batch-size", type=int, choices=(1, 2), required=True)
    speed.add_argument("--speed-inputs", type=Path, required=True)
    for measurement in (accuracy, speed):
        measurement.add_argument("--backend", choices=("pt16", "trt16"), required=True)
        measurement.add_argument("--parity", choices=("not_measured",), required=True)
        measurement.add_argument("--engine-dir", type=Path)
    args = parser.parse_args()
    output = args.output_root / args.model
    batch_size = getattr(args, "batch_size", 1)
    if batch_size < 1:
        parser.error("--batch-size must be positive")
    device = torch.device(args.device)
    if args.command != "export" and device.type != "cuda":
        parser.error("Measurements and engine builds require an explicit CUDA device")
    trt = output / "trt16" / f"batch-{batch_size}"
    engine_path = trt / "engine"
    measuring = args.command in ("evaluate", "speed")
    destination = trt
    if measuring:
        if args.engine_dir is not None:
            if args.backend != "trt16":
                parser.error("--engine-dir requires --backend trt16")
            engine_path = args.engine_dir
        if args.command == "speed":
            destination = output / args.backend / f"speed-b{batch_size}"
        else:
            destination = output / "accuracy" / args.backend
        destination.mkdir(parents=True)
    if repro := os.environ.get("TENNIS_REPRO_DIR"):
        (Path(repro) / "output_dir.txt").write_text(f"{destination.resolve()}\n")
    try:
        model = Model.open(args.weights, args.weights_sha256)
        if args.command == "evaluate":
            evaluate(
                model,
                Backend(args.backend.upper()),
                device,
                destination,
                engine_path,
                args.evaluation_bundle,
                args.manifest_sha256,
                args.source,
            )
            return 0
        if args.command == "speed":
            status = measure_speed(
                model,
                Backend(args.backend.upper()),
                device,
                destination,
                engine_path,
                args.speed_inputs,
                batch_size,
            )
            return 0 if status is Status.OK else 1
        if args.command == "export":
            receipt = measure_conversion(
                Conversion.ONNX_EXPORT,
                lambda: export_onnx(model, engine_path, batch_size),
            )
        else:
            shape = model.input_shape(batch_size)
            receipt = measure_conversion(
                Conversion.ENGINE_BUILD,
                lambda: build_bundle(
                    engine_path,
                    {
                        "input": InputProfile(
                            minimum=shape, optimum=shape, maximum=shape
                        )
                    },
                    device=device,
                    workspace_bytes=WORKSPACE_BYTES,
                ),
                device=device,
            )
        write_json_atomic(trt / f"{receipt.phase}.json", receipt)
        if receipt.failure is not None:
            print(receipt.failure.message, file=sys.stderr)
            return 1
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        status = (
            Status.OOM
            if isinstance(error, torch.OutOfMemoryError)
            else Status.UNAVAILABLE
        )
        if isinstance(error, ConditionUnavailable):
            status = error.status
        write_json_atomic(
            destination / f"{args.command}-failure.json",
            {
                "status": Status.INFEASIBLE,
                "failure": Failure(
                    status, args.command, f"{type(error).__name__}: {error}"
                ),
            },
        )
        raise


if __name__ == "__main__":
    # A direct script invocation otherwise shadows the upstream rfdetr package.
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        del sys.path[0]
    sys.exit(main())
