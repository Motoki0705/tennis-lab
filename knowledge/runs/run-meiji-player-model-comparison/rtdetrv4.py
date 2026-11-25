"""Run the official RT-DETRv4 COCO detector on the fixed Meiji benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import JsonValue
from torch import nn
from torchvision import transforms

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
    BenchmarkAdapter,
    BenchmarkCase,
    Conversion,
    ParityStatus,
    Stage,
    benchmark,
    measure_conversion,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    PLAN_FILE,
    EngineBundle,
    InputProfile,
    TensorSpec,
)
from src.utils.models.tensorrt_build import build_bundle, export_bundle
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.video.reader import RandomAccessVideoReader

UPSTREAM_COMMIT = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
IMAGE_SIDE = 640
NUM_QUERIES = 300
NUM_CLASSES = 80
PERSON_LABEL = 0  # COCO category 1, before upstream's deploy-time category remap.
PT32_NOT_MEASURED = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"
WEIGHT_SHA256 = {
    "rtdetrv4-l": "935c691dbd4f282a6a4ba59a57d77f13ad6963411dbb1b29615a72af2452f6e1",
    "rtdetrv4-x": "2925ec2e53d48e6141db8601c8656566219e696cd63bf8d6955190cae6bfd3f9",
}
ONNX_OPSET = 19
TRACE_BATCH = 2
INPUT_NAME = "pixels"
OUTPUT_NAMES = ("pred_logits", "pred_boxes")
GIB = 1024**3
_TRANSFORM = transforms.Compose(
    [transforms.Resize((IMAGE_SIDE, IMAGE_SIDE)), transforms.ToTensor()]
)

ImageBGR: TypeAlias = NDArray[np.uint8]
ModelOutputs: TypeAlias = tuple[torch.Tensor, torch.Tensor]
Value = TypeVar("Value")


class Stages(Protocol):
    def stage(self, stage: Stage, operation: Callable[[], Value]) -> Value: ...

    def measure_forward(
        self, backend: Backend, operation: Callable[[], Value]
    ) -> Value: ...


class _DirectStages:
    def stage(self, stage: Stage, operation: Callable[[], Value]) -> Value:
        return operation()

    def measure_forward(
        self, backend: Backend, operation: Callable[[], Value]
    ) -> Value:
        return operation()


_DIRECT = _DirectStages()


class _UpstreamConfig(Protocol):
    @property
    def model(self) -> nn.Module: ...


@dataclass(frozen=True)
class Assets:
    model_id: str
    upstream: Path
    weights: Path

    @property
    def weight_sha256(self) -> str:
        return WEIGHT_SHA256[self.model_id]

    @property
    def config(self) -> Path:
        size = self.model_id.rsplit("-", 1)[1]
        return self.upstream / f"configs/rtv4/rtv4_hgnetv2_{size}_coco.yml"

    def verify(self) -> None:
        if self.model_id not in WEIGHT_SHA256:
            raise ValueError(f"Unknown RT-DETRv4 variant: {self.model_id}")
        git = ["git", "-C", str(self.upstream)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True)
        changes = subprocess.check_output([*git, "status", "--porcelain"], text=True)
        if head.strip() != UPSTREAM_COMMIT or changes:
            raise ValueError(f"Expected clean RT-DETRv4 checkout at {UPSTREAM_COMMIT}")
        if dual_sha256(self.weights) != self.weight_sha256:
            raise ValueError(f"RT-DETRv4 weight SHA-256 mismatch: {self.weights}")

    def config_and_postprocessor(self) -> tuple[_UpstreamConfig, nn.Module]:
        sys.path.insert(0, str(self.upstream.resolve()))
        try:
            core = import_module("engine.core")
            dataset = import_module("engine.data.dataset")
        finally:
            sys.path.pop(0)
        if core.__file__ is None or not Path(core.__file__).resolve().is_relative_to(
            self.upstream.resolve()
        ):
            raise RuntimeError("A different upstream 'engine' package is loaded")
        config = core.YAMLConfig(str(self.config))
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
        postprocessor = config.postprocessor.deploy()
        if (
            config.yaml_cfg["eval_spatial_size"] != [IMAGE_SIDE, IMAGE_SIDE]
            or config.yaml_cfg["DFINETransformer"]["num_queries"] != NUM_QUERIES
            or postprocessor.num_top_queries != NUM_QUERIES
            or postprocessor.num_classes != NUM_CLASSES
            or not postprocessor.use_focal_loss
            or dataset.mscoco_label2category[PERSON_LABEL] != 1
        ):
            raise ValueError(
                "Official input, class or query settings differ from the adapter"
            )
        return config, postprocessor


class _RawDetector(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> ModelOutputs:
        output = self.model(pixels)
        return output["pred_logits"], output["pred_boxes"]


def _load_pt_model(assets: Assets, config: _UpstreamConfig) -> _RawDetector:
    # The official L/X .pth files use torch.save's pre-ZIP serialization.
    checkpoint = torch.load(assets.weights, map_location="cpu", weights_only=True)
    weights = (
        checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    )
    model = config.model
    model.load_state_dict(weights, strict=True)
    wrapper = _RawDetector(model.deploy())
    wrapper.eval()
    return wrapper


def _register_encoder_positions(encoder: nn.Module) -> None:
    # Upstream caches the fixed spatial positions as plain CPU attributes.
    for index in encoder.use_encoder_idx:
        name = f"pos_embed{index}"
        position = getattr(encoder, name)
        delattr(encoder, name)
        encoder.register_buffer(name, position, persistent=False)


def preprocess(rows: Sequence[ImageBGR]) -> tuple[torch.Tensor, torch.Tensor]:
    """PIL bilinear stretch plus RGB [0,1], with native sizes ordered as W,H."""
    images, sizes = [], []
    for image in rows:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RT-DETRv4 expects a BGR uint8 image with 3 channels")
        rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
        images.append(_TRANSFORM(rgb))
        sizes.append(rgb.size)
    return torch.stack(images), torch.tensor(sizes, dtype=torch.int64)


def person_predictions(
    labels: NDArray[np.integer],
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    sizes: NDArray[np.integer],
) -> list[Predictions]:
    """Filter official global top-k rows and enforce the native xyxy contract."""
    if (
        scores.ndim != 2
        or labels.shape != scores.shape
        or boxes.shape != (*scores.shape, 4)
        or sizes.shape != (len(scores), 2)
    ):
        raise ValueError("RT-DETRv4 output rows or native image sizes are misaligned")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite RT-DETRv4 deployment output")

    result = []
    for cls, xyxy, confidence, (width, height) in zip(
        labels, boxes, scores, sizes, strict=True
    ):
        people = cls == PERSON_LABEL
        clipped = np.clip(xyxy[people], 0, (width, height, width, height)).astype(
            np.float32
        )
        positive = (clipped[:, 2:] > clipped[:, :2]).all(axis=1)
        result.append(
            Predictions(
                clipped[positive], confidence[people][positive].astype(np.float32)
            )
        )
    return result


@dataclass
class Detector:
    backend: Backend
    device: torch.device
    model: Callable[[torch.Tensor], ModelOutputs]
    postprocessor: nn.Module

    def forward(self, pixels: torch.Tensor) -> ModelOutputs:
        with (
            torch.no_grad(),
            torch.autocast(
                self.device.type,
                dtype=torch.float16,
                enabled=self.backend is Backend.PT16,
            ),
        ):
            return self.model(pixels)

    def decode(
        self, output: ModelOutputs, sizes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, boxes = output
        labels, xyxy, scores = self.postprocessor(
            {"pred_logits": logits.float(), "pred_boxes": boxes.float()}, sizes
        )
        return labels, xyxy, scores

    def __call__(
        self, rows: Sequence[ImageBGR], stages: Stages = _DIRECT
    ) -> list[Predictions]:
        pixels_cpu, sizes_cpu = stages.stage(Stage.PREPROCESS, lambda: preprocess(rows))
        pixels, sizes = stages.stage(
            Stage.H2D, lambda: (pixels_cpu.to(self.device), sizes_cpu.to(self.device))
        )

        output = stages.measure_forward(self.backend, lambda: self.forward(pixels))
        decoded = stages.stage(Stage.POSTPROCESS, lambda: self.decode(output, sizes))
        labels, boxes_cpu, scores = stages.stage(
            Stage.D2H, lambda: tuple(tensor.cpu().numpy() for tensor in decoded)
        )
        return stages.stage(
            Stage.POSTPROCESS,
            lambda: person_predictions(labels, boxes_cpu, scores, sizes_cpu.numpy()),
        )


def _export_settings() -> dict[str, JsonValue]:
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "mode": "official_deploy_raw",
        "opset": ONNX_OPSET,
        "trace_batch": TRACE_BATCH,
        "input_hw": [IMAGE_SIDE, IMAGE_SIDE],
        "io_dtype": "float32",
    }


def load_detector(
    assets: Assets, backend: Backend, device: torch.device, engine_dir: Path | None
) -> Detector:
    if device.type != "cuda" and backend is not Backend.PT32:
        raise ValueError(f"{backend} requires CUDA")
    model: Callable[[torch.Tensor], ModelOutputs]
    if backend is Backend.TRT16:
        if engine_dir is None:
            raise ValueError("TRT16 requires --engine-dir")
        _, postprocessor = assets.config_and_postprocessor()
        bundle = EngineBundle.read(engine_dir).for_model(
            model=assets.model_id,
            settings=_export_settings(),
            sources={"checkpoint": assets.weight_sha256},
            inputs={
                INPUT_NAME: TensorSpec(
                    dtype="float32", shape=(-1, 3, IMAGE_SIDE, IMAGE_SIDE)
                )
            },
            outputs={
                "pred_logits": TensorSpec(
                    dtype="float32", shape=(-1, NUM_QUERIES, NUM_CLASSES)
                ),
                "pred_boxes": TensorSpec(dtype="float32", shape=(-1, NUM_QUERIES, 4)),
            },
        )
        runner = TensorRTRunner(bundle, device)

        def trt(pixels: torch.Tensor) -> ModelOutputs:
            output = runner({INPUT_NAME: pixels})
            return output["pred_logits"], output["pred_boxes"]

        model = trt
    else:
        config, postprocessor = assets.config_and_postprocessor()
        eager = _load_pt_model(assets, config)
        _register_encoder_positions(eager.model.encoder)
        model = eager.to(device)
    return Detector(backend, device, model, postprocessor.to(device))


def benchmark_adapter(
    assets: Assets, backend: Backend, device: torch.device, engine_dir: Path | None
) -> BenchmarkAdapter[Sequence[ImageBGR], list[Predictions]]:
    engines = (
        {"model": engine_dir / PLAN_FILE}
        if backend is Backend.TRT16 and engine_dir is not None
        else {}
    )
    return BenchmarkAdapter(
        backend,
        lambda: load_detector(assets, backend, device, engine_dir),
        {"model": assets.weights},
        engines,
    )


def _export(assets: Assets, directory: Path) -> None:
    onnx = import_module("onnx")
    convert = import_module("onnxruntime.transformers.float16").convert_float_to_float16
    sort_graph = import_module(
        "onnxruntime.transformers.onnx_model"
    ).OnnxModel.graph_topological_sort

    def write_onnx(path: Path) -> None:
        config, _ = assets.config_and_postprocessor()
        wrapper = _load_pt_model(assets, config)
        with TemporaryDirectory(dir=path.parent) as temporary:
            fp32 = Path(temporary) / "fp32.onnx"
            fastpath = torch.backends.mha.get_fastpath_enabled()
            try:
                # The fused eager CPU attention is unavailable to the legacy exporter.
                torch.backends.mha.set_fastpath_enabled(False)
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper,
                        # Trace the upstream anchor repeat with a symbolic batch size.
                        (torch.zeros(TRACE_BATCH, 3, IMAGE_SIDE, IMAGE_SIDE),),
                        str(fp32),
                        input_names=[INPUT_NAME],
                        output_names=list(OUTPUT_NAMES),
                        dynamic_axes={
                            name: {0: "batch"} for name in (INPUT_NAME, *OUTPUT_NAMES)
                        },
                        dynamo=False,
                        opset_version=ONNX_OPSET,
                    )
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath)
            del wrapper, config
            typed = convert(onnx.load(fp32), keep_io_types=True)
            sort_graph(typed.graph)
            onnx.save_model(
                typed, path, save_as_external_data=True, location=f"{path.name}.data"
            )

    export_bundle(
        directory,
        model=assets.model_id,
        settings=_export_settings(),
        sources={"checkpoint": assets.weights},
        export=write_onnx,
    )


def _build(args: argparse.Namespace, device: torch.device) -> None:
    if not 1 <= args.opt_batch <= TRACE_BATCH or args.max_batch != TRACE_BATCH:
        raise ValueError(f"Expected a profile covering batches 1 to {TRACE_BATCH}")
    chw = (3, IMAGE_SIDE, IMAGE_SIDE)
    build_bundle(
        args.engine_dir,
        {
            INPUT_NAME: InputProfile(
                minimum=(1, *chw),
                optimum=(args.opt_batch, *chw),
                maximum=(args.max_batch, *chw),
            )
        },
        device=device,
        workspace_bytes=args.workspace_gib * GIB,
    )


def _accuracy(args: argparse.Namespace, assets: Assets, device: torch.device) -> None:
    from prediction_rows import PredictionHeader, write_predictions

    path = args.output / "predictions.jsonl"
    receipt = args.output / "accuracy.json"
    if path.exists() or receipt.exists():
        raise FileExistsError(f"Accuracy output already exists at {args.output}")
    manifest = load_manifest(args.benchmark, expected_sha256=args.manifest_sha256)
    pinned_file(
        args.source,
        {
            "path": "manifest.json",
            "sha256": manifest["source_benchmark"]["manifest_sha256"],
        },
    )
    references = load_reference_frames(
        args.benchmark, expected_sha256=args.manifest_sha256
    )
    header = PredictionHeader(
        manifest["benchmark_id"],
        args.manifest_sha256,
        assets.model_id,
        args.backend.value.lower(),
        False,
    )
    detector = load_detector(assets, args.backend, device, args.engine_dir)
    predictions = {}
    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        size = (clip["height"], clip["width"], 3)
        for video in clip["source_videos"]:
            media = pinned_file(args.source, video)
            camera = video["camera_id"]
            print(f"{args.model_id} {args.backend} {clip_id} {camera}", flush=True)
            with RandomAccessVideoReader(media) as reader:
                for frame in clip["evaluation_frames"]:
                    image = reader.read(frame)
                    if image.shape != size:
                        raise ValueError(
                            f"Unexpected native frame size: {clip_id} {camera} {frame}"
                        )
                    (row,) = detector([image])
                    predictions[FrameKey(clip_id, camera, frame)] = row

    write_predictions(path, header, predictions, expected_frames=references)
    predictions_sha256 = dual_sha256(path)
    write_json_atomic(
        receipt,
        {
            "condition": header.as_json(),
            "weight_sha256": assets.weight_sha256,
            "upstream_commit": UPSTREAM_COMMIT,
            "runtime": {
                "job_id": os.environ["TENNIS_RUN_ID"],
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "batch_size": 1,
                "tf32": False,
            },
            "engine_sha256": (
                dual_sha256(args.engine_dir / PLAN_FILE)
                if args.backend is Backend.TRT16
                else None
            ),
            "predictions_sha256": predictions_sha256,
            "parity_status": ParityStatus.NOT_EVALUATED,
            "parity_not_evaluated_reason": args.parity_not_evaluated_reason,
            "score": score_players(references, predictions),
        },
    )


def _speed(args: argparse.Namespace, assets: Assets, device: torch.device) -> None:
    path = args.output / f"speed-b{args.batch_size}.json"
    if path.exists():
        raise FileExistsError(f"Speed output already exists at {path}")
    # Use #2496's loader and batching, including the published input order and hashes.
    baseline = import_module("run")
    fixed = baseline.InputSet(args.inputs)
    if fixed.sha256 != args.inputs_sha256:
        raise ValueError(f"Speed manifest SHA-256 mismatch: {args.inputs}")
    schedule = fixed.pins.schedule("frames", args.batch_size)
    batches = baseline.inputs(fixed, "dino", schedule)
    conversion = tuple(baseline.load_conversion(path) for path in args.conversion)
    engine_profiles = {}
    if args.backend is Backend.TRT16 and args.engine_dir is not None:
        manifest = args.engine_dir / "engine.json"
        if manifest.is_file():
            engine_profiles = json.loads(manifest.read_bytes())["profiles"]
    case = BenchmarkCase(
        assets.model_id,
        args.backend,
        args.batch_size,
        "image",
        fixed.sha256,
        {
            "dataset_id": fixed.manifest["id"],
            "input_decode": "predecoded BGR",
            "input_hw": [IMAGE_SIDE, IMAGE_SIDE],
            "cuda_stream": "non-default",
            "timing_protocol_sha256": fixed.pins.timing_sha256,
            "timed_input_ids_sha256": schedule.timed_ids_sha256,
            "host_threads": torch.get_num_threads(),
            "num_queries": NUM_QUERIES,
            "postprocessor": "official PyTorch FP32 sigmoid/global top-k",
            "pytorch_autocast_fp16": args.backend is Backend.PT16,
            "engine_profiles": engine_profiles,
            "upstream_commit": UPSTREAM_COMMIT,
        },
    )
    adapter = benchmark_adapter(assets, args.backend, device, args.engine_dir)

    report = benchmark(
        case,
        batches,
        {args.backend: adapter},
        device=device,
        dedicated_stream=True,
        protocol=schedule.protocol,
        conversion=conversion,
        parity_not_evaluated_reason=args.parity_not_evaluated_reason,
    )
    write_json_atomic(path, report.to_dict())
    print(
        f"{assets.model_id}/{args.backend}/B{args.batch_size} {report.status}",
        flush=True,
    )
    if report.failure is not None:
        raise SystemExit(report.failure.message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=WEIGHT_SHA256)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    export = modes.add_parser("export")
    build = modes.add_parser("build")
    for conversion in (export, build):
        conversion.add_argument("--engine-dir", type=Path, required=True)
        conversion.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--opt-batch", type=int, required=True)
    build.add_argument("--max-batch", type=int, required=True)
    build.add_argument("--workspace-gib", type=int, required=True)
    accuracy = modes.add_parser("accuracy")
    accuracy.add_argument("--benchmark", type=Path, required=True)
    accuracy.add_argument("--manifest-sha256", required=True)
    accuracy.add_argument("--source", type=Path, required=True)
    accuracy.add_argument("--output", type=Path, required=True)
    speed = modes.add_parser("speed")
    speed.add_argument("--inputs", type=Path, required=True)
    speed.add_argument("--inputs-sha256", required=True)
    speed.add_argument("--batch-size", type=int, required=True)
    speed.add_argument("--output", type=Path, required=True)
    speed.add_argument("--conversion", type=Path, action="append", default=[])
    for measurement in (accuracy, speed):
        measurement.add_argument("--engine-dir", type=Path)
        measurement.add_argument(
            "--backend",
            type=lambda value: Backend(value.upper()),
            choices=(Backend.PT16, Backend.TRT16),
            required=True,
        )
        measurement.add_argument(
            "--parity-not-evaluated-reason",
            choices=(PT32_NOT_MEASURED,),
            required=True,
        )
    for cuda in (build, accuracy, speed):
        cuda.add_argument("--device", required=True, type=torch.device)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assets = Assets(args.model_id, args.upstream, args.weights)
    assets.verify()
    if args.mode == "export":
        result = measure_conversion(
            Conversion.ONNX_EXPORT, lambda: _export(assets, args.engine_dir)
        )
    else:
        if args.device.type != "cuda" or not os.environ.get("TENNIS_RUN_ID"):
            raise ValueError("Run this CUDA command in the shared training queue")
        if args.mode == "speed":
            _speed(args, assets, args.device)
            return
        if args.mode == "accuracy":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            _accuracy(args, assets, args.device)
            return
        result = measure_conversion(
            Conversion.ENGINE_BUILD,
            lambda: _build(args, args.device),
            device=args.device,
        )
    write_json_atomic(args.receipt, asdict(result))
    if result.failure is not None:
        raise SystemExit(result.failure.message)


if __name__ == "__main__":
    main()
