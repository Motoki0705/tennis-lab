"""Run the pinned RF-DETR-L detector on the Meiji player evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
    describe,
    score_players,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkProtocol,
    ConditionUnavailable,
    Conversion,
    ConversionResult,
    Failure,
    ParityResult,
    Stage,
    Status,
    benchmark,
    measure_conversion,
)
from src.utils.io import load_json, read_jsonl, write_json_atomic, write_jsonl
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
SPEED_MANIFEST_SHA256 = (
    "64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62"
)
SPEED_PROTOCOL = BenchmarkProtocol(batch_sizes=(1, 2))
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
                Backend.PT32: "PyTorch eager FP32",
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
        else:
            self.network: torch.nn.Module = (
                model.open_upstream(device).model.model.float().eval().to(device)
            )

    def transfer(
        self, data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, sizes_hw = data
        return images.to(self.device), sizes_hw.to(self.device)

    def forward(self, images: torch.Tensor) -> Tensors:
        if self.backend is Backend.TRT16:
            result = self.runner({"input": images})
            return {"pred_boxes": result["dets"], "pred_logits": result["labels"]}
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.backend is Backend.PT16
        ):
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
) -> tuple[str, list[Batch[Sequence[Image]]]]:
    fixed = import_module(f"{SPEED_BASELINE}.speed_inputs").InputSet(input_dir)
    if fixed.sha256 != SPEED_MANIFEST_SHA256:
        raise ValueError(f"Unexpected fixed speed manifest SHA-256: {fixed.sha256}")
    ids = tuple(row["id"] for row in fixed.rows("frames"))
    if (
        len(ids) != 96
        or batch_size not in SPEED_PROTOCOL.batch_sizes
        or len(ids) % batch_size
    ):
        raise ValueError(
            "RF-DETR-L requires the 96 fixed images in complete batches of 1 or 2"
        )
    batches: list[Batch[Sequence[Image]]] = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        batches.append(
            Batch(
                batch_ids,
                tuple(fixed.frame(key) for key in batch_ids),
                items=batch_size,
                processed_items=batch_size,
            )
        )
    return fixed.manifest["id"], batches


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


def read_speed_reference(
    directory: Path, expected_ids: Sequence[str], weights_sha256: str
) -> Mapping[str, Any]:
    report_path = directory / "speed-b1.json"
    predictions_path = directory / "speed-b1-predictions.json"
    if not report_path.is_file() or not predictions_path.is_file():
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE, f"Missing PT32 B=1 speed reference: {directory}"
        )
    report = load_json(report_path)
    expected_case = {
        "name": MODEL_ID,
        "backend": Backend.PT32.value,
        "batch_size": 1,
        "input_manifest_sha256": SPEED_MANIFEST_SHA256,
    }
    if (
        report["status"] != Status.OK
        or any(report["case"][key] != value for key, value in expected_case.items())
        or report["weights"]["model"]["sha256"] != weights_sha256
        or report["parity"]["metrics"]["predictions_sha256"]
        != dual_sha256(predictions_path)
    ):
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 B=1 speed reference identity or checksum differs: {directory}",
        )
    predictions = load_json(predictions_path)
    if (
        predictions["input_manifest_sha256"] != SPEED_MANIFEST_SHA256
        or predictions["input_ids"] != list(expected_ids)
        or predictions["rows"].keys() != set(expected_ids)
    ):
        raise ConditionUnavailable(
            Status.PARITY_UNAVAILABLE,
            f"PT32 B=1 speed reference has different input rows: {predictions_path}",
        )
    return cast(Mapping[str, Any], predictions["rows"])


def measure_speed(
    model: Model,
    backend: Backend,
    device: torch.device,
    output: Path,
    engine: Path,
    input_dir: Path,
    batch_size: int,
) -> None:
    dataset_id, batches = speed_batches(input_dir, batch_size)
    expected_ids = tuple(key for batch in batches for key in batch.ids)
    destination = output / backend.value.lower()
    prediction_path = destination / f"speed-b{batch_size}-predictions.json"
    reference = output / "pt32"
    parity = import_module(f"{SPEED_BASELINE}.parity")
    conversions = speed_conversion(engine.parent) if backend is Backend.TRT16 else ()
    case = BenchmarkCase(
        MODEL_ID,
        backend,
        batch_size,
        "image",
        SPEED_MANIFEST_SHA256,
        {
            **model.settings(backend),
            "dataset_id": dataset_id,
            "input_decode": "predecoded BGR",
            "host_threads": torch.get_num_threads(),
            "cuda_stream": "non-default",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )

    def check(
        predictions: Iterator[tuple[Batch[Sequence[Image]], list[Predictions]]],
    ) -> ParityResult:
        actual = {
            key: parity.serialize("dino", row)
            for batch, values in predictions
            for key, row in zip(batch.ids, values, strict=True)
        }
        if tuple(actual) != expected_ids:
            raise ValueError(
                "RF-DETR speed predictions differ from the fixed input order"
            )
        write_json_atomic(
            prediction_path,
            {
                "input_manifest_sha256": SPEED_MANIFEST_SHA256,
                "input_ids": expected_ids,
                "rows": actual,
            },
        )
        expected = (
            actual
            if backend is Backend.PT32 and batch_size == 1
            else read_speed_reference(reference, expected_ids, model.sha256)
        )
        comparison = parity.compare("dino", expected, actual)
        return ParityResult(
            comparison.passed,
            {**comparison.metrics, "predictions_sha256": dual_sha256(prediction_path)},
            comparison.reason,
        )

    adapter = benchmark_adapter(model, backend, device, batch_size, engine)
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        report = benchmark(
            case,
            batches,
            {backend: adapter},
            device=device,
            protocol=SPEED_PROTOCOL,
            conversion=conversions,
            check_parity=check,
        )
    path = destination / f"speed-b{batch_size}.json"
    write_json_atomic(path, report.to_dict())
    if repro := os.environ.get("TENNIS_REPRO_DIR"):
        (Path(repro) / "output_dir.txt").write_text(f"{destination.resolve()}\n")
    print(f"{MODEL_ID} {backend} B={batch_size} {report.status} {path}", flush=True)
    if report.failure is not None:
        print(f"{report.failure.phase}: {report.failure.message}", file=sys.stderr)


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


def prediction_parity(pairs: Iterable[tuple[Predictions, Predictions]]) -> ParityResult:
    images = expected = observed = matched = 0
    ious, coordinates, scores = [], [], []
    for reference, candidate in pairs:
        images += 1
        expected += len(reference.scores)
        observed += len(candidate.scores)
        overlaps = pairwise_iou(reference.boxes_xyxy, candidate.boxes_xyxy)
        base, trial = match_iou(overlaps, 0.5)
        matched += len(base)
        ious.extend(overlaps[base, trial])
        coordinates.extend(
            np.abs(reference.boxes_xyxy[base] - candidate.boxes_xyxy[trial]).ravel()
        )
        scores.extend(np.abs(reference.scores[base] - candidate.scores[trial]))
    recall = matched / expected if expected else float(observed == 0)
    precision = matched / observed if observed else float(expected == 0)
    iou_p05 = float(np.quantile(ious, 0.05)) if ious else None
    score_stats = describe(np.asarray(scores))
    score_p95 = score_stats["p95"]
    passed = (
        images > 0
        and min(recall, precision) >= 0.98
        and (iou_p05 is None or iou_p05 >= 0.90)
        and (score_p95 is None or score_p95 <= 0.05)
    )
    return ParityResult(
        passed,
        {
            "images": images,
            "pt32_boxes": expected,
            "candidate_boxes": observed,
            "matched": matched,
            "pt32_retained_fraction": recall,
            "candidate_matched_fraction": precision,
            "matched_iou_p05": iou_p05,
            "abs_score_difference": score_stats,
            "abs_xyxy_difference_px": describe(np.asarray(coordinates)),
        },
        None
        if passed
        else "Expected >=98% matches in both directions, matched IoU P05 >=0.90 and score delta P95 <=0.05",
    )


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


def read_pt32_predictions(
    directory: Path, identity: dict[str, Any], weights_sha256: str
) -> dict[FrameKey, Predictions]:
    accuracy = load_json(directory / "accuracy.json")
    if accuracy["weights_sha256"] != weights_sha256 or accuracy[
        "predictions_sha256"
    ] != dual_sha256(directory / "predictions.jsonl"):
        raise ValueError(
            "PT32 checkpoint or predictions do not match the accuracy receipt"
        )
    records = read_jsonl(directory / "predictions.jsonl")
    if not records or records[0] != identity:
        raise ValueError("PT32 prediction header does not match the pinned evaluation")
    result = {}
    for row in records[1:]:
        key = FrameKey(row["clip_id"], row["camera_id"], row["source_frame"])
        if key in result or row["keypoints_xyc"] is not None:
            raise ValueError(f"Duplicate frame or unexpected pose data: {key}")
        boxes = np.asarray(row["boxes_xyxy"], dtype=np.float32)
        scores = np.asarray(row["scores"], dtype=np.float32)
        result[key] = Predictions(
            boxes.reshape(0, 4) if boxes.shape == (0,) else boxes, scores
        )
    return result


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
    pt32 = None
    if backend is not Backend.PT32:
        pt32 = read_pt32_predictions(
            output.parent / "pt32",
            header(benchmark_id, manifest_sha, Backend.PT32),
            model.sha256,
        )
        if pt32.keys() != references.keys():
            raise ValueError("PT32 predictions lack the exact set of evaluation frames")
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
    output.mkdir(parents=True, exist_ok=True)
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
            "parity_vs_pt32": (
                None
                if pt32 is None
                else prediction_parity(
                    (pt32[key], predictions[key]) for key in references
                )
            ),
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
    accuracy.add_argument(
        "--backend",
        choices=[backend.value.lower() for backend in Backend],
        required=True,
    )
    accuracy.add_argument("--evaluation-bundle", type=Path, required=True)
    accuracy.add_argument("--manifest-sha256", required=True)
    accuracy.add_argument("--source", type=Path, required=True)
    speed = commands.add_parser("speed")
    speed.add_argument(
        "--backend",
        choices=[backend.value.lower() for backend in Backend],
        required=True,
    )
    speed.add_argument(
        "--batch-size", type=int, choices=SPEED_PROTOCOL.batch_sizes, required=True
    )
    speed.add_argument("--speed-inputs", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root / args.model
    trt = output / "trt16" / f"batch-{getattr(args, 'batch_size', 1)}"
    engine_path = trt / "engine"
    try:
        model = Model.open(args.weights, args.weights_sha256)
        device = torch.device(args.device)
        if args.command != "export" and device.type != "cuda":
            parser.error("Evaluation and engine builds require an explicit CUDA device")
        if args.command == "evaluate":
            evaluate(
                model,
                Backend(args.backend.upper()),
                device,
                output / args.backend,
                engine_path,
                args.evaluation_bundle,
                args.manifest_sha256,
                args.source,
            )
            return 0
        if args.command == "speed":
            measure_speed(
                model,
                Backend(args.backend.upper()),
                device,
                output,
                engine_path,
                args.speed_inputs,
                args.batch_size,
            )
            return 0
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        if args.command == "export":
            receipt = measure_conversion(
                Conversion.ONNX_EXPORT,
                lambda: export_onnx(model, engine_path, args.batch_size),
            )
        else:
            shape = model.input_shape(args.batch_size)
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
        if args.command == "speed":
            destination = output / args.backend
            failure_file = f"speed-b{args.batch_size}-failure.json"
        else:
            destination = output / args.backend if args.command == "evaluate" else trt
            failure_file = f"{args.command}-failure.json"
        write_json_atomic(
            destination / failure_file,
            {"status": status, "error": type(error).__name__, "reason": str(error)},
        )
        raise


if __name__ == "__main__":
    # A direct script invocation otherwise shadows the upstream rfdetr package.
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        del sys.path[0]
    sys.exit(main())
