"""Pinned RF-DETR-L COCO person detector with native-pixel xyxy output."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as tvf

from src.submodules.configuration import InferencePrecision, require_absolute_path
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models.dino.person_detector import (
    PersonDetectionRequest,
    PersonDetectionResult,
)
from src.submodules.people_model_assets import PEOPLE_MODEL_CHECKPOINTS
from src.utils.inference_recording import span
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle, TensorSpec
from src.utils.models.tensorrt_runtime import TensorRTRunner

MODEL_ID = "rfdetr-l"
PACKAGE_VERSION = "1.12.1"
SOURCE_COMMIT = "f29e2a8bbb7cdcb8920bf74a335062f88a35046d"
RESOLUTION = 704
NUM_QUERIES = 300
NUM_CLASSES = 91
PERSON_ID = 1
ONNX_OPSET = 17
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def require_rf_detr_version() -> None:
    try:
        actual = metadata.version("rfdetr")
    except metadata.PackageNotFoundError as error:
        raise ImportError(f"RF-DETR-L requires rfdetr=={PACKAGE_VERSION}. Install the rfdetr project extra.") from error
    if actual != PACKAGE_VERSION:
        raise ImportError(f"RF-DETR-L requires rfdetr=={PACKAGE_VERSION}, found {actual}.")


def load_rf_detr(checkpoint: Path, device: torch.device) -> Any:
    PEOPLE_MODEL_CHECKPOINTS[MODEL_ID].verify(checkpoint)
    require_rf_detr_version()
    try:
        upstream = import_module("rfdetr").RFDETRLarge(pretrain_weights=str(checkpoint), device=str(device))
        names = import_module("rfdetr.utilities.class_names").prediction_labels(upstream)
    except ImportError as error:
        raise ImportError(f"Could not load RF-DETR-L. Install the rfdetr project extra: {error}") from error
    if (names.class_id_to_name[PERSON_ID] != "person"
        or upstream.model.model.class_embed.out_features != NUM_CLASSES
        or upstream.model.resolution != RESOLUTION
        or upstream.model.postprocess.num_select != NUM_QUERIES):
        raise ValueError("RF-DETR-L checkpoint does not have the pinned sparse COCO head")
    return upstream


def export_settings(batch_size: int) -> dict[str, Any]:
    return {"rfdetr_version": PACKAGE_VERSION, "rfdetr_commit": SOURCE_COMMIT,
            "opset": ONNX_OPSET, "static_batch_size": batch_size, "fp16_cast_io": "float32"}


def bindings(batch_size: int) -> tuple[dict[str, TensorSpec], dict[str, TensorSpec]]:
    return (
        {"input": TensorSpec(dtype="float32", shape=(batch_size, 3, RESOLUTION, RESOLUTION))},
        {
            "dets": TensorSpec(dtype="float32", shape=(batch_size, NUM_QUERIES, 4)),
            "labels": TensorSpec(dtype="float32", shape=(batch_size, NUM_QUERIES, NUM_CLASSES)),
        },
    )


def prepare_images(requests: Sequence[PersonDetectionRequest]) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    sizes = []
    for request in requests:
        frame = request.frame_bgr
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("RF-DETR-L requires nonempty BGR uint8 frames")
        rgb = tvf.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized = tvf.resize(rgb, [RESOLUTION, RESOLUTION], antialias=False)
        images.append(tvf.normalize(resized, IMAGENET_MEAN, IMAGENET_STD))
        sizes.append(frame.shape[:2])
    return torch.stack(images), torch.tensor(sizes, dtype=torch.float32)


def decode_persons(boxes: torch.Tensor, logits: torch.Tensor, sizes_hw: torch.Tensor,
                   *, confidence: float) -> list[PersonDetectionResult]:
    """Apply the upstream stable global query/class selection and project xyxy."""
    count = len(sizes_hw)
    if boxes.shape != (count, NUM_QUERIES, 4) or logits.shape != (count, NUM_QUERIES, NUM_CLASSES):
        raise ValueError(f"Unexpected RF-DETR-L output shapes: {boxes.shape}, {logits.shape}")
    boxes, logits = boxes.float(), logits.float()
    finite = torch.isfinite(boxes).all() & torch.isfinite(logits).all()
    probabilities = logits.sigmoid().flatten(1)
    ranked = torch.argsort(probabilities, dim=1, descending=True, stable=True)[:, :NUM_QUERIES]
    scores = probabilities.gather(1, ranked)
    query = ranked // NUM_CLASSES
    cx, cy, w, h = boxes.unbind(-1)
    corners = torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)
    corners = corners.gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
    height, width = sizes_hw.unbind(1)
    scale = torch.stack((width, height, width, height), dim=1)[:, None, :]
    corners = (corners * scale).clamp_min(0).clamp(max=scale)
    rows = torch.cat((corners, scores[..., None], (ranked % NUM_CLASSES)[..., None]), dim=-1).cpu().numpy()
    if not finite.item():
        raise ValueError("Nonfinite RF-DETR-L output")
    results = []
    for row in rows:
        corners_xyxy, score, label = row[:, :4], row[:, 4], row[:, 5]
        keep = (label == PERSON_ID) & (score >= confidence) & (corners_xyxy[:, 2:] > corners_xyxy[:, :2]).all(axis=1)
        results.append(PersonDetectionResult(corners_xyxy[keep], score[keep]))
    return results


class RFDetrPersonDetector(BaseInferenceModel[PersonDetectionRequest, PersonDetectionResult]):
    """RF-DETR-L with PT32, CUDA FP16 autocast, or explicit fixed-batch TRT16."""

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device, confidence: float = 0.30,
                 precision: InferencePrecision = "fp32", engines: tuple[EngineBundle, ...] = ()) -> None:
        super().__init__(device)
        self.checkpoint = require_absolute_path(checkpoint, name="RF-DETR-L checkpoint")
        if not 0.0 < confidence < 1.0:
            raise ValueError("RF-DETR-L confidence must be in (0, 1)")
        if precision not in {"fp32", "fp16"} or (engines and precision != "fp16"):
            raise ValueError("RF-DETR-L supports PT32, PT16 and TRT16")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("RF-DETR-L FP16 requires CUDA")
        self.confidence = confidence
        self.precision = precision
        self.engines = engines
        self._runners: dict[int, TensorRTRunner] = {}
        self._network: torch.nn.Module | None = None

    def recording_assets(self) -> dict[str, Path]:
        return {"checkpoint": self.checkpoint, **{
            f"engine_{i}": bundle.root / PLAN_FILE for i, bundle in enumerate(self.engines)
        }}

    def _load_impl(self) -> None:
        if not self.engines:
            self._network = load_rf_detr(self.checkpoint, self.device).model.model.float().eval().to(self.device)
            return
        checkpoint = PEOPLE_MODEL_CHECKPOINTS[MODEL_ID]
        checkpoint.verify(self.checkpoint)
        for bundle in self.engines:
            batch = bundle.export.inputs["input"].shape[0]
            if batch < 1 or batch in self._runners:
                raise ValueError(f"RF-DETR-L requires distinct fixed batches, found {batch}")
            inputs, outputs = bindings(batch)
            bound = bundle.for_model(model=MODEL_ID, settings=export_settings(batch),
                                     sources={"checkpoint": checkpoint.sha256}, inputs=inputs, outputs=outputs)
            self._runners[batch] = TensorRTRunner(bound, self.device)

    def _unload_impl(self) -> None:
        self._network = None
        self._runners.clear()

    def predict_batch(self, requests: Sequence[PersonDetectionRequest], *,
                      batch_size: int = 1) -> list[PersonDetectionResult]:
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not requests:
            return []
        self.load()
        with torch.no_grad():
            return [
                result
                for offset in range(0, len(requests), batch_size)
                for result in self._predict_images(requests[offset:offset + batch_size])
            ]

    def _predict_impl(self, request: PersonDetectionRequest) -> PersonDetectionResult:
        return self._predict_images((request,))[0]

    def _predict_images(self, requests: Sequence[PersonDetectionRequest]) -> list[PersonDetectionResult]:
        with span("preprocess", model="RF-DETR-L", operation="resize_and_normalize"):
            images, sizes = prepare_images(requests)
        with span("transfer", model="RF-DETR-L", operation="input_to_device"):
            images, sizes = images.to(self.device), sizes.to(self.device)
        with span("inference", model="RF-DETR-L", batch_shape=list(images.shape)):
            boxes, logits = self._forward(images)
        with span("postprocess", model="RF-DETR-L", operation="decode_boxes"):
            return decode_persons(boxes, logits, sizes, confidence=self.confidence)

    def _forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.engines:
            if (runner := self._runners.get(len(images))) is None:
                raise ValueError(f"No RF-DETR-L TensorRT engine for batch {len(images)}")
            outputs = runner({"input": images})
            return outputs["dets"], outputs["labels"]
        if self._network is None:
            raise RuntimeError("RF-DETR-L model did not load before prediction")
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.precision == "fp16"):
            tensors = cast(dict[str, torch.Tensor], self._network(images))
        return tensors["pred_boxes"], tensors["pred_logits"]
