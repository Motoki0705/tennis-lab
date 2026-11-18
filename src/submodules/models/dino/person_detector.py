"""IDEA-Research DINO frame-level person detector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.configuration import InferencePrecision, require_absolute_path
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models._base.precision import autocast_dtype
from src.submodules.models.dino.architecture import (
    COCO_PERSON_CLASS_ID,
    build_dino,
    dino_4scale_swin_args,
    load_dino_state_dict,
    preprocess_frame,
    use_fp32_deformable_attention,
)


@dataclass(frozen=True)
class PersonDetectionRequest:
    """One BGR uint8 frame to detect people in."""

    frame_bgr: NDArray[np.uint8]


@dataclass(frozen=True)
class PersonDetectionResult:
    """Person detections as xyxy pixel boxes and confidence scores."""

    boxes_xyxy: NDArray[np.float32]
    scores: NDArray[np.float32]


class DinoPersonDetector(
    BaseInferenceModel[PersonDetectionRequest, PersonDetectionResult]
):
    """Run a four-scale Swin-L DINO checkpoint and decode class id 1.

    Class id 1 is COCO person for the released checkpoint and tennis player for
    checkpoints exported by ``src.tasks.player_detection``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        repository: str | Path,
        *,
        device: str | torch.device,
        confidence: float,
        short_side: int,
        max_long_side: int,
        precision: InferencePrecision = "fp32",
    ) -> None:
        super().__init__(device)
        if type(confidence) is not float:
            raise TypeError("confidence must be a float.")
        if type(short_side) is not int or type(max_long_side) is not int:
            raise TypeError("short_side and max_long_side must be integers.")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if short_side <= 0 or max_long_side < short_side:
            raise ValueError(
                "Expected 0 < short_side <= max_long_side, got "
                f"{short_side} and {max_long_side}"
            )
        self.checkpoint = require_absolute_path(checkpoint, name="DINO checkpoint")
        self.repository = require_absolute_path(repository, name="DINO repository")
        self.confidence = confidence
        self.short_side = short_side
        self.max_long_side = max_long_side
        if precision != "fp32" and self.device.type != "cuda":
            raise ValueError("Mixed precision DINO requires a CUDA device")
        self._autocast_dtype = autocast_dtype(precision, self.device)
        self._model: torch.nn.Module | None = None

    def _load_impl(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError(
                f"DINO multi-scale deformable attention requires CUDA, got {self.device}"
            )
        state_dict = load_dino_state_dict(self.checkpoint)
        model, _ = build_dino(
            self.repository, dino_4scale_swin_args(self.device, use_checkpoint=False)
        )
        model.load_state_dict(state_dict, strict=True)
        if self._autocast_dtype is not None:
            use_fp32_deformable_attention(model)
        self._model = model.to(self.device).eval()

    def _unload_impl(self) -> None:
        self._model = None

    def predict_batch(
        self,
        requests: Sequence[PersonDetectionRequest],
        *,
        batch_size: int = 1,
    ) -> list[PersonDetectionResult]:
        """Detect each image, preserving input order and original pixel sizes.

        Only consecutive images with identical preprocessed ``(3,H,W)`` shapes
        share a forward pass, with at most ``batch_size`` images. A shape change
        flushes the pending batch; images are never padded or reordered. Set
        ``batch_size=1`` to explicitly run one image per forward pass. Empty
        requests return an empty list without loading the model.
        """
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not requests:
            return []
        self.load()
        with torch.no_grad():
            return self._predict_batch_impl(requests, batch_size=batch_size)

    def _predict_impl(self, request: PersonDetectionRequest) -> PersonDetectionResult:
        return self._predict_batch_impl([request], batch_size=1)[0]

    def _predict_batch_impl(
        self, requests: Sequence[PersonDetectionRequest], *, batch_size: int
    ) -> list[PersonDetectionResult]:
        results: list[PersonDetectionResult] = []
        images: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int]] = []
        for request in requests:
            frame = request.frame_bgr
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"frame_bgr must have shape (H, W, 3), got {frame.shape}"
                )
            if frame.dtype != np.uint8:
                raise TypeError(f"frame_bgr must have dtype uint8, got {frame.dtype}")
            image = preprocess_frame(
                frame, short_side=self.short_side, max_long_side=self.max_long_side
            )
            if images and image.shape != images[0].shape:
                results.extend(self._predict_compatible_batch(images, image_sizes))
                images.clear()
                image_sizes.clear()
            images.append(image)
            image_sizes.append((frame.shape[0], frame.shape[1]))
            if len(images) == batch_size:
                results.extend(self._predict_compatible_batch(images, image_sizes))
                images.clear()
                image_sizes.clear()
        if images:
            results.extend(self._predict_compatible_batch(images, image_sizes))
        return results

    def _predict_compatible_batch(
        self, images: Sequence[torch.Tensor], image_sizes: Sequence[tuple[int, int]]
    ) -> list[PersonDetectionResult]:
        if self._model is None:
            raise RuntimeError("DINO model did not load before prediction.")
        batch = torch.stack(list(images)).to(self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype, enabled=self._autocast_dtype is not None):
            output = self._model(batch)
        if self._autocast_dtype is not None:
            output = {key: output[key].float() for key in ("pred_logits", "pred_boxes")}
        return decode_person_detections(
            output, image_sizes=image_sizes, confidence=self.confidence
        )


def decode_person_detections(
    output: Mapping[str, torch.Tensor],
    *,
    image_sizes: Sequence[tuple[int, int]],
    confidence: float,
) -> list[PersonDetectionResult]:
    """Decode each batch row using its original ``(height, width)`` in pixels."""
    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    if (
        logits.ndim != 3
        or boxes.ndim != 3
        or logits.shape[:2] != boxes.shape[:2]
        or logits.shape[0] != len(image_sizes)
        or logits.shape[2] <= COCO_PERSON_CLASS_ID
        or boxes.shape[2] != 4
    ):
        raise ValueError(
            "Expected DINO logits (B,Q,C), boxes (B,Q,4), and B image_sizes, got "
            f"logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}, "
            f"image_sizes={image_sizes}"
        )
    results: list[PersonDetectionResult] = []
    for index, (image_height, image_width) in enumerate(image_sizes):
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"Image size must be positive, got {image_sizes[index]}")
        scores = logits[index, :, COCO_PERSON_CLASS_ID].sigmoid()
        keep = scores >= confidence
        scores = scores[keep]
        image_boxes = boxes[index, keep]
        order = scores.argsort(descending=True)
        scores = scores[order]
        image_boxes = image_boxes[order]
        center_x, center_y, box_width, box_height = image_boxes.unbind(-1)
        xyxy = torch.stack(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ],
            dim=-1,
        ).clamp_(0.0, 1.0)
        scale = xyxy.new_tensor([image_width, image_height, image_width, image_height])
        xyxy = xyxy * scale
        results.append(
            PersonDetectionResult(
                boxes_xyxy=xyxy.detach().cpu().numpy().astype(np.float32, copy=False),
                scores=scores.detach().cpu().numpy().astype(np.float32, copy=False),
            )
        )
    return results
