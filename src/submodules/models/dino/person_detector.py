"""IDEA-Research DINO frame-level person detector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.configuration import require_absolute_path
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models.dino.architecture import (
    COCO_PERSON_CLASS_ID,
    build_dino,
    dino_4scale_swin_args,
    load_dino_state_dict,
    preprocess_frame,
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
        self._model = model.to(self.device).eval()

    def _unload_impl(self) -> None:
        self._model = None

    def _predict_impl(self, request: PersonDetectionRequest) -> PersonDetectionResult:
        if request.frame_bgr.ndim != 3 or request.frame_bgr.shape[2] != 3:
            raise ValueError(
                f"frame_bgr must have shape (H, W, 3), got {request.frame_bgr.shape}"
            )
        if request.frame_bgr.dtype != np.uint8:
            raise TypeError(
                f"frame_bgr must have dtype uint8, got {request.frame_bgr.dtype}"
            )
        if self._model is None:
            raise RuntimeError("DINO model did not load before prediction.")
        height, width = request.frame_bgr.shape[:2]
        image = preprocess_frame(
            request.frame_bgr,
            short_side=self.short_side,
            max_long_side=self.max_long_side,
        ).to(self.device)
        output = self._model(image.unsqueeze(0))
        return decode_person_detections(
            output,
            image_width=width,
            image_height=height,
            confidence=self.confidence,
        )


def decode_person_detections(
    output: Mapping[str, torch.Tensor],
    *,
    image_width: int,
    image_height: int,
    confidence: float,
) -> PersonDetectionResult:
    """Decode COCO person logits and normalized cxcywh boxes."""
    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    if (
        logits.ndim != 3
        or boxes.ndim != 3
        or logits.shape[0] != 1
        or boxes.shape[0] != 1
    ):
        raise ValueError(
            "Expected batched DINO outputs with batch size 1, got "
            f"logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}"
        )
    scores = logits[0, :, COCO_PERSON_CLASS_ID].sigmoid()
    keep = scores >= confidence
    scores = scores[keep]
    boxes = boxes[0, keep]
    if scores.numel() == 0:
        return PersonDetectionResult(
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )
    order = scores.argsort(descending=True)
    scores = scores[order]
    boxes = boxes[order]
    center_x, center_y, box_width, box_height = boxes.unbind(-1)
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
    return PersonDetectionResult(
        boxes_xyxy=xyxy.detach().cpu().numpy().astype(np.float32, copy=False),
        scores=scores.detach().cpu().numpy().astype(np.float32, copy=False),
    )
