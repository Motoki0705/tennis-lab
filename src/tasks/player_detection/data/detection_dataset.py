"""Frame-level DINO detection samples drawn from the player frame store."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.submodules.models.dino.architecture import (
    COCO_PERSON_CLASS_ID,
    preprocess_frame,
)
from src.tasks.player_detection.configuration import (
    AugmentationConfig,
    FrameSelectionConfig,
    InputSizeConfig,
)
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    PlayerFrameStore,
    Split,
)
from src.utils.seeding import make_sample_rng

# Fine-tuned checkpoints reuse the COCO person logit for "tennis player" so the
# exported weights are a drop-in replacement for DinoPersonDetector.
PLAYER_CLASS_ID = COCO_PERSON_CLASS_ID


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Selected rows plus explicit accounting of everything excluded."""

    frames: NDArray[np.int64]
    boxes_xyxy: tuple[NDArray[np.float32], ...]
    stats: dict[str, int]


def visible_boxes(
    boxes_xyxy: NDArray[np.float32], *, width: int, height: int, min_side_px: float
) -> tuple[NDArray[np.float32], int]:
    """Clip amodal boxes to the image; drop those thinner than ``min_side_px``.

    Returns the kept boxes and the number dropped as (almost) out of frame.
    """
    clipped = boxes_xyxy.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0.0, float(width))
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0.0, float(height))
    sides = np.minimum(clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1])
    keep = sides >= min_side_px
    return clipped[keep], int((~keep).sum())


def select_detection_frames(
    store: PlayerFrameStore,
    split: Split,
    selection: FrameSelectionConfig,
    *,
    frame_stride: int,
) -> FrameSelection:
    """Choose detection frames of one split and their image-clipped GT boxes."""
    rows = store.split_frames(split)
    stats = {
        "split_frames": int(rows.size),
        "dropped_unreviewed": 0,
        "dropped_unresolved_player": 0,
        "dropped_no_visible_box": 0,
        "dropped_by_stride": 0,
        "dropped_small_instances": 0,
        "instances": 0,
    }
    eligible: list[int] = []
    eligible_boxes: list[NDArray[np.float32]] = []
    unresolved_code = BBOX_SOURCE_CODES["unresolved"]
    for row in rows.tolist():
        if selection.require_reviewed and not bool(store.frames["reviewed"][row]):
            stats["dropped_unreviewed"] += 1
            continue
        instances = store.instances_of(row)
        unresolved = instances.bbox_source == unresolved_code
        if selection.require_located_players and unresolved.any():
            stats["dropped_unresolved_player"] += 1
            continue
        clip = store.clip_of(row)
        boxes, dropped = visible_boxes(
            instances.boxes_xyxy[~unresolved],
            width=clip.width,
            height=clip.height,
            min_side_px=selection.min_visible_box_px,
        )
        if boxes.shape[0] == 0:
            stats["dropped_no_visible_box"] += 1
            continue
        eligible.append(row)
        eligible_boxes.append(boxes)
        stats["dropped_small_instances"] += dropped
    kept = list(range(0, len(eligible), frame_stride))
    stats["dropped_by_stride"] = len(eligible) - len(kept)
    frames = np.asarray([eligible[i] for i in kept], dtype=np.int64)
    boxes_tuple = tuple(eligible_boxes[i] for i in kept)
    stats["frames"] = int(frames.size)
    stats["instances"] = int(sum(box.shape[0] for box in boxes_tuple))
    if frames.size == 0:
        raise ValueError(f"No {split} detection frames remain after selection: {stats}")
    return FrameSelection(frames, boxes_tuple, stats)


@dataclass(frozen=True, slots=True)
class DetectionSample:
    image: torch.Tensor  # (3,h,w) ImageNet-normalized RGB
    boxes_cxcywh: torch.Tensor  # (N,4) normalized by the resized image size
    labels: torch.Tensor  # (N,) int64
    boxes_xyxy_px: torch.Tensor  # (N,4) original-image pixels (unaugmented)
    original_size: tuple[int, int]  # (H,W) of the stored frame
    frame: int


def jitter_color(
    bgr: NDArray[np.uint8], rng: random.Random, config: AugmentationConfig
) -> NDArray[np.uint8]:
    """Brightness, contrast and saturation factors in ``[1-s, 1+s]``."""
    image = bgr.astype(np.float32)
    image *= rng.uniform(1.0 - config.brightness, 1.0 + config.brightness)
    gray = cv2.cvtColor(image.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast = rng.uniform(1.0 - config.contrast, 1.0 + config.contrast)
    image = (image - gray.mean()) * contrast + gray.mean()
    saturation = rng.uniform(1.0 - config.saturation, 1.0 + config.saturation)
    gray = cv2.cvtColor(image.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = (image - gray[..., None]) * saturation + gray[..., None]
    return cast(NDArray[np.uint8], image.clip(0, 255).astype(np.uint8))


class PlayerDetectionDataset(Dataset[DetectionSample]):
    """Selected store frames; ``augmentation=None`` is the deployed eval path."""

    def __init__(
        self,
        store: PlayerFrameStore,
        selection: FrameSelection,
        *,
        input_size: InputSizeConfig,
        augmentation: AugmentationConfig | None,
    ) -> None:
        self.store = store
        self.selection = selection
        self.input_size = input_size
        self.augmentation = augmentation
        # Unique per-sample identities used by test-prediction artifacts.
        self.prediction_ids = [store.frame_key(int(row)) for row in selection.frames]

    def __len__(self) -> int:
        return int(self.selection.frames.size)

    def __getitem__(self, index: int) -> DetectionSample:
        row = int(self.selection.frames[index])
        bgr = self.store.read_bgr(row)
        height, width = bgr.shape[:2]
        boxes = self.selection.boxes_xyxy[index]
        model_boxes = boxes.copy()
        short_side = self.input_size.short_side
        augmentation = self.augmentation
        if augmentation is not None:
            rng = make_sample_rng(index)
            if rng.random() < augmentation.hflip_prob:
                bgr = np.ascontiguousarray(bgr[:, ::-1])
                model_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            bgr = jitter_color(bgr, rng, augmentation)
            short_side = rng.choice(augmentation.short_side_choices)
        image = preprocess_frame(bgr, short_side=short_side, max_long_side=self.input_size.max_long_side)
        scale = np.asarray([width, height, width, height], dtype=np.float32)
        normalized = model_boxes / scale
        cxcywh = np.stack(
            [
                (normalized[:, 0] + normalized[:, 2]) / 2,
                (normalized[:, 1] + normalized[:, 3]) / 2,
                normalized[:, 2] - normalized[:, 0],
                normalized[:, 3] - normalized[:, 1],
            ],
            axis=1,
        )
        return DetectionSample(
            image=image,
            boxes_cxcywh=torch.from_numpy(cxcywh.astype(np.float32)),
            labels=torch.full((boxes.shape[0],), PLAYER_CLASS_ID, dtype=torch.int64),
            boxes_xyxy_px=torch.from_numpy(boxes.astype(np.float32)),
            original_size=(height, width),
            frame=row,
        )


@dataclass
class DetectionBatch:
    """Mutable on purpose: Lightning's transfer rebuilds it on the device
    (frozen dataclasses are kept as CPU metadata by the shared transfer)."""

    images: list[torch.Tensor]
    boxes_cxcywh: list[torch.Tensor]
    labels: list[torch.Tensor]
    boxes_xyxy_px: list[torch.Tensor]
    original_sizes: list[tuple[int, int]]
    frames: list[int]

    def __len__(self) -> int:
        return len(self.images)


def collate_detection(samples: list[DetectionSample]) -> DetectionBatch:
    """Keep per-image sizes; DINO pads them into a masked NestedTensor."""
    return DetectionBatch(
        images=[sample.image for sample in samples],
        boxes_cxcywh=[sample.boxes_cxcywh for sample in samples],
        labels=[sample.labels for sample in samples],
        boxes_xyxy_px=[sample.boxes_xyxy_px for sample in samples],
        original_sizes=[sample.original_size for sample in samples],
        frames=[sample.frame for sample in samples],
    )
