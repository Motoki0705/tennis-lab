"""One pose/appearance pass shared by camera-local methods and downstream consumers.

Decoded frames may come from raw video or the refiner's JPEG store. The caller
owns media provenance. No detector, tracker or identity selection runs here.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import Pose2DFrameSequenceRequest, Pose2DResult
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import crop


class PoseModel(Protocol):
    def predict(self, request: Pose2DFrameSequenceRequest) -> Pose2DResult: ...


class DetectionEncoder(Protocol):
    """KPR can consume ViTPose prompts; unprompted models explicitly ignore them.

    Prompts use crop-normalized x/y (may be outside [0,1]) and original joint
    confidence. They refer to the same rounded, clipped crop used for RGB.
    """

    @property
    def name(self) -> str: ...

    @property
    def input_size(self) -> tuple[int, int]: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, crops: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor: ...


@dataclass
class UnpromptedEncoder:
    encoder: AppearanceEncoder
    dimension: int

    @property
    def name(self) -> str:
        name: str = self.encoder.name
        return name

    @property
    def input_size(self) -> tuple[int, int]:
        size: tuple[int, int] = self.encoder.input_size
        return size

    def embed(self, crops: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed(crops)


@dataclass(frozen=True)
class FeatureConfig:
    bbox_enlarge: float = 1.2
    min_appearance_height_px: float = 16.  # at 1920x1080, scaled by image diagonal
    appearance_batch_size: int = 64

    def __post_init__(self) -> None:
        if not np.isfinite([self.bbox_enlarge, self.min_appearance_height_px]).all() \
                or self.bbox_enlarge <= 0 or self.min_appearance_height_px <= 0 or self.appearance_batch_size < 1:
            raise ValueError("Invalid per-detection feature configuration")


DEFAULT_CONFIG = FeatureConfig()


class FeatureExtractor:
    def __init__(self, pose: PoseModel | None, encoder: DetectionEncoder, config: FeatureConfig = DEFAULT_CONFIG) -> None:
        self.pose, self.encoder, self.config = pose, encoder, config
        if pose is None and not isinstance(encoder, UnpromptedEncoder):
            raise ValueError("Pose-free extraction requires an explicitly unprompted encoder")
        if encoder.dimension < 1:
            raise ValueError("Encoder dimension must be explicit, including empty frames")

    def extract(
        self, frame: int, image: NDArray[np.uint8], rows: NDArray[np.int64],
        boxes: NDArray[np.float32], scores: NDArray[np.float32],
        *, score_origin: NDArray[np.uint8] | None = None,
    ) -> DetectionFeatures:
        n = len(rows)
        poses: NDArray[np.float32] | None = None if self.pose is None else np.zeros((n, 17, 3), np.float32)
        embeddings: NDArray[np.float32] = np.zeros((n, self.encoder.dimension), np.float32)
        valid: NDArray[np.bool_] = np.zeros(n, bool)
        # Validate before invoking either model, including empty input frames.
        DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid, score_origin=score_origin)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
            raise ValueError("Feature image must be uint8 BGR (H,W,3)")
        if not n:
            return DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid, score_origin=score_origin)
        if self.pose is not None:
            poses = infer_poses(frame, image, boxes, self.pose, self.config, rows=rows)
        return encode_appearance(frame, image, rows, boxes, scores, poses, self.encoder, self.config,
                                 score_origin=score_origin)


def infer_poses(frame: int, image: NDArray[np.uint8], boxes: NDArray[np.float32],
                model: PoseModel, config: FeatureConfig = DEFAULT_CONFIG, *,
                rows: NDArray[np.int64] | None = None) -> NDArray[np.float32]:
    """Infer exactly the supplied observations using the shared ViTPose crop geometry."""
    n = len(boxes)
    if not n:
        return np.zeros((0, 17, 3), np.float32)
    squares = pose_boxes(boxes, config)
    result = model.predict(Pose2DFrameSequenceRequest(
        [(frame, image)], squares, torch.full((n,), frame, dtype=torch.int64)))
    if result.keypoints.shape != (n, 17, 3):
        raise ValueError("Pose model changed the detection row axis")
    poses = result.keypoints.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(poses).all():
        bad = np.argwhere(~np.isfinite(poses))
        details = [(int(rows[i]) if rows is not None else int(i), int(j), int(k), str(poses[i, j, k])) for i, j, k in bad]
        raise ValueError(f"Nonfinite pose at frame {frame}: (detection row, joint, channel, value)={details}")
    return poses


def pose_boxes(boxes: NDArray[np.float32], config: FeatureConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """The original full-image square geometry, shared by frame and stream passes."""
    centres = (boxes[:, :2] + boxes[:, 2:]) * .5
    sizes = (boxes[:, 2:] - boxes[:, :2]).max(1) * config.bbox_enlarge
    return torch.from_numpy(np.column_stack((centres, sizes)).astype(np.float32))


@dataclass(frozen=True)
class AppearanceCrop:
    """A prepared crop with its original source address; no image is retained."""

    frame: int
    row: int
    pixels: torch.Tensor
    prompt: torch.Tensor | None


def iter_appearance_crops(frame: int, image: NDArray[np.uint8], rows: NDArray[np.int64],
                          boxes: NDArray[np.float32], poses: NDArray[np.float32] | None,
                          encoder: DetectionEncoder, config: FeatureConfig = DEFAULT_CONFIG) -> Iterator[AppearanceCrop]:
    """Prepare rounded/clipped RGB crops, without invoking an encoder."""
    if poses is None and not isinstance(encoder, UnpromptedEncoder):
        raise ValueError("Pose-free appearance requires an explicitly unprompted encoder")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
        raise ValueError("Feature image must be uint8 BGR (H,W,3)")
    height, width = image.shape[:2]
    clipped = np.rint(boxes).astype(np.float32)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    crop_size = clipped[:, 2:] - clipped[:, :2]
    scale = np.hypot(width, height) / np.hypot(1920, 1080)
    valid = (crop_size[:, 0] > 0) & (crop_size[:, 1] >= config.min_appearance_height_px * scale)
    for index in np.flatnonzero(valid):
        prompt = None
        if poses is not None and not isinstance(encoder, UnpromptedEncoder):
            relative = poses[index].copy()
            relative[..., :2] = (relative[..., :2] - clipped[index, :2]) / crop_size[index]
            prompt = torch.from_numpy(relative)
        yield AppearanceCrop(frame, int(rows[index]), torch.from_numpy(crop(image, boxes[index], encoder.input_size)), prompt)


def appearance_batches(crops: Iterable[AppearanceCrop], batch_size: int) -> Iterator[list[AppearanceCrop]]:
    """Bound crop memory while allowing a batch to span empty/adjacent frames."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Appearance batch size must be a positive integer")
    batch: list[AppearanceCrop] = []
    for item in crops:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def infer_appearance(batch: Sequence[AppearanceCrop], encoder: DetectionEncoder) -> NDArray[np.float32]:
    """Infer one prepared batch; standard CLIP never receives pose prompts."""
    pixels = torch.stack([item.pixels for item in batch])
    if isinstance(encoder, UnpromptedEncoder):
        encoded = encoder.encoder.embed(pixels)
    else:
        if any(item.prompt is None for item in batch):
            raise ValueError("Prompted encoder requires saved pose prompts")
        encoded = encoder.embed(pixels, torch.stack([item.prompt for item in batch if item.prompt is not None]))
    result = encoded.detach().cpu().numpy().astype(np.float32)
    if result.shape != (len(batch), encoder.dimension):
        raise ValueError("Appearance model changed the detection or embedding axis")
    return result


def scatter_appearance(batch: Sequence[AppearanceCrop], encoded: NDArray[np.float32],
                       row_indices: dict[int, int], embeddings: NDArray[np.float32], valid: NDArray[np.bool_]) -> None:
    """Write only inferred source rows; invalid crops keep zero and false."""
    if encoded.shape != (len(batch), embeddings.shape[1]):
        raise ValueError("Appearance batch output changed the row axis")
    indices = [row_indices[item.row] for item in batch]
    if len(set(indices)) != len(indices) or valid[indices].any():
        raise ValueError("Appearance source row was inferred more than once")
    embeddings[indices] = encoded
    valid[indices] = True


def encode_appearance(frame: int, image: NDArray[np.uint8], rows: NDArray[np.int64],
                      boxes: NDArray[np.float32], scores: NDArray[np.float32], poses: NDArray[np.float32] | None,
                      encoder: DetectionEncoder, config: FeatureConfig = DEFAULT_CONFIG, *,
                      score_origin: NDArray[np.uint8] | None = None) -> DetectionFeatures:
    """Encode another backbone using the identical saved pose/detection rows."""
    n = len(rows)
    if poses is None and not isinstance(encoder, UnpromptedEncoder):
        raise ValueError("Pose-free appearance requires an explicitly unprompted encoder")
    embeddings: NDArray[np.float32] = np.zeros((n, encoder.dimension), np.float32)
    valid: NDArray[np.bool_] = np.zeros(n, bool)
    DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid, score_origin=score_origin)
    row_indices = {int(row): index for index, row in enumerate(rows)}
    crops = iter_appearance_crops(frame, image, rows, boxes, poses, encoder, config)
    for batch in appearance_batches(crops, config.appearance_batch_size):
        scatter_appearance(batch, infer_appearance(batch, encoder), row_indices, embeddings, valid)
    return DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid, score_origin=score_origin)
