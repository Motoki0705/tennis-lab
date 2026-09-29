"""One pose/appearance pass shared by camera-local methods and downstream consumers.

Decoded frames may come from raw video or the refiner's JPEG store. The caller
owns media provenance. No detector, tracker or identity selection runs here.
"""

from dataclasses import dataclass
from typing import Protocol, cast

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
        return cast(str, self.encoder.name)

    @property
    def input_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.encoder.input_size)

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
    def __init__(self, pose: PoseModel, encoder: DetectionEncoder, config: FeatureConfig = DEFAULT_CONFIG) -> None:
        self.pose, self.encoder, self.config = pose, encoder, config
        if encoder.dimension < 1:
            raise ValueError("Encoder dimension must be explicit, including empty frames")

    def extract(
        self, frame: int, image: NDArray[np.uint8], rows: NDArray[np.int64],
        boxes: NDArray[np.float32], scores: NDArray[np.float32],
    ) -> DetectionFeatures:
        n = len(rows)
        poses: NDArray[np.float32] = np.zeros((n, 17, 3), np.float32)
        embeddings: NDArray[np.float32] = np.zeros((n, self.encoder.dimension), np.float32)
        valid: NDArray[np.bool_] = np.zeros(n, bool)
        # Validate before invoking either model, including empty input frames.
        DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
            raise ValueError("Feature image must be uint8 BGR (H,W,3)")
        if not n:
            return DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid)
        centres = (boxes[:, :2] + boxes[:, 2:]) * .5
        sizes = (boxes[:, 2:] - boxes[:, :2]).max(1) * self.config.bbox_enlarge
        squares = torch.from_numpy(np.column_stack((centres, sizes)).astype(np.float32))
        result = self.pose.predict(Pose2DFrameSequenceRequest(
            [(frame, image)], squares, torch.full((n,), frame, dtype=torch.int64)))
        if result.keypoints.shape != (n, 17, 3):
            raise ValueError("Pose model changed the detection row axis")
        poses = result.keypoints.detach().cpu().numpy().astype(np.float32)
        # ViTPose returns regression heatmap peaks, not probabilities. Preserve
        # them (including values >1 or <0); clipping would change pose weights
        # and the prompt supplied to a pose-aware appearance encoder.
        if not np.isfinite(poses).all():
            bad = np.argwhere(~np.isfinite(poses))
            details = [(int(rows[i]), int(j), int(k), str(poses[i, j, k])) for i, j, k in bad]
            raise ValueError(f"Nonfinite pose at frame {frame}: (detection row, joint, channel, value)={details}")
        height, width = image.shape[:2]
        clipped = np.rint(boxes).astype(np.float32)
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
        crop_size = clipped[:, 2:] - clipped[:, :2]
        scale = np.hypot(width, height) / np.hypot(1920, 1080)
        valid = (crop_size[:, 0] > 0) & (crop_size[:, 1] >= self.config.min_appearance_height_px * scale)
        selected = np.flatnonzero(valid)
        for start in range(0, len(selected), self.config.appearance_batch_size):
            batch = selected[start:start + self.config.appearance_batch_size]
            pixels = torch.from_numpy(np.stack([crop(image, boxes[row], self.encoder.input_size) for row in batch]))
            prompts = poses[batch].copy()
            prompts[..., :2] = (prompts[..., :2] - clipped[batch, None, :2]) / crop_size[batch, None, :]
            encoded = self.encoder.embed(pixels, torch.from_numpy(prompts)).detach().cpu().numpy()
            if encoded.shape != (len(batch), self.encoder.dimension):
                raise ValueError("Appearance model changed the detection or embedding axis")
            embeddings[batch] = encoded
        return DetectionFeatures(frame, rows, boxes, scores, poses, embeddings, valid)
