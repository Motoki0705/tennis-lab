"""ViTPose 2D keypoint model (typed port of hmr4d.utils.preproc.vitpose)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from src.submodules.configuration import InferencePrecision, require_absolute_path
from src.submodules.models._base.crops import iter_frame_person_crops, iter_person_crops
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models._base.precision import autocast_dtype
from src.submodules.vendor.gvhmr.hmr2.preproc import get_batch
from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps
from src.utils.inference_recording import span
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner


@dataclass(frozen=True)
class Pose2DRequest:
    """Request for per-frame 2D pose estimation of one tracked person.

    Attributes:
        video_path: Source video file.
        bbx_xys: Per-frame square person boxes ``(F, 3)`` as
            (center_x, center_y, size) in pixels.
    """

    video_path: str | Path
    bbx_xys: torch.Tensor
    frame_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class Pose2DFrameSequenceRequest:
    """Top-down poses from an explicit, possibly one-shot BGR frame stream.

    ``frames_bgr`` yields strictly increasing integer IDs and uint8 images.
    ``frame_indices`` is an ordered CPU int64 tensor with one ID per box;
    repeated IDs allow several people in one image. ``bbx_xys`` uses the
    decoded image's pixel coordinates. Only requested frames are cropped,
    and every request row produces one COCO-17 result in the same order.
    """

    frames_bgr: Iterable[tuple[int, NDArray[np.uint8]]]
    bbx_xys: torch.Tensor
    frame_indices: torch.Tensor


@dataclass(frozen=True)
class Pose2DResult:
    """COCO-17 ``(F, 17, 3)``: pixel x/y and raw regression heatmap peak.

    The last channel is a confidence score, not a probability: the linear
    heatmap head does not bound it to [0,1]. No sigmoid/clipping is applied.
    """

    keypoints: torch.Tensor


class ViTPosePose2D(BaseInferenceModel[Pose2DRequest | Pose2DFrameSequenceRequest, Pose2DResult]):
    """ViTPose-H top-down 2D pose estimator on tracked person crops.

    ``show_progress`` controls the crop-batch bar, which is enabled by default.
    Callers reporting their own progress can disable it independently.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        flip_test: bool,
        batch_size: int,
        head_config: ViTPoseHeadConfig,
        precision: InferencePrecision | Literal["float32", "bfloat16"] = "float32",
        show_progress: bool = True,
        engine: EngineBundle | None = None,
    ) -> None:
        super().__init__(device)
        if type(flip_test) is not bool:
            raise TypeError("flip_test must be a bool.")
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if type(show_progress) is not bool:
            raise TypeError("show_progress must be a bool.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.checkpoint = require_absolute_path(checkpoint, name="ViTPose checkpoint")
        self.flip_test = flip_test
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.precision = precision
        if precision == "float32":
            precision = "fp32"
        elif precision == "bfloat16":
            precision = "bf16"
        self._autocast_dtype = autocast_dtype(precision, self.device) if engine is None else None
        if not isinstance(head_config, ViTPoseHeadConfig):
            raise TypeError("head_config must be a validated ViTPoseHeadConfig.")
        self.head_config = head_config
        self._pose: torch.nn.Module | None = None
        self._engine = engine
        self._runner: TensorRTRunner | None = None

    def recording_assets(self) -> dict[str, Path]:
        if self._engine is not None:
            return {"engine": self._engine.root / PLAN_FILE}
        return {"checkpoint": self.checkpoint}

    def _load_impl(self) -> None:
        if self._engine is not None:
            self._runner = TensorRTRunner(self._engine, self.device)
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {self.checkpoint}")
        self._pose = (
            build_vitpose_huge(
                str(self.checkpoint),
                head_config=self.head_config,
            )
            .to(self._device)
            .eval()
        )

    def _unload_impl(self) -> None:
        self._pose = None
        self._runner = None

    def _predict_impl(self, request: Pose2DRequest | Pose2DFrameSequenceRequest) -> Pose2DResult:
        batches: Iterator[tuple[torch.Tensor, torch.Tensor]]
        if isinstance(request, Pose2DFrameSequenceRequest):
            batches = iter_frame_person_crops(
                request.frames_bgr, request.bbx_xys, request.frame_indices, batch_size=self.batch_size,
            )
        elif request.frame_indices is None:
            imgs, bbx_xys = get_batch(str(request.video_path), request.bbx_xys, img_ds=0.5)
            batches = ((imgs[j:j + self.batch_size], bbx_xys[j:j + self.batch_size]) for j in range(0, len(imgs), self.batch_size))
        else:
            batches = iter_person_crops(request.video_path, request.bbx_xys, request.frame_indices, batch_size=self.batch_size)
        keypoints = []
        batch_iterator = tqdm(batches, desc="ViTPose") if self.show_progress else batches
        for images, bbx_xys_batch in batch_iterator:
            keypoints.append(self._predict_crop_batch(images, bbx_xys_batch))

        return Pose2DResult(keypoints=torch.cat(keypoints, dim=0).float() if keypoints else torch.empty(0, 17, 3))

    def predict_crops(self, images: torch.Tensor, boxes_xys: torch.Tensor) -> Pose2DResult:
        """Infer one externally bounded batch of prepared ImageNet person crops."""
        self.load()
        with torch.no_grad():
            return Pose2DResult(self._predict_crop_batch(images, boxes_xys).float())

    def _predict_crop_batch(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        with span("transfer", model="ViTPose", operation="input_to_device", device=str(self.device)):
            imgs_batch = images[:, :, :, 32:224].to(self._device)
        with span("inference", model="ViTPose", batch_shape=[len(images) * (2 if self.flip_test else 1), *imgs_batch.shape[1:]],
                  batch_items=len(images), flip_test=self.flip_test), torch.autocast(
            self.device.type, dtype=self._autocast_dtype, enabled=self._autocast_dtype is not None,
        ):
            if self.flip_test:
                heatmap, heatmap_flipped = self._forward_heatmaps(
                    torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)
                ).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self._forward_heatmaps(imgs_batch.clone())  # (B, J, 64, 48)

        with span("transfer", model="ViTPose", operation="heatmap_to_host", device=str(self.device)):
            heatmap_np = heatmap.float().cpu().numpy()
        with span("postprocess", model="ViTPose", operation="decode_heatmaps"):
            # MMPose-style UDP post-processing back to full-image pixels.
            center = boxes[:, :2].numpy()
            scale = (torch.cat((boxes[:, [2]] * 24 / 32, boxes[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(
                heatmaps=heatmap_np, center=center, scale=scale, use_udp=True
            )
            return torch.from_numpy(np.concatenate((preds, maxvals), axis=-1))

    def _forward_heatmaps(self, pixels: torch.Tensor) -> torch.Tensor:
        if self._runner is not None:
            return self._runner({"pixels": pixels})["heatmaps"]
        if self._pose is None:
            raise RuntimeError("ViTPose model did not load before prediction.")
        heatmaps: torch.Tensor = self._pose(pixels)
        return heatmaps
