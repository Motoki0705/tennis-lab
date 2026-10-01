"""ViTPose 2D keypoint model (typed port of hmr4d.utils.preproc.vitpose)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from src.submodules.configuration import require_absolute_path
from src.submodules.models._base.crops import iter_person_crops
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.vendor.gvhmr.hmr2.preproc import get_batch
from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps


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
class Pose2DResult:
    """COCO-17 keypoints ``(F, 17, 3)`` as (x, y, confidence) in pixels."""

    keypoints: torch.Tensor


class ViTPosePose2D(BaseInferenceModel[Pose2DRequest, Pose2DResult]):
    """ViTPose-H top-down 2D pose estimator on tracked person crops."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        flip_test: bool,
        batch_size: int,
        head_config: ViTPoseHeadConfig,
        precision: Literal["float32", "bfloat16"] = "float32",
    ) -> None:
        super().__init__(device)
        if type(flip_test) is not bool:
            raise TypeError("flip_test must be a bool.")
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.checkpoint = require_absolute_path(checkpoint, name="ViTPose checkpoint")
        self.flip_test = flip_test
        self.batch_size = batch_size
        if precision not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported ViTPose precision: {precision}")
        self.precision = precision
        if not isinstance(head_config, ViTPoseHeadConfig):
            raise TypeError("head_config must be a validated ViTPoseHeadConfig.")
        self.head_config = head_config
        self._pose: torch.nn.Module | None = None

    def _load_impl(self) -> None:
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

    def _predict_impl(self, request: Pose2DRequest) -> Pose2DResult:
        if self._pose is None:
            raise RuntimeError("ViTPose model did not load before prediction.")
        batches: Iterator[tuple[torch.Tensor, torch.Tensor]]
        if request.frame_indices is None:
            imgs, bbx_xys = get_batch(str(request.video_path), request.bbx_xys, img_ds=0.5)
            batches = ((imgs[j:j + self.batch_size], bbx_xys[j:j + self.batch_size]) for j in range(0, len(imgs), self.batch_size))
        else:
            batches = iter_person_crops(request.video_path, request.bbx_xys, request.frame_indices, batch_size=self.batch_size)
        keypoints = []
        for images, bbx_xys_batch in tqdm(batches, desc="ViTPose"):
            imgs_batch = images[:, :, :, 32:224].to(self._device)
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.precision == "bfloat16",
            ):
                if self.flip_test:
                    heatmap, heatmap_flipped = self._pose(
                        torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)
                    ).chunk(2)
                    heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                    heatmap = (heatmap + heatmap_flipped) * 0.5
                    del heatmap_flipped
                else:
                    heatmap = self._pose(imgs_batch.clone())  # (B, J, 64, 48)

            # mmpose-style UDP post-processing back to full-image pixels
            heatmap_np = heatmap.float().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (
                torch.cat(
                    (bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1
                )
                / 200
            ).numpy()
            preds, maxvals = keypoints_from_heatmaps(
                heatmaps=heatmap_np, center=center, scale=scale, use_udp=True
            )
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            keypoints.append(torch.from_numpy(kp2d))

        return Pose2DResult(keypoints=torch.cat(keypoints, dim=0).float() if keypoints else torch.empty(0, 17, 3))
