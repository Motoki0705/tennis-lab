"""ViTPose 2D keypoint model (typed port of hmr4d.utils.preproc.vitpose)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.submodules.models._base import BaseInferenceModel
from src.submodules.vendor.gvhmr.hmr2.preproc import get_batch
from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps
from src.utils.paths import PROJECT_ROOT

DEFAULT_VITPOSE_CHECKPOINT = PROJECT_ROOT / "ckpt/vitpose/vitpose-h-multi-coco.pth"


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


@dataclass(frozen=True)
class Pose2DResult:
    """COCO-17 keypoints ``(F, 17, 3)`` as (x, y, confidence) in pixels."""

    keypoints: torch.Tensor


class ViTPosePose2D(BaseInferenceModel[Pose2DRequest, Pose2DResult]):
    """ViTPose-H top-down 2D pose estimator on tracked person crops."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_VITPOSE_CHECKPOINT,
        device: str | torch.device = "auto",
        flip_test: bool = True,
        batch_size: int = 16,
    ) -> None:
        super().__init__(device)
        self.checkpoint = Path(checkpoint)
        self.flip_test = flip_test
        self.batch_size = batch_size
        self._pose: torch.nn.Module | None = None

    def _load_impl(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {self.checkpoint}")
        self._pose = build_vitpose_huge(str(self.checkpoint)).to(self._device).eval()

    def _unload_impl(self) -> None:
        self._pose = None

    def _predict_impl(self, request: Pose2DRequest) -> Pose2DResult:
        assert self._pose is not None
        imgs, bbx_xys = get_batch(str(request.video_path), request.bbx_xys, img_ds=0.5)

        num_frames = imgs.shape[0]
        keypoints = []
        for j in tqdm(range(0, num_frames, self.batch_size), desc="ViTPose"):
            imgs_batch = imgs[j : j + self.batch_size, :, :, 32:224].to(self._device)
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
            bbx_xys_batch = bbx_xys[j : j + self.batch_size]
            heatmap_np = heatmap.cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (
                torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200
            ).numpy()
            preds, maxvals = keypoints_from_heatmaps(
                heatmaps=heatmap_np, center=center, scale=scale, use_udp=True
            )
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            keypoints.append(torch.from_numpy(kp2d))

        return Pose2DResult(keypoints=torch.cat(keypoints, dim=0).float())  # (F, 17, 3)
