"""HMR2 image-feature model (typed port of hmr4d.utils.preproc.vitfeat_extractor)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.submodules.configuration import require_absolute_path
from src.submodules.models._base import BaseInferenceModel
from src.submodules.vendor.gvhmr.hmr2 import load_hmr2
from src.submodules.vendor.gvhmr.hmr2.preproc import get_batch


@dataclass(frozen=True)
class ImageFeatureRequest:
    """Request for per-frame HMR2 image features of one tracked person.

    Attributes:
        video_path: Source video file.
        bbx_xys: Per-frame square person boxes ``(F, 3)`` as
            (center_x, center_y, size) in pixels.
    """

    video_path: str | Path
    bbx_xys: torch.Tensor


@dataclass(frozen=True)
class ImageFeatureResult:
    """HMR2 feature tokens ``(F, 1024)`` (float32, CPU)."""

    features: torch.Tensor


class Hmr2FeatureExtractor(BaseInferenceModel[ImageFeatureRequest, ImageFeatureResult]):
    """HMR2.0a backbone features used by GVHMR as per-frame image evidence."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        allow_device_fallback: bool,
        batch_size: int,
        mean_params_path: str | Path,
    ) -> None:
        super().__init__(device, allow_device_fallback=allow_device_fallback)
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.checkpoint = require_absolute_path(checkpoint, name="HMR2 checkpoint")
        self.mean_params_path = require_absolute_path(
            mean_params_path,
            name="HMR2 mean-parameter asset",
        )
        self.batch_size = batch_size
        self._model: torch.nn.Module | None = None

    def _load_impl(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"HMR2 checkpoint not found: {self.checkpoint}")
        if not self.mean_params_path.is_file():
            raise FileNotFoundError(
                f"HMR2 mean-parameter asset not found: {self.mean_params_path}"
            )
        self._model = (
            load_hmr2(
                str(self.checkpoint),
                mean_params_path=self.mean_params_path,
            )
            .to(self._device)
            .eval()
        )

    def _unload_impl(self) -> None:
        self._model = None

    def _predict_impl(self, request: ImageFeatureRequest) -> ImageFeatureResult:
        assert self._model is not None
        imgs, _ = get_batch(str(request.video_path), request.bbx_xys, img_ds=0.5)

        num_frames = imgs.shape[0]
        features = []
        for j in tqdm(range(0, num_frames, self.batch_size), desc="HMR2 features"):
            imgs_batch = imgs[j : j + self.batch_size].to(self._device)
            feature = self._model({"img": imgs_batch})
            features.append(feature.detach().cpu())

        return ImageFeatureResult(
            features=torch.cat(features, dim=0).float()
        )  # (F, 1024)
