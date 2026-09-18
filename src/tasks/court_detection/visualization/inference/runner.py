"""One strict Lightning load per checkpoint, shared by every Court head."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from PIL import Image

from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
    CourtLogits,
)
from src.tasks.court_detection.model_io.images import prepare_court_image
from src.tasks.court_detection.models.pose_head import CourtModelOutput
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)

KP_MAX_PEAKS = 4


class CourtHeadRunner:
    """Load one checkpoint and decode every trained head from one forward pass."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
        subpixel_refine: bool = True,
        max_peaks: int = KP_MAX_PEAKS,
    ) -> None:
        if max_peaks <= 0:
            raise ValueError("Court keypoint max_peaks must be positive.")
        module = CourtDetectionLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        adapter = module.model_io
        adapter.validate_model_pair(module.model)
        module.model.to(device)
        module.model.eval()
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.module = module
        self.adapter = adapter
        self.subpixel_refine = subpixel_refine
        self.max_peaks = max_peaks

    @property
    def heads(self) -> tuple[str, ...]:
        return tuple(self.adapter.spec.target_bundle.kinds)

    @property
    def short_side(self) -> int:
        return int(self.adapter.spec.short_side)

    def predict(self, image: Image.Image) -> dict[str, CourtDecodedPrediction]:
        """Return every trained head decoded into original-image geometry."""
        adapter = self.adapter
        images, height, width = prepare_court_image(
            image,
            short_side=adapter.spec.short_side,
            device=self.device,
        )
        with torch.no_grad():
            call = adapter.prepare_images(images)
            raw = self.module.model(*call.model_args)
            adapter.validate_logits(raw, call)
            logits = (
                cast(CourtModelOutput, raw).dense_logits
                if isinstance(raw, CourtModelOutput)
                else cast(CourtLogits, raw)
            )
            decoded: dict[str, CourtDecodedPrediction] = {}
            for kind in adapter.spec.target_bundle.kinds:
                decoded[kind] = adapter.decode_prediction(
                    kind,
                    logits[kind],
                    original_size_hw=(height, width),
                    subpixel_refine=self.subpixel_refine and kind == "kp",
                    max_peaks=self.max_peaks,
                )
        return decoded

    def close(self) -> None:
        """Release the model and any cached CUDA memory."""
        self.module.model.to("cpu")
        del self.module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["KP_MAX_PEAKS", "CourtHeadRunner"]
