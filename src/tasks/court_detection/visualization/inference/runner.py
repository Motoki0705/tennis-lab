"""One strict Lightning load per checkpoint, shared by every Court head."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.tasks.court_detection.inference.predictor import CourtPredictor
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedPrediction,
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
        self.predictor = CourtPredictor.load_from_checkpoint(
            checkpoint_path.resolve(),
            device=device,
            subpixel_refine=subpixel_refine,
            max_peaks=max_peaks,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.adapter = self.predictor.adapter
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
        prediction = self.predictor.predict(image, postprocess="none")
        return dict(prediction.raw_heads)

    def close(self) -> None:
        """Release the model and any cached CUDA memory."""
        self.predictor.model.to("cpu")
        del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["KP_MAX_PEAKS", "CourtHeadRunner"]
