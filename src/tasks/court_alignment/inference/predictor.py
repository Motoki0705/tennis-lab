"""Inference wrapper for the Court Alignment keypoint CNN."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.tasks.court_alignment.inference.decoder import (
    CourtInstances,
    CourtPeakDetections,
    decode_court_instances,
    decode_keypoint_peaks,
)
from src.tasks.court_alignment.models.cnn import (
    validate_court_alignment_input,
    validate_court_alignment_output,
)


class CourtAlignmentPredictor:
    """Run a trained model and decode multi-court KP instances.

    The predictor accepts a batch of one-channel UV heatmaps.  Input tensors
    are intentionally not resized or normalised: those operations belong to
    the versioned dataset contract and changing them at inference would alter
    pixel-space center votes.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        threshold: float = 0.25,
        nms_kernel: int = 3,
        max_peaks: int = 8,
        subpixel_refine: bool = True,
        cluster_distance_px: float = 12.0,
        max_instances: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("CourtAlignmentPredictor model must be a torch module.")
        # Decoder validates the numeric values too; retaining these attributes
        # makes an experiment's decode configuration serialisable and visible.
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0,1].")
        if type(nms_kernel) is not int or nms_kernel <= 0 or nms_kernel % 2 == 0:
            raise ValueError("nms_kernel must be a positive odd integer.")
        if type(max_peaks) is not int or max_peaks <= 0:
            raise ValueError("max_peaks must be a positive integer.")
        if not math.isfinite(float(cluster_distance_px)) or cluster_distance_px <= 0:
            raise ValueError("cluster_distance_px must be finite and positive.")
        if max_instances is not None and (type(max_instances) is not int or max_instances <= 0):
            raise ValueError("max_instances must be a positive integer or None.")
        self.model = model
        self.threshold = float(threshold)
        self.nms_kernel = nms_kernel
        self.max_peaks = max_peaks
        self.subpixel_refine = bool(subpixel_refine)
        self.cluster_distance_px = float(cluster_distance_px)
        self.max_instances = max_instances
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        if not isinstance(image, Tensor):
            raise TypeError("Court alignment image must be a torch.Tensor.")
        if (
            image.ndim != 4
            or image.shape[0] <= 0
            or image.shape[1] != 1
            or any(size <= 0 for size in image.shape[2:])
        ):
            raise ValueError("Court alignment image must have shape (B,1,H,W).")
        if not image.is_floating_point():
            raise TypeError("Court alignment image must be floating point.")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("Court alignment image must contain only finite values.")
        if bool(torch.any((image < 0.0) | (image > 1.0))):
            raise ValueError("Court alignment image values must lie in [0,1].")

    def predict(self, image: Tensor) -> CourtInstances:
        """Return grouped court instances for every input sample."""
        self._validate_image(image)
        expected_dtype = next(self.model.parameters(), torch.empty(0)).dtype
        validate_court_alignment_input(image, expected_dtype=expected_dtype)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image.to(self.device))
        output = validate_court_alignment_output(output)
        heatmap_logits, center_votes = output.heatmap_logits, output.center_votes
        return decode_court_instances(
            heatmap_logits,
            center_votes,
            threshold=self.threshold,
            nms_kernel=self.nms_kernel,
            max_peaks=self.max_peaks,
            subpixel_refine=self.subpixel_refine,
            cluster_distance_px=self.cluster_distance_px,
            max_instances=self.max_instances,
        )

    def decode(self, heatmap_logits: Tensor, center_votes: Tensor) -> CourtInstances:
        """Decode precomputed model outputs without running the network."""
        return decode_court_instances(
            heatmap_logits,
            center_votes,
            threshold=self.threshold,
            nms_kernel=self.nms_kernel,
            max_peaks=self.max_peaks,
            subpixel_refine=self.subpixel_refine,
            cluster_distance_px=self.cluster_distance_px,
            max_instances=self.max_instances,
        )

    def decode_peaks(
        self, heatmap_logits: Tensor, center_votes: Tensor
    ) -> CourtPeakDetections:
        """Expose the pre-grouping peaks for diagnostics and sigma ablations."""
        return decode_keypoint_peaks(
            heatmap_logits,
            center_votes,
            threshold=self.threshold,
            nms_kernel=self.nms_kernel,
            max_peaks=self.max_peaks,
            subpixel_refine=self.subpixel_refine,
        )


CourtAlignmentKP14Predictor = CourtAlignmentPredictor


__all__ = ["CourtAlignmentKP14Predictor", "CourtAlignmentPredictor"]
