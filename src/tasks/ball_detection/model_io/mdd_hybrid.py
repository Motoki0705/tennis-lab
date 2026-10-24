"""Explicit hybrid heatmap/presence boundary, separate from coordinate regression."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import (
    DEFAULT_CANDIDATE_CONFIG,
    BallCandidateConfig,
    BallModelIOError,
    BallPrediction,
)
from src.tasks.ball_detection.model_io.mdd_coordinates import validate_rgb_timestamps
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput
from src.tasks.ball_detection.models.mdd import (
    DeepMDDHybridDetector,
    MDDHybridConfig,
    MDDHybridOutput,
)
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic


@dataclass(frozen=True)
class BallHybridPrediction(BallPrediction):
    """CPU evidence on the native /4 grid, with an explicit presence decision.

    coords uses the BallPrediction endpoint convention for present frames, NaN
    for absent frames. confidence is the heatmap peak score, zero when absent.
    All absent candidate slots are invalid/zero. Raw heatmaps and presence_logits
    remain available for independent auditing; neither is a calibrated probability.
    Consumers must apply present before accessing coordinates or rescale them.
    """

    presence_logits: Tensor
    present: Tensor


class MDDHybridAdapter:
    def __init__(self, config: MDDHybridConfig, *, presence_threshold: float,
                 subpixel: bool = True, candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG) -> None:
        if not isinstance(config, MDDHybridConfig):
            raise BallModelIOError("Hybrid I/O requires MDDHybridConfig")
        if type(presence_threshold) not in (int, float) or not math.isfinite(presence_threshold) or not 0 < presence_threshold < 1:
            raise BallModelIOError("presence_threshold must be finite and strictly between zero and one")
        if type(subpixel) is not bool:
            raise BallModelIOError("subpixel must be boolean")
        self.config = config
        self.presence_threshold = presence_threshold
        self.subpixel, self.candidate_config = subpixel, candidate_config

    @property
    def model_type(self) -> type[nn.Module]:
        model_type: type[nn.Module] = DeepMDDHybridDetector
        return model_type

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, DeepMDDHybridDetector) or model.config != self.config:
            raise BallModelIOError("Hybrid model/adapter configuration mismatch")

    def build_call(self, inputs: MDDQueryInput) -> ModelCall:
        validate_rgb_timestamps(self.config, inputs.rgb, inputs.timestamps)
        return ModelCall(args=(inputs.rgb, inputs.timestamps))

    def decode_output(self, output: MDDHybridOutput) -> BallHybridPrediction:
        if not isinstance(output, MDDHybridOutput):
            raise BallModelIOError("Hybrid decoding requires heatmap and presence logits")
        heatmap, presence = output.heatmap_logits, output.presence_logits
        if (heatmap.ndim != 4 or heatmap.shape[0] < 1 or heatmap.shape[1] != self.config.frames
                or min(heatmap.shape[-2:]) < 2 or presence.shape != heatmap.shape[:2]):
            raise BallModelIOError(f"Hybrid output requires B,{self.config.frames},H,W heatmaps and B,{self.config.frames} presence")
        if (heatmap.device != presence.device or not heatmap.is_floating_point() or not presence.is_floating_point()
                or not bool(torch.isfinite(heatmap).all()) or not bool(torch.isfinite(presence).all())):
            raise BallModelIOError("Hybrid logits must be finite floating tensors on the same device")
        probabilities = heatmap.detach().float().sigmoid().cpu()
        presence = presence.detach().float().cpu()
        present = presence.sigmoid() >= self.presence_threshold
        coords, confidence = heatmaps_to_argmax(probabilities)
        if self.subpixel:
            coords = refine_peaks_log_parabolic(probabilities, coords)
        candidates = decode_candidates(probabilities, config=self.candidate_config, subpixel_refine=self.subpixel)
        valid = candidates.valid & present[..., None]
        candidates = replace(candidates,
            coords=candidates.coords.masked_fill(~valid[..., None], 0),
            scores=candidates.scores.masked_fill(~valid, 0), valid=valid,
            cells=candidates.cells.masked_fill(~valid[..., None], 0),
            patches=candidates.patches.masked_fill(~valid[..., None, None], 0),
            patch_valid=candidates.patch_valid & valid[..., None, None])
        return BallHybridPrediction(
            coords=coords.masked_fill(~present[..., None], float("nan")),
            confidence=confidence.masked_fill(~present, 0), heatmaps=probabilities,
            candidates=candidates, presence_logits=presence, present=present)


def build_mdd_hybrid_detector(config: MDDHybridConfig, *, presence_threshold: float,
                             subpixel: bool = True, candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG,
                             mdd_a: float = .2, mdd_b: float = .15) -> BoundModelIO[MDDQueryInput, MDDHybridOutput, BallHybridPrediction]:
    model = DeepMDDHybridDetector(config, mdd_a=mdd_a, mdd_b=mdd_b)
    adapter = MDDHybridAdapter(config, presence_threshold=presence_threshold,
                               subpixel=subpixel, candidate_config=candidate_config)
    adapter.validate_model_pair(model)
    return bind_model_io(model, adapter)


def load_mdd_hybrid_detector(path: Path) -> BoundModelIO[MDDQueryInput, MDDHybridOutput, BallHybridPrediction]:
    """Load an adopted checkpoint without ancestor files; use its explicit threshold."""
    from src.tasks.ball_detection.checkpoints.hybrid_training import (
        restore_hybrid_model,
    )

    model, recipe = restore_hybrid_model(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    adapter = MDDHybridAdapter(model.config, presence_threshold=recipe["presence_threshold"])
    return bind_model_io(model, adapter)
