"""Explicit variable-length, heatmap-only RGB+MDD model boundary and artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

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
    DeepMDDHeatmapHybridDetector,
    MDDDPTDetector,
    MDDHeatmapConfig,
    MDDHeatmapHybridConfig,
)
from src.tasks.ball_detection.preprocessing.rgb_mdd import input_transform_from_contract
from src.tasks.base.model_io import BoundModelIO, ModelCall, bind_model_io
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic

HEATMAP_MODEL_SCHEMA = "mdd_heatmap_model.v1"
DPT_HEATMAP_KIND = "mdd_dpt_heatmap"
HYBRID_HEATMAP_KIND = "mdd_heatmap_hybrid"
HeatmapModel: TypeAlias = MDDDPTDetector | DeepMDDHeatmapHybridDetector


def heatmap_model_kind(config: MDDHeatmapConfig) -> str:
    if type(config) is MDDHeatmapHybridConfig:
        return HYBRID_HEATMAP_KIND
    if type(config) is MDDHeatmapConfig:
        return DPT_HEATMAP_KIND
    raise BallModelIOError("Heatmap-only models require an explicit temporal heatmap config")


class MDDHeatmapAdapter:
    def __init__(self, config: MDDHeatmapConfig, *, subpixel: bool = True,
                 candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG) -> None:
        heatmap_model_kind(config)
        if type(subpixel) is not bool:
            raise BallModelIOError("subpixel must be boolean")
        self.config = config
        self.subpixel, self.candidate_config = subpixel, candidate_config

    @property
    def model_type(self) -> type[nn.Module]:
        model_type: type[nn.Module] = DeepMDDHeatmapHybridDetector if type(self.config) is MDDHeatmapHybridConfig else MDDDPTDetector
        return model_type

    def validate_model_pair(self, model: nn.Module) -> None:
        if not isinstance(model, (MDDDPTDetector, DeepMDDHeatmapHybridDetector)) or type(model) is not self.model_type or model.config != self.config:
            raise BallModelIOError("Heatmap-only model/adapter configuration mismatch")

    def build_call(self, inputs: MDDQueryInput) -> ModelCall:
        validate_rgb_timestamps(self.config, inputs.rgb, inputs.timestamps)
        args = (inputs.rgb, inputs.timestamps) if type(self.config) is MDDHeatmapHybridConfig else (inputs.rgb,)
        return ModelCall(args=args)

    def decode_output(self, output: Tensor) -> BallPrediction:
        if (not isinstance(output, Tensor) or output.ndim != 4 or min(output.shape) < 1
                or output.shape[1] != self.config.frames or min(output.shape[-2:]) < 2
                or not output.is_floating_point() or not bool(torch.isfinite(output).all())):
            raise BallModelIOError(f"Heatmap-only output requires finite floating B,{self.config.frames},H,W logits")
        probabilities = output.detach().float().sigmoid().cpu()
        coords, confidence = heatmaps_to_argmax(probabilities)
        if self.subpixel:
            coords = refine_peaks_log_parabolic(probabilities, coords)
        return BallPrediction(coords=coords, confidence=confidence, heatmaps=probabilities,
            candidates=decode_candidates(probabilities, config=self.candidate_config, subpixel_refine=self.subpixel))


def build_mdd_heatmap_detector(config: MDDHeatmapConfig, *, mdd_a: float = .2, mdd_b: float = .15,
                               subpixel: bool = True, candidate_config: BallCandidateConfig = DEFAULT_CANDIDATE_CONFIG,
) -> BoundModelIO[MDDQueryInput, Tensor, BallPrediction]:
    heatmap_model_kind(config)
    model: HeatmapModel
    if isinstance(config, MDDHeatmapHybridConfig):
        model = DeepMDDHeatmapHybridDetector(config, mdd_a=mdd_a, mdd_b=mdd_b)
    else:
        model = MDDDPTDetector(config, mdd_a=mdd_a, mdd_b=mdd_b)
    return bind_model_io(model, MDDHeatmapAdapter(config, subpixel=subpixel, candidate_config=candidate_config))


def heatmap_model_payload(model: HeatmapModel) -> dict[str, Any]:
    """Model weights only; training budgets, optimizer and resume state belong to training."""
    if not isinstance(model.config, MDDHeatmapConfig):
        raise BallModelIOError("Heatmap-only artifacts require an explicit temporal heatmap config")
    MDDHeatmapAdapter(model.config).validate_model_pair(model)
    return dict(schema=HEATMAP_MODEL_SCHEMA, model_kind=heatmap_model_kind(model.config),
                model_config=asdict(model.config), input_contract=model.mdd.input_contract(), state_dict=model.state_dict())


def restore_heatmap_model(saved: dict[str, Any]) -> HeatmapModel:
    """Restore the saved T exactly; never infer length or drop a presence head."""
    if saved.get("schema") != HEATMAP_MODEL_SCHEMA:
        raise BallModelIOError("Expected an explicit mdd_heatmap_model.v1 artifact")
    kind = saved.get("model_kind")
    config_type: type[MDDHeatmapConfig]
    if kind == HYBRID_HEATMAP_KIND:
        config_type = MDDHeatmapHybridConfig
    elif kind == DPT_HEATMAP_KIND:
        config_type = MDDHeatmapConfig
    else:
        raise BallModelIOError("Unknown heatmap-only model kind")
    raw = saved["model_config"]
    if not isinstance(raw, dict) or set(raw) != set(config_type.__dataclass_fields__):
        raise BallModelIOError("Heatmap artifact configuration must explicitly declare every field including frames")
    config = config_type(**raw)
    transform = input_transform_from_contract(saved["input_contract"], input_channels=config.input_channels, frames=config.frames)
    pair = build_mdd_heatmap_detector(config, mdd_a=transform.a, mdd_b=transform.b)
    model = pair.model
    if not isinstance(model, (MDDDPTDetector, DeepMDDHeatmapHybridDetector)):
        raise BallModelIOError("Invalid heatmap model type")
    model.load_state_dict(saved["state_dict"], strict=True)
    return model


def load_mdd_heatmap_detector(path: Path) -> BoundModelIO[MDDQueryInput, Tensor, BallPrediction]:
    model = restore_heatmap_model(torch.load(path, map_location="cpu", weights_only=True))
    if not isinstance(model.config, MDDHeatmapConfig):
        raise BallModelIOError("Expected temporal heatmap config")
    return bind_model_io(model.eval(), MDDHeatmapAdapter(model.config))
