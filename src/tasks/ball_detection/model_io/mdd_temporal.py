"""Explicit temporal model and weight contracts for heatmaps and direct coordinates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeAlias, cast

from torch import Tensor

from src.tasks.ball_detection.model_io.mdd_coordinates import validate_rgb_timestamps
from src.tasks.ball_detection.model_io.mdd_heatmap import (
    HeatmapModel,
    build_mdd_heatmap_detector,
    heatmap_model_payload,
    restore_heatmap_model,
)
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDHeatmapConfig,
    MDDHeatmapHybridConfig,
    MDDTemporalQueryConfig,
)
from src.tasks.ball_detection.preprocessing.rgb_mdd import input_transform_from_contract

TemporalConfig: TypeAlias = MDDHeatmapConfig | MDDTemporalQueryConfig
TemporalModel: TypeAlias = HeatmapModel | DeepMDDQueryDetector
QUERY_SCHEMA = "mdd_temporal_coordinates.v1"


def build_temporal_model(
    config: TemporalConfig, *, mdd_a: float = 0.2, mdd_b: float = 0.15
) -> TemporalModel:
    if type(config) is MDDTemporalQueryConfig:
        return DeepMDDQueryDetector(config, mdd_a=mdd_a, mdd_b=mdd_b)
    return build_mdd_heatmap_detector(config, mdd_a=mdd_a, mdd_b=mdd_b).model


def temporal_forward(model: TemporalModel, rgb: Tensor, timestamps: Tensor) -> Tensor:
    validate_rgb_timestamps(model.config, rgb, timestamps)
    if (
        isinstance(model, DeepMDDQueryDetector)
        or type(model.config) is MDDHeatmapHybridConfig
    ):
        return model(rgb, timestamps)
    return model(rgb)


def temporal_model_payload(model: TemporalModel) -> dict[str, Any]:
    if isinstance(model, DeepMDDQueryDetector):
        if type(model.config) is not MDDTemporalQueryConfig:
            raise ValueError(
                "Temporal coordinates require their explicit variable-T config"
            )
        return dict(
            schema=QUERY_SCHEMA,
            model_kind="mdd_temporal_query",
            model_config=asdict(model.config),
            input_contract=model.mdd.input_contract(),
            state_dict=model.state_dict(),
        )
    return cast(dict[str, Any], heatmap_model_payload(model))


def restore_temporal_model(saved: dict[str, Any]) -> TemporalModel:
    if saved.get("schema") != QUERY_SCHEMA:
        return restore_heatmap_model(saved)
    if saved.get("model_kind") != "mdd_temporal_query" or set(
        saved["model_config"]
    ) != set(MDDTemporalQueryConfig.__dataclass_fields__):
        raise ValueError("Invalid explicit temporal coordinate artifact")
    config = MDDTemporalQueryConfig(**saved["model_config"])
    transform = input_transform_from_contract(
        saved["input_contract"], input_channels=5, frames=config.frames
    )
    model = build_temporal_model(config, mdd_a=transform.a, mdd_b=transform.b)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model
