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
    DeepMDDMHCQueryDetector,
    DeepMDDQueryDetector,
    MDDHeatmapConfig,
    MDDHeatmapHybridConfig,
    MDDMHCQueryConfig,
    MDDTemporalQueryConfig,
)
from src.tasks.ball_detection.preprocessing.rgb_mdd import input_transform_from_contract

TemporalConfig: TypeAlias = MDDHeatmapConfig | MDDTemporalQueryConfig
TemporalModel: TypeAlias = HeatmapModel | DeepMDDQueryDetector
QUERY_SCHEMA = "mdd_temporal_coordinates.v1"
MHC_QUERY_SCHEMA = "mdd_mhc_coordinates.v1"
QUERY_CONFIG_TYPES = (MDDTemporalQueryConfig, MDDMHCQueryConfig)


def build_temporal_model(
    config: TemporalConfig, *, mdd_a: float = 0.2, mdd_b: float = 0.15
) -> TemporalModel:
    if type(config) is MDDMHCQueryConfig:
        return DeepMDDMHCQueryDetector(config, mdd_a=mdd_a, mdd_b=mdd_b)
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
        mhc = type(model) is DeepMDDMHCQueryDetector and type(model.config) is MDDMHCQueryConfig
        legacy = type(model) is DeepMDDQueryDetector and type(model.config) is MDDTemporalQueryConfig
        if not (mhc or legacy):
            raise ValueError(
                "Temporal coordinates require their explicit variable-T config"
            )
        return dict(
            schema=MHC_QUERY_SCHEMA if mhc else QUERY_SCHEMA,
            model_kind="mdd_mhc_query" if mhc else "mdd_temporal_query",
            model_config=asdict(model.config),
            input_contract=model.mdd.input_contract(),
            state_dict=model.state_dict(),
        )
    return cast(dict[str, Any], heatmap_model_payload(model))


def restore_temporal_model(saved: dict[str, Any]) -> TemporalModel:
    kinds: dict[str, tuple[str, type[MDDTemporalQueryConfig]]] = {
        QUERY_SCHEMA: ("mdd_temporal_query", MDDTemporalQueryConfig),
        MHC_QUERY_SCHEMA: ("mdd_mhc_query", MDDMHCQueryConfig),
    }
    schema = saved.get("schema")
    if schema not in kinds:
        return restore_heatmap_model(saved)
    kind, config_type = kinds[schema]
    raw = saved.get("model_config")
    if saved.get("model_kind") != kind or not isinstance(raw, dict) or set(raw) != set(config_type.__dataclass_fields__):
        raise ValueError("Invalid explicit temporal coordinate artifact")
    config = config_type(**raw)
    transform = input_transform_from_contract(
        saved["input_contract"], input_channels=5, frames=config.frames
    )
    model = build_temporal_model(config, mdd_a=transform.a, mdd_b=transform.b)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model
