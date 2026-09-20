"""Inference-only PLCS composition with exact architecture/coordinate checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.model_io.axial_reference import validate_axial_reference_checkpoint
from src.tasks.plcs.model_io.factory import PLCSBoundModelIO, build_plcs_model_io
from src.utils.schema.court_normalization import validate_court_coordinate_normalization


@dataclass(frozen=True)
class PLCSInferenceData:
    num_court_tokens: int
    adapter_camera_index: int
    values: Mapping[str, object]


@dataclass(frozen=True)
class PLCSInferenceConfig:
    model: PLCSModelConfig
    data: PLCSInferenceData
    court_keypoint_contract: CourtKeypointContract

    @classmethod
    def from_config(
        cls, config: object, contract: CourtKeypointContract
    ) -> PLCSInferenceConfig:
        root = as_config_mapping(config, path="checkpoint.config")
        model = PLCSModelConfig.from_mapping(
            require_config_mapping(root, "model", path="checkpoint.config")
        )
        raw = require_config_mapping(root, "data", path="checkpoint.config")
        count = cast(int, require_config_value(raw, "num_court_kp", int, path="data"))
        camera = cast(
            int, require_config_value(raw, "adapter_camera_index", int, path="data")
        )
        mode = cast(str, require_config_value(raw, "mode", str, path="data"))
        if count <= 0 or camera < 0:
            raise ValueError(
                "PLCS inference requires positive court count and nonnegative camera index"
            )
        if mode not in {"frame", "sequence", "multiview_sequence"}:
            raise ValueError(f"Invalid PLCS input mode {mode!r}")
        if (model.input_profile == "multiview") != (mode == "multiview_sequence"):
            raise ValueError("PLCS checkpoint model/data input profiles disagree")
        return cls(
            model,
            PLCSInferenceData(count, camera, MappingProxyType({"mode": mode})),
            contract,
        )


def load_plcs_pair(
    checkpoint: Mapping[str, Any], config: object, contract: CourtKeypointContract
) -> PLCSBoundModelIO:
    runtime = PLCSInferenceConfig.from_config(config, contract)
    validate_court_coordinate_normalization(checkpoint, artifact="PLCS checkpoint")
    validate_axial_reference_checkpoint(checkpoint, model_name=runtime.model.name)
    pair = build_plcs_model_io(runtime)
    weights = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not weights:
        raise ValueError("PLCS checkpoint contains no model state")
    pair.model.load_state_dict(weights, strict=True)
    return pair
