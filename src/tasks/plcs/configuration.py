"""Strict typed configuration and path contracts for PLCS runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal, TypeAlias, cast

from omegaconf import DictConfig, OmegaConf

import src.tasks.plcs.configuration_contracts as configuration_contracts
from src.tasks.base.configuration import (
    ChunkDataConfig,
    SceneVisualizationConfig,
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
    parse_view_3d,
)
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.hydra import register_boundary_validator
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES
from src.utils.paths import PROJECT_ROOT
from src.utils.rendering.camera_view import CameraController
from src.utils.schema.player import NUM_HUMAN_KP

PLCSValue: TypeAlias = (
    str | int | float | bool | None | tuple[object, ...] | Mapping[str, object]
)
PLCSFineTuneMode: TypeAlias = Literal[
    "all",
    "presence_head",
    "presence_competition",
]
PLCSPresenceCompetitionMode: TypeAlias = Literal["none", "deepsets"]


def _plain(value: object, *, path: str) -> Mapping[str, object]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    mapping: Mapping[str, object] = as_config_mapping(value, path=path)
    return mapping


def _exact(
    value: object,
    *,
    path: str,
    required: set[str] | frozenset[str],
    allowed: set[str] | frozenset[str],
) -> Mapping[str, object]:
    mapping = _plain(value, path=path)
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): {', '.join(f'{path}.{key}' for key in unknown)}."
        )
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): {', '.join(f'{path}.{key}' for key in missing)}."
        )
    return mapping


def _number(mapping: Mapping[str, object], key: str, *, path: str) -> float:
    value = require_config_value(mapping, key, (float, int), path=path)
    try:
        number = float(cast("float | int", value))
    except OverflowError as error:
        raise SemanticConfigurationError(f"{path}.{key} must be finite.") from error
    if not math.isfinite(number):
        raise SemanticConfigurationError(f"{path}.{key} must be finite.")
    return number


def _integer(mapping: Mapping[str, object], key: str, *, path: str) -> int:
    return cast("int", require_config_value(mapping, key, int, path=path))


def _boolean(mapping: Mapping[str, object], key: str, *, path: str) -> bool:
    return cast("bool", require_config_value(mapping, key, bool, path=path))


def _string(mapping: Mapping[str, object], key: str, *, path: str) -> str:
    return cast("str", require_config_value(mapping, key, str, path=path))


def _sequence(
    mapping: Mapping[str, object],
    key: str,
    *,
    path: str,
    item_types: tuple[type[object], ...],
    length: int | None = None,
    non_empty: bool = False,
) -> tuple[object, ...]:
    raw = require_config_value(mapping, key, (list, tuple), path=path)
    values = tuple(cast("Sequence[object]", raw))
    if length is not None and len(values) != length:
        raise ConfigurationTypeError(
            f"{path}.{key}: expected exactly {length} values, got {len(values)}."
        )
    if non_empty and not values:
        raise SemanticConfigurationError(f"{path}.{key} must not be empty.")
    if any(type(item) not in item_types for item in values):
        expected = " | ".join(candidate.__name__ for candidate in item_types)
        raise ConfigurationTypeError(
            f"{path}.{key}: every value must be exactly {expected}."
        )
    if item_types in {(float, int), (int, float)} and any(
        not math.isfinite(float(cast("float | int", item))) for item in values
    ):
        raise SemanticConfigurationError(f"{path}.{key} values must be finite.")
    return values


def _positive(number: float, *, path: str, allow_zero: bool = False) -> None:
    if number < 0.0 if allow_zero else number <= 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SemanticConfigurationError(f"{path} must be {qualifier}.")


def _probability(number: float, *, path: str) -> None:
    if not 0.0 <= number <= 1.0:
        raise SemanticConfigurationError(f"{path} must be within [0, 1].")


def _ordered_numeric_range(
    mapping: Mapping[str, object],
    key: str,
    *,
    path: str,
    positive: bool = False,
) -> tuple[float, float]:
    values = _sequence(mapping, key, path=path, item_types=(float, int), length=2)
    lo, hi = (float(cast("float | int", value)) for value in values)
    if lo > hi:
        raise SemanticConfigurationError(f"{path}.{key} must be ordered low-to-high.")
    if positive and lo <= 0.0:
        raise SemanticConfigurationError(f"{path}.{key} values must be positive.")
    return lo, hi


def _simple_name(value: str, *, path: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SemanticConfigurationError(f"{path} must be a non-empty file name.")
    return value


def _resolved_device(value: object, *, path: str, nullable: bool = False) -> str:
    expected: tuple[type[object], ...] = (str, type(None)) if nullable else (str,)
    if type(value) not in expected:
        names = " | ".join(candidate.__name__ for candidate in expected)
        raise ConfigurationTypeError(
            f"{path}: expected {names}, got {type(value).__name__}."
        )
    requested = "auto" if value is None else cast("str", value)
    if not requested.strip():
        raise SemanticConfigurationError(f"{path} must not be empty.")
    try:
        device = resolve_device(requested)
    except DeviceSelectionError as error:
        raise SemanticConfigurationError(
            f"{path} is not an available device: {requested!r}."
        ) from error
    return str(device)


_MODEL_COMMON = {
    "name",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "ffn_dim",
    "ffn_type",
    "dropout",
    "invisible_init_std",
}
_MODEL_FIELDS: dict[str, frozenset[str]] = {
    "plcs": frozenset(
        _MODEL_COMMON
        | {
            "io",
            "num_register_tokens",
            "use_kp_id_embedding",
            "use_rope",
            "rope_dim",
            "rope_theta",
            "rope_theta_time",
            "rope_theta_camera",
            "rope_theta_type",
            "predict_canonical_pose",
        }
    ),
    "plcs_multiview": frozenset(
        _MODEL_COMMON
        | {
            "io",
            "max_views",
            "max_seq_len",
            "rope_dim",
            "rope_theta",
            "rope_theta_time",
            "rope_theta_camera",
            "rope_theta_type",
            "predict_canonical_pose",
        }
    ),
    "plcs_multiview_axial": frozenset(
        _MODEL_COMMON
        | {
            "io",
            "max_views",
            "max_seq_len",
            "rope_dim",
            "rope_theta_time",
            "rope_theta_camera",
            "predict_canonical_pose",
            "canonical_pose_readout",
        }
    ),
    "plcs_multiview_axial_split": frozenset(
        _MODEL_COMMON
        | {
            "io",
            "max_views",
            "max_seq_len",
            "rope_dim",
            "rope_theta_time",
            "rope_theta_camera",
            "predict_canonical_pose",
            "canonical_pose_readout",
            "num_task_layers",
            "rot_num_task_layers",
            "pose_num_task_layers",
            "canonical_on_rotation_branch",
            "aux_position_on_rotation_branch",
            "detach_pose_branch",
        }
    ),
    "plcs_multiview_axial_camtoken": frozenset(
        _MODEL_COMMON
        | {
            "io",
            "max_views",
            "max_seq_len",
            "rope_dim",
            "rope_theta_time",
            "rope_theta_camera",
            "predict_canonical_pose",
            "canonical_pose_readout",
        }
    ),
    "plcs_track_query": frozenset(
        {
            "name",
            "hidden_dim",
            "num_heads",
            "ffn_dim",
            "num_queries",
            "num_stages",
            "num_joints",
            "rope_dim",
            "rope_theta",
            "ffn_type",
            "dropout",
            "predict_canonical_pose",
            "presence_competition",
            "role_rope_enabled",
            "invisible_init_std",
            "mhc",
            "cswa",
        }
    ),
    "plcs_track_query_ablation": frozenset(
        {
            "name",
            "hidden_dim",
            "num_heads",
            "ffn_dim",
            "num_queries",
            "num_stages",
            "num_joints",
            "rope_dim",
            "rope_theta",
            "ffn_type",
            "dropout",
            "predict_canonical_pose",
            "presence_competition",
            "role_rope_enabled",
            "invisible_init_std",
            "ffn_mode",
            "mhc_writeback",
            "mhc",
            "cswa",
        }
    ),
    "plcs_track_query_reference": frozenset(
        {
            "name",
            "hidden_dim",
            "num_heads",
            "ffn_dim",
            "num_queries",
            "num_stages",
            "num_joints",
            "rope_dim",
            "rope_theta",
            "ffn_type",
            "dropout",
            "predict_canonical_pose",
            "presence_competition",
            "invisible_init_std",
            "target_frame_contract",
            "track_query_rope_contract",
            "reference_selector_mode",
            "mhc",
            "cswa",
        }
    ),
    "plcs_track_query_reference_ablation": frozenset(
        {
            "name",
            "hidden_dim",
            "num_heads",
            "ffn_dim",
            "num_queries",
            "num_stages",
            "num_joints",
            "rope_dim",
            "rope_theta",
            "ffn_type",
            "dropout",
            "predict_canonical_pose",
            "presence_competition",
            "invisible_init_std",
            "target_frame_contract",
            "track_query_rope_contract",
            "reference_selector_mode",
            "ffn_mode",
            "mhc_writeback",
            "mhc",
            "cswa",
        }
    ),
}

_TRACK_QUERY_MODEL_NAMES = frozenset(
    {
        "plcs_track_query",
        "plcs_track_query_ablation",
        "plcs_track_query_reference",
        "plcs_track_query_reference_ablation",
    }
)
_TRACK_QUERY_OPTIONAL_MODEL_FIELDS = frozenset(
    {"predict_canonical_pose", "presence_competition"}
)
_REFERENCE_TRACK_QUERY_MODEL_NAMES = frozenset(
    {
        "plcs_track_query_reference",
        "plcs_track_query_reference_ablation",
    }
)
_TRACK_QUERY_ABLATION_MODEL_NAMES = frozenset(
    {
        "plcs_track_query_ablation",
        "plcs_track_query_reference_ablation",
    }
)


@dataclass(frozen=True, slots=True)
class PLCSTrackQueryMHCConfig:
    """Strict manifold-constrained hyper-connection configuration."""

    coefficient_dim: int
    sinkhorn_iters: int
    eps: float
    residual_identity_bias: float
    update_scale_init: float


@dataclass(frozen=True, slots=True)
class PLCSTrackQueryCSWAConfig:
    """Strict compressed sliding-window attention configuration."""

    compression_ratio: int
    window_radius: int
    backend: Literal["reference", "cuda"]


@dataclass(frozen=True, slots=True)
class PLCSModelConfig:
    """Exact model-variant mapping, validated once before model construction."""

    name: str
    input_profile: str | None
    values: Mapping[str, object]
    track_query_mhc: PLCSTrackQueryMHCConfig | None
    track_query_cswa: PLCSTrackQueryCSWAConfig | None
    track_query_presence_competition: PLCSPresenceCompetitionMode | None

    @classmethod
    def from_mapping(cls, value: object) -> PLCSModelConfig:
        initial = _plain(value, path="model")
        name = _string(initial, "name", path="model")
        try:
            fields = _MODEL_FIELDS[name]
        except KeyError as error:
            raise SemanticConfigurationError(
                f"model.name: unsupported PLCS model {name!r}."
            ) from error
        optional_fields = (
            _TRACK_QUERY_OPTIONAL_MODEL_FIELDS
            if name in _TRACK_QUERY_MODEL_NAMES
            else frozenset()
        )
        mapping = _exact(
            initial,
            path="model",
            required=fields - optional_fields,
            allowed=fields,
        )
        input_profile: str | None = None
        if "io" in mapping:
            io = _exact(
                mapping["io"],
                path="model.io",
                required={"input_profile"},
                allowed={"input_profile"},
            )
            input_profile = _string(io, "input_profile", path="model.io")
            if input_profile not in {"frame", "multiview"}:
                raise SemanticConfigurationError(
                    "model.io.input_profile must be 'frame' or 'multiview'."
                )
        expected_profile = {
            "plcs": "frame",
            "plcs_multiview": "multiview",
            "plcs_multiview_axial": "multiview",
            "plcs_multiview_axial_split": "multiview",
            "plcs_multiview_axial_camtoken": "multiview",
            "plcs_track_query": None,
            "plcs_track_query_ablation": None,
            "plcs_track_query_reference": None,
            "plcs_track_query_reference_ablation": None,
        }[name]
        if input_profile != expected_profile:
            raise SemanticConfigurationError(
                f"model.name={name!r} requires model.io.input_profile="
                f"{expected_profile!r}."
            )
        hidden = _integer(mapping, "hidden_dim", path="model")
        heads = _integer(mapping, "num_heads", path="model")
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise SemanticConfigurationError(
                "model.hidden_dim and model.num_heads must be positive and divisible."
            )
        dropout = _number(mapping, "dropout", path="model")
        if not 0.0 <= dropout < 1.0:
            raise SemanticConfigurationError("model.dropout must be within [0, 1).")
        positive_integer_fields = {
            "ffn_dim",
            "max_views",
            "max_seq_len",
            "num_task_layers",
            "rot_num_task_layers",
            "pose_num_task_layers",
            "num_queries",
            "num_stages",
            "num_joints",
        }
        for key in positive_integer_fields & set(mapping):
            if _integer(mapping, key, path="model") <= 0:
                raise SemanticConfigurationError(f"model.{key} must be positive.")
        for key in {"num_layers", "num_register_tokens"} & set(mapping):
            if _integer(mapping, key, path="model") < 0:
                raise SemanticConfigurationError(f"model.{key} must be non-negative.")
        rope_dim = _integer(mapping, "rope_dim", path="model")
        head_dim = hidden // heads
        if rope_dim < 0 or rope_dim % 2 or rope_dim > head_dim:
            raise SemanticConfigurationError(
                "model.rope_dim must be non-negative, even, and no larger than "
                f"the attention head dimension ({head_dim})."
            )
        number_fields = {
            "rope_theta",
            "rope_theta_time",
            "rope_theta_camera",
            "rope_theta_type",
            "invisible_init_std",
        }
        for key in number_fields & set(mapping):
            number = _number(mapping, key, path="model")
            if key == "invisible_init_std":
                _positive(number, path=f"model.{key}", allow_zero=True)
            else:
                _positive(number, path=f"model.{key}")
        for key in {
            "use_kp_id_embedding",
            "use_rope",
            "predict_canonical_pose",
            "canonical_on_rotation_branch",
            "aux_position_on_rotation_branch",
            "detach_pose_branch",
            "role_rope_enabled",
        } & set(mapping):
            _boolean(mapping, key, path="model")
        if "ffn_type" in mapping:
            ffn_type = _string(mapping, "ffn_type", path="model")
            if ffn_type not in SUPPORTED_FFN_TYPES:
                raise SemanticConfigurationError(
                    "model.ffn_type must be one of "
                    f"{sorted(SUPPORTED_FFN_TYPES)!r}."
                )
        if "canonical_pose_readout" in mapping:
            canonical_pose_readout = _string(
                mapping, "canonical_pose_readout", path="model"
            )
            if canonical_pose_readout not in {"direct", "temporal_decomposition"}:
                raise SemanticConfigurationError(
                    "model.canonical_pose_readout must be 'direct' or "
                    "'temporal_decomposition'."
                )
            if canonical_pose_readout == "temporal_decomposition" and not _boolean(
                mapping, "predict_canonical_pose", path="model"
            ):
                raise SemanticConfigurationError(
                    "model.canonical_pose_readout='temporal_decomposition' requires "
                    "model.predict_canonical_pose=true."
                )
        if (
            "num_joints" in mapping
            and _integer(mapping, "num_joints", path="model") != NUM_HUMAN_KP
        ):
            raise SemanticConfigurationError(
                f"model.num_joints must equal the canonical COCO joint count ({NUM_HUMAN_KP})."
            )
        track_query_mhc: PLCSTrackQueryMHCConfig | None = None
        track_query_cswa: PLCSTrackQueryCSWAConfig | None = None
        track_query_presence_competition: PLCSPresenceCompetitionMode | None = None
        if name in _TRACK_QUERY_MODEL_NAMES:
            presence_competition_value = (
                _string(mapping, "presence_competition", path="model")
                if "presence_competition" in mapping
                else "none"
            )
            if presence_competition_value not in {"none", "deepsets"}:
                raise SemanticConfigurationError(
                    "model.presence_competition must be one of 'none', "
                    f"'deepsets'; got {presence_competition_value!r}."
                )
            track_query_presence_competition = cast(
                "PLCSPresenceCompetitionMode",
                presence_competition_value,
            )
            num_stages = _integer(mapping, "num_stages", path="model")
            if num_stages % 4 != 0:
                raise SemanticConfigurationError(
                    "model.num_stages must be a positive multiple of 4."
                )

            raw_mhc = _exact(
                mapping["mhc"],
                path="model.mhc",
                required={
                    "coefficient_dim",
                    "sinkhorn_iters",
                    "eps",
                    "residual_identity_bias",
                    "update_scale_init",
                },
                allowed={
                    "coefficient_dim",
                    "sinkhorn_iters",
                    "eps",
                    "residual_identity_bias",
                    "update_scale_init",
                },
            )
            track_query_mhc = PLCSTrackQueryMHCConfig(
                coefficient_dim=_integer(raw_mhc, "coefficient_dim", path="model.mhc"),
                sinkhorn_iters=_integer(raw_mhc, "sinkhorn_iters", path="model.mhc"),
                eps=_number(raw_mhc, "eps", path="model.mhc"),
                residual_identity_bias=_number(
                    raw_mhc, "residual_identity_bias", path="model.mhc"
                ),
                update_scale_init=_number(
                    raw_mhc, "update_scale_init", path="model.mhc"
                ),
            )
            if (
                track_query_mhc.coefficient_dim <= 0
                or track_query_mhc.sinkhorn_iters <= 0
            ):
                raise SemanticConfigurationError(
                    "model.mhc.coefficient_dim and model.mhc.sinkhorn_iters "
                    "must be positive."
                )
            _positive(track_query_mhc.eps, path="model.mhc.eps")
            _positive(
                track_query_mhc.residual_identity_bias,
                path="model.mhc.residual_identity_bias",
                allow_zero=True,
            )

            raw_cswa = _exact(
                mapping["cswa"],
                path="model.cswa",
                required={"compression_ratio", "window_radius", "backend"},
                allowed={"compression_ratio", "window_radius", "backend"},
            )
            backend = _string(raw_cswa, "backend", path="model.cswa")
            if backend not in {"reference", "cuda"}:
                raise SemanticConfigurationError(
                    "model.cswa.backend must be 'reference' or 'cuda'."
                )
            track_query_cswa = PLCSTrackQueryCSWAConfig(
                compression_ratio=_integer(
                    raw_cswa, "compression_ratio", path="model.cswa"
                ),
                window_radius=_integer(raw_cswa, "window_radius", path="model.cswa"),
                backend=cast("Literal['reference', 'cuda']", backend),
            )
            if track_query_cswa.compression_ratio < 2:
                raise SemanticConfigurationError(
                    "model.cswa.compression_ratio must be at least 2."
                )
            if track_query_cswa.window_radius < 0:
                raise SemanticConfigurationError(
                    "model.cswa.window_radius must be non-negative."
                )
            if name in _REFERENCE_TRACK_QUERY_MODEL_NAMES:
                target_frame_contract = _string(
                    mapping, "target_frame_contract", path="model"
                )
                if target_frame_contract != "reference_camera_court_rzpi_v1":
                    raise SemanticConfigurationError(
                        "model.target_frame_contract must be "
                        "'reference_camera_court_rzpi_v1' for reference track-query models."
                    )
                track_query_rope_contract = _string(
                    mapping, "track_query_rope_contract", path="model"
                )
                if (
                    track_query_rope_contract
                    != "time_camera_reference_selector_v1"
                ):
                    raise SemanticConfigurationError(
                        "model.track_query_rope_contract must be "
                        "'time_camera_reference_selector_v1' for reference "
                        "track-query models."
                    )
                selector_mode = _string(
                    mapping, "reference_selector_mode", path="model"
                )
                if selector_mode not in {"reference", "selector_zero"}:
                    raise SemanticConfigurationError(
                        "model.reference_selector_mode must be 'reference' or "
                        "'selector_zero'."
                    )
                if rope_dim < 6:
                    raise SemanticConfigurationError(
                        "Reference track-query model.rope_dim must be at least 6 "
                        "so every spatial axis receives a rotary pair."
                    )
                if (
                    name == "plcs_track_query_reference"
                    and selector_mode != "reference"
                ):
                    raise SemanticConfigurationError(
                        "The normal reference track-query model requires "
                        "model.reference_selector_mode='reference'."
                    )
            if name in _TRACK_QUERY_ABLATION_MODEL_NAMES:
                if _string(mapping, "ffn_mode", path="model") not in {
                    "per_attention",
                    "shared",
                }:
                    raise SemanticConfigurationError(
                        "model.ffn_mode must be 'per_attention' or 'shared'."
                    )
                if _string(mapping, "mhc_writeback", path="model") not in {
                    "after_object_temporal",
                    "layer_end",
                }:
                    raise SemanticConfigurationError(
                        "model.mhc_writeback must be 'after_object_temporal' or "
                        "'layer_end'."
                    )
        return cls(
            name=name,
            input_profile=input_profile,
            values=MappingProxyType(dict(mapping)),
            track_query_mhc=track_query_mhc,
            track_query_cswa=track_query_cswa,
            track_query_presence_competition=track_query_presence_competition,
        )

    def integer(self, key: str) -> int:
        return _integer(self.values, key, path="model")

    def number(self, key: str) -> float:
        return _number(self.values, key, path="model")

    def boolean(self, key: str) -> bool:
        return _boolean(self.values, key, path="model")

    def string(self, key: str) -> str:
        return _string(self.values, key, path="model")


_AUGMENTATION_BLOCK_FIELDS: dict[str, frozenset[str]] = {
    "uv_scale": frozenset(
        {"enabled", "prob", "scale_range", "apply_to_human", "apply_to_court"}
    ),
    "gaussian_noise": frozenset({"enabled", "prob", "human_std", "court_std"}),
    "visibility_dropout": frozenset(
        {"enabled", "prob", "human_drop_prob", "court_drop_prob"}
    ),
    "temporal_jitter": frozenset(
        {
            "enabled",
            "prob",
            "human_jitter_std",
            "human_drift_std",
            "court_jitter_std",
            "court_drift_std",
            "drift_decay",
        }
    ),
    "burst_dropout": frozenset(
        {
            "enabled",
            "prob",
            "human_track_prob",
            "court_track_prob",
            "min_len",
            "max_len",
            "max_bursts",
        }
    ),
    "false_positive": frozenset(
        {
            "enabled",
            "prob",
            "human_prob_absent",
            "human_prob_after_dropout",
            "human_after_dropout_window",
            "court_prob_absent",
            "court_prob_after_dropout",
            "court_after_dropout_window",
        }
    ),
    "edge_degradation": frozenset(
        {
            "enabled",
            "prob",
            "edge_margin",
            "human_noise_std",
            "human_drop_prob",
            "human_clip_out_prob",
            "court_noise_std",
            "court_drop_prob",
            "court_clip_out_prob",
        }
    ),
    "speed_conditioned": frozenset(
        {
            "enabled",
            "prob",
            "human_frame_prob",
            "human_speed_threshold",
            "human_lag_overshoot_range",
            "human_noise_std",
            "court_frame_prob",
            "court_speed_threshold",
            "court_lag_overshoot_range",
            "court_noise_std",
        }
    ),
}


def validate_augmentation(value: object) -> Mapping[str, object]:
    fields = frozenset({"enabled", *_AUGMENTATION_BLOCK_FIELDS})
    augmentation = _exact(
        value, path="data.augmentation", required=fields, allowed=fields
    )
    _boolean(augmentation, "enabled", path="data.augmentation")
    for block_name, block_fields in _AUGMENTATION_BLOCK_FIELDS.items():
        block = _exact(
            augmentation[block_name],
            path=f"data.augmentation.{block_name}",
            required=block_fields,
            allowed=block_fields,
        )
        probability = _number(block, "prob", path=f"data.augmentation.{block_name}")
        if not 0.0 <= probability <= 1.0:
            raise SemanticConfigurationError(
                f"data.augmentation.{block_name}.prob must be within [0, 1]."
            )
        for key, item in block.items():
            item_path = f"data.augmentation.{block_name}"
            if key in {"enabled", "apply_to_human", "apply_to_court"}:
                _boolean(block, key, path=item_path)
            elif key in {
                "min_len",
                "max_len",
                "max_bursts",
                "human_after_dropout_window",
                "court_after_dropout_window",
            }:
                _integer(block, key, path=item_path)
            elif key.endswith("_range"):
                lo, _hi = _ordered_numeric_range(
                    block,
                    key,
                    path=item_path,
                    positive=key == "scale_range",
                )
                del lo
            elif key != "prob":
                number = _number(block, key, path=item_path)
                if "prob" in key or key == "drift_decay":
                    _probability(number, path=f"{item_path}.{key}")
                else:
                    _positive(number, path=f"{item_path}.{key}", allow_zero=True)
                del item
        if block_name == "burst_dropout":
            min_len = _integer(block, "min_len", path=item_path)
            max_len = _integer(block, "max_len", path=item_path)
            max_bursts = _integer(block, "max_bursts", path=item_path)
            if min_len <= 0 or max_len < min_len or max_bursts <= 0:
                raise SemanticConfigurationError(
                    f"{item_path} lengths/count must define a positive ordered range."
                )
        if block_name == "false_positive":
            for key in {"human_after_dropout_window", "court_after_dropout_window"}:
                if _integer(block, key, path=item_path) < 0:
                    raise SemanticConfigurationError(
                        f"{item_path}.{key} must be non-negative."
                    )
        if block_name == "edge_degradation":
            edge_margin = _number(block, "edge_margin", path=item_path)
            if edge_margin > 0.5:
                raise SemanticConfigurationError(
                    f"{item_path}.edge_margin must be within [0, 0.5]."
                )
    return MappingProxyType(dict(augmentation))


_DATA_COMMON = {
    "backend",
    "scene_dir",
    "batch_size",
    "num_workers",
    "pin_memory",
    "camera_mode",
    "num_views_range",
    "seq_len_range",
    "augmentation",
    "adapter_camera_index",
    "min_cameras",
    "evaluation_reference_camera_id",
}


@dataclass(frozen=True, slots=True)
class PLCSDataConfig:
    backend: str
    scene_dir: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    input_profile: str | None
    adapter_camera_index: int
    num_court_tokens: int | None
    evaluation_reference_camera_id: str | None
    values: Mapping[str, object]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver, model: PLCSModelConfig
    ) -> PLCSDataConfig:
        initial = _plain(value, path="data")
        backend = _string(initial, "backend", path="data")
        tracking = model.name in _TRACK_QUERY_MODEL_NAMES
        allowed = set(_DATA_COMMON)
        if tracking:
            allowed.add("lifecycle")
        else:
            allowed.update({"mode", "num_court_kp"})
            if model.input_profile == "multiview":
                allowed.add("min_cameras")
            configured_mode = _string(initial, "mode", path="data")
            if model.input_profile == "frame" and configured_mode == "frame":
                allowed.discard("seq_stride")
            else:
                allowed.add("seq_stride")
        if backend == "chunked":
            allowed.update({"generator_device", "chunk"})
        elif backend != "default":
            raise SemanticConfigurationError(
                "data.backend must be 'default' or 'chunked'."
            )
        mapping = _exact(
            initial,
            path="data",
            required=(
                allowed
                - {
                    "seq_stride",
                    "min_cameras",
                    "evaluation_reference_camera_id",
                }
                | (
                    {"evaluation_reference_camera_id"}
                    if model.name in _REFERENCE_TRACK_QUERY_MODEL_NAMES
                    else set()
                )
            ),
            allowed=allowed,
        )
        if backend == "chunked":
            chunk = ChunkDataConfig.from_validated_task_mapping(
                mapping,
                resolver=resolver,
            )
            generator_device = _resolved_device(
                chunk.generator_device,
                path="data.generator_device",
            )
            if generator_device != "cpu":
                raise SemanticConfigurationError(
                    "data.generator_device must resolve to 'cpu' for parallel "
                    "PLCS chunk generation."
                )
        augmentation = validate_augmentation(mapping["augmentation"])
        num_views_values = _sequence(
            mapping,
            "num_views_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        seq_len_values = _sequence(
            mapping,
            "seq_len_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        camera_mode = require_config_value(
            mapping, "camera_mode", (str, int), path="data"
        )
        if isinstance(camera_mode, str):
            if camera_mode not in {"random", "first"}:
                raise SemanticConfigurationError(
                    "data.camera_mode must be 'random', 'first', or a non-negative index."
                )
        elif cast("int", camera_mode) < 0:
            raise SemanticConfigurationError(
                "data.camera_mode camera index must be non-negative."
            )
        num_views_range = tuple(cast("int", value) for value in num_views_values)
        seq_len_range = tuple(cast("int", value) for value in seq_len_values)
        for key, range_value in (
            ("num_views_range", num_views_range),
            ("seq_len_range", seq_len_range),
        ):
            if range_value[0] <= 0 or range_value[1] < range_value[0]:
                raise SemanticConfigurationError(
                    f"data.{key} must be a positive ordered range."
                )
        if "max_views" in model.values and num_views_range[1] > model.integer(
            "max_views"
        ):
            raise SemanticConfigurationError(
                "data.num_views_range cannot exceed model.max_views."
            )
        if "max_seq_len" in model.values and seq_len_range[1] > model.integer(
            "max_seq_len"
        ):
            raise SemanticConfigurationError(
                "data.seq_len_range cannot exceed model.max_seq_len."
            )
        if "min_cameras" in mapping:
            min_cameras = _integer(mapping, "min_cameras", path="data")
            if min_cameras <= 0 or min_cameras > num_views_range[1]:
                raise SemanticConfigurationError(
                    "data.min_cameras must be within data.num_views_range capacity."
                )
        if (
            "seq_stride" in mapping
            and _integer(mapping, "seq_stride", path="data") <= 0
        ):
            raise SemanticConfigurationError("data.seq_stride must be positive.")
        if not tracking:
            mode = _string(mapping, "mode", path="data")
            allowed_modes = (
                {"multiview_sequence"}
                if model.input_profile == "multiview"
                else {"frame", "sequence"}
            )
            if mode not in allowed_modes:
                raise SemanticConfigurationError(
                    f"data.mode={mode!r} is incompatible with "
                    f"model.io.input_profile={model.input_profile!r}."
                )
        if tracking:
            lifecycle_fields = {
                "pack_to_query_slots",
                "min_reuse_gap_frames",
                "randomize_slots_train",
            }
            lifecycle = _exact(
                mapping["lifecycle"],
                path="data.lifecycle",
                required=lifecycle_fields,
                allowed=lifecycle_fields,
            )
            if not _boolean(lifecycle, "pack_to_query_slots", path="data.lifecycle"):
                raise SemanticConfigurationError(
                    "data.lifecycle.pack_to_query_slots must be true for fixed-Q "
                    "PLCS tracking."
                )
            _integer(lifecycle, "min_reuse_gap_frames", path="data.lifecycle")
            _boolean(lifecycle, "randomize_slots_train", path="data.lifecycle")
            if _integer(lifecycle, "min_reuse_gap_frames", path="data.lifecycle") < 0:
                raise SemanticConfigurationError(
                    "data.lifecycle.min_reuse_gap_frames must be non-negative."
                )
        scene_dir = _string(mapping, "scene_dir", path="data")
        batch_size = _integer(mapping, "batch_size", path="data")
        workers = _integer(mapping, "num_workers", path="data")
        adapter = _integer(mapping, "adapter_camera_index", path="data")
        if batch_size <= 0 or workers < 0 or adapter < 0:
            raise SemanticConfigurationError(
                "data.batch_size must be positive; num_workers and adapter_camera_index must be non-negative."
            )
        num_court_tokens: int | None = None
        if not tracking:
            num_court_tokens = _integer(mapping, "num_court_kp", path="data")
            if num_court_tokens <= 0:
                raise SemanticConfigurationError("data.num_court_kp must be positive.")
        evaluation_reference_camera_id = (
            _string(mapping, "evaluation_reference_camera_id", path="data")
            if "evaluation_reference_camera_id" in mapping
            else None
        )
        if (
            evaluation_reference_camera_id is not None
            and not evaluation_reference_camera_id.strip()
        ):
            raise SemanticConfigurationError(
                "data.evaluation_reference_camera_id must be a non-empty stable "
                "camera identity."
            )
        resolved = dict(mapping)
        resolved["augmentation"] = augmentation
        return cls(
            backend=backend,
            scene_dir=resolver.resolve(PathRole.DATA, scene_dir),
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=_boolean(mapping, "pin_memory", path="data"),
            input_profile=model.input_profile,
            adapter_camera_index=adapter,
            num_court_tokens=num_court_tokens,
            evaluation_reference_camera_id=evaluation_reference_camera_id,
            values=MappingProxyType(resolved),
        )


@dataclass(frozen=True, slots=True)
class PLCSTrainingConfig:
    """Complete typed PLCS training boundary."""

    shared: TrainingRuntimeConfig
    court_keypoint_contract: CourtKeypointContract
    paths: configuration_contracts.PLCSPathConfig
    model: PLCSModelConfig
    data: PLCSDataConfig
    fine_tune_mode: PLCSFineTuneMode
    tracking_reprojection_enabled: bool
    tracking_metrics: TrackingMetricConfig | None
    qualitative_style: SceneStyleConfig
    qualitative_view_3d: CameraController
    qualitative_fps: float
    raw: DictConfig

    @classmethod
    def from_config(cls, value: object) -> PLCSTrainingConfig:
        if not isinstance(value, DictConfig):
            raise ConfigurationTypeError("PLCS training boundary requires DictConfig.")
        root = _exact(
            value,
            path="configuration",
            required={
                "court_keypoints",
                "model",
                "data",
                "training",
                "loss",
                "run",
                "paths",
                "external_assets",
                "qualitative",
            },
            allowed={
                "court_keypoints",
                "model",
                "data",
                "training",
                "loss",
                "metrics",
                "run",
                "paths",
                "external_assets",
                "qualitative",
                "tracking_metrics",
                "camera",
                "motion_sources",
                "generation",
                "simulation",
            },
        )
        paths = configuration_contracts.PLCSPathConfig.from_config(value)
        model = PLCSModelConfig.from_mapping(
            require_config_mapping(root, "model", path="configuration")
        )
        court_keypoint_contract = (
            PLCSCourtKeypointRuntimeConfig.from_config(value).contract
        )
        if model.name in _REFERENCE_TRACK_QUERY_MODEL_NAMES:
            if court_keypoint_contract.selector != "camera_view_v2":
                raise SemanticConfigurationError(
                    "Reference track-query models require "
                    "court_keypoints.selector='camera_view_v2'."
                )
        elif (
            model.name in _TRACK_QUERY_MODEL_NAMES
            and court_keypoint_contract.selector != "physical_v1"
        ):
            raise SemanticConfigurationError(
                "Legacy track-query models require "
                "court_keypoints.selector='physical_v1'; select an explicit "
                "reference model for camera_view_v2."
            )
        data = PLCSDataConfig.from_mapping(
            require_config_mapping(root, "data", path="configuration"),
            resolver=paths.resolver,
            model=model,
        )
        exact_root_fields = {
            "court_keypoints",
            "model",
            "data",
            "training",
            "loss",
            "run",
            "paths",
            "external_assets",
            "qualitative",
            "tracking_metrics"
            if model.name in _TRACK_QUERY_MODEL_NAMES
            else "metrics",
        }
        if data.backend == "chunked":
            exact_root_fields.update({"camera", "motion_sources", "generation"})
        _exact(
            root,
            path="configuration",
            required=exact_root_fields,
            allowed=exact_root_fields,
        )
        training_fields = {
            "trainer",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "warmup_epochs",
            "min_lr",
            "steps_per_epoch",
            "optimizer",
            "compile",
            "matmul_precision",
            "allow_tf32",
            "checkpoint",
            "early_stopping",
            "lr_monitor",
            "qualitative_logging",
            "gan",
            "mcmc",
        }
        training_optional_fields = {"fine_tune_mode"}
        training_mapping = _exact(
            require_config_mapping(root, "training", path="configuration"),
            path="training",
            required=training_fields,
            allowed=training_fields | training_optional_fields,
        )
        fine_tune_mode_value = (
            _string(training_mapping, "fine_tune_mode", path="training")
            if "fine_tune_mode" in training_mapping
            else "all"
        )
        if fine_tune_mode_value not in {
            "all",
            "presence_head",
            "presence_competition",
        }:
            raise SemanticConfigurationError(
                "training.fine_tune_mode must be one of 'all', 'presence_head', "
                "'presence_competition'; "
                f"got {fine_tune_mode_value!r}."
            )
        fine_tune_mode = cast("PLCSFineTuneMode", fine_tune_mode_value)
        if (
            fine_tune_mode == "presence_head"
            and model.name not in _TRACK_QUERY_MODEL_NAMES
        ):
            raise SemanticConfigurationError(
                "training.fine_tune_mode='presence_head' requires a PLCS "
                "track-query model with an independent presence_head."
            )
        if (
            fine_tune_mode == "presence_competition"
            and model.name not in _TRACK_QUERY_MODEL_NAMES
        ):
            raise SemanticConfigurationError(
                "training.fine_tune_mode='presence_competition' requires a "
                "PLCS track-query model."
            )
        if (
            fine_tune_mode == "presence_competition"
            and model.track_query_presence_competition != "deepsets"
        ):
            raise SemanticConfigurationError(
                "training.fine_tune_mode='presence_competition' requires "
                "model.presence_competition='deepsets'."
            )
        gan_fields = {
            "enabled",
            "target_weight",
            "warmup_epochs",
            "generator_gradient_clip_val",
            "discriminator_gradient_clip_val",
            "transition",
        }
        gan_mapping = _exact(
            require_config_mapping(training_mapping, "gan", path="training"),
            path="training.gan",
            required=gan_fields,
            allowed=gan_fields | {"discriminator"},
        )
        run_fields = {
            "output_dir",
            "seed",
            "gpus",
            "resume",
            "init_weights",
            "fast_dev_run",
            "dry_run",
            "test_after_fit",
        }
        _exact(
            require_config_mapping(root, "run", path="configuration"),
            path="run",
            required=run_fields,
            allowed=run_fields,
        )
        shared = TrainingRuntimeConfig.from_config(value, repository_root=PROJECT_ROOT)
        if fine_tune_mode in {"presence_head", "presence_competition"}:
            if shared.run.resume is not None:
                raise SemanticConfigurationError(
                    f"training.fine_tune_mode={fine_tune_mode!r} forbids "
                    "run.resume; "
                    "use run.init_weights with a fresh optimizer."
                )
            if shared.run.init_weights is None:
                raise SemanticConfigurationError(
                    f"training.fine_tune_mode={fine_tune_mode!r} requires "
                    "run.init_weights."
                )
        external_assets = _exact(
            require_config_mapping(root, "external_assets", path="configuration"),
            path="external_assets",
            required={"smplh_model_path"},
            allowed={"smplh_model_path"},
        )
        paths.resolver.resolve(
            PathRole.EXTERNAL_ASSET,
            _string(external_assets, "smplh_model_path", path="external_assets"),
        )
        if data.backend == "chunked":
            generation_components = (
                configuration_contracts.PLCSGenerationComponents.from_config(root)
            )
            generation_mode = generation_components.mode
            if model.name in _TRACK_QUERY_MODEL_NAMES:
                if generation_mode != "multi_object":
                    raise SemanticConfigurationError(
                        "Chunked PLCS tracking requires generation.mode='multi_object'."
                    )
                generation = require_config_mapping(
                    root, "generation", path="configuration"
                )
                timeline = require_config_mapping(
                    generation, "timeline", path="generation"
                )
                if _integer(
                    timeline, "max_concurrent", path="generation.timeline"
                ) > model.integer("num_queries"):
                    raise SemanticConfigurationError(
                        "generation.timeline.max_concurrent cannot exceed "
                        "model.num_queries."
                    )
                data_mapping = require_config_mapping(
                    root, "data", path="configuration"
                )
                lifecycle = require_config_mapping(
                    data_mapping, "lifecycle", path="data"
                )
                if _integer(
                    timeline, "min_reuse_gap_frames", path="generation.timeline"
                ) < _integer(lifecycle, "min_reuse_gap_frames", path="data.lifecycle"):
                    raise SemanticConfigurationError(
                        "generation.timeline.min_reuse_gap_frames cannot be smaller "
                        "than data.lifecycle.min_reuse_gap_frames."
                    )
            elif generation_mode != "single_object":
                raise SemanticConfigurationError(
                    "Chunked non-tracking PLCS training requires "
                    "generation.mode='single_object'."
                )
        qualitative = _exact(
            require_config_mapping(root, "qualitative", path="configuration"),
            path="qualitative",
            required={"fps", "style", "view_3d"},
            allowed={"fps", "style", "view_3d"},
        )
        qualitative_fps = _number(qualitative, "fps", path="qualitative")
        if qualitative_fps <= 0.0:
            raise SemanticConfigurationError("qualitative.fps must be positive.")
        qualitative_style = parse_scene_style(qualitative["style"])
        qualitative_view_3d = parse_view_3d(qualitative["view_3d"])
        from src.tasks.plcs.training.mcmc import MCMCConfig

        MCMCConfig.from_dict(
            dict(require_config_mapping(training_mapping, "mcmc", path="training"))
        )
        tracking_metric_config: TrackingMetricConfig | None = None
        tracking_reprojection_enabled = False
        if model.name not in _TRACK_QUERY_MODEL_NAMES:
            from src.tasks.plcs.training.losses import PLCSLossConfig

            PLCSLossConfig.from_dict(
                dict(require_config_mapping(root, "loss", path="configuration"))
            )
            metrics_fields = {
                "position_threshold_m",
                "angle_threshold_deg",
                "velocity_threshold_m",
            }
            metrics = _exact(
                require_config_mapping(root, "metrics", path="configuration"),
                path="metrics",
                required=metrics_fields,
                allowed=metrics_fields,
            )
            for key in metrics_fields:
                _positive(_number(metrics, key, path="metrics"), path=f"metrics.{key}")
        else:
            tracking_metric_config = TrackingMetricConfig.from_mapping(
                require_config_mapping(root, "tracking_metrics", path="configuration")
            )
            tracking_loss_fields = {
                "position_weight",
                "rotation_weight",
                "presence_weight",
                "presence_inactive_weight",
                "presence_active_weight",
                "presence_transition_weight",
                "transition_radius",
                "track_smoothness_weight",
                "match_position_weight",
                "match_rotation_weight",
                "match_presence_weight",
            }
            tracking_optional_loss_fields = {
                "match_presence_inactive_weight",
                "cardinality_weight",
                "cardinality_nll_weight",
                "presence_hard_negative_weight",
                "presence_hard_negative_gamma",
                "presence_pairwise_weight",
                "presence_pairwise_margin",
            }
            tracking_allowed_loss_fields = (
                tracking_loss_fields | tracking_optional_loss_fields
            )
            tracking_pose_loss_fields = {
                "position_smooth_l1_beta",
                "angle_weight",
                "position_smoothness_weight",
                "canonical_pose_weight",
                "canonical_pose_smooth_l1_beta",
                "reprojection_weight",
                "reprojection_smooth_l1_beta",
                "joint_angle_weight",
                "torsion_angle_weight",
                "torso_twist_weight",
                "bone_length_weight",
                "joint_angle_velocity_weight",
                "torsion_angle_velocity_weight",
                "torso_twist_velocity_weight",
                "joint_angle_velocity_angle_weights",
                "torsion_angle_velocity_angle_weights",
            }
            tracking_loss = _exact(
                require_config_mapping(root, "loss", path="configuration"),
                path="loss",
                required=tracking_loss_fields,
                allowed=tracking_allowed_loss_fields | tracking_pose_loss_fields,
            )
            present_pose_fields = tracking_pose_loss_fields & set(tracking_loss)
            if present_pose_fields != tracking_pose_loss_fields and present_pose_fields:
                missing_pose_fields = sorted(
                    tracking_pose_loss_fields - present_pose_fields
                )
                raise MissingConfigurationKeyError(
                    "Tracking pose supervision requires the complete standard "
                    "PLCS loss contract; missing configuration key(s): "
                    + ", ".join(f"loss.{key}" for key in missing_pose_fields)
                    + "."
                )
            # ``match_presence_inactive_weight`` was introduced after tracking
            # checkpoints already existed. An otherwise complete v1 mapping is
            # the sole supported legacy shape; all other missing/unknown fields
            # remain rejected by the exact contracts above.
            validated_weight_fields = tracking_allowed_loss_fields & set(tracking_loss)
            for key in validated_weight_fields - {"transition_radius"}:
                _positive(
                    _number(tracking_loss, key, path="loss"),
                    path=f"loss.{key}",
                    allow_zero=True,
                )
            if _integer(tracking_loss, "transition_radius", path="loss") < 0:
                raise SemanticConfigurationError(
                    "loss.transition_radius must be non-negative."
                )
            if present_pose_fields:
                from src.tasks.plcs.training.losses import PLCSLossConfig

                standard_loss_fields = tracking_pose_loss_fields | {
                    "position_weight",
                    "rotation_weight",
                }
                PLCSLossConfig.from_dict(
                    {key: tracking_loss[key] for key in standard_loss_fields}
                )
                tracking_reprojection_enabled = (
                    _number(tracking_loss, "reprojection_weight", path="loss") > 0.0
                )
                canonical_pose_required = (
                    _number(tracking_loss, "canonical_pose_weight", path="loss")
                    > 0.0
                    or tracking_reprojection_enabled
                )
                if canonical_pose_required and not bool(
                    model.values.get("predict_canonical_pose", False)
                ):
                    raise SemanticConfigurationError(
                        "Tracking canonical-pose or reprojection supervision requires "
                        "model.predict_canonical_pose=true."
                    )
            if all(
                _number(tracking_loss, key, path="loss") == 0.0
                for key in {
                    "match_position_weight",
                    "match_rotation_weight",
                    "match_presence_weight",
                }
            ):
                raise SemanticConfigurationError(
                    "At least one tracking match cost weight must be positive."
                )
        if shared.training.gan.enabled:
            discriminator_fields = {
                "name",
                "hidden_dim",
                "num_layers",
                "num_heads",
                "ffn_dim",
                "ffn_type",
                "dropout",
                "rope_dim",
                "rope_theta",
                "max_seq_len",
                "invalid_init_std",
                "cls_init_std",
            }
            discriminator = _exact(
                require_config_mapping(
                    gan_mapping, "discriminator", path="training.gan"
                ),
                path="training.gan.discriminator",
                required=discriminator_fields,
                allowed=discriminator_fields,
            )
            if (
                _string(discriminator, "name", path="training.gan.discriminator")
                != "pose_sequence_transformer"
            ):
                raise SemanticConfigurationError(
                    "training.gan.discriminator.name must be 'pose_sequence_transformer'."
                )
            hidden = _integer(
                discriminator, "hidden_dim", path="training.gan.discriminator"
            )
            heads = _integer(
                discriminator, "num_heads", path="training.gan.discriminator"
            )
            if hidden <= 0 or heads <= 0 or hidden % heads:
                raise SemanticConfigurationError(
                    "training.gan.discriminator hidden_dim and num_heads must be "
                    "positive and divisible."
                )
            ffn_dim = _integer(
                discriminator, "ffn_dim", path="training.gan.discriminator"
            )
            if ffn_dim <= 0:
                raise SemanticConfigurationError(
                    "training.gan.discriminator.ffn_dim must be positive."
                )
            if (
                _integer(discriminator, "num_layers", path="training.gan.discriminator")
                < 0
            ):
                raise SemanticConfigurationError(
                    "training.gan.discriminator.num_layers must be non-negative."
                )
            rope_dim = _integer(
                discriminator, "rope_dim", path="training.gan.discriminator"
            )
            if rope_dim < 0 or rope_dim % 2 or rope_dim > hidden // heads:
                raise SemanticConfigurationError(
                    "training.gan.discriminator.rope_dim must be non-negative, "
                    "even, and no larger than the attention head dimension."
                )
            if _string(
                discriminator, "ffn_type", path="training.gan.discriminator"
) not in SUPPORTED_FFN_TYPES:
                raise SemanticConfigurationError(
                    "training.gan.discriminator.ffn_type must be one of "
                    f"{sorted(SUPPORTED_FFN_TYPES)!r}."
                )
            dropout = _number(
                discriminator, "dropout", path="training.gan.discriminator"
            )
            if not 0.0 <= dropout < 1.0:
                raise SemanticConfigurationError(
                    "training.gan.discriminator.dropout must be within [0, 1)."
                )
            for key in {"rope_theta"}:
                _positive(
                    _number(discriminator, key, path="training.gan.discriminator"),
                    path=f"training.gan.discriminator.{key}",
                )
            for key in {"invalid_init_std", "cls_init_std"}:
                _positive(
                    _number(discriminator, key, path="training.gan.discriminator"),
                    path=f"training.gan.discriminator.{key}",
                    allow_zero=True,
                )
            if (
                _integer(
                    discriminator, "max_seq_len", path="training.gan.discriminator"
                )
                <= 0
            ):
                raise SemanticConfigurationError(
                    "training.gan.discriminator.max_seq_len must be positive."
                )
        if (
            model.name in _TRACK_QUERY_MODEL_NAMES
            and model.input_profile is not None
        ):
            raise SemanticConfigurationError(
                "Tracking models must not define model.io."
            )
        return cls(
            shared=shared,
            court_keypoint_contract=court_keypoint_contract,
            paths=paths,
            model=model,
            data=data,
            fine_tune_mode=fine_tune_mode,
            tracking_reprojection_enabled=tracking_reprojection_enabled,
            tracking_metrics=tracking_metric_config,
            qualitative_style=qualitative_style,
            qualitative_view_3d=qualitative_view_3d,
            qualitative_fps=qualitative_fps,
            raw=value,
        )


def _validate_training_boundary(config: DictConfig) -> None:
    PLCSTrainingConfig.from_config(config)


def _validate_visualization_boundary(config: DictConfig) -> None:
    root = _exact(
        config,
        path="configuration",
        required={
            "court_keypoints",
            "visualization",
            "run",
            "paths",
        },
        allowed={
            "court_keypoints",
            "visualization",
            "run",
            "paths",
        },
    )
    court_keypoint_contract = PLCSCourtKeypointRuntimeConfig.from_config(
        config
    ).contract
    resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
    visualization_fields = {
        "mode",
        "scene_path",
        "camera",
        "cameras",
        "animation_view",
        "fps",
        "save",
        "info",
        "checkpoint",
        "canonical_pose_source",
        "device",
        "style",
        "view_3d",
        "reference_camera_id",
    }
    visualization = _exact(
        require_config_mapping(root, "visualization", path="configuration"),
        path="visualization",
        required=visualization_fields,
        allowed=visualization_fields,
    )
    mode = _string(visualization, "mode", path="visualization")
    if mode not in {"visualize", "predict"}:
        raise SemanticConfigurationError(
            "visualization.mode must be 'visualize' or 'predict'."
        )
    if mode == "predict" and visualization["checkpoint"] is None:
        raise SemanticConfigurationError(
            "visualization.checkpoint is required for predict mode."
        )
    SceneVisualizationConfig.from_mapping(
        visualization,
        resolver=resolver,
        extension_keys={"canonical_pose_source", "reference_camera_id"},
    )
    reference_camera_id = visualization["reference_camera_id"]
    if court_keypoint_contract.camera_view_semantics and mode == "predict":
        if not isinstance(reference_camera_id, str) or not reference_camera_id.strip():
            raise SemanticConfigurationError(
                "camera_view_v2 prediction visualization requires an explicit "
                "visualization.reference_camera_id."
            )
    elif reference_camera_id is not None:
        raise SemanticConfigurationError(
            "visualization.reference_camera_id is only valid for camera_view_v2."
        )
    source = _string(visualization, "canonical_pose_source", path="visualization")
    if source not in {"gt", "prediction"}:
        raise SemanticConfigurationError(
            "visualization.canonical_pose_source must be 'gt' or 'prediction'."
        )
    animation_view = _string(visualization, "animation_view", path="visualization")
    if animation_view not in {"3d", "2d_topdown", "camera"}:
        raise SemanticConfigurationError(
            "visualization.animation_view must be '3d', '2d_topdown', or 'camera'."
        )
    if mode == "predict" and animation_view == "camera":
        raise SemanticConfigurationError(
            "visualization.animation_view='camera' is unavailable in predict mode."
        )
    _resolved_device(visualization["device"], path="visualization.device")
    parse_scene_style(visualization["style"])
    parse_view_3d(visualization["view_3d"])


def _validate_script_boundary(
    config: DictConfig,
    *,
    required_sections: set[str],
    section_fields: Mapping[str, set[str]],
) -> None:
    required_sections = {
        *required_sections,
        "court_keypoints",
    }
    root = _exact(
        config,
        path="configuration",
        required=required_sections,
        allowed=required_sections,
    )
    PLCSCourtKeypointRuntimeConfig.from_config(config)
    resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
    if "data" in root:
        data_fields = {
            "backend",
            "scene_dir",
            "num_court_kp",
            "augmentation",
            "min_cameras",
            "camera_mode",
            "num_views_range",
            "batch_size",
            "num_workers",
            "pin_memory",
            "mode",
            "seq_len_range",
            "adapter_camera_index",
        }
        data = _exact(
            require_config_mapping(root, "data", path="configuration"),
            path="data",
            required=data_fields,
            allowed=data_fields,
        )
        validate_augmentation(data["augmentation"])
        backend = _string(data, "backend", path="data")
        if backend != "default":
            raise SemanticConfigurationError(
                "PLCS analysis/preview boundaries require data.backend='default'."
            )
        _string(data, "scene_dir", path="data")
        resolver.resolve(PathRole.DATA, _string(data, "scene_dir", path="data"))
        if _integer(data, "num_court_kp", path="data") <= 0:
            raise SemanticConfigurationError("data.num_court_kp must be positive.")
        min_cameras = _integer(data, "min_cameras", path="data")
        if min_cameras <= 0:
            raise SemanticConfigurationError("data.min_cameras must be positive.")
        camera_mode = require_config_value(data, "camera_mode", (str, int), path="data")
        if isinstance(camera_mode, str):
            if camera_mode not in {"random", "first"}:
                raise SemanticConfigurationError(
                    "data.camera_mode must be 'random', 'first', or a non-negative index."
                )
        elif cast("int", camera_mode) < 0:
            raise SemanticConfigurationError(
                "data.camera_mode index must be non-negative."
            )
        views = _sequence(
            data,
            "num_views_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        lengths = _sequence(
            data,
            "seq_len_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        for key, values in (("num_views_range", views), ("seq_len_range", lengths)):
            lo, hi = (cast("int", item) for item in values)
            if lo <= 0 or hi < lo:
                raise SemanticConfigurationError(
                    f"data.{key} must be a positive ordered range."
                )
        if min_cameras > cast("int", views[1]):
            raise SemanticConfigurationError(
                "data.min_cameras cannot exceed data.num_views_range capacity."
            )
        if _integer(data, "batch_size", path="data") <= 0:
            raise SemanticConfigurationError("data.batch_size must be positive.")
        if _integer(data, "num_workers", path="data") < 0:
            raise SemanticConfigurationError("data.num_workers must be non-negative.")
        _boolean(data, "pin_memory", path="data")
        if _string(data, "mode", path="data") not in {
            "frame",
            "sequence",
            "multiview_sequence",
        }:
            raise SemanticConfigurationError(
                "data.mode must be 'frame', 'sequence', or 'multiview_sequence'."
            )
        if _integer(data, "adapter_camera_index", path="data") < 0:
            raise SemanticConfigurationError(
                "data.adapter_camera_index must be non-negative."
            )
    for section, fields in section_fields.items():
        _exact(
            require_config_mapping(root, section, path="configuration"),
            path=section,
            required=fields,
            allowed=fields,
        )
    if "preview" in root:
        preview = require_config_mapping(root, "preview", path="configuration")
        split = _string(preview, "split", path="preview")
        if split not in {"train", "val", "test"}:
            raise SemanticConfigurationError(
                "preview.split must be 'train', 'val', or 'test'."
            )
        sample_indices = _sequence(
            preview,
            "sample_indices",
            path="preview",
            item_types=(int,),
        )
        if any(cast("int", index) < 0 for index in sample_indices):
            raise SemanticConfigurationError(
                "preview.sample_indices must contain only non-negative indices."
            )
        for key in {
            "max_samples",
            "num_augmented",
            "seed",
            "max_cameras",
            "pose_frames",
        }:
            value = _integer(preview, key, path="preview")
            if key != "seed" and value <= 0:
                raise SemanticConfigurationError(f"preview.{key} must be positive.")
        resolver.resolve(
            PathRole.OUTPUT, _string(preview, "output_dir", path="preview")
        )
        figure = _exact(
            preview["figure"],
            path="preview.figure",
            required={"panel_width", "panel_height", "dpi"},
            allowed={"panel_width", "panel_height", "dpi"},
        )
        for key in {"panel_width", "panel_height"}:
            _positive(
                _number(figure, key, path="preview.figure"),
                path=f"preview.figure.{key}",
            )
        if _integer(figure, "dpi", path="preview.figure") <= 0:
            raise SemanticConfigurationError("preview.figure.dpi must be positive.")
    if "analysis" in root:
        analysis = require_config_mapping(root, "analysis", path="configuration")
        if "output_filename" in analysis:
            split = _string(analysis, "split", path="analysis")
            if split not in {"train", "val", "test"}:
                raise SemanticConfigurationError(
                    "analysis.split must be 'train', 'val', or 'test'."
                )
            max_batches = require_config_value(
                analysis, "max_batches", (int, type(None)), path="analysis"
            )
            if max_batches is not None and cast("int", max_batches) <= 0:
                raise SemanticConfigurationError(
                    "analysis.max_batches must be null or positive."
                )
            _simple_name(
                _string(analysis, "output_filename", path="analysis"),
                path="analysis.output_filename",
            )
        elif "xy_hist" in analysis:
            mode = _string(analysis, "mode", path="analysis")
            if mode not in {"per_frame", "initial_only"}:
                raise SemanticConfigurationError(
                    "analysis.mode must be 'per_frame' or 'initial_only'."
                )
            for key in {"max_scenes", "max_frames_per_scene"}:
                limit = require_config_value(
                    analysis, key, (int, type(None)), path="analysis"
                )
                if limit is not None and cast("int", limit) <= 0:
                    raise SemanticConfigurationError(
                        f"analysis.{key} must be null or positive."
                    )
            thresholds = _sequence(
                analysis,
                "radius_thresholds_m",
                path="analysis",
                item_types=(float, int),
                non_empty=True,
            )
            threshold_values = tuple(
                float(cast("float | int", threshold)) for threshold in thresholds
            )
            if any(value <= 0.0 for value in threshold_values) or any(
                current <= previous
                for previous, current in zip(
                    threshold_values, threshold_values[1:], strict=False
                )
            ):
                raise SemanticConfigurationError(
                    "analysis.radius_thresholds_m must be positive and strictly increasing."
                )
            xy_hist = _exact(
                analysis["xy_hist"],
                path="analysis.xy_hist",
                required={"bins_x", "bins_y", "x_range_m", "y_range_m"},
                allowed={"bins_x", "bins_y", "x_range_m", "y_range_m"},
            )
            for key in {"bins_x", "bins_y"}:
                if _integer(xy_hist, key, path="analysis.xy_hist") <= 0:
                    raise SemanticConfigurationError(
                        f"analysis.xy_hist.{key} must be positive."
                    )
            for key in {"x_range_m", "y_range_m"}:
                range_lo, range_hi = _ordered_numeric_range(
                    xy_hist, key, path="analysis.xy_hist"
                )
                if range_lo == range_hi:
                    raise SemanticConfigurationError(
                        f"analysis.xy_hist.{key} must have non-zero width."
                    )
            yaw_hist = _exact(
                analysis["yaw_hist"],
                path="analysis.yaw_hist",
                required={"bins"},
                allowed={"bins"},
            )
            if _integer(yaw_hist, "bins", path="analysis.yaw_hist") <= 0:
                raise SemanticConfigurationError(
                    "analysis.yaw_hist.bins must be positive."
                )
            plots = require_config_mapping(root, "plots", path="configuration")
            _boolean(plots, "enabled", path="plots")
            if _integer(plots, "dpi", path="plots") <= 0:
                raise SemanticConfigurationError("plots.dpi must be positive.")
        elif "loss_config" in analysis:
            split = _string(analysis, "split", path="analysis")
            if split not in {"train", "val", "test"}:
                raise SemanticConfigurationError(
                    "analysis.split must be 'train', 'val', or 'test'."
                )
            _resolved_device(analysis["device"], path="analysis.device", nullable=True)
            max_batches = require_config_value(
                analysis, "max_batches", (int, type(None)), path="analysis"
            )
            if max_batches is not None and cast("int", max_batches) <= 0:
                raise SemanticConfigurationError(
                    "analysis.max_batches must be null or positive."
                )
            for key in {"report_filename", "plot_filename"}:
                filename = require_config_value(
                    analysis, key, (str, type(None)), path="analysis"
                )
                if key == "report_filename" and filename is None:
                    raise SemanticConfigurationError(
                        "analysis.report_filename must be a file name."
                    )
                if isinstance(filename, str):
                    _simple_name(filename, path=f"analysis.{key}")
            if _integer(analysis, "plot_dpi", path="analysis") <= 0:
                raise SemanticConfigurationError("analysis.plot_dpi must be positive.")
            loss_config = require_config_value(
                analysis, "loss_config", (str, type(None)), path="analysis"
            )
            if isinstance(loss_config, str):
                resolver.resolve(PathRole.PROJECT, loss_config)
        elif "top_k" in analysis:
            for key in {
                "split",
                "device",
                "animation_view",
                "scene_subdir",
                "report_filename",
                "output_suffix",
            }:
                text = _string(analysis, key, path="analysis")
                if not text:
                    raise SemanticConfigurationError(
                        f"analysis.{key} must not be empty."
                    )
            if analysis["split"] not in {"train", "val", "test"}:
                raise SemanticConfigurationError(
                    "analysis.split must be 'train', 'val', or 'test'."
                )
            if analysis["animation_view"] not in {"3d", "2d_topdown"}:
                raise SemanticConfigurationError(
                    "analysis.animation_view must be '3d' or '2d_topdown'."
                )
            _resolved_device(analysis["device"], path="analysis.device")
            _simple_name(
                cast("str", analysis["report_filename"]),
                path="analysis.report_filename",
            )
            _simple_name(
                cast("str", analysis["output_suffix"]), path="analysis.output_suffix"
            )
            for key in {
                "top_k",
                "candidates_per_scene",
                "clip_half_window",
                "fps",
            }:
                value = _integer(analysis, key, path="analysis")
                if value <= 0:
                    raise SemanticConfigurationError(
                        f"analysis.{key} must be positive."
                    )
            for key in {
                "unique_scenes",
                "render_visualizations",
                "overwrite",
            }:
                _boolean(analysis, key, path="analysis")
            cameras = require_config_value(
                analysis, "cameras", (str, list, tuple), path="analysis"
            )
            if isinstance(cameras, str):
                if cameras != "all":
                    try:
                        indices = tuple(
                            int(part.strip()) for part in cameras.split(",")
                        )
                    except ValueError as error:
                        raise SemanticConfigurationError(
                            "analysis.cameras must be 'all' or comma-separated integers."
                        ) from error
                    if not indices or any(index < 0 for index in indices):
                        raise SemanticConfigurationError(
                            "analysis.cameras indices must be non-negative."
                        )
            else:
                camera_values = tuple(cast("Sequence[object]", cameras))
                if not camera_values:
                    raise SemanticConfigurationError(
                        "analysis.cameras must not be empty."
                    )
                if any(type(item) is not int for item in camera_values):
                    raise ConfigurationTypeError(
                        "analysis.cameras must contain exact int values."
                    )
                if any(cast("int", item) < 0 for item in camera_values):
                    raise SemanticConfigurationError(
                        "analysis.cameras indices must be non-negative."
                    )
            visualization = require_config_mapping(
                root, "visualization", path="configuration"
            )
            SceneVisualizationConfig.from_mapping(
                visualization,
                resolver=resolver,
                extension_keys={"canonical_pose_source", "reference_camera_id"},
            )
            parse_scene_style(visualization["style"])
            parse_view_3d(visualization["view_3d"])
    if "run" in root:
        run = require_config_mapping(root, "run", path="configuration")
        _integer(run, "seed", path="run")
        if "output_dir" in run:
            output_relative = _string(run, "output_dir", path="run")
            resolver.resolve(PathRole.OUTPUT, output_relative)
            if "analysis" in root:
                analysis = require_config_mapping(
                    root, "analysis", path="configuration"
                )
                for key in {"output_filename", "report_filename", "plot_filename"}:
                    if key not in analysis:
                        continue
                    child = analysis[key]
                    if isinstance(child, str):
                        resolver.resolve(PathRole.OUTPUT, output_relative, child)
                if "scene_subdir" in analysis:
                    resolver.resolve(
                        PathRole.OUTPUT,
                        output_relative,
                        _string(analysis, "scene_subdir", path="analysis"),
                    )
        if "scene_dir" in run:
            resolver.resolve(PathRole.DATA, _string(run, "scene_dir", path="run"))
        if "checkpoint" in run:
            resolver.resolve(
                PathRole.CHECKPOINT, _string(run, "checkpoint", path="run")
            )
        if "hparams" in run:
            resolver.resolve(PathRole.ARTIFACT, _string(run, "hparams", path="run"))


@dataclass(frozen=True, slots=True)
class PLCSPreviewRuntimeConfig:
    """Resolved paths for the PLCS augmentation-preview entrypoint."""

    OUTPUT_ROLE: ClassVar[PathRole] = PathRole.OUTPUT

    resolver: PathResolver
    scene_dir: Path
    output_dir: Path
    raw: DictConfig

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSPreviewRuntimeConfig:
        _validate_script_boundary(
            config,
            required_sections={"data", "preview", "paths"},
            section_fields={
                "preview": {
                    "split",
                    "sample_indices",
                    "max_samples",
                    "num_augmented",
                    "seed",
                    "max_cameras",
                    "pose_frames",
                    "output_dir",
                    "figure",
                }
            },
        )
        resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
        return cls(
            resolver=resolver,
            scene_dir=resolver.resolve(PathRole.DATA, str(config.data.scene_dir)),
            output_dir=resolver.resolve(
                cls.OUTPUT_ROLE, str(config.preview.output_dir)
            ),
            raw=config,
        )


@dataclass(frozen=True, slots=True)
class PLCSAnalysisRuntimeConfig:
    """Resolved role-aware paths for a PLCS analysis entrypoint."""

    OUTPUT_ROLE: ClassVar[PathRole] = PathRole.OUTPUT

    resolver: PathResolver
    output_dir: Path
    scene_dir: Path | None
    scene_records_dir: Path | None
    split_path: Path | None
    checkpoint: Path | None
    hparams: Path | None
    loss_config: Path | None
    result_path: Path | None
    plot_path: Path | None
    scene_output_dir: Path | None
    device: str | None
    raw: DictConfig

    @classmethod
    def angle_velocity(cls, config: DictConfig) -> PLCSAnalysisRuntimeConfig:
        _validate_script_boundary(
            config,
            required_sections={"data", "analysis", "run", "paths"},
            section_fields={
                "analysis": {"split", "max_batches", "output_filename"},
                "run": {"output_dir", "seed"},
            },
        )
        resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
        output_dir = resolver.resolve(cls.OUTPUT_ROLE, str(config.run.output_dir))
        return cls(
            resolver=resolver,
            output_dir=output_dir,
            scene_dir=resolver.resolve(PathRole.DATA, str(config.data.scene_dir)),
            scene_records_dir=None,
            split_path=None,
            checkpoint=None,
            hparams=None,
            loss_config=None,
            result_path=resolver.resolve(
                cls.OUTPUT_ROLE,
                str(config.run.output_dir),
                str(config.analysis.output_filename),
            ),
            plot_path=None,
            scene_output_dir=None,
            device=None,
            raw=config,
        )

    @classmethod
    def distribution(cls, config: DictConfig) -> PLCSAnalysisRuntimeConfig:
        _validate_script_boundary(
            config,
            required_sections={"data", "analysis", "run", "plots", "paths"},
            section_fields={
                "analysis": {
                    "mode",
                    "max_scenes",
                    "max_frames_per_scene",
                    "radius_thresholds_m",
                    "xy_hist",
                    "yaw_hist",
                },
                "run": {"output_dir", "seed"},
                "plots": {"enabled", "dpi"},
            },
        )
        resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
        return cls(
            resolver=resolver,
            output_dir=resolver.resolve(cls.OUTPUT_ROLE, str(config.run.output_dir)),
            scene_dir=resolver.resolve(PathRole.DATA, str(config.data.scene_dir)),
            scene_records_dir=resolver.resolve(
                PathRole.DATA,
                str(config.data.scene_dir),
                "scenes",
            ),
            split_path=None,
            checkpoint=None,
            hparams=None,
            loss_config=None,
            result_path=None,
            plot_path=None,
            scene_output_dir=None,
            device=None,
            raw=config,
        )

    @classmethod
    def loss_dominance(cls, config: DictConfig) -> PLCSAnalysisRuntimeConfig:
        _validate_script_boundary(
            config,
            required_sections={"analysis", "run", "paths"},
            section_fields={
                "analysis": {
                    "split",
                    "device",
                    "max_batches",
                    "report_filename",
                    "plot_filename",
                    "plot_dpi",
                    "loss_config",
                },
                "run": {"checkpoint", "hparams", "output_dir", "seed"},
            },
        )
        resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
        output_relative = str(config.run.output_dir)
        plot_filename = config.analysis.plot_filename
        loss_config = config.analysis.loss_config
        return cls(
            resolver=resolver,
            output_dir=resolver.resolve(cls.OUTPUT_ROLE, output_relative),
            scene_dir=None,
            scene_records_dir=None,
            split_path=None,
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, str(config.run.checkpoint)
            ),
            hparams=resolver.resolve(PathRole.ARTIFACT, str(config.run.hparams)),
            loss_config=(
                None
                if loss_config is None
                else resolver.resolve(PathRole.PROJECT, str(loss_config))
            ),
            result_path=resolver.resolve(
                cls.OUTPUT_ROLE,
                output_relative,
                str(config.analysis.report_filename),
            ),
            plot_path=(
                None
                if plot_filename is None
                else resolver.resolve(
                    cls.OUTPUT_ROLE, output_relative, str(plot_filename)
                )
            ),
            scene_output_dir=None,
            device=_resolved_device(
                config.analysis.device,
                path="analysis.device",
                nullable=True,
            ),
            raw=config,
        )

    @classmethod
    def rotation_error(cls, config: DictConfig) -> PLCSAnalysisRuntimeConfig:
        _validate_script_boundary(
            config,
            required_sections={"analysis", "run", "paths", "visualization"},
            section_fields={
                "analysis": {
                    "split",
                    "device",
                    "top_k",
                    "unique_scenes",
                    "candidates_per_scene",
                    "clip_half_window",
                    "cameras",
                    "render_visualizations",
                    "animation_view",
                    "fps",
                    "scene_subdir",
                    "report_filename",
                    "output_suffix",
                    "overwrite",
                },
                "run": {"checkpoint", "scene_dir", "output_dir", "seed"},
                "visualization": {
                    "mode",
                    "scene_path",
                    "camera",
                    "cameras",
                    "animation_view",
                    "fps",
                    "save",
                    "info",
                    "checkpoint",
                    "device",
                    "canonical_pose_source",
                    "style",
                    "view_3d",
                },
            },
        )
        resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
        output_relative = str(config.run.output_dir)
        scene_relative = str(config.run.scene_dir)
        return cls(
            resolver=resolver,
            output_dir=resolver.resolve(cls.OUTPUT_ROLE, output_relative),
            scene_dir=resolver.resolve(PathRole.DATA, scene_relative),
            scene_records_dir=resolver.resolve(
                PathRole.DATA,
                scene_relative,
                "scenes",
            ),
            split_path=resolver.resolve(
                PathRole.DATA,
                scene_relative,
                f"{config.analysis.split}.txt",
            ),
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, str(config.run.checkpoint)
            ),
            hparams=None,
            loss_config=None,
            result_path=resolver.resolve(
                cls.OUTPUT_ROLE,
                output_relative,
                str(config.analysis.report_filename),
            ),
            plot_path=None,
            scene_output_dir=resolver.resolve(
                cls.OUTPUT_ROLE,
                output_relative,
                str(config.analysis.scene_subdir),
            ),
            device=_resolved_device(
                config.analysis.device,
                path="analysis.device",
            ),
            raw=config,
        )


def _validate_preview_boundary(config: DictConfig) -> None:
    PLCSPreviewRuntimeConfig.from_config(config)


def _validate_angle_velocity_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.angle_velocity(config)


def _validate_distribution_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.distribution(config)


def _validate_loss_dominance_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.loss_dominance(config)


def _validate_rotation_error_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.rotation_error(config)


def _register_validators() -> None:
    register_boundary_validator("plcs.train", _validate_training_boundary)
    register_boundary_validator("plcs.visualize", _validate_visualization_boundary)
    register_boundary_validator("plcs.preview_augmentation", _validate_preview_boundary)
    register_boundary_validator(
        "plcs.analyze_angle_velocity", _validate_angle_velocity_boundary
    )
    register_boundary_validator(
        "plcs.analyze_dataset_distribution", _validate_distribution_boundary
    )
    register_boundary_validator(
        "plcs.analyze_loss_dominance", _validate_loss_dominance_boundary
    )
    register_boundary_validator(
        "plcs.analyze_rotation_error_samples", _validate_rotation_error_boundary
    )


_register_validators()


__all__ = [
    "PLCSAnalysisRuntimeConfig",
    "PLCSDataConfig",
    "PLCSModelConfig",
    "PLCSPreviewRuntimeConfig",
    "PLCSTrainingConfig",
    "validate_augmentation",
]
