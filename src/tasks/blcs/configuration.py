"""Strict typed configuration contracts for BLCS runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    SceneVisualizationConfig,
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    resolve_court_keypoint_contract,
)
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
    parse_view_3d,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.device import resolve_device
from src.utils.hydra import register_boundary_validator
from src.utils.models.components.ffn_layers import (
    SUPPORTED_FFN_TYPES,
    FFNType,
)
from src.utils.paths import PROJECT_ROOT

Scalar: TypeAlias = str | int | float | bool | None


def _exact(mapping: Mapping[str, object], required: set[str], *, path: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): "
            f"{', '.join(f'{path}.{key}' for key in missing)}."
        )
    unknown = sorted(set(mapping) - required)
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): "
            f"{', '.join(f'{path}.{key}' for key in unknown)}."
        )


def _value(
    mapping: Mapping[str, object],
    key: str,
    expected: type[object] | tuple[type[object], ...],
    *,
    path: str,
) -> object:
    if key not in mapping:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key: {path}.{key}."
        )
    value = mapping[key]
    accepted = expected if isinstance(expected, tuple) else (expected,)
    if type(value) not in accepted:
        names = " | ".join(candidate.__name__ for candidate in accepted)
        raise ConfigurationTypeError(
            f"{path}.{key}: expected {names}, got {type(value).__name__}."
        )
    return value


def _int_sequence(value: object, *, path: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(type(item) is not int for item in value)
    ):
        raise ConfigurationTypeError(f"{path}: expected a sequence of int values.")
    return tuple(cast("int", item) for item in value)


def _bool_sequence(value: object, *, path: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(type(item) is not bool for item in value)
    ):
        raise ConfigurationTypeError(f"{path}: expected a sequence of bool values.")
    return tuple(cast("bool", item) for item in value)


def _numeric_sequence(
    value: object, *, path: str, length: int | None = None
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationTypeError(f"{path}: expected a sequence of numbers.")
    sequence = tuple(value)
    if any(type(item) not in (float, int) for item in sequence):
        raise ConfigurationTypeError(f"{path}: expected a sequence of numbers.")
    if length is not None and len(sequence) != length:
        raise SemanticConfigurationError(
            f"{path}: expected exactly {length} values, got {len(sequence)}."
        )
    values = tuple(float(cast("float | int", item)) for item in sequence)
    if any(not math.isfinite(item) for item in values):
        raise SemanticConfigurationError(f"{path}: values must be finite.")
    return values


def _optional_numeric_sequence(
    value: object, *, path: str, length: int | None = None
) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _numeric_sequence(value, path=path, length=length)


def _validate_types(
    mapping: Mapping[str, object],
    specifications: Mapping[str, type[object] | tuple[type[object], ...]],
    *,
    path: str,
) -> None:
    for key, expected in specifications.items():
        _value(mapping, key, expected, path=path)


def _finite(value: float, *, path: str) -> float:
    if not math.isfinite(value):
        raise SemanticConfigurationError(f"{path} must be finite.")
    return value


def _non_negative(value: float, *, path: str) -> float:
    value = _finite(value, path=path)
    if value < 0.0:
        raise SemanticConfigurationError(f"{path} must be non-negative.")
    return value


def _positive(value: float, *, path: str) -> float:
    value = _finite(value, path=path)
    if value <= 0.0:
        raise SemanticConfigurationError(f"{path} must be positive.")
    return value


def _probability(value: float, *, path: str) -> float:
    value = _finite(value, path=path)
    if not 0.0 <= value <= 1.0:
        raise SemanticConfigurationError(f"{path} must be within [0, 1].")
    return value


def _ordered_range(
    value: object,
    *,
    path: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    positive: bool = False,
) -> tuple[float, float]:
    lo, hi = _numeric_sequence(value, path=path, length=2)
    if lo > hi:
        raise SemanticConfigurationError(f"{path} must be ordered low-to-high.")
    if positive and lo <= 0.0:
        raise SemanticConfigurationError(f"{path} values must be positive.")
    if lower_bound is not None and lo < lower_bound:
        raise SemanticConfigurationError(f"{path} values must be >= {lower_bound}.")
    if upper_bound is not None and hi > upper_bound:
        raise SemanticConfigurationError(f"{path} values must be <= {upper_bound}.")
    return lo, hi


def _validate_transformer_dimensions(
    *,
    hidden_dim: int,
    num_heads: int,
    ffn_dim: int,
    rope_dim: int,
    dropout: float,
    path: str,
) -> None:
    if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
        raise SemanticConfigurationError(
            f"{path}.hidden_dim and {path}.num_heads must be positive and divisible."
        )
    if ffn_dim <= 0:
        raise SemanticConfigurationError(f"{path}.ffn_dim must be positive.")
    head_dim = hidden_dim // num_heads
    if rope_dim < 0 or rope_dim % 2 or rope_dim > head_dim:
        raise SemanticConfigurationError(
            f"{path}.rope_dim must be non-negative, even, and no larger than "
            f"the attention head dimension ({head_dim})."
        )
    dropout = _finite(dropout, path=f"{path}.dropout")
    if not 0.0 <= dropout < 1.0:
        raise SemanticConfigurationError(f"{path}.dropout must be within [0, 1).")


def build_path_resolver(config: object) -> PathResolver:
    """Validate all seven BLCS roots and build the shared resolver."""
    root = as_config_mapping(config, path="configuration")
    return PathResolver(
        RuntimePathRoots.from_mapping(
            require_config_mapping(root, "paths", path="configuration"),
            repository_root=PROJECT_ROOT,
        )
    )


def parse_court_keypoint_contract(config: object) -> CourtKeypointContract:
    """Resolve the independent, explicitly selected CourtKP20 contract."""
    root = as_config_mapping(config, path="configuration")
    section = require_config_mapping(root, "court_keypoints", path="configuration")
    _exact(section, {"selector"}, path="court_keypoints")
    selector = _value(section, "selector", str, path="court_keypoints")
    try:
        return resolve_court_keypoint_contract(cast("str", selector))
    except ValueError as error:
        raise SemanticConfigurationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class SingleModelConfig:
    name: Literal["blcs"]
    input_profile: Literal["single"]
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    ffn_type: FFNType
    dropout: float
    max_seq_len: int
    invisible_init_std: float
    rope_dim: int
    rope_theta: float
    rope_theta_time: float
    rope_theta_camera: float
    rope_theta_type: float
    predict_velocity: bool
    num_court_tokens: int


@dataclass(frozen=True, slots=True)
class AxialModelConfig:
    name: Literal["blcs_multiview_axial"]
    input_profile: Literal["multiview"]
    hidden_dim: int
    num_layers: int
    num_heads: int
    attention_type: Literal["mha", "gqa"]
    num_kv_heads: int | None
    ffn_dim: int
    ffn_type: FFNType
    dropout: float
    rope_dim: int
    rope_theta_time: float
    rope_theta_camera: float
    max_seq_len: int
    max_num_cameras: int
    predict_velocity: bool
    invisible_init_std: float
    num_court_tokens: int
    time_window_radius: int
    camera_layers_per_stage: tuple[int, ...]
    time_layers_per_stage: tuple[int, ...]
    time_global_stage_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class TrackQueryMHCConfig:
    coefficient_dim: int
    sinkhorn_iters: int
    eps: float
    residual_identity_bias: float
    update_scale_init: float


@dataclass(frozen=True, slots=True)
class TrackQueryCSWAConfig:
    compression_ratio: int
    window_radius: int
    backend: Literal["reference", "cuda"]


@dataclass(frozen=True, slots=True)
class TrackQueryModelConfig:
    name: Literal["blcs_track_query"]
    hidden_dim: int
    num_heads: int
    num_stages: int
    ffn_dim: int
    ffn_type: FFNType
    num_queries: int
    rope_dim: int
    dropout: float
    invisible_init_std: float
    mhc: TrackQueryMHCConfig
    cswa: TrackQueryCSWAConfig


@dataclass(frozen=True, slots=True)
class TrackQueryReferenceModelConfig:
    """Reference-conditioned v2 track-query architecture contract."""

    name: Literal["blcs_track_query_reference"]
    hidden_dim: int
    num_heads: int
    num_stages: int
    ffn_dim: int
    ffn_type: FFNType
    num_queries: int
    rope_dim: int
    dropout: float
    invisible_init_std: float
    target_frame_contract: Literal["reference_camera_court_rzpi_v1"]
    track_query_rope_contract: Literal["time_camera_reference_selector_v1"]
    reference_selector_mode: Literal["reference"]
    mhc: TrackQueryMHCConfig
    cswa: TrackQueryCSWAConfig


BLCSModelConfig: TypeAlias = (
    SingleModelConfig
    | AxialModelConfig
    | TrackQueryModelConfig
    | TrackQueryReferenceModelConfig
)

_TRACK_QUERY_MODEL_CONFIG_TYPES = (
    TrackQueryModelConfig,
    TrackQueryReferenceModelConfig,
)
_TRACK_QUERY_MODEL_NAMES = frozenset(
    {
        "blcs_track_query",
        "blcs_track_query_reference",
    }
)


def _io(mapping: Mapping[str, object]) -> str:
    io = as_config_mapping(_value(mapping, "io", dict, path="model"), path="model.io")
    _exact(io, {"input_profile"}, path="model.io")
    return cast("str", _value(io, "input_profile", str, path="model.io"))


def _model_mapping(config: object) -> Mapping[str, Any]:
    root = as_config_mapping(config, path="configuration")
    return cast(
        "Mapping[str, Any]",
        require_config_mapping(root, "model", path="configuration"),
    )


def parse_model_config(config: object) -> BLCSModelConfig:
    """Parse one exact BLCS model variant without cross-section defaults."""
    model = _model_mapping(config)
    name = cast("str", _value(model, "name", str, path="model"))
    result: BLCSModelConfig
    if name == "blcs":
        keys = {
            "name",
            "io",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "ffn_type",
            "dropout",
            "max_seq_len",
            "invisible_init_std",
            "rope_dim",
            "rope_theta",
            "rope_theta_time",
            "rope_theta_camera",
            "rope_theta_type",
            "predict_velocity",
            "num_court_tokens",
        }
        _exact(model, keys, path="model")
        _validate_types(
            model,
            {
                "name": str,
                "io": dict,
                "hidden_dim": int,
                "num_layers": int,
                "num_heads": int,
                "ffn_dim": int,
                "ffn_type": str,
                "dropout": (float, int),
                "max_seq_len": int,
                "invisible_init_std": (float, int),
                "rope_dim": int,
                "rope_theta": (float, int),
                "rope_theta_time": (float, int),
                "rope_theta_camera": (float, int),
                "rope_theta_type": (float, int),
                "predict_velocity": bool,
                "num_court_tokens": int,
            },
            path="model",
        )
        profile = _io(model)
        if profile != "single" or model["ffn_type"] not in SUPPORTED_FFN_TYPES:
            raise SemanticConfigurationError(
                "Invalid single-view model profile or ffn_type."
            )
        result = SingleModelConfig(
            name="blcs",
            input_profile="single",
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            ffn_dim=cast("int", model["ffn_dim"]),
            ffn_type=cast("FFNType", model["ffn_type"]),
            dropout=float(model["dropout"]),
            max_seq_len=int(model["max_seq_len"]),
            invisible_init_std=float(model["invisible_init_std"]),
            rope_dim=cast("int", model["rope_dim"]),
            rope_theta=float(model["rope_theta"]),
            rope_theta_time=float(model["rope_theta_time"]),
            rope_theta_camera=float(model["rope_theta_camera"]),
            rope_theta_type=float(model["rope_theta_type"]),
            predict_velocity=bool(model["predict_velocity"]),
            num_court_tokens=int(model["num_court_tokens"]),
        )
        _validate_transformer_dimensions(
            hidden_dim=result.hidden_dim,
            num_heads=result.num_heads,
            ffn_dim=result.ffn_dim,
            rope_dim=result.rope_dim,
            dropout=result.dropout,
            path="model",
        )
        if result.num_layers < 0:
            raise SemanticConfigurationError("model.num_layers must be non-negative.")
        if result.max_seq_len <= 0 or result.num_court_tokens <= 0:
            raise SemanticConfigurationError(
                "model.max_seq_len and model.num_court_tokens must be positive."
            )
        _non_negative(result.invisible_init_std, path="model.invisible_init_std")
        for key, value in (
            ("rope_theta", result.rope_theta),
            ("rope_theta_time", result.rope_theta_time),
            ("rope_theta_camera", result.rope_theta_camera),
            ("rope_theta_type", result.rope_theta_type),
        ):
            _positive(value, path=f"model.{key}")
        return result
    if name == "blcs_multiview_axial":
        keys = {
            "name",
            "io",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "attention_type",
            "num_kv_heads",
            "ffn_dim",
            "ffn_type",
            "dropout",
            "rope_dim",
            "rope_theta_time",
            "rope_theta_camera",
            "max_seq_len",
            "max_num_cameras",
            "predict_velocity",
            "invisible_init_std",
            "num_court_tokens",
            "time_window_radius",
            "camera_layers_per_stage",
            "time_layers_per_stage",
            "time_global_stage_mask",
        }
        _exact(model, keys, path="model")
        _validate_types(
            model,
            {
                "name": str,
                "io": dict,
                "hidden_dim": int,
                "num_layers": int,
                "num_heads": int,
                "attention_type": str,
                "num_kv_heads": (int, type(None)),
                "ffn_dim": int,
                "ffn_type": str,
                "dropout": (float, int),
                "rope_dim": int,
                "rope_theta_time": (float, int),
                "rope_theta_camera": (float, int),
                "max_seq_len": int,
                "max_num_cameras": int,
                "predict_velocity": bool,
                "invisible_init_std": (float, int),
                "num_court_tokens": int,
                "time_window_radius": int,
                "camera_layers_per_stage": (list, tuple),
                "time_layers_per_stage": (list, tuple),
                "time_global_stage_mask": (list, tuple),
            },
            path="model",
        )
        if (
            _io(model) != "multiview"
            or model["attention_type"] not in {"mha", "gqa"}
            or model["ffn_type"] not in SUPPORTED_FFN_TYPES
        ):
            raise SemanticConfigurationError(
                "Invalid axial model profile, attention_type, or ffn_type."
            )
        result = AxialModelConfig(
            name="blcs_multiview_axial",
            input_profile="multiview",
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            attention_type=cast("Literal['mha', 'gqa']", model["attention_type"]),
            num_kv_heads=cast("int | None", model["num_kv_heads"]),
            ffn_dim=cast("int", model["ffn_dim"]),
            ffn_type=cast("FFNType", model["ffn_type"]),
            dropout=float(model["dropout"]),
            rope_dim=cast("int", model["rope_dim"]),
            rope_theta_time=float(model["rope_theta_time"]),
            rope_theta_camera=float(model["rope_theta_camera"]),
            max_seq_len=int(model["max_seq_len"]),
            max_num_cameras=int(model["max_num_cameras"]),
            predict_velocity=bool(model["predict_velocity"]),
            invisible_init_std=float(model["invisible_init_std"]),
            num_court_tokens=int(model["num_court_tokens"]),
            time_window_radius=int(model["time_window_radius"]),
            camera_layers_per_stage=_int_sequence(
                model["camera_layers_per_stage"], path="model.camera_layers_per_stage"
            ),
            time_layers_per_stage=_int_sequence(
                model["time_layers_per_stage"], path="model.time_layers_per_stage"
            ),
            time_global_stage_mask=_bool_sequence(
                model["time_global_stage_mask"], path="model.time_global_stage_mask"
            ),
        )
        if not (
            len(result.camera_layers_per_stage)
            == len(result.time_layers_per_stage)
            == len(result.time_global_stage_mask)
            == result.num_layers
        ):
            raise SemanticConfigurationError(
                "Axial per-stage sequences must have model.num_layers entries."
            )
        _validate_transformer_dimensions(
            hidden_dim=result.hidden_dim,
            num_heads=result.num_heads,
            ffn_dim=result.ffn_dim,
            rope_dim=result.rope_dim,
            dropout=result.dropout,
            path="model",
        )
        if result.num_layers < 0:
            raise SemanticConfigurationError("model.num_layers must be non-negative.")
        if result.max_seq_len <= 0 or result.max_num_cameras <= 0:
            raise SemanticConfigurationError(
                "model.max_seq_len and model.max_num_cameras must be positive."
            )
        if result.num_court_tokens <= 0 or result.time_window_radius < 0:
            raise SemanticConfigurationError(
                "model.num_court_tokens must be positive and time_window_radius "
                "must be non-negative."
            )
        if any(value <= 0 for value in result.camera_layers_per_stage):
            raise SemanticConfigurationError(
                "model.camera_layers_per_stage values must be positive."
            )
        if any(value <= 0 for value in result.time_layers_per_stage):
            raise SemanticConfigurationError(
                "model.time_layers_per_stage values must be positive."
            )
        if result.attention_type == "mha" and result.num_kv_heads is not None:
            raise SemanticConfigurationError(
                "model.num_kv_heads must be null when attention_type='mha'."
            )
        if result.attention_type == "gqa" and (
            result.num_kv_heads is None
            or result.num_kv_heads <= 0
            or result.num_heads % result.num_kv_heads
        ):
            raise SemanticConfigurationError(
                "model.num_kv_heads must be positive and divide model.num_heads "
                "when attention_type='gqa'."
            )
        _non_negative(result.invisible_init_std, path="model.invisible_init_std")
        _positive(result.rope_theta_time, path="model.rope_theta_time")
        _positive(result.rope_theta_camera, path="model.rope_theta_camera")
        return result
    if name in _TRACK_QUERY_MODEL_NAMES:
        is_reference = name == "blcs_track_query_reference"
        base_keys = {
            "name",
            "hidden_dim",
            "num_heads",
            "num_stages",
            "ffn_dim",
            "ffn_type",
            "num_queries",
            "rope_dim",
            "dropout",
            "invisible_init_std",
            "mhc",
            "cswa",
        }
        if is_reference:
            base_keys |= {
                "target_frame_contract",
                "track_query_rope_contract",
                "reference_selector_mode",
            }
        _exact(model, base_keys, path="model")
        _validate_types(
            model,
            {
                "name": str,
                "hidden_dim": int,
                "num_heads": int,
                "num_stages": int,
                "ffn_dim": int,
                "ffn_type": str,
                "num_queries": int,
                "rope_dim": int,
                "dropout": (float, int),
                "invisible_init_std": (float, int),
            },
            path="model",
        )
        raw_ffn_type = cast("str", model["ffn_type"])
        if raw_ffn_type not in SUPPORTED_FFN_TYPES:
            raise SemanticConfigurationError(
                f"model.ffn_type must be one of {sorted(SUPPORTED_FFN_TYPES)!r}."
            )
        if is_reference:
            _validate_types(
                model,
                {
                    "target_frame_contract": str,
                    "track_query_rope_contract": str,
                    "reference_selector_mode": str,
                },
                path="model",
            )
            if model["target_frame_contract"] != "reference_camera_court_rzpi_v1":
                raise SemanticConfigurationError(
                    "model.target_frame_contract must be "
                    "'reference_camera_court_rzpi_v1' for BLCS reference v2."
                )
            if (
                model["track_query_rope_contract"]
                != "time_camera_reference_selector_v1"
            ):
                raise SemanticConfigurationError(
                    "model.track_query_rope_contract must be "
                    "'time_camera_reference_selector_v1' for BLCS reference v2."
                )
            if model["reference_selector_mode"] != "reference":
                raise SemanticConfigurationError(
                    "model.reference_selector_mode must be 'reference'."
                )
        raw_mhc = as_config_mapping(model["mhc"], path="model.mhc")
        mhc_keys = {
            "coefficient_dim",
            "sinkhorn_iters",
            "eps",
            "residual_identity_bias",
            "update_scale_init",
        }
        _exact(raw_mhc, mhc_keys, path="model.mhc")
        _validate_types(
            raw_mhc,
            {
                "coefficient_dim": int,
                "sinkhorn_iters": int,
                "eps": (float, int),
                "residual_identity_bias": (float, int),
                "update_scale_init": (float, int),
            },
            path="model.mhc",
        )
        mhc = TrackQueryMHCConfig(
            coefficient_dim=cast("int", raw_mhc["coefficient_dim"]),
            sinkhorn_iters=cast("int", raw_mhc["sinkhorn_iters"]),
            eps=float(cast("float | int", raw_mhc["eps"])),
            residual_identity_bias=float(
                cast("float | int", raw_mhc["residual_identity_bias"])
            ),
            update_scale_init=float(cast("float | int", raw_mhc["update_scale_init"])),
        )
        if mhc.coefficient_dim <= 0 or mhc.sinkhorn_iters <= 0:
            raise SemanticConfigurationError(
                "model.mhc.coefficient_dim and sinkhorn_iters must be positive."
            )
        _positive(mhc.eps, path="model.mhc.eps")
        _non_negative(
            mhc.residual_identity_bias,
            path="model.mhc.residual_identity_bias",
        )
        _finite(mhc.update_scale_init, path="model.mhc.update_scale_init")

        raw_cswa = as_config_mapping(model["cswa"], path="model.cswa")
        cswa_keys = {"compression_ratio", "window_radius", "backend"}
        _exact(raw_cswa, cswa_keys, path="model.cswa")
        _validate_types(
            raw_cswa,
            {
                "compression_ratio": int,
                "window_radius": int,
                "backend": str,
            },
            path="model.cswa",
        )
        backend = cast("str", raw_cswa["backend"])
        if backend not in {"reference", "cuda"}:
            raise SemanticConfigurationError(
                "model.cswa.backend must be 'reference' or 'cuda'."
            )
        cswa = TrackQueryCSWAConfig(
            compression_ratio=cast("int", raw_cswa["compression_ratio"]),
            window_radius=cast("int", raw_cswa["window_radius"]),
            backend=cast("Literal['reference', 'cuda']", backend),
        )
        if cswa.compression_ratio < 2:
            raise SemanticConfigurationError(
                "model.cswa.compression_ratio must be at least 2."
            )
        if cswa.window_radius < 0:
            raise SemanticConfigurationError(
                "model.cswa.window_radius must be non-negative."
            )
        if is_reference:
            result = TrackQueryReferenceModelConfig(
                name="blcs_track_query_reference",
                hidden_dim=int(model["hidden_dim"]),
                num_heads=int(model["num_heads"]),
                num_stages=int(model["num_stages"]),
                ffn_dim=cast("int", model["ffn_dim"]),
                ffn_type=cast("FFNType", model["ffn_type"]),
                num_queries=int(model["num_queries"]),
                rope_dim=cast("int", model["rope_dim"]),
                dropout=float(model["dropout"]),
                invisible_init_std=float(model["invisible_init_std"]),
                target_frame_contract=cast(
                    "Literal['reference_camera_court_rzpi_v1']",
                    model["target_frame_contract"],
                ),
                track_query_rope_contract=cast(
                    "Literal['time_camera_reference_selector_v1']",
                    model["track_query_rope_contract"],
                ),
                reference_selector_mode=cast(
                    "Literal['reference']", model["reference_selector_mode"]
                ),
                mhc=mhc,
                cswa=cswa,
            )
        else:
            result = TrackQueryModelConfig(
                name="blcs_track_query",
                hidden_dim=int(model["hidden_dim"]),
                num_heads=int(model["num_heads"]),
                num_stages=int(model["num_stages"]),
                ffn_dim=cast("int", model["ffn_dim"]),
                ffn_type=cast("FFNType", model["ffn_type"]),
                num_queries=int(model["num_queries"]),
                rope_dim=cast("int", model["rope_dim"]),
                dropout=float(model["dropout"]),
                invisible_init_std=float(model["invisible_init_std"]),
                mhc=mhc,
                cswa=cswa,
            )
        _validate_transformer_dimensions(
            hidden_dim=result.hidden_dim,
            num_heads=result.num_heads,
            ffn_dim=result.ffn_dim,
            rope_dim=result.rope_dim,
            dropout=result.dropout,
            path="model",
        )
        if is_reference and result.rope_dim < 6:
            raise SemanticConfigurationError(
                "model.rope_dim must be at least 6 for all three reference-v2 axes."
            )
        if (
            result.num_stages <= 0
            or result.num_stages % 4 != 0
            or result.num_queries <= 0
        ):
            raise SemanticConfigurationError(
                "model.num_stages must be a positive multiple of 4 and "
                "model.num_queries must be positive."
            )
        _non_negative(result.invisible_init_std, path="model.invisible_init_std")
        return result
    raise SemanticConfigurationError(f"Unsupported model.name={name!r}.")


@dataclass(frozen=True, slots=True)
class GenerationRunConfig:
    output_dir: Path
    seed: int
    device: str
    num_workers: int
    chunksize: int
    train_ratio: float
    val_ratio: float


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    output_dir: Path
    scene_dir: Path
    split: str
    sample_indices: tuple[int, ...]
    max_samples: int
    num_augmented: int
    seed: int
    max_cameras: int
    panel_width: float
    panel_height: float
    dpi: int


@dataclass(frozen=True, slots=True)
class QualitativeRenderingConfig:
    """BLCS-owned qualitative rendering values selected by composition."""

    style: SceneStyleConfig
    fps: float


def _validate_augmentation(config: Mapping[str, object]) -> None:
    _validate_types(
        config,
        {"enabled": bool, "preserve_clean_targets": bool},
        path="data.augmentation",
    )
    block_names = (
        "uv_scale",
        "gaussian_noise",
        "visibility_dropout",
        "temporal_jitter",
        "burst_dropout",
        "false_positive",
        "edge_degradation",
        "speed_conditioned",
    )
    blocks = {
        name: require_config_mapping(config, name, path="data.augmentation")
        for name in block_names
    }
    for name, block in blocks.items():
        _value(block, "enabled", bool, path=f"data.augmentation.{name}")
        probability = float(
            cast(
                "float | int",
                _value(
                    block,
                    "prob",
                    (float, int),
                    path=f"data.augmentation.{name}",
                ),
            )
        )
        _probability(probability, path=f"data.augmentation.{name}.prob")

    uv_scale = blocks["uv_scale"]
    _ordered_range(
        uv_scale["scale_range"],
        path="data.augmentation.uv_scale.scale_range",
        positive=True,
    )
    for key in ("apply_to_ball", "apply_to_court"):
        _value(uv_scale, key, bool, path="data.augmentation.uv_scale")

    gaussian = blocks["gaussian_noise"]
    for key in ("ball_std", "court_std"):
        _non_negative(
            float(
                cast(
                    "float | int",
                    _value(
                        gaussian,
                        key,
                        (float, int),
                        path="data.augmentation.gaussian_noise",
                    ),
                )
            ),
            path=f"data.augmentation.gaussian_noise.{key}",
        )

    visibility = blocks["visibility_dropout"]
    _probability(
        float(
            cast(
                "float | int",
                _value(
                    visibility,
                    "drop_prob",
                    (float, int),
                    path="data.augmentation.visibility_dropout",
                ),
            )
        ),
        path="data.augmentation.visibility_dropout.drop_prob",
    )

    temporal = blocks["temporal_jitter"]
    for key in ("jitter_std", "drift_std"):
        _non_negative(
            float(
                cast(
                    "float | int",
                    _value(
                        temporal,
                        key,
                        (float, int),
                        path="data.augmentation.temporal_jitter",
                    ),
                )
            ),
            path=f"data.augmentation.temporal_jitter.{key}",
        )
    _probability(
        float(
            cast(
                "float | int",
                _value(
                    temporal,
                    "drift_decay",
                    (float, int),
                    path="data.augmentation.temporal_jitter",
                ),
            )
        ),
        path="data.augmentation.temporal_jitter.drift_decay",
    )

    burst = blocks["burst_dropout"]
    _probability(
        float(
            cast(
                "float | int",
                _value(
                    burst,
                    "track_prob",
                    (float, int),
                    path="data.augmentation.burst_dropout",
                ),
            )
        ),
        path="data.augmentation.burst_dropout.track_prob",
    )
    burst_min = cast(
        "int", _value(burst, "min_len", int, path="data.augmentation.burst_dropout")
    )
    burst_max = cast(
        "int", _value(burst, "max_len", int, path="data.augmentation.burst_dropout")
    )
    max_bursts = cast(
        "int",
        _value(burst, "max_bursts", int, path="data.augmentation.burst_dropout"),
    )
    if burst_min <= 0 or burst_max < burst_min or max_bursts <= 0:
        raise SemanticConfigurationError(
            "data.augmentation.burst_dropout lengths/count must define a positive ordered range."
        )

    false_positive = blocks["false_positive"]
    for key in ("prob_absent", "prob_after_dropout"):
        _probability(
            float(
                cast(
                    "float | int",
                    _value(
                        false_positive,
                        key,
                        (float, int),
                        path="data.augmentation.false_positive",
                    ),
                )
            ),
            path=f"data.augmentation.false_positive.{key}",
        )
    if (
        cast(
            "int",
            _value(
                false_positive,
                "after_dropout_window",
                int,
                path="data.augmentation.false_positive",
            ),
        )
        < 0
    ):
        raise SemanticConfigurationError(
            "data.augmentation.false_positive.after_dropout_window must be non-negative."
        )

    edge = blocks["edge_degradation"]
    edge_margin = float(
        cast(
            "float | int",
            _value(
                edge,
                "edge_margin",
                (float, int),
                path="data.augmentation.edge_degradation",
            ),
        )
    )
    if not 0.0 <= edge_margin <= 0.5:
        raise SemanticConfigurationError(
            "data.augmentation.edge_degradation.edge_margin must be within [0, 0.5]."
        )
    _non_negative(
        float(
            cast(
                "float | int",
                _value(
                    edge,
                    "noise_std",
                    (float, int),
                    path="data.augmentation.edge_degradation",
                ),
            )
        ),
        path="data.augmentation.edge_degradation.noise_std",
    )
    for key in ("drop_prob", "clip_out_prob"):
        _probability(
            float(
                cast(
                    "float | int",
                    _value(
                        edge,
                        key,
                        (float, int),
                        path="data.augmentation.edge_degradation",
                    ),
                )
            ),
            path=f"data.augmentation.edge_degradation.{key}",
        )

    speed = blocks["speed_conditioned"]
    _probability(
        float(
            cast(
                "float | int",
                _value(
                    speed,
                    "frame_prob",
                    (float, int),
                    path="data.augmentation.speed_conditioned",
                ),
            )
        ),
        path="data.augmentation.speed_conditioned.frame_prob",
    )
    for key in ("speed_threshold", "noise_std"):
        _non_negative(
            float(
                cast(
                    "float | int",
                    _value(
                        speed,
                        key,
                        (float, int),
                        path="data.augmentation.speed_conditioned",
                    ),
                )
            ),
            path=f"data.augmentation.speed_conditioned.{key}",
        )
    _ordered_range(
        speed["lag_overshoot_range"],
        path="data.augmentation.speed_conditioned.lag_overshoot_range",
    )


def parse_qualitative_rendering(config: object) -> QualitativeRenderingConfig:
    """Parse the exact BLCS qualitative rendering extension."""
    root = as_config_mapping(config, path="configuration")
    training = require_config_mapping(root, "training", path="configuration")
    rendering = require_config_mapping(
        training, "qualitative_rendering", path="training"
    )
    _exact(rendering, {"style", "fps"}, path="training.qualitative_rendering")
    style = require_config_mapping(
        rendering, "style", path="training.qualitative_rendering"
    )
    _exact(
        style,
        {
            "theme",
            "show_shadow",
            "show_trail",
            "trail_length",
            "show_hud",
            "show_minimap",
        },
        path="training.qualitative_rendering.style",
    )
    _validate_types(
        style,
        {
            "theme": str,
            "show_shadow": bool,
            "show_trail": bool,
            "trail_length": int,
            "show_hud": bool,
            "show_minimap": bool,
        },
        path="training.qualitative_rendering.style",
    )
    parsed_style = SceneStyleConfig(
        theme=cast("str", style["theme"]),
        show_shadow=cast("bool", style["show_shadow"]),
        show_trail=cast("bool", style["show_trail"]),
        trail_length=cast("int", style["trail_length"]),
        show_hud=cast("bool", style["show_hud"]),
        show_minimap=cast("bool", style["show_minimap"]),
    )
    # Use the shared strict parser to enforce theme and trail semantics.
    parse_scene_style(dict(style))
    fps = cast(
        "float",
        _value(rendering, "fps", float, path="training.qualitative_rendering"),
    )
    if fps <= 0.0:
        raise SemanticConfigurationError(
            "training.qualitative_rendering.fps must be > 0."
        )
    return QualitativeRenderingConfig(style=parsed_style, fps=fps)


def parse_preview_config(config: object) -> PreviewConfig:
    """Validate augmentation-preview settings and resolve data/output paths."""
    root = as_config_mapping(config, path="configuration")
    preview = require_config_mapping(root, "preview", path="configuration")
    data = require_config_mapping(root, "data", path="configuration")
    figure = require_config_mapping(preview, "figure", path="preview")
    _exact(
        preview,
        {
            "split",
            "sample_indices",
            "max_samples",
            "num_augmented",
            "seed",
            "max_cameras",
            "output_dir",
            "figure",
        },
        path="preview",
    )
    _exact(figure, {"panel_width", "panel_height", "dpi"}, path="preview.figure")
    resolver = build_path_resolver(config)
    result = PreviewConfig(
        output_dir=resolver.resolve(
            PathRole.OUTPUT,
            cast("str", _value(preview, "output_dir", str, path="preview")),
        ),
        scene_dir=resolver.resolve(
            PathRole.DATA, cast("str", _value(data, "scene_dir", str, path="data"))
        ),
        split=cast("str", _value(preview, "split", str, path="preview")),
        sample_indices=_int_sequence(
            preview["sample_indices"], path="preview.sample_indices"
        ),
        max_samples=cast("int", _value(preview, "max_samples", int, path="preview")),
        num_augmented=cast(
            "int", _value(preview, "num_augmented", int, path="preview")
        ),
        seed=cast("int", _value(preview, "seed", int, path="preview")),
        max_cameras=cast("int", _value(preview, "max_cameras", int, path="preview")),
        panel_width=float(
            cast(
                "float | int",
                _value(figure, "panel_width", (float, int), path="preview.figure"),
            )
        ),
        panel_height=float(
            cast(
                "float | int",
                _value(figure, "panel_height", (float, int), path="preview.figure"),
            )
        ),
        dpi=cast("int", _value(figure, "dpi", int, path="preview.figure")),
    )
    if (
        min(result.max_samples, result.num_augmented, result.max_cameras, result.dpi)
        < 1
    ):
        raise SemanticConfigurationError(
            "BLCS preview counts and dpi must be positive."
        )
    return result


def parse_generation_run(config: object) -> tuple[GenerationRunConfig, PathResolver]:
    """Validate generation run fields and resolve its data output."""
    root = as_config_mapping(config, path="configuration")
    _exact(
        root,
        {
            "paths",
            "court_keypoints",
            "generation",
            "physics",
            "rally",
            "camera",
            "targeted_velocity",
            "generator",
            "run",
        },
        path="configuration",
    )
    parse_court_keypoint_contract(config)
    validate_generator_sections(config)
    run = require_config_mapping(root, "run", path="configuration")
    keys = {
        "output_dir",
        "seed",
        "device",
        "num_workers",
        "chunksize",
        "train_ratio",
        "val_ratio",
    }
    _exact(run, keys, path="run")
    resolver = build_path_resolver(config)
    requested_device = cast("str", _value(run, "device", str, path="run"))
    try:
        device = str(resolve_device(requested_device))
    except (RuntimeError, ValueError) as error:
        raise SemanticConfigurationError(
            f"run.device is not an available device: {requested_device!r}."
        ) from error
    result = GenerationRunConfig(
        output_dir=resolver.resolve(
            PathRole.DATA, cast("str", _value(run, "output_dir", str, path="run"))
        ),
        seed=cast("int", _value(run, "seed", int, path="run")),
        device=device,
        num_workers=cast("int", _value(run, "num_workers", int, path="run")),
        chunksize=cast("int", _value(run, "chunksize", int, path="run")),
        train_ratio=float(
            cast("float | int", _value(run, "train_ratio", (float, int), path="run"))
        ),
        val_ratio=float(
            cast("float | int", _value(run, "val_ratio", (float, int), path="run"))
        ),
    )
    _probability(result.train_ratio, path="run.train_ratio")
    _probability(result.val_ratio, path="run.val_ratio")
    if (
        result.num_workers < 1
        or result.chunksize < 1
        or result.train_ratio + result.val_ratio > 1
    ):
        raise SemanticConfigurationError(
            "Invalid BLCS generation worker count or split ratios."
        )
    if result.device != "cpu":
        raise SemanticConfigurationError(
            "run.device must resolve to 'cpu' for parallel BLCS generation."
        )
    return result, resolver


def validate_generator_sections(
    config: object, *, include_generation: bool = True
) -> None:
    """Reject missing, unknown, and legacy BLCS generator section keys."""
    root = as_config_mapping(config, path="configuration")
    schemas = {
        "physics": {
            "gravity",
            "k_drag",
            "k_magnus",
            "e_z",
            "mu",
            "alpha_net",
            "alpha_net_cord",
            "alpha_fence",
            "net_half_thickness",
            "net_cord_radius",
            "dt",
            "use_drag",
            "use_magnus",
            "wind",
            "gravity_range",
            "k_drag_range",
            "k_magnus_range",
            "e_z_range",
            "mu_range",
            "wind_speed_range",
            "wind_direction_range_deg",
        },
        "rally": {
            "z_range",
            "spin_x_range",
            "spin_y_range",
            "spin_z_range",
            "max_sim_frames",
            "output_fps",
            "sim_fps",
            "max_rallies",
            "max_total_frames",
            "hit_timing_range",
            "return_z_range",
            "serve_probability",
            "serve_z_range",
            "toss_vz_range",
            "toss_xy_noise_range",
            "toss_max_frames",
            "toss_z0_tolerance",
            "volley_probability",
            "normal_return_probability",
            "late_return_probability",
            "out_court_target_probability",
        },
        "camera": {
            "layout",
            "z_min",
            "z_max",
            "hfov_deg",
            "image_size",
            "fixed_look_at",
            "fixed_baseline_clear_extra",
            "fixed_position_noise_radius",
            "fixed_look_at_xy_radius",
            "broadcast_setback",
            "broadcast_height",
            "broadcast_hfov_deg",
            "broadcast_look_at_y",
            "broadcast_look_at_height",
            "broadcast_position_noise_radius",
            "broadcast_look_at_xy_radius",
            "broadcast_hfov_jitter_deg",
            "broadcast_setback_range",
            "broadcast_height_range",
            "broadcast_court_width_frac_range",
        },
        "targeted_velocity": {
            "drive_elevation_range_deg",
            "lob_elevation_range_deg",
            "lob_probability",
            "max_ballistic_apex_height_m",
            "gravity",
            "net_elevation_step_deg",
            "landing_refine_enabled",
            "landing_refine_max_iters",
            "landing_refine_tolerance_m",
            "landing_sim_max_frames",
            "target_margin_m",
        },
        "generator": {"num_scenes", "court"},
    }
    sections = {
        section: require_config_mapping(root, section, path="configuration")
        for section in schemas
    }
    for section, keys in schemas.items():
        _exact(sections[section], keys, path=section)

    physics = sections["physics"]
    _validate_types(
        physics,
        {
            **{
                key: float
                for key in (
                    "gravity",
                    "k_drag",
                    "k_magnus",
                    "e_z",
                    "mu",
                    "alpha_net",
                    "alpha_net_cord",
                    "alpha_fence",
                    "net_half_thickness",
                    "net_cord_radius",
                    "dt",
                )
            },
            "use_drag": bool,
            "use_magnus": bool,
            **{
                key: list
                for key in (
                    "wind",
                    "gravity_range",
                    "k_drag_range",
                    "k_magnus_range",
                    "e_z_range",
                    "mu_range",
                    "wind_speed_range",
                    "wind_direction_range_deg",
                )
            },
        },
        path="physics",
    )
    _numeric_sequence(physics["wind"], path="physics.wind", length=3)
    _positive(cast("float", physics["gravity"]), path="physics.gravity")
    for key in ("k_drag", "k_magnus", "mu"):
        _non_negative(cast("float", physics[key]), path=f"physics.{key}")
    for key in ("e_z", "alpha_net", "alpha_net_cord", "alpha_fence"):
        _probability(cast("float", physics[key]), path=f"physics.{key}")
    for key in ("net_half_thickness", "net_cord_radius", "dt"):
        _positive(cast("float", physics[key]), path=f"physics.{key}")
    _ordered_range(
        physics["gravity_range"], path="physics.gravity_range", positive=True
    )
    for key in ("k_drag_range", "k_magnus_range", "mu_range", "wind_speed_range"):
        _ordered_range(physics[key], path=f"physics.{key}", lower_bound=0.0)
    _ordered_range(
        physics["e_z_range"],
        path="physics.e_z_range",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    _ordered_range(
        physics["wind_direction_range_deg"],
        path="physics.wind_direction_range_deg",
    )

    rally = sections["rally"]
    rally_ranges = (
        "z_range",
        "spin_x_range",
        "spin_y_range",
        "spin_z_range",
        "hit_timing_range",
        "return_z_range",
        "serve_z_range",
        "toss_vz_range",
        "toss_xy_noise_range",
    )
    _validate_types(
        rally,
        {
            **{key: list for key in rally_ranges},
            **{
                key: int
                for key in (
                    "max_sim_frames",
                    "output_fps",
                    "sim_fps",
                    "max_rallies",
                    "max_total_frames",
                    "toss_max_frames",
                )
            },
            **{
                key: float
                for key in (
                    "serve_probability",
                    "toss_z0_tolerance",
                    "volley_probability",
                    "normal_return_probability",
                    "late_return_probability",
                    "out_court_target_probability",
                )
            },
        },
        path="rally",
    )
    for key in rally_ranges:
        _ordered_range(rally[key], path=f"rally.{key}")
    for key in ("z_range", "return_z_range", "serve_z_range", "toss_vz_range"):
        _ordered_range(rally[key], path=f"rally.{key}", positive=True)
    _ordered_range(
        rally["hit_timing_range"],
        path="rally.hit_timing_range",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    for key in (
        "max_sim_frames",
        "output_fps",
        "sim_fps",
        "max_rallies",
        "max_total_frames",
        "toss_max_frames",
    ):
        if cast("int", rally[key]) <= 0:
            raise SemanticConfigurationError(f"rally.{key} must be positive.")
    output_fps = cast("int", rally["output_fps"])
    sim_fps = cast("int", rally["sim_fps"])
    if sim_fps < output_fps or sim_fps % output_fps:
        raise SemanticConfigurationError(
            "rally.sim_fps must be >= and divisible by rally.output_fps."
        )
    probability_keys = (
        "serve_probability",
        "volley_probability",
        "normal_return_probability",
        "late_return_probability",
        "out_court_target_probability",
    )
    for key in probability_keys:
        _probability(cast("float", rally[key]), path=f"rally.{key}")
    if (
        sum(
            cast("float", rally[key])
            for key in (
                "volley_probability",
                "normal_return_probability",
                "late_return_probability",
            )
        )
        <= 0.0
    ):
        raise SemanticConfigurationError(
            "At least one rally return-type probability must be positive."
        )
    _non_negative(
        cast("float", rally["toss_z0_tolerance"]),
        path="rally.toss_z0_tolerance",
    )

    camera = sections["camera"]
    _validate_types(
        camera,
        {
            "layout": str,
            **{
                key: float
                for key in (
                    "z_min",
                    "z_max",
                    "hfov_deg",
                    "fixed_baseline_clear_extra",
                    "fixed_position_noise_radius",
                    "fixed_look_at_xy_radius",
                    "broadcast_setback",
                    "broadcast_height",
                    "broadcast_hfov_deg",
                    "broadcast_look_at_y",
                    "broadcast_look_at_height",
                    "broadcast_position_noise_radius",
                    "broadcast_look_at_xy_radius",
                    "broadcast_hfov_jitter_deg",
                )
            },
            "image_size": list,
            "fixed_look_at": list,
            "broadcast_setback_range": (list, type(None)),
            "broadcast_height_range": (list, type(None)),
            "broadcast_court_width_frac_range": (list, type(None)),
        },
        path="camera",
    )
    _int_sequence(camera["image_size"], path="camera.image_size")
    if len(cast("Sequence[object]", camera["image_size"])) != 2:
        raise SemanticConfigurationError("camera.image_size must contain two values.")
    _numeric_sequence(camera["fixed_look_at"], path="camera.fixed_look_at", length=3)
    if camera["layout"] not in {"fixed", "broadcast"}:
        raise SemanticConfigurationError(
            "camera.layout must be 'fixed' or 'broadcast'."
        )
    image_size = _int_sequence(camera["image_size"], path="camera.image_size")
    if any(value <= 0 for value in image_size):
        raise SemanticConfigurationError("camera.image_size values must be positive.")
    z_min = cast("float", camera["z_min"])
    z_max = cast("float", camera["z_max"])
    if z_min <= 0.0 or z_max < z_min:
        raise SemanticConfigurationError(
            "camera z range must satisfy 0 < z_min <= z_max."
        )
    for key in ("hfov_deg", "broadcast_hfov_deg"):
        value = _finite(cast("float", camera[key]), path=f"camera.{key}")
        if not 0.0 < value < 180.0:
            raise SemanticConfigurationError(f"camera.{key} must be within (0, 180).")
    for key in (
        "fixed_baseline_clear_extra",
        "fixed_position_noise_radius",
        "fixed_look_at_xy_radius",
        "broadcast_setback",
        "broadcast_position_noise_radius",
        "broadcast_look_at_xy_radius",
        "broadcast_hfov_jitter_deg",
        "broadcast_look_at_height",
    ):
        _non_negative(cast("float", camera[key]), path=f"camera.{key}")
    _positive(cast("float", camera["broadcast_height"]), path="camera.broadcast_height")
    jitter = cast("float", camera["broadcast_hfov_jitter_deg"])
    broadcast_hfov = cast("float", camera["broadcast_hfov_deg"])
    if broadcast_hfov - jitter <= 0.0 or broadcast_hfov + jitter >= 180.0:
        raise SemanticConfigurationError(
            "camera.broadcast_hfov_jitter_deg must keep every sampled HFOV in (0, 180)."
        )
    for key in (
        "broadcast_setback_range",
        "broadcast_height_range",
        "broadcast_court_width_frac_range",
    ):
        range_value = camera[key]
        if range_value is not None:
            _ordered_range(
                range_value,
                path=f"camera.{key}",
                positive=True,
                upper_bound=(
                    1.0 if key == "broadcast_court_width_frac_range" else None
                ),
            )
    if camera["broadcast_court_width_frac_range"] is not None and jitter != 0.0:
        raise SemanticConfigurationError(
            "camera.broadcast_court_width_frac_range and non-zero "
            "camera.broadcast_hfov_jitter_deg are mutually exclusive."
        )

    targeted_velocity = sections["targeted_velocity"]
    targeted_ranges = ("drive_elevation_range_deg", "lob_elevation_range_deg")
    _validate_types(
        targeted_velocity,
        {
            **{key: list for key in targeted_ranges},
            **{
                key: float
                for key in (
                    "lob_probability",
                    "max_ballistic_apex_height_m",
                    "gravity",
                    "net_elevation_step_deg",
                    "landing_refine_tolerance_m",
                    "target_margin_m",
                )
            },
            **{
                key: int
                for key in (
                    "landing_refine_max_iters",
                    "landing_sim_max_frames",
                )
            },
            "landing_refine_enabled": bool,
        },
        path="targeted_velocity",
    )
    for key in targeted_ranges:
        lo, hi = _ordered_range(
            targeted_velocity[key],
            path=f"targeted_velocity.{key}",
            positive=True,
        )
        if hi >= 90.0:
            raise SemanticConfigurationError(
                f"targeted_velocity.{key} must stay below 90 degrees."
            )
        del lo
    _probability(
        cast("float", targeted_velocity["lob_probability"]),
        path="targeted_velocity.lob_probability",
    )
    for key in ("max_ballistic_apex_height_m", "gravity"):
        _positive(
            cast("float", targeted_velocity[key]), path=f"targeted_velocity.{key}"
        )
    for key in ("landing_sim_max_frames",):
        if cast("int", targeted_velocity[key]) <= 0:
            raise SemanticConfigurationError(
                f"targeted_velocity.{key} must be positive."
            )
    refine_iters = cast("int", targeted_velocity["landing_refine_max_iters"])
    if refine_iters < 0 or (
        cast("bool", targeted_velocity["landing_refine_enabled"]) and refine_iters == 0
    ):
        raise SemanticConfigurationError(
            "targeted_velocity.landing_refine_max_iters must be non-negative and "
            "positive when refinement is enabled."
        )
    for key in (
        "net_elevation_step_deg",
        "landing_refine_tolerance_m",
    ):
        _positive(
            cast("float", targeted_velocity[key]), path=f"targeted_velocity.{key}"
        )
    _non_negative(
        cast("float", targeted_velocity["target_margin_m"]),
        path="targeted_velocity.target_margin_m",
    )

    generator = sections["generator"]
    _value(generator, "num_scenes", int, path="generator")
    if cast("int", generator["num_scenes"]) <= 0:
        raise SemanticConfigurationError("generator.num_scenes must be positive.")
    court = require_config_mapping(generator, "court", path="generator")
    _exact(
        court, {"net_post_offset_x", "net_post_offset_x_range"}, path="generator.court"
    )
    _validate_types(
        court,
        {"net_post_offset_x": float, "net_post_offset_x_range": list},
        path="generator.court",
    )
    _ordered_range(
        court["net_post_offset_x_range"],
        path="generator.court.net_post_offset_x_range",
    )
    _finite(
        cast("float", court["net_post_offset_x"]),
        path="generator.court.net_post_offset_x",
    )
    if not include_generation:
        return
    generation = require_config_mapping(root, "generation", path="configuration")
    mode = cast("str", _value(generation, "mode", str, path="generation"))
    if mode == "single_object":
        _exact(
            generation,
            {
                "mode",
                "min_balls",
                "max_balls",
                "maximum_physics_attempts_per_scene",
            },
            path="generation",
        )
        _validate_types(
            generation,
            {
                "mode": str,
                "min_balls": int,
                "max_balls": int,
                "maximum_physics_attempts_per_scene": int,
            },
            path="generation",
        )
        min_balls = cast("int", generation["min_balls"])
        max_balls = cast("int", generation["max_balls"])
        if min_balls <= 0 or max_balls < min_balls:
            raise SemanticConfigurationError(
                "generation ball counts must satisfy 1 <= min_balls <= max_balls."
            )
        maximum_attempts = cast(
            "int",
            _value(
                generation,
                "maximum_physics_attempts_per_scene",
                int,
                path="generation",
            ),
        )
        if maximum_attempts <= 0:
            raise SemanticConfigurationError(
                "generation.maximum_physics_attempts_per_scene must be positive."
            )
    elif mode == "multi_object":
        _exact(
            generation,
            {"mode", "maximum_physics_attempts_per_object", "timeline"},
            path="generation",
        )
        maximum_attempts = cast(
            "int",
            _value(
                generation,
                "maximum_physics_attempts_per_object",
                int,
                path="generation",
            ),
        )
        if maximum_attempts <= 0:
            raise SemanticConfigurationError(
                "generation.maximum_physics_attempts_per_object must be positive."
            )
        timeline = require_config_mapping(generation, "timeline", path="generation")
        _exact(
            timeline,
            {
                "num_frames",
                "min_tracks",
                "max_tracks",
                "max_concurrent",
                "min_reuse_gap_frames",
                "start_index_range",
                "min_active_frames",
                "overlap_probability",
                "min_gap_frames",
                "max_gap_frames",
            },
            path="generation.timeline",
        )
        _validate_types(
            timeline,
            {
                **{
                    key: int
                    for key in (
                        "num_frames",
                        "min_tracks",
                        "max_tracks",
                        "max_concurrent",
                        "min_reuse_gap_frames",
                        "min_active_frames",
                        "min_gap_frames",
                        "max_gap_frames",
                    )
                },
                "start_index_range": list,
                "overlap_probability": float,
            },
            path="generation.timeline",
        )
        _int_sequence(
            timeline["start_index_range"], path="generation.timeline.start_index_range"
        )
        if len(cast("Sequence[object]", timeline["start_index_range"])) != 2:
            raise SemanticConfigurationError(
                "generation.timeline.start_index_range must contain two values."
            )
        try:
            TimelineConfig.from_mapping(cast("Mapping[str, Any]", timeline))
        except (TypeError, ValueError) as error:
            raise SemanticConfigurationError(str(error)) from error
    else:
        raise SemanticConfigurationError(
            "generation.mode must be 'single_object' or 'multi_object'."
        )


def validate_training_boundary(config: object) -> BLCSModelConfig:
    """Validate BLCS-specific model and the seven-root contract before training."""
    root = as_config_mapping(config, path="configuration")
    model = parse_model_config(config)
    if isinstance(model, _TRACK_QUERY_MODEL_CONFIG_TYPES):
        allowed = {
            "paths",
            "court_keypoints",
            "model",
            "data",
            "training",
            "loss",
            "tracking_metrics",
            "run",
        }
        data = require_config_mapping(root, "data", path="configuration")
        backend = cast("str", _value(data, "backend", str, path="data"))
        if backend == "chunked":
            allowed |= {
                "generation",
                "physics",
                "rally",
                "camera",
                "targeted_velocity",
                "generator",
            }
    else:
        allowed = {
            "paths",
            "court_keypoints",
            "model",
            "data",
            "training",
            "metrics",
            "run",
            "physics",
            "rally",
            "camera",
            "targeted_velocity",
            "generator",
        }
    _exact(root, allowed, path="configuration")
    court_keypoint_contract = parse_court_keypoint_contract(config)
    if isinstance(
        model,
        TrackQueryReferenceModelConfig,
    ):
        if court_keypoint_contract.selector != "camera_view_v2":
            raise SemanticConfigurationError(
                "BLCS reference track-query models require "
                "court_keypoints.selector='camera_view_v2'."
            )
        if model.target_frame_contract != court_keypoint_contract.target_frame_id:
            raise SemanticConfigurationError(
                "BLCS reference model target-frame and CourtKP20 contracts "
                "must match exactly."
            )
    elif isinstance(model, TrackQueryModelConfig):
        if court_keypoint_contract.selector != "physical_v1":
            raise SemanticConfigurationError(
                "Canonical BLCS track-query models require "
                "court_keypoints.selector='physical_v1'; select an explicit "
                "reference-v2 model for camera_view_v2."
            )
    build_path_resolver(config)
    data = require_config_mapping(root, "data", path="configuration")
    backend = cast("str", _value(data, "backend", str, path="data"))
    data_keys = {
        "backend",
        "scene_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "camera_mode",
        "num_views_range",
        "seq_len_range",
        "augmentation",
    }
    if isinstance(model, _TRACK_QUERY_MODEL_CONFIG_TYPES):
        data_keys.update({"association", "lifecycle", "evaluation_reference_camera_id"})
        lifecycle = require_config_mapping(data, "lifecycle", path="data")
        _exact(
            lifecycle,
            {"pack_to_query_slots", "min_reuse_gap_frames"},
            path="data.lifecycle",
        )
        association = ObservationTrackingConfig.from_mapping(
            require_config_mapping(data, "association", path="data")
        )
    else:
        data_keys.add("num_court_kp")
    if backend == "chunked":
        data_keys |= {"generator_device", "chunk"}
        _value(data, "generator_device", str, path="data")
        chunk = require_config_mapping(data, "chunk", path="data")
        _exact(
            chunk,
            {
                "scenes_per_chunk",
                "epochs_per_chunk",
                "prefetch_chunks",
                "chunks_dir",
                "generation_workers",
                "generation_chunksize",
            },
            path="data.chunk",
        )
        _validate_types(
            chunk,
            {
                "scenes_per_chunk": int,
                "epochs_per_chunk": int,
                "prefetch_chunks": int,
                "chunks_dir": str,
                "generation_workers": int,
                "generation_chunksize": int,
            },
            path="data.chunk",
        )
        for key in (
            "scenes_per_chunk",
            "epochs_per_chunk",
            "prefetch_chunks",
            "generation_workers",
            "generation_chunksize",
        ):
            if cast("int", chunk[key]) <= 0:
                raise SemanticConfigurationError(f"data.chunk.{key} must be positive.")
        validate_generator_sections(
            config,
            include_generation=model.name in _TRACK_QUERY_MODEL_NAMES,
        )
        if isinstance(
            model,
            (TrackQueryModelConfig, TrackQueryReferenceModelConfig),
        ):
            generation = require_config_mapping(
                root, "generation", path="configuration"
            )
            if generation["mode"] != "multi_object":
                raise SemanticConfigurationError(
                    "Chunked BLCS tracking requires generation.mode='multi_object'."
                )
            timeline = require_config_mapping(generation, "timeline", path="generation")
            if cast("int", timeline["max_concurrent"]) > model.num_queries:
                raise SemanticConfigurationError(
                    "generation.timeline.max_concurrent cannot exceed model.num_queries."
                )
            lifecycle_gap = cast("int", lifecycle["min_reuse_gap_frames"])
            if cast("int", timeline["min_reuse_gap_frames"]) < lifecycle_gap:
                raise SemanticConfigurationError(
                    "generation.timeline.min_reuse_gap_frames cannot be smaller than "
                    "data.lifecycle.min_reuse_gap_frames."
                )
    _exact(data, data_keys, path="data")
    data_types: dict[str, type[object]] = {
        "backend": str,
        "scene_dir": str,
        "batch_size": int,
        "num_workers": int,
        "pin_memory": bool,
        "camera_mode": str,
        "num_views_range": list,
        "seq_len_range": list,
        "augmentation": dict,
    }
    if isinstance(model, _TRACK_QUERY_MODEL_CONFIG_TYPES):
        data_types["evaluation_reference_camera_id"] = str
    _validate_types(data, data_types, path="data")
    if isinstance(model, _TRACK_QUERY_MODEL_CONFIG_TYPES):
        evaluation_reference_camera_id = cast(
            "str", data["evaluation_reference_camera_id"]
        )
        if not evaluation_reference_camera_id.strip():
            raise SemanticConfigurationError(
                "data.evaluation_reference_camera_id must be a non-empty stable "
                "camera identity."
            )
    if backend not in {"default", "chunked"}:
        raise SemanticConfigurationError("data.backend must be 'default' or 'chunked'.")
    if data["camera_mode"] not in {"random", "first"}:
        raise SemanticConfigurationError(
            "data.camera_mode must be 'random' or 'first'."
        )
    num_views_range = _int_sequence(
        data["num_views_range"], path="data.num_views_range"
    )
    seq_len_range = _int_sequence(data["seq_len_range"], path="data.seq_len_range")
    if len(num_views_range) != 2 or len(seq_len_range) != 2:
        raise SemanticConfigurationError(
            "data.num_views_range and data.seq_len_range must contain two values."
        )
    for name, values in (
        ("num_views_range", num_views_range),
        ("seq_len_range", seq_len_range),
    ):
        if values[0] <= 0 or values[1] < values[0]:
            raise SemanticConfigurationError(
                f"data.{name} must be a positive ordered range."
            )
    batch_size = cast("int", data["batch_size"])
    num_workers = cast("int", data["num_workers"])
    if batch_size <= 0 or num_workers < 0:
        raise SemanticConfigurationError(
            "data.batch_size must be positive and data.num_workers non-negative."
        )
    if (
        isinstance(model, (SingleModelConfig, AxialModelConfig))
        and seq_len_range[1] > model.max_seq_len
    ):
        raise SemanticConfigurationError(
            "data.seq_len_range cannot exceed model.max_seq_len."
        )
    if (
        isinstance(model, AxialModelConfig)
        and num_views_range[1] > model.max_num_cameras
    ):
        raise SemanticConfigurationError(
            "data.num_views_range cannot exceed model.max_num_cameras."
        )
    if isinstance(model, SingleModelConfig) and num_views_range != (1, 1):
        raise SemanticConfigurationError(
            "Single-view BLCS models require data.num_views_range=[1, 1]."
        )
    if isinstance(model, _TRACK_QUERY_MODEL_CONFIG_TYPES):
        _validate_types(
            lifecycle,
            {
                "pack_to_query_slots": bool,
                "min_reuse_gap_frames": int,
            },
            path="data.lifecycle",
        )
        if lifecycle["pack_to_query_slots"] is not True:
            raise SemanticConfigurationError(
                "BLCS track-query training requires "
                "data.lifecycle.pack_to_query_slots=true."
            )
        if cast("int", lifecycle["min_reuse_gap_frames"]) < 0:
            raise SemanticConfigurationError(
                "data.lifecycle.min_reuse_gap_frames must be non-negative."
            )
        if association.min_common_keypoints != 1:
            raise SemanticConfigurationError(
                "BLCS point tracking requires data.association.min_common_keypoints=1."
            )
        if association.cost_reduction != "mean":
            raise SemanticConfigurationError(
                "BLCS point tracking requires data.association.cost_reduction='mean'."
            )
    else:
        num_court_kp = cast("int", _value(data, "num_court_kp", int, path="data"))
        if num_court_kp <= 0 or num_court_kp != model.num_court_tokens:
            raise SemanticConfigurationError(
                "data.num_court_kp must be positive and equal model.num_court_tokens."
            )
    from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation

    augmentation = require_config_mapping(data, "augmentation", path="data")
    BLCSBallObservationAugmentation(augmentation)
    _validate_augmentation(augmentation)
    training = require_config_mapping(root, "training", path="configuration")
    training_keys = {
        "trainer",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "warmup_epochs",
        "min_lr",
        "steps_per_epoch",
        "optimizer",
        "compile",
        "position_loss_weight",
        "position_axis_weights",
        "reprojection_loss_weight",
        "smoothness_loss_weight",
        "gravity_loss_weight",
        "smoothness_order",
        "smoothness_beta",
        "gravity_beta",
        "smoothness_axis_weights",
        "checkpoint",
        "early_stopping",
        "lr_monitor",
        "qualitative_logging",
        "qualitative_rendering",
        "matmul_precision",
        "allow_tf32",
        "gan",
    }
    _exact(training, training_keys, path="training")
    _validate_types(
        training,
        {
            "position_loss_weight": float,
            "position_axis_weights": (list, type(None)),
            "reprojection_loss_weight": float,
            "smoothness_loss_weight": float,
            "gravity_loss_weight": float,
            "smoothness_order": int,
            "smoothness_beta": float,
            "gravity_beta": float,
            "smoothness_axis_weights": (list, type(None)),
            "qualitative_rendering": dict,
        },
        path="training",
    )
    for key in ("position_axis_weights", "smoothness_axis_weights"):
        weights = _optional_numeric_sequence(
            training[key], path=f"training.{key}", length=3
        )
        if weights is not None and any(weight < 0.0 for weight in weights):
            raise SemanticConfigurationError(
                f"training.{key} values must be non-negative."
            )
    for key in (
        "position_loss_weight",
        "reprojection_loss_weight",
        "smoothness_loss_weight",
        "gravity_loss_weight",
        "smoothness_beta",
        "gravity_beta",
    ):
        _non_negative(cast("float", training[key]), path=f"training.{key}")
    if cast("int", training["smoothness_order"]) < 1:
        raise SemanticConfigurationError("training.smoothness_order must be >= 1.")
    TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    parse_qualitative_rendering(config)
    gan = require_config_mapping(training, "gan", path="training")
    _exact(
        gan,
        {
            "enabled",
            "target_weight",
            "warmup_epochs",
            "generator_gradient_clip_val",
            "discriminator_gradient_clip_val",
            "transition",
            "discriminator",
        },
        path="training.gan",
    )
    discriminator = require_config_mapping(gan, "discriminator", path="training.gan")
    _exact(
        discriminator,
        {
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
        },
        path="training.gan.discriminator",
    )
    _validate_types(
        discriminator,
        {
            "name": str,
            "hidden_dim": int,
            "num_layers": int,
            "num_heads": int,
            "ffn_dim": int,
            "ffn_type": str,
            "dropout": float,
            "rope_dim": int,
            "rope_theta": float,
            "max_seq_len": int,
            "invalid_init_std": float,
            "cls_init_std": float,
        },
        path="training.gan.discriminator",
    )
    if discriminator["name"] != "trajectory_transformer":
        raise SemanticConfigurationError(
            "training.gan.discriminator.name must be 'trajectory_transformer'."
        )
    _validate_transformer_dimensions(
        hidden_dim=cast("int", discriminator["hidden_dim"]),
        num_heads=cast("int", discriminator["num_heads"]),
        ffn_dim=cast("int", discriminator["ffn_dim"]),
        rope_dim=cast("int", discriminator["rope_dim"]),
        dropout=cast("float", discriminator["dropout"]),
        path="training.gan.discriminator",
    )
    if cast("int", discriminator["num_layers"]) < 0:
        raise SemanticConfigurationError(
            "training.gan.discriminator.num_layers must be non-negative."
        )
    if cast("int", discriminator["max_seq_len"]) <= 0:
        raise SemanticConfigurationError(
            "training.gan.discriminator.max_seq_len must be positive."
        )
    _positive(
        cast("float", discriminator["rope_theta"]),
        path="training.gan.discriminator.rope_theta",
    )
    for key in ("invalid_init_std", "cls_init_std"):
        _non_negative(
            cast("float", discriminator[key]),
            path=f"training.gan.discriminator.{key}",
        )
    if model.name in _TRACK_QUERY_MODEL_NAMES:
        loss = require_config_mapping(root, "loss", path="configuration")
        _exact(
            loss,
            {
                "position_weight",
                "position_axis_weights",
                "presence_weight",
                "presence_inactive_weight",
                "presence_active_weight",
                "presence_transition_weight",
                "transition_radius",
                "smoothness_weight",
                "gravity_weight",
                "gravity_mps2",
                "frame_dt_seconds",
                "match_position_weight",
                "match_presence_weight",
            },
            path="loss",
        )
        _validate_types(
            loss,
            {
                "position_weight": float,
                "position_axis_weights": list,
                "presence_weight": float,
                "presence_inactive_weight": float,
                "presence_active_weight": float,
                "presence_transition_weight": float,
                "transition_radius": int,
                "smoothness_weight": float,
                "gravity_weight": float,
                "gravity_mps2": float,
                "frame_dt_seconds": float,
                "match_position_weight": float,
                "match_presence_weight": float,
            },
            path="loss",
        )
        for key in (
            "position_weight",
            "presence_weight",
            "presence_inactive_weight",
            "presence_active_weight",
            "presence_transition_weight",
            "smoothness_weight",
            "gravity_weight",
            "match_position_weight",
            "match_presence_weight",
        ):
            _non_negative(cast("float", loss[key]), path=f"loss.{key}")
        axis_weights = _numeric_sequence(
            loss["position_axis_weights"],
            path="loss.position_axis_weights",
            length=3,
        )
        if any(weight <= 0.0 for weight in axis_weights):
            raise SemanticConfigurationError(
                "loss.position_axis_weights values must be positive."
            )
        if cast("int", loss["transition_radius"]) < 0:
            raise SemanticConfigurationError(
                "loss.transition_radius must be non-negative."
            )
        if (
            cast("float", loss["match_position_weight"]) == 0.0
            and cast("float", loss["match_presence_weight"]) == 0.0
        ):
            raise SemanticConfigurationError(
                "At least one tracking match cost weight must be positive."
            )
        _positive(cast("float", loss["gravity_mps2"]), path="loss.gravity_mps2")
        _positive(
            cast("float", loss["frame_dt_seconds"]),
            path="loss.frame_dt_seconds",
        )
        TrackingMetricConfig.from_mapping(
            require_config_mapping(root, "tracking_metrics", path="configuration")
        )
    else:
        metrics = require_config_mapping(root, "metrics", path="configuration")
        _exact(
            metrics,
            {"position_threshold_m", "endpoint_threshold_m"},
            path="metrics",
        )
        _validate_types(
            metrics,
            {"position_threshold_m": float, "endpoint_threshold_m": float},
            path="metrics",
        )
        for key in ("position_threshold_m", "endpoint_threshold_m"):
            _positive(cast("float", metrics[key]), path=f"metrics.{key}")
        validate_generator_sections(config, include_generation=False)
    return model


def validate_visualization_boundary(config: object) -> None:
    """Validate the complete BLCS visualization contract before scene I/O."""
    root = as_config_mapping(config, path="configuration")
    _exact(
        root,
        {
            "paths",
            "court_keypoints",
            "visualization",
        },
        path="configuration",
    )
    court_keypoint_contract = parse_court_keypoint_contract(config)
    visualization = require_config_mapping(root, "visualization", path="configuration")
    common = SceneVisualizationConfig.from_mapping(
        visualization,
        resolver=build_path_resolver(config),
        extension_keys=frozenset({"reference_camera_id"}),
    )
    reference_camera_id = visualization.get("reference_camera_id")
    if court_keypoint_contract.camera_view_semantics and common.mode == "predict":
        if not isinstance(reference_camera_id, str) or not reference_camera_id.strip():
            raise SemanticConfigurationError(
                "camera_view_v2 prediction visualization requires an explicit "
                "visualization.reference_camera_id."
            )
    elif reference_camera_id is not None:
        raise SemanticConfigurationError(
            "visualization.reference_camera_id is only valid for camera_view_v2."
        )
    parse_scene_style(
        require_config_mapping(visualization, "style", path="visualization")
    )
    parse_view_3d(
        require_config_mapping(visualization, "view_3d", path="visualization")
    )
    if common.mode not in {"visualize", "predict"}:
        raise SemanticConfigurationError(
            "visualization.mode must be 'visualize' or 'predict'."
        )
    if common.animation_view not in {"2d", "3d"}:
        raise SemanticConfigurationError(
            "visualization.animation_view must be '2d' or '3d'."
        )
    if common.fps is None:
        raise SemanticConfigurationError("visualization.fps must be an explicit value.")
    if common.mode == "predict" and common.checkpoint is None:
        raise SemanticConfigurationError(
            "visualization.checkpoint is required when mode='predict'."
        )


def validate_api_boundary(config: object) -> None:
    """Validate API server and canonical simulator config before app creation."""
    root = as_config_mapping(config, path="configuration")
    _exact(
        root,
        {
            "paths",
            "court_keypoints",
            "physics",
            "rally",
            "camera",
            "targeted_velocity",
            "generator",
            "server",
        },
        path="configuration",
    )
    parse_court_keypoint_contract(config)
    build_path_resolver(config)
    validate_generator_sections(config, include_generation=False)
    server = require_config_mapping(root, "server", path="configuration")
    _exact(server, {"host", "port", "log_level"}, path="server")
    _validate_types(server, {"host": str, "port": int, "log_level": str}, path="server")
    port = cast("int", server["port"])
    if not 0 < port < 65536:
        raise SemanticConfigurationError("server.port must be in [1, 65535].")


def validate_generation_boundary(config: object) -> None:
    """Validate dataset-generation configuration before filesystem writes."""
    parse_generation_run(config)


def validate_preview_boundary(config: object) -> None:
    """Validate augmentation-preview configuration before dataset I/O."""
    parse_court_keypoint_contract(config)
    parse_preview_config(config)


def _validate_training_for_hydra(config: DictConfig) -> None:
    validate_training_boundary(config)


register_boundary_validator("blcs.train", _validate_training_for_hydra)
register_boundary_validator("blcs.visualize", validate_visualization_boundary)
register_boundary_validator("blcs.generate_dataset", validate_generation_boundary)
register_boundary_validator("blcs.preview_augmentation", validate_preview_boundary)
register_boundary_validator("blcs.api_server", validate_api_boundary)


def run_negative_matrix() -> None:
    """Exercise representative BLCS missing/unknown/type/path failures."""
    from copy import deepcopy

    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from src.utils.configuration import ConfigurationError, PathContractError

    valid_paths = {
        "project_root": ".",
        "data_root": "data",
        "checkpoint_root": "ckpt",
        "artifact_root": "artifacts",
        "output_root": "outputs",
        "cache_root": ".cache",
        "external_asset_root": "third_party",
    }
    invalid = (
        (
            {
                "paths": {
                    key: value
                    for key, value in valid_paths.items()
                    if key != "data_root"
                }
            },
            MissingConfigurationKeyError,
        ),
        (
            {"paths": {**valid_paths, "legacy_root": "legacy"}},
            UnknownConfigurationKeyError,
        ),
        ({"paths": {**valid_paths, "data_root": 3}}, ConfigurationTypeError),
    )
    for candidate, expected in invalid:
        try:
            build_path_resolver(candidate)
        except expected:
            continue
        except ConfigurationError:
            raise
        raise AssertionError(f"BLCS negative matrix accepted {candidate!r}.")
    resolver = build_path_resolver({"paths": valid_paths})
    try:
        resolver.resolve(PathRole.DATA, "../escape")
    except PathContractError:
        pass
    else:
        raise AssertionError("BLCS negative matrix accepted a root-escaping path.")

    def expect_failure(
        name: str,
        operation: Callable[[], object],
        expected: type[BaseException],
        message_fragment: str,
    ) -> None:
        try:
            operation()
        except expected as error:
            if message_fragment not in str(error):
                raise AssertionError(
                    f"BLCS case {name!r} raised {expected.__name__} without "
                    f"{message_fragment!r}: {error}"
                ) from error
            return
        raise AssertionError(f"BLCS negative matrix accepted {name}.")

    config_dir = str((Path(__file__).parent / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        training = compose(config_name="train")
        gan_training = compose(config_name="train_chunked_gan")
        generation = compose(config_name="generate_dataset")
        visualization = compose(config_name="visualize")

    nested_trainer_typo = deepcopy(training)
    with open_dict(nested_trainer_typo.training.trainer):
        nested_trainer_typo.training.trainer["__unknown__"] = 1
    expect_failure(
        "nested-training-trainer-typo",
        lambda: validate_training_boundary(nested_trainer_typo),
        UnknownConfigurationKeyError,
        "training.trainer.__unknown__",
    )

    gan_global_clip = deepcopy(gan_training)
    gan_global_clip.training.trainer.gradient_clip_val = 1.0
    expect_failure(
        "gan-global-gradient-clip-conflict",
        lambda: validate_training_boundary(gan_global_clip),
        SemanticConfigurationError,
        "must be null when GAN is enabled",
    )

    gan_early_stop = deepcopy(gan_training)
    gan_early_stop.training.early_stopping.enabled = True
    expect_failure(
        "gan-early-stopping-conflict",
        lambda: validate_training_boundary(gan_early_stop),
        SemanticConfigurationError,
        "must be false when GAN is enabled",
    )

    unsupported_generation_device = deepcopy(generation)
    unsupported_generation_device.run.device = "cuda"
    expect_failure(
        "unsupported-generation-device",
        lambda: parse_generation_run(unsupported_generation_device),
        SemanticConfigurationError,
        "run.device",
    )

    visualization_unknown = deepcopy(visualization)
    with open_dict(visualization_unknown.visualization):
        visualization_unknown.visualization["__unknown__"] = 1
    expect_failure(
        "visualization-unknown",
        lambda: validate_visualization_boundary(visualization_unknown),
        UnknownConfigurationKeyError,
        "visualization.__unknown__",
    )

    visualization_wrong_type = deepcopy(visualization)
    visualization_wrong_type.visualization.style.show_shadow = "true"
    expect_failure(
        "visualization-style-wrong-type",
        lambda: validate_visualization_boundary(visualization_wrong_type),
        ConfigurationTypeError,
        "visualization.style.show_shadow",
    )

    removed_run_device = deepcopy(visualization)
    with open_dict(removed_run_device):
        removed_run_device["run"] = {"device": "cpu"}
    expect_failure(
        "removed-visualization-run-device",
        lambda: validate_visualization_boundary(removed_run_device),
        UnknownConfigurationKeyError,
        "configuration.run",
    )


__all__ = [
    "AxialModelConfig",
    "BLCSModelConfig",
    "GenerationRunConfig",
    "PreviewConfig",
    "QualitativeRenderingConfig",
    "SingleModelConfig",
    "TrackQueryCSWAConfig",
    "TrackQueryMHCConfig",
    "TrackQueryModelConfig",
    "TrackQueryReferenceModelConfig",
    "build_path_resolver",
    "parse_court_keypoint_contract",
    "parse_generation_run",
    "parse_model_config",
    "parse_preview_config",
    "parse_qualitative_rendering",
    "run_negative_matrix",
    "validate_api_boundary",
    "validate_generator_sections",
    "validate_training_boundary",
    "validate_visualization_boundary",
]
