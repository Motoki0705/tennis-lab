"""Strict typed configuration contracts for BLCS runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.base.visualization.style import (
    SceneStyleConfig,
    parse_scene_style,
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
from src.utils.hydra import register_boundary_validator
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


@dataclass(frozen=True, slots=True)
class SingleModelConfig:
    name: Literal["blcs"]
    input_profile: Literal["single"]
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    ffn_type: Literal["swiglu", "mlp"]
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
class MultiViewModelConfig:
    name: Literal["blcs_multiview"]
    input_profile: Literal["multiview"]
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    ffn_type: Literal["swiglu", "mlp"]
    dropout: float
    rope_dim: int
    rope_theta: float
    rope_theta_time: float
    rope_theta_camera: float
    rope_theta_type: float
    max_seq_len: int
    max_num_cameras: int
    predict_velocity: bool
    num_court_tokens: int
    invisible_init_std: float
    query_init_std: float


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
    ffn_type: Literal["swiglu", "mlp"]
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
class PointFusionConfig:
    token_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    rope_dim: int
    dropout: float


@dataclass(frozen=True, slots=True)
class TrackQueryModelConfig:
    name: Literal["blcs_track_query"]
    hidden_dim: int
    num_heads: int
    num_stages: int
    ffn_dim: int
    num_queries: int
    rope_dim: int
    dropout: float
    role_rope_enabled: bool
    mask_invisible_observations: bool
    invisible_init_std: float
    observation_fusion: Literal["linear", "point_attention"]
    point_fusion: PointFusionConfig | None


BLCSModelConfig: TypeAlias = (
    SingleModelConfig | MultiViewModelConfig | AxialModelConfig | TrackQueryModelConfig
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
        if profile != "single" or model["ffn_type"] not in {"swiglu", "mlp"}:
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
            ffn_type=cast("Literal['swiglu', 'mlp']", model["ffn_type"]),
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
    if name == "blcs_multiview":
        keys = {
            "name",
            "io",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "ffn_type",
            "dropout",
            "rope_dim",
            "rope_theta",
            "rope_theta_time",
            "rope_theta_camera",
            "rope_theta_type",
            "max_seq_len",
            "max_num_cameras",
            "predict_velocity",
            "num_court_tokens",
            "invisible_init_std",
            "query_init_std",
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
                "rope_dim": int,
                "rope_theta": (float, int),
                "rope_theta_time": (float, int),
                "rope_theta_camera": (float, int),
                "rope_theta_type": (float, int),
                "max_seq_len": int,
                "max_num_cameras": int,
                "predict_velocity": bool,
                "num_court_tokens": int,
                "invisible_init_std": (float, int),
                "query_init_std": (float, int),
            },
            path="model",
        )
        if _io(model) != "multiview" or model["ffn_type"] not in {"swiglu", "mlp"}:
            raise SemanticConfigurationError(
                "Invalid multiview model profile or ffn_type."
            )
        result = MultiViewModelConfig(
            name="blcs_multiview",
            input_profile="multiview",
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            ffn_dim=cast("int", model["ffn_dim"]),
            ffn_type=cast("Literal['swiglu', 'mlp']", model["ffn_type"]),
            dropout=float(model["dropout"]),
            rope_dim=cast("int", model["rope_dim"]),
            rope_theta=float(model["rope_theta"]),
            rope_theta_time=float(model["rope_theta_time"]),
            rope_theta_camera=float(model["rope_theta_camera"]),
            rope_theta_type=float(model["rope_theta_type"]),
            max_seq_len=int(model["max_seq_len"]),
            max_num_cameras=int(model["max_num_cameras"]),
            predict_velocity=bool(model["predict_velocity"]),
            num_court_tokens=int(model["num_court_tokens"]),
            invisible_init_std=float(model["invisible_init_std"]),
            query_init_std=float(model["query_init_std"]),
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
        if result.num_court_tokens <= 0:
            raise SemanticConfigurationError("model.num_court_tokens must be positive.")
        for key, value in (
            ("invisible_init_std", result.invisible_init_std),
            ("query_init_std", result.query_init_std),
        ):
            _non_negative(value, path=f"model.{key}")
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
            or model["ffn_type"] not in {"swiglu", "mlp"}
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
            ffn_type=cast("Literal['swiglu', 'mlp']", model["ffn_type"]),
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
    if name == "blcs_track_query":
        base_keys = {
            "name",
            "hidden_dim",
            "num_heads",
            "num_stages",
            "ffn_dim",
            "num_queries",
            "rope_dim",
            "dropout",
            "role_rope_enabled",
            "mask_invisible_observations",
            "invisible_init_std",
            "observation_fusion",
        }
        fusion_name = cast(
            "str", _value(model, "observation_fusion", str, path="model")
        )
        expected = base_keys | (
            {"point_fusion"} if fusion_name == "point_attention" else set()
        )
        _exact(model, expected, path="model")
        _validate_types(
            model,
            {
                "name": str,
                "hidden_dim": int,
                "num_heads": int,
                "num_stages": int,
                "ffn_dim": int,
                "num_queries": int,
                "rope_dim": int,
                "dropout": (float, int),
                "role_rope_enabled": bool,
                "mask_invisible_observations": bool,
                "invisible_init_std": (float, int),
                "observation_fusion": str,
            },
            path="model",
        )
        point: PointFusionConfig | None = None
        if fusion_name == "point_attention":
            raw = as_config_mapping(model["point_fusion"], path="model.point_fusion")
            point_keys = {
                "token_dim",
                "num_heads",
                "num_layers",
                "ffn_dim",
                "rope_dim",
                "dropout",
            }
            _exact(raw, point_keys, path="model.point_fusion")
            _validate_types(
                raw,
                {
                    "token_dim": int,
                    "num_heads": int,
                    "num_layers": int,
                    "ffn_dim": int,
                    "rope_dim": int,
                    "dropout": float,
                },
                path="model.point_fusion",
            )
            point = PointFusionConfig(
                token_dim=cast("int", raw["token_dim"]),
                num_heads=cast("int", raw["num_heads"]),
                num_layers=cast("int", raw["num_layers"]),
                ffn_dim=cast("int", raw["ffn_dim"]),
                rope_dim=cast("int", raw["rope_dim"]),
                dropout=float(cast("float | int", raw["dropout"])),
            )
            _validate_transformer_dimensions(
                hidden_dim=point.token_dim,
                num_heads=point.num_heads,
                ffn_dim=point.ffn_dim,
                rope_dim=point.rope_dim,
                dropout=point.dropout,
                path="model.point_fusion",
            )
            if point.num_layers <= 0:
                raise SemanticConfigurationError(
                    "model.point_fusion.num_layers must be positive."
                )
        if fusion_name not in {"linear", "point_attention"}:
            raise SemanticConfigurationError(
                "model.observation_fusion must be 'linear' or 'point_attention'."
            )
        result = TrackQueryModelConfig(
            name="blcs_track_query",
            hidden_dim=int(model["hidden_dim"]),
            num_heads=int(model["num_heads"]),
            num_stages=int(model["num_stages"]),
            ffn_dim=cast("int", model["ffn_dim"]),
            num_queries=int(model["num_queries"]),
            rope_dim=cast("int", model["rope_dim"]),
            dropout=float(model["dropout"]),
            role_rope_enabled=bool(model["role_rope_enabled"]),
            mask_invisible_observations=bool(model["mask_invisible_observations"]),
            invisible_init_std=float(model["invisible_init_std"]),
            observation_fusion=cast(
                "Literal['linear', 'point_attention']", fusion_name
            ),
            point_fusion=point,
        )
        _validate_transformer_dimensions(
            hidden_dim=result.hidden_dim,
            num_heads=result.num_heads,
            ffn_dim=result.ffn_dim,
            rope_dim=result.rope_dim,
            dropout=result.dropout,
            path="model",
        )
        if result.num_stages <= 0 or result.num_queries <= 0:
            raise SemanticConfigurationError(
                "model.num_stages and model.num_queries must be positive."
            )
        _non_negative(result.invisible_init_std, path="model.invisible_init_std")
        return result
    raise SemanticConfigurationError(f"Unsupported model.name={name!r}.")


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    output_dir: Path
    dataset_dir: Path
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
        dataset_dir=resolver.resolve(
            PathRole.DATA, cast("str", _value(data, "dataset_dir", str, path="data"))
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
            "profile",
            "image_size",
            "expected_camera_count",
            "slots",
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
    try:
        CameraProfileConfig.from_mapping(camera)
    except (TypeError, ValueError) as error:
        raise SemanticConfigurationError(
            f"Invalid canonical camera profile: {error}"
        ) from error

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
        _exact(generation, {"mode", "min_balls", "max_balls"}, path="generation")
        _validate_types(
            generation,
            {"mode": str, "min_balls": int, "max_balls": int},
            path="generation",
        )
        min_balls = cast("int", generation["min_balls"])
        max_balls = cast("int", generation["max_balls"])
        if min_balls <= 0 or max_balls < min_balls:
            raise SemanticConfigurationError(
                "generation ball counts must satisfy 1 <= min_balls <= max_balls."
            )
    elif mode == "multi_object":
        _exact(generation, {"mode", "timeline"}, path="generation")
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
    if model.name == "blcs_track_query":
        allowed = {
            "paths",
            "model",
            "data",
            "training",
            "loss",
            "tracking_metrics",
            "run",
        }
        data = require_config_mapping(root, "data", path="configuration")
        backend = cast("str", _value(data, "backend", str, path="data"))
    else:
        allowed = {
            "paths",
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
    build_path_resolver(config)
    data = require_config_mapping(root, "data", path="configuration")
    backend = cast("str", _value(data, "backend", str, path="data"))
    data_keys = {
        "backend",
        "dataset_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "seq_len_range",
        "augmentation",
    }
    if model.name == "blcs_track_query":
        data_keys.add("lifecycle")
        lifecycle = require_config_mapping(data, "lifecycle", path="data")
        _exact(
            lifecycle,
            {"pack_to_query_slots", "min_reuse_gap_frames", "randomize_slots_train"},
            path="data.lifecycle",
        )
    else:
        data_keys.add("num_court_kp")
    _exact(data, data_keys, path="data")
    _validate_types(
        data,
        {
            "backend": str,
            "dataset_dir": str,
            "batch_size": int,
            "num_workers": int,
            "pin_memory": bool,
            "seq_len_range": list,
            "augmentation": dict,
        },
        path="data",
    )
    if backend != "default":
        raise SemanticConfigurationError("data.backend must be 'default'.")
    seq_len_range = _int_sequence(data["seq_len_range"], path="data.seq_len_range")
    if len(seq_len_range) != 2:
        raise SemanticConfigurationError("data.seq_len_range must contain two values.")
    for name, values in (("seq_len_range", seq_len_range),):
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
        isinstance(model, (SingleModelConfig, MultiViewModelConfig, AxialModelConfig))
        and seq_len_range[1] > model.max_seq_len
    ):
        raise SemanticConfigurationError(
            "data.seq_len_range cannot exceed model.max_seq_len."
        )
    if model.name == "blcs_track_query":
        _validate_types(
            lifecycle,
            {
                "pack_to_query_slots": bool,
                "min_reuse_gap_frames": int,
                "randomize_slots_train": bool,
            },
            path="data.lifecycle",
        )
        if cast("int", lifecycle["min_reuse_gap_frames"]) < 0:
            raise SemanticConfigurationError(
                "data.lifecycle.min_reuse_gap_frames must be non-negative."
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
    if model.name == "blcs_track_query":
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
                "gravity_target",
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
                "gravity_target": float,
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
        _finite(cast("float", loss["gravity_target"]), path="loss.gravity_target")
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


def validate_api_boundary(config: object) -> None:
    """Validate API server and canonical simulator config before app creation."""
    root = as_config_mapping(config, path="configuration")
    _exact(
        root,
        {
            "paths",
            "physics",
            "rally",
            "camera",
            "targeted_velocity",
            "generator",
            "server",
        },
        path="configuration",
    )
    build_path_resolver(config)
    validate_generator_sections(config, include_generation=False)
    server = require_config_mapping(root, "server", path="configuration")
    _exact(server, {"host", "port", "log_level"}, path="server")
    _validate_types(server, {"host": str, "port": int, "log_level": str}, path="server")
    port = cast("int", server["port"])
    if not 0 < port < 65536:
        raise SemanticConfigurationError("server.port must be in [1, 65535].")


def validate_preview_boundary(config: object) -> None:
    """Validate augmentation-preview configuration before dataset I/O."""
    parse_preview_config(config)


def _validate_training_for_hydra(config: DictConfig) -> None:
    validate_training_boundary(config)


register_boundary_validator("blcs.train", _validate_training_for_hydra)
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
        gan_training = compose(config_name="train", overrides=["training=gan_base"])

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


__all__ = [
    "AxialModelConfig",
    "BLCSModelConfig",
    "MultiViewModelConfig",
    "PointFusionConfig",
    "QualitativeRenderingConfig",
    "SingleModelConfig",
    "TrackQueryModelConfig",
    "build_path_resolver",
    "parse_model_config",
    "parse_qualitative_rendering",
    "run_negative_matrix",
    "validate_api_boundary",
    "validate_generator_sections",
    "validate_training_boundary",
]
