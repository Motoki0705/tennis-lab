"""Strict typed configuration and path contracts for PLCS runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, TypeAlias, cast

from omegaconf import DictConfig, OmegaConf

import src.tasks.plcs.configuration_contracts as configuration_contracts
from src.tasks.base.configuration import (
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
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
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT
from src.utils.rendering.camera_view import CameraController
from src.utils.schema.player import NUM_HUMAN_KP

PLCSValue: TypeAlias = (
    str | int | float | bool | None | tuple[object, ...] | Mapping[str, object]
)


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
    number = float(cast("float | int", value))
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
            "role_rope_enabled",
            "mask_invisible_observations",
            "invisible_init_std",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class PLCSModelConfig:
    """Exact model-variant mapping, validated once before model construction."""

    name: str
    input_profile: str | None
    values: Mapping[str, object]

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
        mapping = _exact(
            initial,
            path="model",
            required=fields,
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
            "mask_invisible_observations",
        } & set(mapping):
            _boolean(mapping, key, path="model")
        if "ffn_type" in mapping:
            ffn_type = _string(mapping, "ffn_type", path="model")
            if ffn_type not in {"swiglu", "mlp"}:
                raise SemanticConfigurationError(
                    "model.ffn_type must be 'swiglu' or 'mlp'."
                )
        if (
            "num_joints" in mapping
            and _integer(mapping, "num_joints", path="model") != NUM_HUMAN_KP
        ):
            raise SemanticConfigurationError(
                f"model.num_joints must equal the canonical COCO joint count ({NUM_HUMAN_KP})."
            )
        return cls(
            name=name,
            input_profile=input_profile,
            values=MappingProxyType(dict(mapping)),
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
    "dataset_dir",
    "batch_size",
    "num_workers",
    "pin_memory",
    "seq_len_range",
    "augmentation",
}


@dataclass(frozen=True, slots=True)
class PLCSDataConfig:
    backend: str
    dataset_dir: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    input_profile: str | None
    num_court_tokens: int | None
    values: Mapping[str, object]

    @classmethod
    def from_mapping(
        cls, value: object, *, resolver: PathResolver, model: PLCSModelConfig
    ) -> PLCSDataConfig:
        initial = _plain(value, path="data")
        backend = _string(initial, "backend", path="data")
        tracking = model.name == "plcs_track_query"
        allowed = set(_DATA_COMMON)
        if tracking:
            allowed.add("lifecycle")
        else:
            allowed.update({"mode", "num_court_kp"})
            configured_mode = _string(initial, "mode", path="data")
            if model.input_profile == "frame" and configured_mode == "frame":
                allowed.discard("seq_stride")
            else:
                allowed.add("seq_stride")
        if backend != "default":
            raise SemanticConfigurationError("data.backend must be 'default'.")
        mapping = _exact(
            initial,
            path="data",
            required=allowed - {"seq_stride"},
            allowed=allowed,
        )
        augmentation = validate_augmentation(mapping["augmentation"])
        seq_len_values = _sequence(
            mapping,
            "seq_len_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        seq_len_range = tuple(cast("int", value) for value in seq_len_values)
        for key, range_value in (("seq_len_range", seq_len_range),):
            if range_value[0] <= 0 or range_value[1] < range_value[0]:
                raise SemanticConfigurationError(
                    f"data.{key} must be a positive ordered range."
                )
        if "max_seq_len" in model.values and seq_len_range[1] > model.integer(
            "max_seq_len"
        ):
            raise SemanticConfigurationError(
                "data.seq_len_range cannot exceed model.max_seq_len."
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
            _boolean(lifecycle, "pack_to_query_slots", path="data.lifecycle")
            _integer(lifecycle, "min_reuse_gap_frames", path="data.lifecycle")
            _boolean(lifecycle, "randomize_slots_train", path="data.lifecycle")
            if _integer(lifecycle, "min_reuse_gap_frames", path="data.lifecycle") < 0:
                raise SemanticConfigurationError(
                    "data.lifecycle.min_reuse_gap_frames must be non-negative."
                )
        dataset_dir = _string(mapping, "dataset_dir", path="data")
        batch_size = _integer(mapping, "batch_size", path="data")
        workers = _integer(mapping, "num_workers", path="data")
        if batch_size <= 0 or workers < 0:
            raise SemanticConfigurationError(
                "data.batch_size must be positive and num_workers non-negative."
            )
        num_court_tokens: int | None = None
        if not tracking:
            num_court_tokens = _integer(mapping, "num_court_kp", path="data")
            if num_court_tokens <= 0:
                raise SemanticConfigurationError("data.num_court_kp must be positive.")
        resolved = dict(mapping)
        resolved["augmentation"] = augmentation
        return cls(
            backend=backend,
            dataset_dir=resolver.resolve(PathRole.DATA, dataset_dir),
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=_boolean(mapping, "pin_memory", path="data"),
            input_profile=model.input_profile,
            num_court_tokens=num_court_tokens,
            values=MappingProxyType(resolved),
        )


@dataclass(frozen=True, slots=True)
class PLCSTrainingConfig:
    """Complete typed PLCS training boundary."""

    shared: TrainingRuntimeConfig
    paths: configuration_contracts.PLCSPathConfig
    model: PLCSModelConfig
    data: PLCSDataConfig
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
            },
        )
        paths = configuration_contracts.PLCSPathConfig.from_config(value)
        model = PLCSModelConfig.from_mapping(
            require_config_mapping(root, "model", path="configuration")
        )
        data = PLCSDataConfig.from_mapping(
            require_config_mapping(root, "data", path="configuration"),
            resolver=paths.resolver,
            model=model,
        )
        exact_root_fields = {
            "model",
            "data",
            "training",
            "loss",
            "run",
            "paths",
            "external_assets",
            "qualitative",
            "tracking_metrics" if model.name == "plcs_track_query" else "metrics",
        }
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
            "matmul_precision",
            "allow_tf32",
            "checkpoint",
            "early_stopping",
            "lr_monitor",
            "qualitative_logging",
            "gan",
            "mcmc",
        }
        training_mapping = _exact(
            require_config_mapping(root, "training", path="configuration"),
            path="training",
            required=training_fields,
            allowed=training_fields,
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
        if model.name != "plcs_track_query":
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
            tracking_loss = _exact(
                require_config_mapping(root, "loss", path="configuration"),
                path="loss",
                required=tracking_loss_fields,
                allowed=tracking_loss_fields,
            )
            for key in tracking_loss_fields - {"transition_radius"}:
                _positive(
                    _number(tracking_loss, key, path="loss"),
                    path=f"loss.{key}",
                    allow_zero=True,
                )
            if _integer(tracking_loss, "transition_radius", path="loss") < 0:
                raise SemanticConfigurationError(
                    "loss.transition_radius must be non-negative."
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
            ) not in {"swiglu", "mlp"}:
                raise SemanticConfigurationError(
                    "training.gan.discriminator.ffn_type must be 'swiglu' or 'mlp'."
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
        if model.name == "plcs_track_query" and model.input_profile is not None:
            raise SemanticConfigurationError(
                "Tracking models must not define model.io."
            )
        return cls(
            shared=shared,
            paths=paths,
            model=model,
            data=data,
            tracking_metrics=tracking_metric_config,
            qualitative_style=qualitative_style,
            qualitative_view_3d=qualitative_view_3d,
            qualitative_fps=qualitative_fps,
            raw=value,
        )


def _validate_training_boundary(config: DictConfig) -> None:
    PLCSTrainingConfig.from_config(config)


def _validate_script_boundary(
    config: DictConfig,
    *,
    required_sections: set[str],
    section_fields: Mapping[str, set[str]],
) -> None:
    root = _exact(
        config,
        path="configuration",
        required=required_sections,
        allowed=required_sections,
    )
    resolver = configuration_contracts.PLCSPathConfig.from_config(config).resolver
    if "data" in root:
        data_fields = {
            "backend",
            "dataset_dir",
            "num_court_kp",
            "augmentation",
            "batch_size",
            "num_workers",
            "pin_memory",
            "mode",
            "seq_len_range",
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
        _string(data, "dataset_dir", path="data")
        resolver.resolve(PathRole.DATA, _string(data, "dataset_dir", path="data"))
        if _integer(data, "num_court_kp", path="data") <= 0:
            raise SemanticConfigurationError("data.num_court_kp must be positive.")
        lengths = _sequence(
            data,
            "seq_len_range",
            path="data",
            item_types=(int,),
            length=2,
        )
        for key, values in (("seq_len_range", lengths),):
            lo, hi = (cast("int", item) for item in values)
            if lo <= 0 or hi < lo:
                raise SemanticConfigurationError(
                    f"data.{key} must be a positive ordered range."
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
        if "dataset_dir" in run:
            resolver.resolve(PathRole.DATA, _string(run, "dataset_dir", path="run"))
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
    dataset_dir: Path
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
            dataset_dir=resolver.resolve(PathRole.DATA, str(config.data.dataset_dir)),
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
    dataset_dir: Path | None
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
            dataset_dir=resolver.resolve(PathRole.DATA, str(config.data.dataset_dir)),
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
            dataset_dir=resolver.resolve(PathRole.DATA, str(config.data.dataset_dir)),
            scene_records_dir=resolver.resolve(
                PathRole.DATA,
                str(config.data.dataset_dir),
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
            dataset_dir=None,
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


def _validate_preview_boundary(config: DictConfig) -> None:
    PLCSPreviewRuntimeConfig.from_config(config)


def _validate_angle_velocity_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.angle_velocity(config)


def _validate_distribution_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.distribution(config)


def _validate_loss_dominance_boundary(config: DictConfig) -> None:
    PLCSAnalysisRuntimeConfig.loss_dominance(config)


def _register_validators() -> None:
    register_boundary_validator("plcs.train", _validate_training_boundary)


_register_validators()


__all__ = [
    "PLCSDataConfig",
    "PLCSModelConfig",
    "PLCSTrainingConfig",
    "validate_augmentation",
]
