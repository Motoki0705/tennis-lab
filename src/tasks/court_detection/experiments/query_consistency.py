"""Deterministic three-phase scaling manifest for Issue #790."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.court_detection.configuration import (
    ConfigMapping,
    validate_paths_boundary,
)
from src.utils.configuration import PathRole

JsonValue: TypeAlias = Any
DecoderFamily: TypeAlias = Literal["linear", "progressive", "dpt"]
DecoderSize: TypeAlias = Literal["tiny", "small", "base"]
ConsistencyCondition: TypeAlias = Literal[
    "direct-all",
    "joint-both",
    "joint-stopgrad-pose",
    "joint-stopgrad-dense",
]

MANIFEST_SCHEMA = "court_query_consistency_manifest_v2"
PHASE_ORDER = ("encoder_scaling", "decoder_scaling", "consistency_ablation")
SEEDS = (42, 43, 44)
PYTHON_EXECUTABLE = "/home/kamimura/projects/tennis-lab/.venv/bin/python"
SHARED_DATA_ROOT = "/home/kamimura/projects/tennis-lab/data"
SHARED_EXTERNAL_ASSET_ROOT = "/home/kamimura/projects/tennis-lab/third_party"
V3_DERIVED_TARGET_ROOT = "court_detection/derived_targets_issue790_v3"
V3_WORKSPACE_ROOT = (
    "issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes"
)
RESULT_METRIC_NAMES = (
    "kp_mean_distance_px",
    "kp_median_distance_px",
    "pose_reprojection_mean_distance_px",
    "pose_translation_l2_m",
    "pose_rotation_geodesic_deg",
    "pose_focal_relative_error",
    "line_dice",
    "seg_miou",
    "kp_pose_consistency_distance_px",
    "invalid_depth_rate",
    "visible_point_count",
)
TRAIN_DIAGNOSTIC_NAMES = (
    "kp_gradient_finite",
    "seg_gradient_finite",
    "line_gradient_finite",
    "pose_gradient_finite",
    "train_step_time_ms",
    "cuda_peak_memory_bytes",
)

_FAMILIES: tuple[DecoderFamily, ...] = ("linear", "progressive", "dpt")
_SIZES: tuple[DecoderSize, ...] = ("tiny", "small", "base")
_CONDITIONS: tuple[ConsistencyCondition, ...] = (
    "direct-all",
    "joint-both",
    "joint-stopgrad-pose",
    "joint-stopgrad-dense",
)
_LOSS_BY_CONDITION: Mapping[ConsistencyCondition, str] = {
    "direct-all": "query_direct_all",
    "joint-both": "query_joint_both",
    "joint-stopgrad-pose": "query_joint_stopgrad_pose",
    "joint-stopgrad-dense": "query_joint_stopgrad_dense",
}


def _exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
    if set(mapping) != keys:
        raise ValueError(f"{path} requires exactly {sorted(keys)}.")


def _string(mapping: ConfigMapping, key: str, *, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty trimmed string.")
    return value


def _integer(mapping: ConfigMapping, key: str, *, path: str) -> int:
    return cast(int, require_config_value(mapping, key, int, path=path))


def _number(mapping: ConfigMapping, key: str, *, path: str) -> float:
    value = require_config_value(mapping, key, (float, int), path=path)
    if type(value) not in (float, int):  # pragma: no cover - helper is strict
        raise TypeError(f"{path}.{key} must be numeric.")
    result = float(cast(float | int, value))
    if not math.isfinite(result):
        raise ValueError(f"{path}.{key} must be finite.")
    return result


def _boolean(mapping: ConfigMapping, key: str, *, path: str) -> bool:
    return cast(bool, require_config_value(mapping, key, bool, path=path))


def _integers(mapping: ConfigMapping, key: str, *, path: str) -> tuple[int, ...]:
    values = require_config_value(mapping, key, (list, tuple), path=path)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{path}.{key} must be a sequence.")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{path}.{key} must contain integers.")
    return tuple(cast(int, value) for value in values)


def _strings(mapping: ConfigMapping, key: str, *, path: str) -> tuple[str, ...]:
    values = require_config_value(mapping, key, (list, tuple), path=path)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{path}.{key} must be a sequence.")
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"{path}.{key} must contain non-empty strings.")
    return tuple(cast(str, value) for value in values)


def _normalized_relative_descendant(
    mapping: ConfigMapping, key: str, *, path: str
) -> str:
    value = _string(mapping, key, path=path)
    parts = value.split("/")
    if "\\" in value or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"{path}.{key} must be a normalized relative descendant path."
        )
    return value


@dataclass(frozen=True, slots=True)
class EncoderScalingConfig:
    """Frozen first-phase encoder matrix and selection threshold."""

    name: Literal["encoder_scaling"]
    order: Literal[1]
    depths: tuple[int, ...]
    decoder_family: Literal["linear"]
    decoder_size: Literal["base"]
    condition: Literal["joint-both"]
    reference_depth: Literal[8]
    tolerance_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> EncoderScalingConfig:
        path = "consistency_ablation.encoder_scaling"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "name",
                "order",
                "depths",
                "decoder_family",
                "decoder_size",
                "condition",
                "reference_depth",
                "tolerance_ratio",
            },
            path=path,
        )
        result = cls(
            name=cast(Literal["encoder_scaling"], _string(mapping, "name", path=path)),
            order=cast(Literal[1], _integer(mapping, "order", path=path)),
            depths=_integers(mapping, "depths", path=path),
            decoder_family=cast(
                Literal["linear"], _string(mapping, "decoder_family", path=path)
            ),
            decoder_size=cast(
                Literal["base"], _string(mapping, "decoder_size", path=path)
            ),
            condition=cast(
                Literal["joint-both"], _string(mapping, "condition", path=path)
            ),
            reference_depth=cast(
                Literal[8], _integer(mapping, "reference_depth", path=path)
            ),
            tolerance_ratio=_number(mapping, "tolerance_ratio", path=path),
        )
        if result != cls(
            name="encoder_scaling",
            order=1,
            depths=(1, 2, 4, 8),
            decoder_family="linear",
            decoder_size="base",
            condition="joint-both",
            reference_depth=8,
            tolerance_ratio=0.05,
        ):
            raise ValueError("Encoder scaling preset changed from the frozen matrix.")
        return result


@dataclass(frozen=True, slots=True)
class DecoderScalingConfig:
    """Frozen second-phase decoder matrix and selection thresholds."""

    name: Literal["decoder_scaling"]
    order: Literal[2]
    families: tuple[DecoderFamily, ...]
    sizes: tuple[DecoderSize, ...]
    condition: Literal["joint-both"]
    reference_family: Literal["dpt"]
    reference_size: Literal["base"]
    tolerance_ratio: float
    dense_absolute_tolerance: float

    @classmethod
    def from_mapping(cls, value: object) -> DecoderScalingConfig:
        path = "consistency_ablation.decoder_scaling"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "name",
                "order",
                "families",
                "sizes",
                "condition",
                "reference_family",
                "reference_size",
                "tolerance_ratio",
                "dense_absolute_tolerance",
            },
            path=path,
        )
        result = cls(
            name=cast(Literal["decoder_scaling"], _string(mapping, "name", path=path)),
            order=cast(Literal[2], _integer(mapping, "order", path=path)),
            families=cast(
                tuple[DecoderFamily, ...], _strings(mapping, "families", path=path)
            ),
            sizes=cast(tuple[DecoderSize, ...], _strings(mapping, "sizes", path=path)),
            condition=cast(
                Literal["joint-both"], _string(mapping, "condition", path=path)
            ),
            reference_family=cast(
                Literal["dpt"], _string(mapping, "reference_family", path=path)
            ),
            reference_size=cast(
                Literal["base"], _string(mapping, "reference_size", path=path)
            ),
            tolerance_ratio=_number(mapping, "tolerance_ratio", path=path),
            dense_absolute_tolerance=_number(
                mapping, "dense_absolute_tolerance", path=path
            ),
        )
        if result != cls(
            name="decoder_scaling",
            order=2,
            families=_FAMILIES,
            sizes=_SIZES,
            condition="joint-both",
            reference_family="dpt",
            reference_size="base",
            tolerance_ratio=0.05,
            dense_absolute_tolerance=0.01,
        ):
            raise ValueError("Decoder scaling preset changed from the frozen matrix.")
        return result


@dataclass(frozen=True, slots=True)
class ConsistencyAblationPhaseConfig:
    """Frozen third-phase formal comparison and adoption thresholds."""

    name: Literal["consistency_ablation"]
    order: Literal[3]
    conditions: tuple[ConsistencyCondition, ...]
    baseline: Literal["direct-all"]
    candidate: Literal["joint-both"]
    improvement_ratio: float
    maximum_degradation_ratio: float
    dense_absolute_tolerance: float
    maximum_cost_overhead_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> ConsistencyAblationPhaseConfig:
        path = "consistency_ablation.consistency_ablation"
        mapping = as_config_mapping(value, path=path)
        _exact(
            mapping,
            {
                "name",
                "order",
                "conditions",
                "baseline",
                "candidate",
                "improvement_ratio",
                "maximum_degradation_ratio",
                "dense_absolute_tolerance",
                "maximum_cost_overhead_ratio",
            },
            path=path,
        )
        result = cls(
            name=cast(
                Literal["consistency_ablation"], _string(mapping, "name", path=path)
            ),
            order=cast(Literal[3], _integer(mapping, "order", path=path)),
            conditions=cast(
                tuple[ConsistencyCondition, ...],
                _strings(mapping, "conditions", path=path),
            ),
            baseline=cast(
                Literal["direct-all"], _string(mapping, "baseline", path=path)
            ),
            candidate=cast(
                Literal["joint-both"], _string(mapping, "candidate", path=path)
            ),
            improvement_ratio=_number(mapping, "improvement_ratio", path=path),
            maximum_degradation_ratio=_number(
                mapping, "maximum_degradation_ratio", path=path
            ),
            dense_absolute_tolerance=_number(
                mapping, "dense_absolute_tolerance", path=path
            ),
            maximum_cost_overhead_ratio=_number(
                mapping, "maximum_cost_overhead_ratio", path=path
            ),
        )
        if result != cls(
            name="consistency_ablation",
            order=3,
            conditions=_CONDITIONS,
            baseline="direct-all",
            candidate="joint-both",
            improvement_ratio=0.05,
            maximum_degradation_ratio=0.05,
            dense_absolute_tolerance=0.01,
            maximum_cost_overhead_ratio=0.10,
        ):
            raise ValueError("Consistency ablation preset changed from the frozen matrix.")
        return result


@dataclass(frozen=True, slots=True)
class QueryConsistencyAblationConfig:
    """Resolved strict Hydra contract for the staged Issue #790 manifest."""

    output_path: Path
    data_root: Path
    external_asset_root: Path
    derived_target_root: Path
    derived_target_relative_path: str
    workspace_root: Path
    workspace_relative_path: str
    python_executable: str
    train_module: str
    profile_module: str
    seeds: tuple[int, ...]
    epochs: int
    image_height: int
    image_width: int
    batch_size: int
    selected_encoder_depth: int | None
    selected_decoder_family: DecoderFamily | None
    selected_decoder_size: DecoderSize | None
    encoder_scaling: EncoderScalingConfig
    decoder_scaling: DecoderScalingConfig
    consistency_ablation: ConsistencyAblationPhaseConfig

    @classmethod
    def from_config(cls, value: object) -> QueryConsistencyAblationConfig:
        if not isinstance(value, DictConfig):
            raise TypeError("Query consistency boundary requires a composed DictConfig.")
        root, resolver = validate_paths_boundary(
            value, expected_sections={"consistency_ablation"}
        )
        section = require_config_mapping(
            root, "consistency_ablation", path="configuration"
        )
        _exact(
            section,
            {
                "output_path",
                "python_executable",
                "train_module",
                "profile_module",
                "seeds",
                "epochs",
                "input",
                "batch_size",
                "selected",
                "composition",
                "optimizer",
                "direct_weights",
                "auxiliary",
                "encoder_scaling",
                "decoder_scaling",
                "consistency_ablation",
            },
            path="consistency_ablation",
        )
        tensor = require_config_mapping(section, "input", path="consistency_ablation")
        _exact(tensor, {"height", "width"}, path="consistency_ablation.input")
        selected = require_config_mapping(
            section, "selected", path="consistency_ablation"
        )
        _exact(
            selected,
            {"encoder_depth", "decoder_family", "decoder_size"},
            path="consistency_ablation.selected",
        )
        raw_depth = require_config_value(
            selected,
            "encoder_depth",
            (int, type(None)),
            path="consistency_ablation.selected",
        )
        raw_family = require_config_value(
            selected,
            "decoder_family",
            (str, type(None)),
            path="consistency_ablation.selected",
        )
        raw_size = require_config_value(
            selected,
            "decoder_size",
            (str, type(None)),
            path="consistency_ablation.selected",
        )
        composition = require_config_mapping(
            section, "composition", path="consistency_ablation"
        )
        workspace_relative_path = _normalized_relative_descendant(
            composition,
            "workspace_root",
            path="consistency_ablation.composition",
        )
        derived_target_relative_path = _normalized_relative_descendant(
            composition,
            "derived_target_root",
            path="consistency_ablation.composition",
        )
        result = cls(
            output_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(section, "output_path", path="consistency_ablation"),
            ),
            data_root=resolver.validate(
                PathRole.DATA, resolver.roots.root(PathRole.DATA)
            ),
            external_asset_root=resolver.validate(
                PathRole.EXTERNAL_ASSET,
                resolver.roots.root(PathRole.EXTERNAL_ASSET),
            ),
            derived_target_root=resolver.resolve(
                PathRole.DATA,
                derived_target_relative_path,
            ),
            derived_target_relative_path=derived_target_relative_path,
            workspace_root=resolver.resolve(
                PathRole.DATA,
                workspace_relative_path,
            ),
            workspace_relative_path=workspace_relative_path,
            python_executable=_string(
                section, "python_executable", path="consistency_ablation"
            ),
            train_module=_string(section, "train_module", path="consistency_ablation"),
            profile_module=_string(
                section, "profile_module", path="consistency_ablation"
            ),
            seeds=_integers(section, "seeds", path="consistency_ablation"),
            epochs=_integer(section, "epochs", path="consistency_ablation"),
            image_height=_integer(tensor, "height", path="consistency_ablation.input"),
            image_width=_integer(tensor, "width", path="consistency_ablation.input"),
            batch_size=_integer(section, "batch_size", path="consistency_ablation"),
            selected_encoder_depth=cast(int | None, raw_depth),
            selected_decoder_family=cast(DecoderFamily | None, raw_family),
            selected_decoder_size=cast(DecoderSize | None, raw_size),
            encoder_scaling=EncoderScalingConfig.from_mapping(
                require_config_mapping(
                    section, "encoder_scaling", path="consistency_ablation"
                )
            ),
            decoder_scaling=DecoderScalingConfig.from_mapping(
                require_config_mapping(
                    section, "decoder_scaling", path="consistency_ablation"
                )
            ),
            consistency_ablation=ConsistencyAblationPhaseConfig.from_mapping(
                require_config_mapping(
                    section, "consistency_ablation", path="consistency_ablation"
                )
            ),
        )
        result._validate_fixed_sections(section)
        result.validate()
        return result

    def _validate_fixed_sections(self, section: ConfigMapping) -> None:
        composition = require_config_mapping(
            section, "composition", path="consistency_ablation"
        )
        if dict(composition) != {
            "source": "synthetic_court",
            "keypoint_court_scope": "target_court",
            "processing": "all",
            "augmentation": "pose_safe",
            "model": "query_encoder_base",
            "task_encoder": "query_base",
            "heads": "query_base",
            "backbone": "query_dinov3",
            "backbone_checkpoint": (
                "dinov3/checkpoints/"
                "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ),
            "backbone_train_mode": "frozen",
            "derived_target_root": V3_DERIVED_TARGET_ROOT,
            "workspace_root": V3_WORKSPACE_ROOT,
        }:
            raise ValueError("Consistency composition changed from its frozen V3 route.")
        optimizer = require_config_mapping(
            section, "optimizer", path="consistency_ablation"
        )
        if dict(optimizer) != {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
        }:
            raise ValueError("Consistency optimizer changed from the frozen contract.")
        direct = require_config_mapping(
            section, "direct_weights", path="consistency_ablation"
        )
        if dict(direct) != {
            "kp": 1.0,
            "line": 1.0,
            "seg": 1.0,
            "translation": 1.0,
            "rotation": 1.0,
            "focal": 1.0,
        }:
            raise ValueError("Consistency direct weights changed from the frozen contract.")
        auxiliary = require_config_mapping(
            section, "auxiliary", path="consistency_ablation"
        )
        if dict(auxiliary) != {
            "weight": 1.0,
            "temperature": 1.0,
            "huber_delta": 0.01,
            "min_depth_m": 0.1,
            "depth_scale_m": 1.0,
            "cheirality_weight": 0.1,
            "warmup_fraction": 0.1,
        }:
            raise ValueError("Consistency auxiliary values changed from the frozen contract.")

    def validate(self) -> None:
        if str(self.data_root) != SHARED_DATA_ROOT:
            raise ValueError(
                "Consistency runs require the shared declared data root "
                f"{SHARED_DATA_ROOT!r}."
            )
        if str(self.external_asset_root) != SHARED_EXTERNAL_ASSET_ROOT:
            raise ValueError(
                "Consistency runs require the shared declared external asset root "
                f"{SHARED_EXTERNAL_ASSET_ROOT!r}."
            )
        if self.workspace_relative_path != V3_WORKSPACE_ROOT:
            raise ValueError(
                "Consistency runs require the frozen V3 workspace relative path."
            )
        if self.derived_target_relative_path != V3_DERIVED_TARGET_ROOT:
            raise ValueError(
                "Consistency runs require the isolated V3 derived-target relative path."
            )
        if not self.derived_target_root.is_relative_to(self.data_root):
            raise ValueError(
                "Consistency V3 derived-target store must remain below the shared data root."
            )
        if not self.workspace_root.is_relative_to(self.data_root):
            raise ValueError(
                "Consistency V3 workspace must remain below the shared data root."
            )
        if self.python_executable != PYTHON_EXECUTABLE:
            raise ValueError(
                "Consistency commands must use the original repository's absolute "
                ".venv Python so root/worktree queue CWDs share one interpreter."
            )
        if self.train_module != "src.tasks.court_detection.scripts.train":
            raise ValueError("Consistency runs must use the Court training entrypoint.")
        if self.profile_module != "src.tasks.court_detection.scripts.profile_query_model":
            raise ValueError("Consistency capacity must use the query profiler.")
        if self.seeds != SEEDS or self.epochs != 15:
            raise ValueError("Consistency runs require seeds 42/43/44 and 15 epochs.")
        if (self.image_height, self.image_width) != (256, 256):
            raise ValueError("Consistency runs require the 256x256 pose-safe contract.")
        if self.batch_size != 8:
            raise ValueError("Consistency comparison batch size must remain eight.")
        if self.selected_encoder_depth is not None and (
            self.selected_encoder_depth not in self.encoder_scaling.depths
        ):
            raise ValueError("Selected encoder depth is outside the scaling matrix.")
        if self.selected_encoder_depth == 1:
            raise ValueError(
                "Selected encoder depth 1 cannot resolve the required two-or-more-tap "
                "DPT decoder phase; no implicit architecture substitution is allowed."
            )
        family_missing = self.selected_decoder_family is None
        size_missing = self.selected_decoder_size is None
        if family_missing != size_missing:
            raise ValueError("Selected decoder family and size must resolve together.")
        if self.selected_decoder_family is not None and (
            self.selected_encoder_depth is None
            or self.selected_decoder_family not in self.decoder_scaling.families
            or self.selected_decoder_size not in self.decoder_scaling.sizes
        ):
            raise ValueError("Selected decoder requires a valid selected encoder/matrix entry.")


def validate_query_consistency_ablation_boundary(config: DictConfig) -> None:
    """Validate the manifest Hydra boundary before writes."""
    QueryConsistencyAblationConfig.from_config(config)


def build_query_consistency_manifest(
    config: QueryConsistencyAblationConfig,
) -> dict[str, JsonValue]:
    """Build the strict 12 + 27 + 12 staged matrix without launching work."""
    runs: list[dict[str, JsonValue]] = []
    for depth in config.encoder_scaling.depths:
        for seed in config.seeds:
            runs.append(
                _run_record(
                    config,
                    run_id=f"encoder-depth-{depth:02d}-seed-{seed}",
                    phase="encoder_scaling",
                    seed=seed,
                    depth=depth,
                    family="linear",
                    size="base",
                    condition="joint-both",
                    unresolved=(),
                )
            )
    for family in config.decoder_scaling.families:
        for size in config.decoder_scaling.sizes:
            for seed in config.seeds:
                decoder_unresolved = (
                    ()
                    if config.selected_encoder_depth is not None
                    else ("selected_encoder_depth",)
                )
                runs.append(
                    _run_record(
                        config,
                        run_id=f"decoder-{family}-{size}-seed-{seed}",
                        phase="decoder_scaling",
                        seed=seed,
                        depth=config.selected_encoder_depth,
                        family=family,
                        size=size,
                        condition="joint-both",
                        unresolved=decoder_unresolved,
                    )
                )
    for condition in config.consistency_ablation.conditions:
        for seed in config.seeds:
            formal_unresolved: list[str] = []
            if config.selected_encoder_depth is None:
                formal_unresolved.append("selected_encoder_depth")
            if (
                config.selected_decoder_family is None
                or config.selected_decoder_size is None
            ):
                formal_unresolved.append("selected_decoder")
            runs.append(
                _run_record(
                    config,
                    run_id=f"condition-{condition}-seed-{seed}",
                    phase="consistency_ablation",
                    seed=seed,
                    depth=config.selected_encoder_depth,
                    family=config.selected_decoder_family,
                    size=config.selected_decoder_size,
                    condition=condition,
                    unresolved=tuple(formal_unresolved),
                )
            )
    manifest: dict[str, JsonValue] = {
        "schema": MANIFEST_SCHEMA,
        "phase_order": list(PHASE_ORDER),
        "fixed_contract": _fixed_contract(config),
        "selection_rules": _selection_rules(),
        "selected": {
            "encoder_depth": config.selected_encoder_depth,
            "decoder_family": config.selected_decoder_family,
            "decoder_size": config.selected_decoder_size,
        },
        "result_contract": {
            "metrics": list(RESULT_METRIC_NAMES),
            "diagnostics": list(TRAIN_DIAGNOSTIC_NAMES),
            "diagnostic_source": "tensorboard_scalar_events",
            "loss_curve_source": "tensorboard_scalar_train/loss",
            "capacity_source": "court_query_profile_v1",
            "line_metric_aliases_permitted": [],
        },
        "runs": runs,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    validate_query_consistency_manifest(manifest, require_resolved=False)
    return manifest


def _fixed_contract(config: QueryConsistencyAblationConfig) -> dict[str, JsonValue]:
    if (
        config.seeds != SEEDS
        or config.epochs != 15
        or (config.image_height, config.image_width) != (256, 256)
        or config.batch_size != 8
        or config.python_executable != PYTHON_EXECUTABLE
        or str(config.data_root) != SHARED_DATA_ROOT
        or str(config.external_asset_root) != SHARED_EXTERNAL_ASSET_ROOT
        or config.derived_target_relative_path != V3_DERIVED_TARGET_ROOT
        or config.workspace_relative_path != V3_WORKSPACE_ROOT
    ):  # pragma: no cover - typed config validates before manifest construction
        raise AssertionError("Manifest received a non-frozen runtime contract.")
    return _expected_fixed_contract()


def _expected_fixed_contract() -> dict[str, JsonValue]:
    return {
        "seeds": list(SEEDS),
        "epochs": 15,
        "input_hw": [256, 256],
        "batch_size": 8,
        "python_executable": PYTHON_EXECUTABLE,
        "data_root": SHARED_DATA_ROOT,
        "external_asset_root": SHARED_EXTERNAL_ASSET_ROOT,
        "derived_target_root": V3_DERIVED_TARGET_ROOT,
        "workspace_root": V3_WORKSPACE_ROOT,
        "dataset": "synthetic_court_v3_target_court_singleton_kp14",
        "dense_targets": ["kp", "line", "seg"],
        "pose_direct_targets": ["translation", "rotation", "focal"],
        "augmentation": "pose_safe_256_isotropic_fx_eq_fy",
        "backbone": {
            "name": "dinov3_vitb16",
            "checkpoint": (
                "dinov3/checkpoints/"
                "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ),
            "train_mode": "frozen",
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
        },
        "direct_weights": {
            "kp": 1.0,
            "line": 1.0,
            "seg": 1.0,
            "translation": 1.0,
            "rotation": 1.0,
            "focal": 1.0,
        },
        "auxiliary_enabled": {
            "weight": 1.0,
            "temperature": 1.0,
            "huber_delta": 0.01,
            "min_depth_m": 0.1,
            "depth_scale_m": 1.0,
            "cheirality_weight": 0.1,
            "warmup_fraction": 0.1,
        },
        "auxiliary_disabled": {
            "weight": 0.0,
            "temperature": 1.0,
            "huber_delta": 0.01,
            "min_depth_m": 0.1,
            "depth_scale_m": 1.0,
            "cheirality_weight": 0.0,
            "warmup_fraction": 0.0,
        },
        "test_after_fit": True,
        "gpu_execution_policy": "repository_root_shared_training_queue_only",
    }


def _selection_rules() -> dict[str, JsonValue]:
    return {
        "encoder": {
            "reference_depth": 8,
            "relative_tolerance": 0.05,
            "rule": (
                "smallest depth within 5% of depth-8 for GT KP mean and pose-only "
                "reprojection mean, with translation/rotation/focal each no more "
                "than 5% worse; otherwise depth-8"
            ),
        },
        "decoder": {
            "reference": {"family": "dpt", "size": "base"},
            "relative_tolerance": 0.05,
            "dense_absolute_tolerance": 0.01,
            "rule": (
                "minimum decoder MACs then decoder parameters among candidates "
                "within 5% on GT KP and pose-only reprojection/direct-pose metrics "
                "and within 0.01 absolute LINE Dice/SEG mIoU loss; otherwise DPT-base"
            ),
        },
        "adoption": {
            "baseline": "direct-all",
            "candidate": "joint-both",
            "required_gt_improvement": 0.05,
            "maximum_relative_degradation": 0.05,
            "maximum_dense_absolute_degradation": 0.01,
            "maximum_train_cost_overhead": 0.10,
            "consistency_only_improvement_is_rejected": True,
            "requires_all_finite_branch_gradients": True,
        },
    }


def _run_record(
    config: QueryConsistencyAblationConfig,
    *,
    run_id: str,
    phase: str,
    seed: int,
    depth: int | None,
    family: DecoderFamily | None,
    size: DecoderSize | None,
    condition: ConsistencyCondition,
    unresolved: tuple[str, ...],
) -> dict[str, JsonValue]:
    ready = not unresolved
    architecture: dict[str, JsonValue] = {
        "encoder_depth": depth,
        "hidden_dim": 256,
        "num_heads": 8,
        "decoder_family": family,
        "decoder_size": size,
        "encoder_taps": (
            _decoder_taps(depth, family=family, size=size)
            if ready and depth is not None and family is not None and size is not None
            else None
        ),
    }
    relative_dir = f"court_detection/query_consistency_ablation/{run_id}"
    return {
        "run_id": run_id,
        "phase": phase,
        "phase_order": PHASE_ORDER.index(phase) + 1,
        "seed": seed,
        "condition": condition,
        "architecture": architecture,
        "queue_ready": ready,
        "unresolved": list(unresolved),
        "command_argv": (
            _training_argv(
                config,
                run_id=run_id,
                relative_dir=relative_dir,
                seed=seed,
                depth=cast(int, depth),
                family=cast(DecoderFamily, family),
                size=cast(DecoderSize, size),
                condition=condition,
            )
            if ready
            else None
        ),
        "profile_command_argv": (
            _profile_argv(
                config,
                relative_dir=relative_dir,
                depth=cast(int, depth),
                family=cast(DecoderFamily, family),
                size=cast(DecoderSize, size),
            )
            if ready
            else None
        ),
        "evidence_paths": {
            "test_metrics": f"{run_id}/artifacts/test_predictions/metrics.json",
            "tensorboard_log_dir": f"{run_id}/logs/version_0",
            "capacity_profile": f"{run_id}/capacity_profile.json",
        },
    }


def _training_argv(
    config: QueryConsistencyAblationConfig,
    *,
    run_id: str,
    relative_dir: str,
    seed: int,
    depth: int,
    family: DecoderFamily,
    size: DecoderSize,
    condition: ConsistencyCondition,
) -> list[str]:
    loss = _LOSS_BY_CONDITION[condition]
    argv = [
        config.python_executable,
        "-m",
        config.train_module,
        "data/source=synthetic_court",
        f"paths.data_root={config.data_root}",
        f"paths.external_asset_root={config.external_asset_root}",
        f"data.source.workspace_root={config.workspace_relative_path}",
        "data.source.keypoint_court_scope=target_court",
        "data/processing=all",
        f"data.processing.derived_target_root={config.derived_target_relative_path}",
        "data/augmentation=pose_safe",
        f"data.batch_size={config.batch_size}",
        "model=query_encoder_base",
        "model.preset=raw",
        "model/backbone=query_dinov3",
        "model.backbone.train_mode=frozen",
        "model.backbone.last_n_blocks=0",
        "model.backbone.lora.enabled=false",
        "model/task_encoder=query_base",
        f"model/decoder=query_{family}_{size}",
        "model/heads=query_base",
        "model.heads.dense_targets=[kp,seg,line]",
        f"loss={loss}",
        "loss.kp.weight=1.0",
        "loss.line.weight=1.0",
        "loss.seg.weight=1.0",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
        f"training.trainer.max_epochs={config.epochs}",
        "training.learning_rate=0.001",
        "training.weight_decay=0.0001",
        "training.optimizer.name=adamw",
        "training.optimizer.betas=[0.9,0.999]",
        f"run.seed={seed}",
        f"run.output_dir={relative_dir}",
        "run.test_after_fit=true",
        f"paths.artifact_root=outputs/{relative_dir}/artifacts",
    ]
    argv.extend(_architecture_overrides(depth=depth, family=family, size=size))
    argv.extend(_consistency_overrides(condition))
    if run_id not in relative_dir:  # pragma: no cover - internal construction
        raise AssertionError("Run output path must bind its run ID.")
    return argv


def _profile_argv(
    config: QueryConsistencyAblationConfig,
    *,
    relative_dir: str,
    depth: int,
    family: DecoderFamily,
    size: DecoderSize,
) -> list[str]:
    argv = [
        config.python_executable,
        "-m",
        config.profile_module,
        f"paths.external_asset_root={config.external_asset_root}",
        "model=query_encoder_base",
        "model.preset=raw",
        "model/backbone=query_dinov3",
        "model.backbone.train_mode=frozen",
        "model.backbone.last_n_blocks=0",
        "model.backbone.lora.enabled=false",
        "model/task_encoder=query_base",
        f"model/decoder=query_{family}_{size}",
        "model/heads=query_base",
        "model.heads.dense_targets=[kp,seg,line]",
        "loss=query_joint_both",
        f"profile.output_path={relative_dir}/capacity_profile.json",
        "profile.device=cuda",
        "profile.allow_cpu_diagnostic=false",
        "profile.input.batch_size=1",
        "profile.input.channels=3",
        "profile.input.height=256",
        "profile.input.width=256",
        f"profile.candidate.family={family}",
        f"profile.candidate.size={size}",
    ]
    argv.extend(_architecture_overrides(depth=depth, family=family, size=size))
    return argv


def _architecture_overrides(
    *, depth: int, family: DecoderFamily, size: DecoderSize
) -> list[str]:
    taps = _decoder_taps(depth, family=family, size=size)
    rendered = "[" + ",".join(str(value) for value in taps) + "]"
    overrides = [
        f"model.task_encoder.depth={depth}",
        f"model.task_encoder.tap_indices={rendered}",
        f"model.decoder.tap_indices={rendered}",
    ]
    if family == "dpt":
        factors = {
            2: "[2.0,1.0]",
            3: "[4.0,2.0,1.0]",
            4: "[4.0,2.0,1.0,0.5]",
        }
        overrides.extend(
            (
                f"model.decoder.fusion_levels={len(taps)}",
                f"model.decoder.reassemble_factors={factors[len(taps)]}",
            )
        )
    return overrides


def _consistency_overrides(condition: ConsistencyCondition) -> list[str]:
    if condition == "direct-all":
        return [
            "loss.consistency.enabled=false",
            "loss.consistency.weight=0.0",
            "loss.consistency.temperature=1.0",
            "loss.consistency.huber_delta=0.01",
            "loss.consistency.min_depth_m=0.1",
            "loss.consistency.depth_scale_m=1.0",
            "loss.consistency.cheirality_weight=0.0",
            "loss.consistency.warmup_fraction=0.0",
            "loss.consistency.gradient_flow=both",
        ]
    gradient_flow = {
        "joint-both": "both",
        "joint-stopgrad-pose": "stopgrad_pose",
        "joint-stopgrad-dense": "stopgrad_dense",
    }[condition]
    return [
        "loss.consistency.enabled=true",
        "loss.consistency.weight=1.0",
        "loss.consistency.temperature=1.0",
        "loss.consistency.huber_delta=0.01",
        "loss.consistency.min_depth_m=0.1",
        "loss.consistency.depth_scale_m=1.0",
        "loss.consistency.cheirality_weight=0.1",
        "loss.consistency.warmup_fraction=0.1",
        f"loss.consistency.gradient_flow={gradient_flow}",
    ]


def _decoder_taps(
    depth: int,
    *,
    family: DecoderFamily,
    size: DecoderSize,
) -> list[int]:
    if depth <= 0:
        raise ValueError("Encoder depth must be positive.")
    if family in {"linear", "progressive"}:
        return [depth - 1]
    levels = min(2 if size == "tiny" else 4, depth)
    if levels < 2:
        raise ValueError(
            "DPT scaling requires at least two encoder taps; depth 1 cannot be "
            "silently substituted."
        )
    if levels == depth:
        return list(range(depth))
    return [round(index * (depth - 1) / (levels - 1)) for index in range(levels)]


def _manifest_digest(manifest: Mapping[str, JsonValue]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_query_consistency_manifest(
    value: Mapping[str, object],
    *,
    require_resolved: bool,
) -> None:
    """Validate schema, exact order/counts, readiness, and executable argv."""
    expected = {
        "schema",
        "phase_order",
        "fixed_contract",
        "selection_rules",
        "selected",
        "result_contract",
        "runs",
        "manifest_sha256",
    }
    if set(value) != expected or value["schema"] != MANIFEST_SCHEMA:
        raise ValueError("Query consistency manifest fields/schema changed.")
    if value["phase_order"] != list(PHASE_ORDER):
        raise ValueError("Query consistency phase order changed.")
    if value["selection_rules"] != _selection_rules():
        raise ValueError("Query consistency selection rules changed.")
    if value["fixed_contract"] != _expected_fixed_contract():
        raise ValueError("Query consistency fixed contract changed.")
    result_contract = _mapping(value["result_contract"], name="result_contract")
    if result_contract != {
        "metrics": list(RESULT_METRIC_NAMES),
        "diagnostics": list(TRAIN_DIAGNOSTIC_NAMES),
        "diagnostic_source": "tensorboard_scalar_events",
        "loss_curve_source": "tensorboard_scalar_train/loss",
        "capacity_source": "court_query_profile_v1",
        "line_metric_aliases_permitted": [],
    }:
        raise ValueError("Query consistency result contract changed.")
    digest = value["manifest_sha256"]
    if not isinstance(digest, str) or digest != _manifest_digest(
        cast(Mapping[str, JsonValue], value)
    ):
        raise ValueError("Query consistency manifest digest mismatch.")
    selected = _mapping(value["selected"], name="selected")
    if set(selected) != {"encoder_depth", "decoder_family", "decoder_size"}:
        raise ValueError("Query consistency selected fields changed.")
    selected_depth = selected["encoder_depth"]
    selected_family = selected["decoder_family"]
    selected_size = selected["decoder_size"]
    if selected_depth is not None and selected_depth not in {2, 4, 8}:
        raise ValueError("Query consistency selected encoder is invalid/incompatible.")
    if (selected_family is None) != (selected_size is None) or (
        selected_family is not None
        and (
            selected_depth is None
            or selected_family not in _FAMILIES
            or selected_size not in _SIZES
        )
    ):
        raise ValueError("Query consistency selected decoder is invalid.")
    raw_runs = value["runs"]
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("Query consistency runs must be a sequence.")
    runs = tuple(_mapping(run, name="run") for run in raw_runs)
    if len(runs) != 51:
        raise ValueError("Query consistency manifest must contain exactly 51 runs.")
    expected_ids = [
        f"encoder-depth-{depth:02d}-seed-{seed}"
        for depth in (1, 2, 4, 8)
        for seed in SEEDS
    ]
    expected_ids.extend(
        f"decoder-{family}-{size}-seed-{seed}"
        for family in _FAMILIES
        for size in _SIZES
        for seed in SEEDS
    )
    expected_ids.extend(
        f"condition-{condition}-seed-{seed}"
        for condition in _CONDITIONS
        for seed in SEEDS
    )
    if [run["run_id"] for run in runs] != expected_ids:
        raise ValueError("Query consistency run identity/order matrix changed.")
    phase_counts = {phase: 0 for phase in PHASE_ORDER}
    for run in runs:
        _validate_run_record(run, require_resolved=require_resolved)
        phase = cast(str, run["phase"])
        phase_counts[phase] += 1
    if phase_counts != {
        "encoder_scaling": 12,
        "decoder_scaling": 27,
        "consistency_ablation": 12,
    }:
        raise ValueError("Query consistency phase counts changed.")
    selected_decoder = selected["decoder_family"] is not None
    if any(bool(run["queue_ready"]) for run in runs[12:39]) is (
        selected_depth is None
    ):
        raise ValueError("Decoder readiness disagrees with encoder selection.")
    if any(bool(run["queue_ready"]) for run in runs[39:]) is (
        selected_depth is None or not selected_decoder
    ):
        raise ValueError("Formal readiness disagrees with architecture selection.")


def _validate_run_record(
    run: Mapping[str, object], *, require_resolved: bool
) -> None:
    if set(run) != {
        "run_id",
        "phase",
        "phase_order",
        "seed",
        "condition",
        "architecture",
        "queue_ready",
        "unresolved",
        "command_argv",
        "profile_command_argv",
        "evidence_paths",
    }:
        raise ValueError("Query consistency run fields changed.")
    phase = run["phase"]
    if phase not in PHASE_ORDER or run["phase_order"] != PHASE_ORDER.index(phase) + 1:
        raise ValueError("Query consistency run phase/order is invalid.")
    if run["seed"] not in SEEDS or run["condition"] not in _CONDITIONS:
        raise ValueError("Query consistency run seed/condition is invalid.")
    run_id = run["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Query consistency run ID must be a non-empty string.")
    evidence_paths = _mapping(run["evidence_paths"], name="run.evidence_paths")
    if evidence_paths != {
        "test_metrics": f"{run_id}/artifacts/test_predictions/metrics.json",
        "tensorboard_log_dir": f"{run_id}/logs/version_0",
        "capacity_profile": f"{run_id}/capacity_profile.json",
    }:
        raise ValueError("Query consistency evidence layout changed.")
    unresolved = run["unresolved"]
    if not isinstance(unresolved, Sequence) or isinstance(unresolved, (str, bytes)):
        raise ValueError("Query consistency unresolved fields must be a sequence.")
    ready = run["queue_ready"]
    if type(ready) is not bool or ready is not (len(unresolved) == 0):
        raise ValueError("Query consistency queue readiness is inconsistent.")
    for field in ("command_argv", "profile_command_argv"):
        argv = run[field]
        if not ready:
            if argv is not None:
                raise ValueError("Unresolved phases must not expose placeholder argv.")
            continue
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes))
            or list(argv[:2]) != [PYTHON_EXECUTABLE, "-m"]
            or any("__SELECTED" in str(token) for token in argv)
        ):
            raise ValueError("Ready query consistency argv is not directly executable.")
    if ready:
        train_argv = cast(Sequence[object], run["command_argv"])
        required_tokens = {
            f"paths.data_root={SHARED_DATA_ROOT}",
            f"paths.external_asset_root={SHARED_EXTERNAL_ASSET_ROOT}",
            f"data.source.workspace_root={V3_WORKSPACE_ROOT}",
            "data/processing=all",
            f"data.processing.derived_target_root={V3_DERIVED_TARGET_ROOT}",
            "data/augmentation=pose_safe",
            "model.heads.dense_targets=[kp,seg,line]",
            "training.trainer.max_epochs=15",
            "run.test_after_fit=true",
        }
        if not required_tokens.issubset(set(train_argv)):
            raise ValueError("Ready query consistency command lost fixed overrides.")
        condition = run["condition"]
        expected_loss = f"loss={_LOSS_BY_CONDITION[condition]}"
        if expected_loss not in train_argv:
            raise ValueError("Ready query consistency command uses the wrong loss route.")
        relative_dir = f"court_detection/query_consistency_ablation/{run_id}"
        if (
            f"run.output_dir={relative_dir}" not in train_argv
            or f"paths.artifact_root=outputs/{relative_dir}/artifacts"
            not in train_argv
        ):
            raise ValueError("Ready query consistency output paths changed.")
        profile_argv = cast(Sequence[object], run["profile_command_argv"])
        if (
            f"profile.output_path={relative_dir}/capacity_profile.json"
            not in profile_argv
            or "profile.device=cuda" not in profile_argv
            or f"paths.external_asset_root={SHARED_EXTERNAL_ASSET_ROOT}"
            not in profile_argv
        ):
            raise ValueError("Ready query consistency profile evidence path changed.")
        if any(
            str(token).startswith(("paths.data_root=", "data.source.workspace_root="))
            for token in profile_argv
        ):
            raise ValueError("Query consistency profiles must not carry data overrides.")
        consistency_tokens = set(_consistency_overrides(condition))
        if not consistency_tokens.issubset(set(train_argv)):
            raise ValueError("Ready query consistency auxiliary constants changed.")
    if require_resolved and not ready:
        raise ValueError("Requested query consistency phase is unresolved.")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


__all__ = [
    "ConsistencyAblationPhaseConfig",
    "ConsistencyCondition",
    "DecoderFamily",
    "DecoderScalingConfig",
    "DecoderSize",
    "EncoderScalingConfig",
    "MANIFEST_SCHEMA",
    "PHASE_ORDER",
    "PYTHON_EXECUTABLE",
    "QueryConsistencyAblationConfig",
    "RESULT_METRIC_NAMES",
    "SEEDS",
    "SHARED_DATA_ROOT",
    "SHARED_EXTERNAL_ASSET_ROOT",
    "TRAIN_DIAGNOSTIC_NAMES",
    "V3_DERIVED_TARGET_ROOT",
    "V3_WORKSPACE_ROOT",
    "build_query_consistency_manifest",
    "validate_query_consistency_ablation_boundary",
    "validate_query_consistency_manifest",
]
