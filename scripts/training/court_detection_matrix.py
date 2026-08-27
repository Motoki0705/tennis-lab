"""Build and verify the refreshed Issue #790 Court Detection experiment matrix.

Usage:
    python scripts/training/court_detection_matrix.py validate-configs
    python scripts/training/court_detection_matrix.py generate-manifest --output outputs/court_detection/matrix/manifest.json
    python scripts/training/court_detection_matrix.py collect-results --manifest manifest.json --queue-dir .training_queue --output results.json
    python scripts/training/court_detection_matrix.py summarize-results --results results.json --output-dir summary

Notes:
    - The manifest is deterministic and binds every queue command to the
      refreshed hierarchy phase; collectors reject old, foreign, or incomplete
      queue/repro bundles.
    - This script never starts a training worker and never invents missing
      metrics, diagnostics, Colab artifacts, or adoption decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hydra import compose, initialize_config_dir

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MANIFEST_SCHEMA = "court_detection_hierarchy_matrix_manifest_v1"
RESULTS_SCHEMA = "court_detection_hierarchy_matrix_results_v1"
RUN_EVIDENCE_SCHEMA = "court_detection_hierarchy_matrix_run_evidence_v1"
SUMMARY_SCHEMA = "court_detection_hierarchy_matrix_summary_v1"
PHASE = "issue-790-hierarchy-refresh-v1"
ISSUE = 790
ISSUE_SHA256 = "6987820c8009dc9e4fa6081fcfb6fe9c723600b53126aa926c8f03799fb9fe42"
SEED = 42
DEPTHS = (1, 8)
INPUT_SIZES = (256, 384)
DPT_CHANNELS: Mapping[str, int] = {
    "tiny": 64,
    "small": 128,
    "base": 256,
    "large": 512,
}
DPT_CONFIGS: Mapping[str, str] = {
    "tiny": "dpt_tiny",
    "small": "dpt_small",
    "base": "dpt_base",
    "large": "dpt_large",
}
CONDITIONS = ("kp-only", "line-only", "seg-only", "pose-only", "pure", "weighted")
CONFIG_ROOT = (
    REPOSITORY_ROOT / "src/tasks/court_detection/configs"
)

_COMMON_METRIC_KEYS = {
    "kp_mean_distance_px",
    "kp_median_distance_px",
    "line_dice",
    "seg_miou",
    "pose_translation_l2_m",
    "pose_rotation_geodesic_deg",
    "pose_focal_relative_error",
    "pose_reprojection_mean_distance_px",
    "kp_pose_consistency_distance_px",
}
_METRICS_BY_CONDITION: Mapping[str, frozenset[str]] = {
    "kp-only": frozenset({"kp_mean_distance_px", "kp_median_distance_px"}),
    "line-only": frozenset({"line_dice"}),
    "seg-only": frozenset({"seg_miou"}),
    "pose-only": frozenset(
        {
            "pose_translation_l2_m",
            "pose_rotation_geodesic_deg",
            "pose_focal_relative_error",
            "pose_reprojection_mean_distance_px",
        }
    ),
    "pure": frozenset(_COMMON_METRIC_KEYS),
    "weighted": frozenset(_COMMON_METRIC_KEYS),
}
_LOSSES_BY_CONDITION: Mapping[str, frozenset[str]] = {
    "kp-only": frozenset({"kp_direct", "weighted_total"}),
    "line-only": frozenset({"line_direct", "weighted_total"}),
    "seg-only": frozenset({"seg_direct", "weighted_total"}),
    "pose-only": frozenset(
        {
            "pose_translation_direct",
            "pose_rotation_direct",
            "pose_focal_direct",
            "weighted_total",
        }
    ),
    "pure": frozenset(
        {
            "kp_direct",
            "line_direct",
            "seg_direct",
            "pose_translation_direct",
            "pose_rotation_direct",
            "pose_focal_direct",
            "consistency_coordinate",
            "consistency_cheirality",
            "weighted_total",
        }
    ),
    "weighted": frozenset(
        {
            "kp_direct",
            "line_direct",
            "seg_direct",
            "pose_translation_direct",
            "pose_rotation_direct",
            "pose_focal_direct",
            "consistency_coordinate",
            "consistency_cheirality",
            "weighted_total",
        }
    ),
}
_GRADIENT_BRANCHES_BY_CONDITION: Mapping[str, frozenset[str]] = {
    "kp-only": frozenset({"kp"}),
    "line-only": frozenset({"line"}),
    "seg-only": frozenset({"seg"}),
    "pose-only": frozenset({"pose"}),
    "pure": frozenset({"kp", "line", "seg", "pose"}),
    "weighted": frozenset({"kp", "line", "seg", "pose"}),
}


class MatrixError(RuntimeError):
    """Raised when a manifest or result bundle violates the matrix contract."""


@dataclass(frozen=True, slots=True)
class MatrixOptions:
    """Values that are allowed to vary before a manifest is frozen."""

    python: str = ".venv/bin/python"
    max_epochs: int = 15
    batch_size: int = 8
    seed: int = SEED
    run_prefix: str = "court_detection/matrix"
    manifest_path: str = str(
        REPOSITORY_ROOT / "outputs/court_detection/matrix/manifest.json"
    )
    consistency_weight: float = 1.0
    weighted_kp: float = 1.0
    weighted_line: float = 0.5
    weighted_seg: float = 0.5
    # The pose targets are expressed in metres/radians/log-pixels while the
    # dense objectives are mean per-pixel losses.  These defaults compensate
    # the predictable scale gap (translation is O(10^2), focal is O(10^1),
    # rotation is O(1), and dense terms are O(1)) without changing the pure
    # objective.  They are frozen into the weighted run manifest.
    weighted_pose_translation: float = 0.01
    weighted_pose_rotation: float = 0.5
    weighted_pose_focal: float = 0.05
    weighted_consistency: float = 0.25

    def validate(self) -> None:
        """Reject values that cannot identify a reproducible training protocol."""
        if not self.python:
            raise MatrixError("Training Python must be non-empty.")
        if self.max_epochs != 15 or self.batch_size != 8:
            raise MatrixError("The refreshed protocol fixes max_epochs=15 and batch_size=8.")
        if self.seed != SEED:
            raise MatrixError(f"The refreshed initial matrix seed is fixed to {SEED}.")
        if self.consistency_weight != 1.0:
            raise MatrixError("The pure multimodal consistency coefficient is fixed to 1.0.")
        prefix = Path(self.run_prefix)
        if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
            raise MatrixError("run_prefix must be a non-empty repository-relative path.")
        manifest_path = Path(self.manifest_path)
        if not manifest_path.is_absolute():
            raise MatrixError("manifest_path must be absolute for queued producers.")
        weights = (
            self.consistency_weight,
            self.weighted_kp,
            self.weighted_line,
            self.weighted_seg,
            self.weighted_pose_translation,
            self.weighted_pose_rotation,
            self.weighted_pose_focal,
            self.weighted_consistency,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise MatrixError("Every matrix loss weight must be finite and positive.")


@dataclass(frozen=True, slots=True)
class RunDefinition:
    """One unique queue job in the condition/scaling matrix."""

    entry_id: str
    roles: tuple[Literal["condition", "scaling"], ...]
    condition: str
    processing: str
    loss_strategy: str
    depth: int | None
    input_size: int
    dpt_size: str
    overrides: tuple[str, ...]
    output_dir: str
    command: str

    @property
    def queue_name(self) -> str:
        return f"i790-court-{self.entry_id}"

    def to_json(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "queue_name": self.queue_name,
            "roles": list(self.roles),
            "condition": self.condition,
            "processing": self.processing,
            "loss_strategy": self.loss_strategy,
            "depth": self.depth,
            "input_size": self.input_size,
            "dpt_size": self.dpt_size,
            "dpt_channels": DPT_CHANNELS[self.dpt_size],
            "output_dir": self.output_dir,
            "overrides": list(self.overrides),
            "command": self.command,
        }


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _base_overrides(options: MatrixOptions, *, input_size: int) -> tuple[str, ...]:
    return (
        "data/source=synthetic_court",
        "data.source.schema=v3",
        "data.source.keypoint_court_scope=target_court",
        (
            "data.source.workspace_root="
            "issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes"
        ),
        "data.source.scene_ids=[B00]",
        "data.processing.derived_target_root=court_detection/derived_targets_issue790_v3",
        "paths.artifact_root=${oc.env:TENNIS_REPRO_DIR}",
        "data/augmentation=pose_safe",
        f"data.augmentation.train_scales=[{input_size}]",
        f"data.augmentation.val_short_side={input_size}",
        "data.augmentation.preserve_fx_fy=true",
        f"data.batch_size={options.batch_size}",
        "model/encoder=dinov3",
        "model.encoder.train_mode=frozen",
        "model.encoder.last_n_blocks=0",
        "model.encoder.lora.enabled=false",
        f"training.trainer.max_epochs={options.max_epochs}",
        "training.learning_rate=0.001",
        "training.weight_decay=0.0001",
        "training.optimizer.name=adamw",
        "training.optimizer.betas=[0.9,0.999]",
        f"run.seed={options.seed}",
        "run.test_after_fit=true",
    )


def _pose_overrides() -> tuple[str, ...]:
    return (
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
    )


def _pose_only_loss_overrides() -> tuple[str, ...]:
    return (
        "loss=default",
        "loss.kp.weight=0.0",
        "loss.seg.weight=0.0",
        "loss.line.weight=0.0",
        *_pose_overrides(),
        "loss.consistency.enabled=false",
    )


def _consistency_overrides(weight: float) -> tuple[str, ...]:
    return (
        "loss.consistency.enabled=true",
        f"loss.consistency.weight={weight}",
        "loss.consistency.temperature=1.0",
        "loss.consistency.huber_delta=0.01",
        "loss.consistency.min_depth_m=0.1",
        "loss.consistency.depth_scale_m=1.0",
        "loss.consistency.cheirality_weight=0.1",
        "loss.consistency.warmup_fraction=0.1",
        "loss.consistency.gradient_flow=both",
    )


def _condition_overrides(condition: str, options: MatrixOptions) -> tuple[str, ...]:
    if condition in {"kp-only", "line-only", "seg-only"}:
        processing = condition.removesuffix("-only")
        return (
            f"data/processing={processing}",
            "loss=default",
            "model/transformer_encoder=none",
        )
    if condition == "pose-only":
        return (
            "data/processing=kp",
            *_pose_only_loss_overrides(),
            "model/transformer_encoder=default",
        )
    if condition == "pure":
        return (
            "data/processing=all",
            "loss=default",
            "model/transformer_encoder=default",
            *_pose_overrides(),
            *_consistency_overrides(options.consistency_weight),
            "loss.kp.weight=1.0",
            "loss.line.weight=1.0",
            "loss.seg.weight=1.0",
        )
    if condition == "weighted":
        return (
            "data/processing=all",
            "loss=default",
            "model/transformer_encoder=default",
            "loss.pose.enabled=true",
            f"loss.pose.translation_weight={options.weighted_pose_translation}",
            f"loss.pose.rotation_weight={options.weighted_pose_rotation}",
            f"loss.pose.focal_weight={options.weighted_pose_focal}",
            *_consistency_overrides(options.weighted_consistency),
            f"loss.kp.weight={options.weighted_kp}",
            f"loss.line.weight={options.weighted_line}",
            f"loss.seg.weight={options.weighted_seg}",
        )
    raise MatrixError(f"Unknown Court matrix condition: {condition!r}.")


def _processing_for(condition: str) -> str:
    return {
        "kp-only": "kp",
        "line-only": "line",
        "seg-only": "seg",
        "pose-only": "kp",
        "pure": "all",
        "weighted": "all",
    }[condition]


def _make_run(
    *,
    options: MatrixOptions,
    entry_id: str,
    roles: tuple[Literal["condition", "scaling"], ...],
    condition: str,
    depth: int | None,
    input_size: int,
    dpt_size: str,
) -> RunDefinition:
    if dpt_size not in DPT_CONFIGS:
        raise MatrixError(f"Unknown DPT size: {dpt_size!r}.")
    output_dir = f"{options.run_prefix}/runs/{entry_id}"
    overrides = (
        *_base_overrides(options, input_size=input_size),
        f"model/decoder={DPT_CONFIGS[dpt_size]}",
        *_condition_overrides(condition, options),
        *((f"model.transformer_encoder.depth={depth}",) if depth is not None else ()),
        f"run.output_dir={output_dir}",
    )
    training_command = shlex.join(
        (options.python, "-m", "src.tasks.court_detection.scripts.train", *overrides)
    )
    command = (
        "TENNIS_COURT_MATRIX_MANIFEST_PATH="
        f"{shlex.quote(options.manifest_path)} {training_command}"
    )
    return RunDefinition(
        entry_id=entry_id,
        roles=roles,
        condition=condition,
        processing=_processing_for(condition),
        loss_strategy=condition,
        depth=depth,
        input_size=input_size,
        dpt_size=dpt_size,
        overrides=overrides,
        output_dir=output_dir,
        command=command,
    )


def build_runs(options: MatrixOptions) -> tuple[RunDefinition, ...]:
    """Return six primary conditions plus the complete pure scaling matrix."""
    options.validate()
    runs: list[RunDefinition] = []
    for condition in CONDITIONS:
        depth = None if condition in {"kp-only", "line-only", "seg-only"} else 8
        roles: tuple[Literal["condition", "scaling"], ...] = (
            ("condition", "scaling") if condition == "pure" else ("condition",)
        )
        runs.append(
            _make_run(
                options=options,
                entry_id=(
                    "condition-pure-d8-i256-dpt-large"
                    if condition == "pure"
                    else f"condition-{condition}-i256-dpt-large"
                ),
                roles=roles,
                condition=condition,
                depth=depth,
                input_size=256,
                dpt_size="large",
            )
        )
    for depth in DEPTHS:
        for input_size in INPUT_SIZES:
            for dpt_size in DPT_CHANNELS:
                if (depth, input_size, dpt_size) == (8, 256, "large"):
                    continue
                runs.append(
                    _make_run(
                        options=options,
                        entry_id=(
                            f"scaling-pure-d{depth}-i{input_size}-dpt-{dpt_size}"
                        ),
                        roles=("scaling",),
                        condition="pure",
                        depth=depth,
                        input_size=input_size,
                        dpt_size=dpt_size,
                    )
                )
    _validate_run_coverage(runs)
    return tuple(runs)


def _validate_run_coverage(runs: Sequence[RunDefinition]) -> None:
    expected_count = len(CONDITIONS) + len(DEPTHS) * len(INPUT_SIZES) * len(DPT_CHANNELS) - 1
    if len(runs) != expected_count:
        raise MatrixError(f"Expected {expected_count} unique jobs, got {len(runs)}.")
    if len({run.entry_id for run in runs}) != len(runs):
        raise MatrixError("Matrix entry IDs must be unique.")
    if len({run.output_dir for run in runs}) != len(runs):
        raise MatrixError("Matrix output directories must be unique.")
    condition_set = {run.condition for run in runs if "condition" in run.roles}
    if condition_set != set(CONDITIONS):
        raise MatrixError("The primary condition coverage is incomplete.")
    scaling = {
        (run.depth, run.input_size, run.dpt_size)
        for run in runs
        if "scaling" in run.roles
    }
    expected_scaling = {
        (depth, input_size, dpt_size)
        for depth in DEPTHS
        for input_size in INPUT_SIZES
        for dpt_size in DPT_CHANNELS
    }
    if scaling != expected_scaling:
        raise MatrixError("The depth/input/DPT Cartesian scaling coverage is incomplete.")


def build_manifest(options: MatrixOptions) -> dict[str, object]:
    """Create a deterministic self-digesting manifest."""
    runs = build_runs(options)
    body: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "phase": PHASE,
        "issue": ISSUE,
        "issue_sha256": ISSUE_SHA256,
        "run_evidence_schema": RUN_EVIDENCE_SCHEMA,
        "protocol": {
            "source": "synthetic_court_v3_attempt9_target_court_B00",
            "augmentation": "pose_safe",
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
            },
            "max_epochs": options.max_epochs,
            "batch_size": options.batch_size,
            "seed": options.seed,
            "test_after_fit": True,
            "pure_weights": {
                "kp": 1.0,
                "line": 1.0,
                "seg": 1.0,
                "pose_translation": 1.0,
                "pose_rotation": 1.0,
                "pose_focal": 1.0,
                "consistency": options.consistency_weight,
            },
            "weighted_weights": {
                "kp": options.weighted_kp,
                "line": options.weighted_line,
                "seg": options.weighted_seg,
                "pose_translation": options.weighted_pose_translation,
                "pose_rotation": options.weighted_pose_rotation,
                "pose_focal": options.weighted_pose_focal,
                "consistency": options.weighted_consistency,
            },
            "weighted_strategy": (
                "Scale-aware fixed weights: translation=0.01, rotation=0.5, "
                "focal=0.05, KP=1, LINE=0.5, SEG=0.5, consistency=0.25. "
                "No automatic loss balancing; values are frozen by this manifest."
            ),
        },
        "coverage": {
            "conditions": list(CONDITIONS),
            "depths": list(DEPTHS),
            "input_sizes": list(INPUT_SIZES),
            "dpt_channels": dict(DPT_CHANNELS),
            "condition_job_count": len(CONDITIONS),
            "scaling_combination_count": len(DEPTHS) * len(INPUT_SIZES) * len(DPT_CHANNELS),
            "unique_job_count": len(runs),
        },
        "entries": [run.to_json() for run in runs],
    }
    return {**body, "manifest_sha256": _digest(body)}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MatrixError(f"Required artifact is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise MatrixError(f"Invalid JSON artifact {path}: {error}") from error


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MatrixError(f"{path} must be a JSON object with string keys.")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, *, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise MatrixError(f"{path} must be a JSON array.")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise MatrixError(f"{path} must be a non-empty string.")
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MatrixError(f"{path} must be finite.")
    return result


def validate_manifest(value: object) -> Mapping[str, object]:
    """Validate identity, self-digest, phase, and complete matrix coverage."""
    manifest = _mapping(value, path="manifest")
    expected_keys = {
        "schema",
        "phase",
        "issue",
        "issue_sha256",
        "run_evidence_schema",
        "protocol",
        "coverage",
        "entries",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise MatrixError("Manifest keys do not match the refreshed schema exactly.")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["phase"] != PHASE:
        raise MatrixError("Old-phase or foreign Court matrix manifest rejected.")
    if manifest["run_evidence_schema"] != RUN_EVIDENCE_SCHEMA:
        raise MatrixError("Manifest run-evidence schema is foreign.")
    if manifest["issue"] != ISSUE or manifest["issue_sha256"] != ISSUE_SHA256:
        raise MatrixError("Manifest is not bound to the frozen refreshed Issue #790.")
    supplied_digest = _string(manifest["manifest_sha256"], path="manifest.manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied_digest != _digest(body):
        raise MatrixError("Manifest SHA-256 does not match its canonical contents.")
    protocol = _mapping(manifest["protocol"], path="manifest.protocol")
    expected_protocol_keys = {
        "source",
        "augmentation",
        "optimizer",
        "max_epochs",
        "batch_size",
        "seed",
        "test_after_fit",
        "pure_weights",
        "weighted_weights",
        "weighted_strategy",
    }
    if set(protocol) != expected_protocol_keys:
        raise MatrixError("Manifest protocol keys are incomplete or foreign.")
    if (
        protocol["source"] != "synthetic_court_v3_attempt9_target_court_B00"
        or protocol["augmentation"] != "pose_safe"
        or protocol["seed"] != SEED
        or protocol["test_after_fit"] is not True
    ):
        raise MatrixError("Manifest protocol is not the refreshed fixed protocol.")
    for key in ("max_epochs", "batch_size"):
        value_item = protocol[key]
        if isinstance(value_item, bool) or not isinstance(value_item, int) or value_item <= 0:
            raise MatrixError(f"manifest.protocol.{key} must be a positive integer.")
    if protocol["max_epochs"] != 15 or protocol["batch_size"] != 8:
        raise MatrixError("Manifest epoch/batch protocol is foreign.")
    optimizer = _mapping(protocol["optimizer"], path="manifest.protocol.optimizer")
    if optimizer != {
        "name": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
    }:
        raise MatrixError("Manifest optimizer protocol is foreign.")
    weight_keys = {
        "kp",
        "line",
        "seg",
        "pose_translation",
        "pose_rotation",
        "pose_focal",
        "consistency",
    }
    for weight_family in ("pure_weights", "weighted_weights"):
        weights = _mapping(protocol[weight_family], path=f"manifest.protocol.{weight_family}")
        if set(weights) != weight_keys:
            raise MatrixError(f"Manifest {weight_family} keys are incomplete or foreign.")
        for key, raw_weight in weights.items():
            if _number(raw_weight, path=f"manifest.protocol.{weight_family}.{key}") <= 0.0:
                raise MatrixError(f"Manifest {weight_family}.{key} must be positive.")
    if any(
        value_item != 1.0
        for value_item in cast("Mapping[str, object]", protocol["pure_weights"]).values()
    ):
        raise MatrixError("Pure multimodal weights must all equal 1.0.")
    _string(protocol["weighted_strategy"], path="manifest.protocol.weighted_strategy")
    expected_coverage = {
        "conditions": list(CONDITIONS),
        "depths": list(DEPTHS),
        "input_sizes": list(INPUT_SIZES),
        "dpt_channels": dict(DPT_CHANNELS),
        "condition_job_count": len(CONDITIONS),
        "scaling_combination_count": len(DEPTHS) * len(INPUT_SIZES) * len(DPT_CHANNELS),
        "unique_job_count": len(CONDITIONS)
        + len(DEPTHS) * len(INPUT_SIZES) * len(DPT_CHANNELS)
        - 1,
    }
    if manifest["coverage"] != expected_coverage:
        raise MatrixError("Manifest coverage declaration is incomplete or foreign.")
    entries = _sequence(manifest["entries"], path="manifest.entries")
    descriptors: list[RunDefinition] = []
    command_python: str | None = None
    command_manifest_path: str | None = None
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, path=f"manifest.entries[{index}]")
        expected_entry_keys = {
            "entry_id",
            "queue_name",
            "roles",
            "condition",
            "processing",
            "loss_strategy",
            "depth",
            "input_size",
            "dpt_size",
            "dpt_channels",
            "output_dir",
            "overrides",
            "command",
        }
        if set(entry) != expected_entry_keys:
            raise MatrixError(f"Manifest entry {index} has unknown or missing keys.")
        condition = _string(entry["condition"], path=f"entries[{index}].condition")
        if condition not in CONDITIONS:
            raise MatrixError(f"Manifest entry {index} uses an unknown condition.")
        dpt_size = _string(entry["dpt_size"], path=f"entries[{index}].dpt_size")
        if dpt_size not in DPT_CHANNELS or entry["dpt_channels"] != DPT_CHANNELS[dpt_size]:
            raise MatrixError(f"Manifest entry {index} has a foreign DPT size mapping.")
        raw_depth = entry["depth"]
        if raw_depth is not None and raw_depth not in DEPTHS:
            raise MatrixError(f"Manifest entry {index} has an unsupported depth.")
        input_size = entry["input_size"]
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size not in INPUT_SIZES:
            raise MatrixError(f"Manifest entry {index} has an unsupported input size.")
        raw_roles = _sequence(entry["roles"], path=f"entries[{index}].roles")
        if not raw_roles or any(role not in {"condition", "scaling"} for role in raw_roles):
            raise MatrixError(f"Manifest entry {index} has invalid roles.")
        raw_overrides = _sequence(entry["overrides"], path=f"entries[{index}].overrides")
        if any(not isinstance(item, str) or not item for item in raw_overrides):
            raise MatrixError(f"Manifest entry {index} has invalid Hydra overrides.")
        descriptor = RunDefinition(
            entry_id=_string(entry["entry_id"], path=f"entries[{index}].entry_id"),
            roles=cast("tuple[Literal['condition', 'scaling'], ...]", tuple(raw_roles)),
            condition=condition,
            processing=_string(entry["processing"], path=f"entries[{index}].processing"),
            loss_strategy=_string(
                entry["loss_strategy"], path=f"entries[{index}].loss_strategy"
            ),
            depth=raw_depth,
            input_size=input_size,
            dpt_size=dpt_size,
            overrides=cast("tuple[str, ...]", tuple(raw_overrides)),
            output_dir=_string(entry["output_dir"], path=f"entries[{index}].output_dir"),
            command=_string(entry["command"], path=f"entries[{index}].command"),
        )
        if entry["queue_name"] != descriptor.queue_name:
            raise MatrixError(f"Manifest entry {index} queue name is not canonical.")
        if descriptor.processing != _processing_for(condition):
            raise MatrixError(f"Manifest entry {index} target bundle is foreign.")
        if descriptor.loss_strategy != condition:
            raise MatrixError(f"Manifest entry {index} loss strategy is foreign.")
        output_path = Path(descriptor.output_dir)
        if output_path.is_absolute() or ".." in output_path.parts:
            raise MatrixError(f"Manifest entry {index} output path is unsafe.")
        expected_suffix = Path("runs") / descriptor.entry_id
        if tuple(output_path.parts[-2:]) != tuple(expected_suffix.parts):
            raise MatrixError(f"Manifest entry {index} output path is not entry-bound.")
        argv = shlex.split(descriptor.command)
        assignment_prefix = "TENNIS_COURT_MATRIX_MANIFEST_PATH="
        manifest_assignment = argv[0] if argv else ""
        options_manifest_path = manifest_assignment.removeprefix(assignment_prefix)
        if (
            len(argv) < 5
            or not manifest_assignment.startswith(assignment_prefix)
            or not options_manifest_path
            or not Path(options_manifest_path).is_absolute()
            or argv[2:4] != [
                "-m",
                "src.tasks.court_detection.scripts.train",
            ]
        ):
            raise MatrixError(f"Manifest entry {index} command entrypoint is foreign.")
        if tuple(argv[4:]) != descriptor.overrides:
            raise MatrixError(f"Manifest entry {index} command/override binding is invalid.")
        if command_python is None:
            command_python = argv[1]
        elif argv[1] != command_python:
            raise MatrixError("Manifest entries must share one Python executable.")
        if command_manifest_path is None:
            command_manifest_path = options_manifest_path
        elif options_manifest_path != command_manifest_path:
            raise MatrixError("Manifest entries must share one manifest path.")
        required_overrides = {
            "data/source=synthetic_court",
            "data.source.schema=v3",
            "data.source.keypoint_court_scope=target_court",
            (
                "data.source.workspace_root="
                "issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes"
            ),
            "data.processing.derived_target_root=court_detection/derived_targets_issue790_v3",
            "paths.artifact_root=${oc.env:TENNIS_REPRO_DIR}",
            "data/augmentation=pose_safe",
            f"data.augmentation.train_scales=[{descriptor.input_size}]",
            f"data.augmentation.val_short_side={descriptor.input_size}",
            f"model/decoder={DPT_CONFIGS[descriptor.dpt_size]}",
            f"data/processing={descriptor.processing}",
            f"run.seed={SEED}",
            "run.test_after_fit=true",
            f"run.output_dir={descriptor.output_dir}",
        }
        if not required_overrides.issubset(descriptor.overrides):
            raise MatrixError(f"Manifest entry {index} fixed overrides are incomplete.")
        if descriptor.depth is None:
            if condition not in {"kp-only", "line-only", "seg-only"} or (
                "model/transformer_encoder=none" not in descriptor.overrides
            ):
                raise MatrixError(f"Manifest entry {index} disabled Transformer is foreign.")
        elif (
            "model/transformer_encoder=default" not in descriptor.overrides
            or f"model.transformer_encoder.depth={descriptor.depth}"
            not in descriptor.overrides
        ):
            raise MatrixError(f"Manifest entry {index} Transformer depth is not explicit.")
        descriptors.append(descriptor)
    _validate_run_coverage(descriptors)
    return manifest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json(path: Path, value: object, *, refuse_changed_existing: bool = False) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if refuse_changed_existing and path.exists() and path.read_bytes() != payload:
        raise MatrixError(f"Refusing to replace a different frozen manifest: {path}")
    _atomic_write(path, payload)


def validate_configs(options: MatrixOptions) -> None:
    """Compose every entry through Hydra and its strict typed boundary."""
    from src.tasks.court_detection.configuration import (
        CourtModelConfig,
        CourtTrainingConfig,
    )

    runs = build_runs(options)
    for run in runs:
        previous_repro_dir = os.environ.get("TENNIS_REPRO_DIR")
        os.environ["TENNIS_REPRO_DIR"] = str(
            REPOSITORY_ROOT / ".training_queue/repro/_matrix_config_validation"
        )
        try:
            with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
                config = compose(config_name="train", overrides=list(run.overrides))
            runtime = CourtTrainingConfig.from_config(config)
        finally:
            if previous_repro_dir is None:
                os.environ.pop("TENNIS_REPRO_DIR", None)
            else:
                os.environ["TENNIS_REPRO_DIR"] = previous_repro_dir
        if not isinstance(runtime.model, CourtModelConfig):
            raise MatrixError(f"{run.entry_id}: expected CourtModelConfig.")
        decoder = runtime.model.decoder
        if decoder.name != "dpt" or decoder.channels != DPT_CHANNELS[run.dpt_size]:
            raise MatrixError(f"{run.entry_id}: DPT size/channel contract mismatch.")
        decoder_size = getattr(decoder, "size", None)
        if decoder_size != run.dpt_size:
            raise MatrixError(f"{run.entry_id}: strict DPT size identity mismatch.")
        transformer = runtime.model.transformer_encoder
        if run.depth is None:
            if transformer.enabled:
                raise MatrixError(f"{run.entry_id}: dense-only standalone must disable pose query.")
        elif not transformer.enabled or transformer.depth != run.depth:
            raise MatrixError(f"{run.entry_id}: Transformer depth contract mismatch.")
        if runtime.data.augmentation.train_scales != (run.input_size,):
            raise MatrixError(f"{run.entry_id}: training input size mismatch.")
        if runtime.data.augmentation.val_short_side != run.input_size:
            raise MatrixError(f"{run.entry_id}: evaluation input size mismatch.")
        kinds = tuple(target.kind for target in runtime.data.processing.targets)
        expected_kinds = {
            "kp": ("kp",),
            "line": ("line",),
            "seg": ("seg",),
            "all": ("kp", "seg", "line"),
        }[run.processing]
        if kinds != expected_kinds:
            raise MatrixError(f"{run.entry_id}: resolved target bundle mismatch.")
        if run.condition == "pose-only":
            pose_weights = (
                runtime.loss.pose.translation_weight,
                runtime.loss.pose.rotation_weight,
                runtime.loss.pose.focal_weight,
            )
            if (
                not runtime.loss.pose.enabled
                or pose_weights != (1.0, 1.0, 1.0)
                or any(runtime.loss.dense_weights.values())
                or runtime.loss.consistency.enabled
            ):
                raise MatrixError(
                    "pose-only must use KP geometry, zero dense objective, unit pose "
                    "weights, and disabled consistency."
                )
        if run.condition in {"pure", "weighted"} and not runtime.loss.consistency.enabled:
            raise MatrixError(f"{run.entry_id}: multimodal consistency must be enabled.")


def check_removals() -> None:
    """Verify only the explicitly retired Court query and Colab train paths."""
    from src.tasks.court_detection.configuration import (
        CourtModelConfig,
        CourtTrainingConfig,
    )

    forbidden_exact = (
        REPOSITORY_ROOT / "src/tasks/court_detection/experiments",
        REPOSITORY_ROOT / "src/tasks/court_detection/models/query_encoder",
        REPOSITORY_ROOT / "src/tasks/court_detection/configs/query_ablation",
        REPOSITORY_ROOT / "src/tasks/court_detection/configs/query_consistency_ablation",
        REPOSITORY_ROOT / "scripts/colab/train",
    )
    present = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in forbidden_exact
        if path.is_file()
        or path.is_symlink()
        or (path.is_dir() and any(child.is_file() for child in path.rglob("*")))
    ]
    glob_patterns = (
        "src/tasks/court_detection/configs/loss/query_*.yaml",
        "src/tasks/court_detection/configs/model/query_encoder*.yaml",
        "src/tasks/court_detection/configs/model/backbone/query_*.yaml",
        "src/tasks/court_detection/configs/model/decoder/query_*.yaml",
        "src/tasks/court_detection/configs/model/heads/query_*.yaml",
        "src/tasks/court_detection/configs/model/task_encoder/query_*.yaml",
        "src/tasks/court_detection/configs/*query_ablation*.yaml",
        "src/tasks/court_detection/configs/*query_consistency*.yaml",
        "src/tasks/court_detection/scripts/*query_ablation*.py",
        "src/tasks/court_detection/scripts/profile_query_model.py",
    )
    for pattern in glob_patterns:
        present.extend(
            str(path.relative_to(REPOSITORY_ROOT))
            for path in REPOSITORY_ROOT.glob(pattern)
        )
    if present:
        rendered = "\n  - ".join(sorted(set(present)))
        raise MatrixError(f"Retired Court paths remain:\n  - {rendered}")
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
        config = compose(config_name="train")
    runtime = CourtTrainingConfig.from_config(config)
    if not isinstance(runtime.model, CourtModelConfig):
        raise MatrixError("Production default no longer resolves to CourtModelConfig.")
    if runtime.model.transformer_encoder.enabled or runtime.model.decoder.name != "fpn":
        raise MatrixError("Production default must remain Transformer none plus FPN.")


def _validate_metrics(
    value: object,
    *,
    condition: str,
    path: Path,
) -> dict[str, float]:
    metrics = _mapping(value, path=str(path))
    required = _METRICS_BY_CONDITION[condition]
    missing = required - set(metrics)
    if missing:
        raise MatrixError(f"{path}: missing required metrics {sorted(missing)}.")
    return {
        key: _number(raw_value, path=f"{path}:{key}")
        for key, raw_value in metrics.items()
    }


def _validate_run_evidence(
    value: object,
    *,
    entry: Mapping[str, object],
    manifest_sha256: str,
    path: Path,
) -> Mapping[str, object]:
    evidence = _mapping(value, path=str(path))
    expected_keys = {
        "schema",
        "phase",
        "manifest_sha256",
        "entry_id",
        "complete",
        "loss_terms",
        "diagnostics",
    }
    if set(evidence) != expected_keys:
        raise MatrixError(f"{path}: run evidence keys are incomplete or foreign.")
    if (
        evidence["schema"] != RUN_EVIDENCE_SCHEMA
        or evidence["phase"] != PHASE
        or evidence["manifest_sha256"] != manifest_sha256
        or evidence["entry_id"] != entry["entry_id"]
    ):
        raise MatrixError(f"{path}: old-phase or foreign run evidence rejected.")
    if evidence["complete"] is not True:
        raise MatrixError(f"{path}: incomplete run evidence rejected.")
    condition = cast(str, entry["condition"])
    losses = _mapping(evidence["loss_terms"], path=f"{path}:loss_terms")
    missing_losses = _LOSSES_BY_CONDITION[condition] - set(losses)
    if missing_losses:
        raise MatrixError(f"{path}: missing loss terms {sorted(missing_losses)}.")
    for key, value_item in losses.items():
        _number(value_item, path=f"{path}:loss_terms.{key}")
    diagnostics = _mapping(evidence["diagnostics"], path=f"{path}:diagnostics")
    expected_diagnostic_keys = {
        "gradient_finite",
        "parameter_count",
        "train_step_time_ms",
        "peak_memory_bytes",
    }
    if set(diagnostics) != expected_diagnostic_keys:
        raise MatrixError(f"{path}: diagnostics keys are incomplete or foreign.")
    gradients = _mapping(
        diagnostics["gradient_finite"], path=f"{path}:diagnostics.gradient_finite"
    )
    required_branches = _GRADIENT_BRANCHES_BY_CONDITION[condition]
    if required_branches - set(gradients):
        raise MatrixError(f"{path}: gradient diagnostics are incomplete.")
    if any(value_item is not True for key, value_item in gradients.items() if key in required_branches):
        raise MatrixError(f"{path}: non-finite or missing active-branch gradient rejected.")
    parameter_count = diagnostics["parameter_count"]
    peak_memory = diagnostics["peak_memory_bytes"]
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise MatrixError(f"{path}: parameter_count must be a positive integer.")
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory <= 0:
        raise MatrixError(f"{path}: peak_memory_bytes must be a positive integer.")
    if _number(diagnostics["train_step_time_ms"], path=f"{path}:train_step_time_ms") <= 0.0:
        raise MatrixError(f"{path}: train_step_time_ms must be positive.")
    return evidence


def collect_results(*, manifest_path: Path, queue_dir: Path) -> dict[str, object]:
    """Collect only complete queue jobs whose exact command matches the manifest."""
    manifest = validate_manifest(_load_json(manifest_path))
    manifest_sha256 = cast(str, manifest["manifest_sha256"])
    repro_root = queue_dir / "repro"
    if not repro_root.is_dir():
        raise MatrixError(f"Queue repro directory is missing: {repro_root}")
    run_metadata_by_name: dict[str, list[tuple[Path, Mapping[str, object]]]] = {}
    for run_json_path in sorted(repro_root.glob("*/run.json")):
        metadata = _mapping(_load_json(run_json_path), path=str(run_json_path))
        name = metadata.get("name")
        if isinstance(name, str):
            run_metadata_by_name.setdefault(name, []).append((run_json_path, metadata))
    rows: list[dict[str, object]] = []
    for raw_entry in cast("Sequence[object]", manifest["entries"]):
        entry = _mapping(raw_entry, path="manifest entry")
        queue_name = cast(str, entry["queue_name"])
        candidates = run_metadata_by_name.get(queue_name, [])
        exact = [
            candidate
            for candidate in candidates
            if candidate[1].get("command") == entry["command"]
            and candidate[1].get("issue") == str(ISSUE)
        ]
        if len(exact) != 1:
            raise MatrixError(
                f"{entry['entry_id']}: expected exactly one matching Issue #790 "
                f"queue repro, found {len(exact)}."
            )
        run_json_path, metadata = exact[0]
        repro_dir = run_json_path.parent
        run_id = _string(metadata.get("run_id"), path=f"{run_json_path}:run_id")
        if run_id != repro_dir.name:
            raise MatrixError(f"{run_json_path}: run_id does not match repro directory.")
        done_marker = queue_dir / "done" / f"{run_id}.job"
        conflicting = [
            queue_dir / state / f"{run_id}.job"
            for state in ("failed", "running", "jobs")
            if (queue_dir / state / f"{run_id}.job").exists()
        ]
        if not done_marker.is_file() or conflicting:
            raise MatrixError(f"{entry['entry_id']}: queue job is not uniquely complete.")
        # The queue-owned Lightning contract stores reusable test artifacts
        # below the per-run ``predictions`` directory.  Keep the matrix
        # evidence sidecar at repro-root because it describes queue-level
        # diagnostics rather than a test prediction file.
        prediction_dir = repro_dir / "predictions"
        metrics_path = prediction_dir / "metrics.json"
        evidence_path = repro_dir / "court_matrix_evidence.json"
        prediction_path = prediction_dir / "pred_test.npz"
        if not prediction_path.is_file() or prediction_path.stat().st_size <= 0:
            raise MatrixError(f"{entry['entry_id']}: pred_test.npz is missing or empty.")
        metrics = _validate_metrics(
            _load_json(metrics_path),
            condition=cast(str, entry["condition"]),
            path=metrics_path,
        )
        evidence = _validate_run_evidence(
            _load_json(evidence_path),
            entry=entry,
            manifest_sha256=manifest_sha256,
            path=evidence_path,
        )
        rows.append(
            {
                "entry_id": entry["entry_id"],
                "queue_run_id": run_id,
                "condition": entry["condition"],
                "roles": entry["roles"],
                "depth": entry["depth"],
                "input_size": entry["input_size"],
                "dpt_size": entry["dpt_size"],
                "dpt_channels": entry["dpt_channels"],
                "metrics": metrics,
                "loss_terms": evidence["loss_terms"],
                "diagnostics": evidence["diagnostics"],
                "repro_dir": str(repro_dir),
                "predictions": str(prediction_path),
            }
        )
    if len(rows) != len(cast("Sequence[object]", manifest["entries"])):
        raise MatrixError("Result collection did not cover every manifest entry.")
    body: dict[str, object] = {
        "schema": RESULTS_SCHEMA,
        "phase": PHASE,
        "issue": ISSUE,
        "manifest_sha256": manifest_sha256,
        "protocol": manifest["protocol"],
        "complete": True,
        "rows": rows,
    }
    return {**body, "results_sha256": _digest(body)}


def validate_results(value: object) -> Mapping[str, object]:
    """Validate the collected result identity before producing a summary."""
    results = _mapping(value, path="results")
    expected_keys = {
        "schema",
        "phase",
        "issue",
        "manifest_sha256",
        "protocol",
        "complete",
        "rows",
        "results_sha256",
    }
    if set(results) != expected_keys:
        raise MatrixError("Results keys are incomplete or foreign.")
    if (
        results["schema"] != RESULTS_SCHEMA
        or results["phase"] != PHASE
        or results["issue"] != ISSUE
        or results["complete"] is not True
    ):
        raise MatrixError("Old-phase, foreign, or incomplete results rejected.")
    body = {key: value_item for key, value_item in results.items() if key != "results_sha256"}
    if results["results_sha256"] != _digest(body):
        raise MatrixError("Results SHA-256 does not match canonical contents.")
    rows = _sequence(results["rows"], path="results.rows")
    if len(rows) != len(CONDITIONS) + len(DEPTHS) * len(INPUT_SIZES) * len(DPT_CHANNELS) - 1:
        raise MatrixError("Incomplete result row count rejected.")
    entry_ids = {
        _string(_mapping(row, path="result row").get("entry_id"), path="row.entry_id")
        for row in rows
    }
    if len(entry_ids) != len(rows):
        raise MatrixError("Duplicate result entry IDs rejected.")
    return results


def summarize_results(
    results: Mapping[str, object],
    *,
    decision: Mapping[str, object] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Flatten a complete collection without making an automatic adoption claim."""
    validated = validate_results(results)
    flattened: list[dict[str, object]] = []
    for raw_row in cast("Sequence[object]", validated["rows"]):
        row = _mapping(raw_row, path="result row")
        diagnostics = _mapping(row["diagnostics"], path="row.diagnostics")
        values: dict[str, object] = {
            "entry_id": row["entry_id"],
            "queue_run_id": row["queue_run_id"],
            "condition": row["condition"],
            "roles": ",".join(cast("Sequence[str]", row["roles"])),
            "depth": row["depth"],
            "input_size": row["input_size"],
            "dpt_size": row["dpt_size"],
            "dpt_channels": row["dpt_channels"],
            "parameter_count": diagnostics["parameter_count"],
            "train_step_time_ms": diagnostics["train_step_time_ms"],
            "peak_memory_bytes": diagnostics["peak_memory_bytes"],
            "repro_dir": row["repro_dir"],
            "predictions": row["predictions"],
        }
        for prefix, source in (
            ("metric", _mapping(row["metrics"], path="row.metrics")),
            ("loss", _mapping(row["loss_terms"], path="row.loss_terms")),
        ):
            for name, raw_value in source.items():
                values[f"{prefix}_{name}"] = raw_value
        gradients = _mapping(
            diagnostics["gradient_finite"], path="row.diagnostics.gradient_finite"
        )
        for name, raw_value in gradients.items():
            values[f"gradient_finite_{name}"] = raw_value
        flattened.append(values)
    reviewed_decision: Mapping[str, object] | None = None
    if decision is not None:
        allowed = {"decision", "rationale", "selected_entry_id"}
        if set(decision) != allowed:
            raise MatrixError("Decision file must contain exactly decision/rationale/selected_entry_id.")
        if decision["decision"] not in {"adopted", "rejected", "inconclusive"}:
            raise MatrixError("Decision must be adopted, rejected, or inconclusive.")
        _string(decision["rationale"], path="decision.rationale")
        selected = decision["selected_entry_id"]
        if selected is not None:
            selected_id = _string(selected, path="decision.selected_entry_id")
            if selected_id not in {cast(str, row["entry_id"]) for row in flattened}:
                raise MatrixError("Decision selected_entry_id is not in the results.")
        reviewed_decision = decision
    body: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "phase": PHASE,
        "issue": ISSUE,
        "manifest_sha256": validated["manifest_sha256"],
        "results_sha256": validated["results_sha256"],
        "complete": True,
        "row_count": len(flattened),
        "pure_weights": _mapping(
            _mapping(validated["protocol"], path="results.protocol")["pure_weights"],
            path="results.protocol.pure_weights",
        ),
        "weighted_weights": _mapping(
            _mapping(validated["protocol"], path="results.protocol")["weighted_weights"],
            path="results.protocol.weighted_weights",
        ),
        "weighted_strategy": _mapping(
            validated["protocol"], path="results.protocol"
        )["weighted_strategy"],
        "adoption_decision": reviewed_decision,
        "decision_status": (
            "reviewed"
            if reviewed_decision is not None
            else "not_recorded_no_automatic_adoption"
        ),
    }
    return {**body, "summary_sha256": _digest(body)}, flattened


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _add_matrix_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-prefix", default="court_detection/matrix")
    parser.add_argument(
        "--manifest-path",
        default=str(
            REPOSITORY_ROOT / "outputs/court_detection/matrix/manifest.json"
        ),
    )
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--weighted-kp", type=float, default=1.0)
    parser.add_argument("--weighted-line", type=float, default=0.5)
    parser.add_argument("--weighted-seg", type=float, default=0.5)
    parser.add_argument("--weighted-pose-translation", type=float, default=1.0)
    parser.add_argument("--weighted-pose-rotation", type=float, default=1.0)
    parser.add_argument("--weighted-pose-focal", type=float, default=0.5)
    parser.add_argument("--weighted-consistency", type=float, default=0.25)


def _options(namespace: argparse.Namespace) -> MatrixOptions:
    return MatrixOptions(
        python=cast(str, namespace.python),
        max_epochs=cast(int, namespace.max_epochs),
        batch_size=cast(int, namespace.batch_size),
        seed=cast(int, namespace.seed),
        run_prefix=cast(str, namespace.run_prefix),
        manifest_path=cast(str, namespace.manifest_path),
        consistency_weight=cast(float, namespace.consistency_weight),
        weighted_kp=cast(float, namespace.weighted_kp),
        weighted_line=cast(float, namespace.weighted_line),
        weighted_seg=cast(float, namespace.weighted_seg),
        weighted_pose_translation=cast(float, namespace.weighted_pose_translation),
        weighted_pose_rotation=cast(float, namespace.weighted_pose_rotation),
        weighted_pose_focal=cast(float, namespace.weighted_pose_focal),
        weighted_consistency=cast(float, namespace.weighted_consistency),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-configs")
    _add_matrix_options(validate_parser)
    subparsers.add_parser("check-removals")
    manifest_parser = subparsers.add_parser("generate-manifest")
    _add_matrix_options(manifest_parser)
    manifest_parser.add_argument("--output", type=Path, required=True)
    emit_parser = subparsers.add_parser("emit-jobs")
    _add_matrix_options(emit_parser)
    collect_parser = subparsers.add_parser("collect-results")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--queue-dir", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    summary_parser = subparsers.add_parser("summarize-results")
    summary_parser.add_argument("--results", type=Path, required=True)
    summary_parser.add_argument("--output-dir", type=Path, required=True)
    summary_parser.add_argument("--decision-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic matrix tooling command."""
    namespace = _parser().parse_args(argv)
    command = cast(str, namespace.command)
    if command == "validate-configs":
        runs = build_runs(_options(namespace))
        validate_configs(_options(namespace))
        print(f"validated {len(runs)} unique jobs (6 conditions; 16 scaling combinations)")
        return 0
    if command == "check-removals":
        check_removals()
        print("retired Court query/Colab train paths are absent; production defaults unchanged")
        return 0
    if command == "generate-manifest":
        options = _options(namespace)
        output = cast(Path, namespace.output)
        if output.resolve() != Path(options.manifest_path).resolve():
            raise MatrixError(
                "--output and --manifest-path must identify the same frozen manifest."
            )
        manifest = build_manifest(options)
        _write_json(output, manifest, refuse_changed_existing=True)
        print(f"wrote {output} ({manifest['manifest_sha256']})")
        return 0
    if command == "emit-jobs":
        for run in build_runs(_options(namespace)):
            print(f"{run.queue_name}\t{run.command}")
        return 0
    if command == "collect-results":
        output = cast(Path, namespace.output)
        result = collect_results(
            manifest_path=cast(Path, namespace.manifest),
            queue_dir=cast(Path, namespace.queue_dir),
        )
        _write_json(output, result)
        print(f"wrote {output} ({len(cast(Sequence[object], result['rows']))} complete runs)")
        return 0
    if command == "summarize-results":
        results = validate_results(_load_json(cast(Path, namespace.results)))
        decision_path = cast("Path | None", namespace.decision_file)
        decision = (
            _mapping(_load_json(decision_path), path=str(decision_path))
            if decision_path is not None
            else None
        )
        summary, rows = summarize_results(results, decision=decision)
        output_dir = cast(Path, namespace.output_dir)
        _write_json(output_dir / "summary.json", summary)
        _write_csv(output_dir / "all_runs.csv", rows)
        print(f"wrote {output_dir / 'summary.json'} and {output_dir / 'all_runs.csv'}")
        return 0
    raise MatrixError(f"Unsupported command: {command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
