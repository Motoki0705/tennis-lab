"""Fail-closed collection and staged summary for Issue #790 evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    require_config_mapping,
    require_config_value,
)
from src.tasks.court_detection.configuration import (
    ConfigMapping,
    validate_paths_boundary,
)
from src.tasks.court_detection.experiments.query_consistency import (
    MANIFEST_SCHEMA,
    PHASE_ORDER,
    RESULT_METRIC_NAMES,
    SEEDS,
    TRAIN_DIAGNOSTIC_NAMES,
    validate_query_consistency_manifest,
)
from src.tasks.court_detection.models.query_encoder.profiling import (
    validate_profile_record,
)
from src.utils.configuration import PathRole
from src.utils.io import save_json_atomic

JsonValue: TypeAlias = Any
SummaryPhase: TypeAlias = Literal[
    "encoder_scaling",
    "decoder_scaling",
    "consistency_ablation",
]

RESULTS_SCHEMA = "court_query_consistency_results_v1"
SUMMARY_SCHEMA = "court_query_consistency_summary_v1"
CAPACITY_NAMES = (
    "decoder_params",
    "trainable_params",
    "total_params",
    "decoder_macs",
    "decoder_latency_ms",
    "end_to_end_latency_ms",
    "inference_peak_memory_bytes",
)
_STATIC_CAPACITY_NAMES = (
    "decoder_params",
    "trainable_params",
    "total_params",
    "decoder_macs",
)
_PHASE_RESULT_COUNTS: Mapping[SummaryPhase, int] = {
    "encoder_scaling": 12,
    "decoder_scaling": 39,
    "consistency_ablation": 51,
}
_PHASE_GROUP_COUNTS: Mapping[str, int] = {
    "encoder_scaling": 4,
    "decoder_scaling": 9,
    "consistency_ablation": 4,
}


def _exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
    if set(mapping) != keys:
        raise ValueError(f"{path} requires exactly {sorted(keys)}.")


def _string(mapping: ConfigMapping, key: str, *, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty trimmed string.")
    return value


def _boolean(mapping: ConfigMapping, key: str, *, path: str) -> bool:
    return cast(bool, require_config_value(mapping, key, bool, path=path))


@dataclass(frozen=True, slots=True)
class QueryConsistencySummaryConfig:
    """Resolved staged evidence collection and summary boundary."""

    phase: SummaryPhase
    manifest_path: Path
    evidence_root: Path
    results_path: Path
    output_dir: Path
    require_gpu_profiles: bool

    @classmethod
    def from_config(cls, value: object) -> QueryConsistencySummaryConfig:
        if not isinstance(value, DictConfig):
            raise TypeError("Query consistency summary requires a composed DictConfig.")
        root, resolver = validate_paths_boundary(
            value, expected_sections={"summary"}
        )
        summary = require_config_mapping(root, "summary", path="configuration")
        _exact(
            summary,
            {
                "phase",
                "manifest_path",
                "evidence_root",
                "results_path",
                "output_dir",
                "require_gpu_profiles",
            },
            path="summary",
        )
        raw_phase = _string(summary, "phase", path="summary")
        if raw_phase not in PHASE_ORDER:
            raise ValueError("summary.phase is outside the ordered #790 phases.")
        result = cls(
            phase=cast(SummaryPhase, raw_phase),
            manifest_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "manifest_path", path="summary"),
            ),
            evidence_root=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "evidence_root", path="summary"),
            ),
            results_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "results_path", path="summary"),
            ),
            output_dir=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "output_dir", path="summary"),
            ),
            require_gpu_profiles=_boolean(
                summary, "require_gpu_profiles", path="summary"
            ),
        )
        if not result.require_gpu_profiles:
            raise ValueError("Formal #790 summaries require GPU profile evidence.")
        return result


def validate_query_consistency_summary_boundary(config: DictConfig) -> None:
    """Validate the summary Hydra boundary before evidence reads/writes."""
    QueryConsistencySummaryConfig.from_config(config)


def load_json_mapping(path: Path) -> Mapping[str, object]:
    """Load one required JSON object without accepting arrays/scalars."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required #790 evidence is missing: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Required #790 JSON must contain an object: {path}")
    return cast(Mapping[str, object], value)


def collect_query_consistency_results(
    manifest: Mapping[str, object],
    *,
    evidence_root: Path,
    phase: SummaryPhase,
    require_gpu_profiles: bool,
) -> dict[str, JsonValue]:
    """Collect test metrics, TensorBoard diagnostics, and capacity profiles.

    The collector has one explicit source for every result field. It never fills
    missing diagnostics with defaults and never aliases the legacy ``line_iou``.
    """
    validate_query_consistency_manifest(manifest, require_resolved=False)
    _validate_manifest_phase_state(manifest, phase=phase)
    evidence_root = evidence_root.resolve(strict=False)
    raw_runs = cast(Sequence[object], manifest["runs"])
    ready_runs = [
        _mapping(run, name="manifest.run")
        for run in raw_runs
        if isinstance(run, Mapping) and bool(run["queue_ready"])
    ]
    expected_count = _PHASE_RESULT_COUNTS[phase]
    if len(ready_runs) != expected_count:
        raise ValueError(
            f"Summary phase {phase!r} requires exactly {expected_count} queue-ready runs."
        )
    records: list[dict[str, JsonValue]] = []
    for run in ready_runs:
        paths = _mapping(run["evidence_paths"], name="run.evidence_paths")
        _exact(
            paths,
            {"test_metrics", "tensorboard_log_dir", "capacity_profile"},
            path="run.evidence_paths",
        )
        metrics_path = _resolve_evidence_path(
            evidence_root, paths["test_metrics"], name="test_metrics"
        )
        log_dir = _resolve_evidence_path(
            evidence_root, paths["tensorboard_log_dir"], name="tensorboard_log_dir"
        )
        profile_path = _resolve_evidence_path(
            evidence_root, paths["capacity_profile"], name="capacity_profile"
        )
        diagnostics, loss_curve = collect_tensorboard_training_evidence(log_dir)
        records.append(
            build_query_consistency_result_record(
                run,
                test_metrics=load_json_mapping(metrics_path),
                diagnostics=diagnostics,
                loss_curve=loss_curve,
                profile=load_json_mapping(profile_path),
                require_gpu_profile=require_gpu_profiles,
            )
        )
    results: dict[str, JsonValue] = {
        "schema": RESULTS_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "phase": phase,
        "runs": records,
    }
    validate_query_consistency_results(results, manifest=manifest, phase=phase)
    return results


def _resolve_evidence_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Evidence path {name!r} must be a non-empty relative path.")
    resolved = root.joinpath(value).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Evidence path {name!r} escapes the configured evidence root.")
    return resolved


def collect_tensorboard_training_evidence(
    log_dir: Path,
) -> tuple[dict[str, float], list[dict[str, JsonValue]]]:
    """Reduce exact Lightning scalar tags under a deterministic contract.

    Gradient status is the minimum over logged steps, step time is the arithmetic
    mean, and CUDA memory is the maximum. The loss curve retains each strictly
    increasing ``train/loss`` step. Missing/duplicate/nonfinite evidence raises.
    """
    if not log_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory is missing: {log_dir}")
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(cast(Sequence[str], accumulator.Tags().get("scalars", ())))
    required_tags = {f"train/{name}" for name in TRAIN_DIAGNOSTIC_NAMES}
    required_tags.add("train/loss")
    missing = required_tags - available
    if missing:
        raise ValueError(
            "TensorBoard evidence lacks required exact scalar tags: "
            f"{sorted(missing)}."
        )
    diagnostics: dict[str, float] = {}
    for name in TRAIN_DIAGNOSTIC_NAMES:
        values = _scalar_values(accumulator, f"train/{name}")
        if name.endswith("_gradient_finite"):
            if any(value not in {0.0, 1.0} for _, value in values):
                raise ValueError(f"Gradient diagnostic {name!r} must be exactly 0/1.")
            diagnostics[name] = min(value for _, value in values)
        elif name == "train_step_time_ms":
            diagnostics[name] = statistics.fmean(value for _, value in values)
        else:
            diagnostics[name] = max(value for _, value in values)
    loss_values = _scalar_values(accumulator, "train/loss")
    steps = [step for step, _ in loss_values]
    if steps != sorted(set(steps)):
        raise ValueError("TensorBoard train/loss steps must be unique and increasing.")
    loss_curve = [
        {"step": step, "loss": value} for step, value in loss_values
    ]
    _validate_diagnostics(diagnostics)
    _validate_loss_curve(loss_curve)
    return diagnostics, loss_curve


def _scalar_values(accumulator: Any, tag: str) -> list[tuple[int, float]]:
    events = cast(Sequence[Any], accumulator.Scalars(tag))
    if not events:
        raise ValueError(f"TensorBoard scalar tag {tag!r} has no events.")
    values: list[tuple[int, float]] = []
    for event in events:
        step = event.step
        value = event.value
        if type(step) is not int or step < 0:
            raise ValueError(f"TensorBoard scalar tag {tag!r} has an invalid step.")
        values.append((step, _finite_number(value, name=tag)))
    return values


def build_query_consistency_result_record(
    run: Mapping[str, object],
    *,
    test_metrics: Mapping[str, object],
    diagnostics: Mapping[str, object],
    loss_curve: Sequence[Mapping[str, object]],
    profile: Mapping[str, object],
    require_gpu_profile: bool,
) -> dict[str, JsonValue]:
    """Build one result from explicit fixtures without omissions or aliases."""
    if "line_iou" in test_metrics:
        raise ValueError("Legacy line_iou is not an accepted #790 metric alias.")
    missing = set(RESULT_METRIC_NAMES) - set(test_metrics)
    if missing:
        raise ValueError(f"Test metrics lack required #790 keys: {sorted(missing)}.")
    metrics = {
        name: _finite_number(test_metrics[name], name=f"metrics.{name}")
        for name in RESULT_METRIC_NAMES
    }
    _validate_metrics(metrics)
    diagnostic_values = {
        name: _finite_number(diagnostics[name], name=f"diagnostics.{name}")
        for name in TRAIN_DIAGNOSTIC_NAMES
        if name in diagnostics
    }
    if set(diagnostic_values) != set(TRAIN_DIAGNOSTIC_NAMES):
        raise ValueError("Diagnostics must contain every exact #790 diagnostic key.")
    _validate_diagnostics(diagnostic_values)
    normalized_curve = [dict(point) for point in loss_curve]
    _validate_loss_curve(normalized_curve)
    validate_profile_record(profile, require_gpu_evidence=require_gpu_profile)
    architecture = _mapping(run["architecture"], name="run.architecture")
    candidate = _mapping(profile["candidate"], name="profile.candidate")
    if (
        candidate["family"] != architecture["decoder_family"]
        or candidate["size"] != architecture["decoder_size"]
    ):
        raise ValueError("Capacity profile candidate disagrees with the manifest run.")
    parameters = _mapping(profile["parameters"], name="profile.parameters")
    macs = _mapping(profile["decoder_macs"], name="profile.decoder_macs")
    latency = _mapping(profile["latency_ms"], name="profile.latency_ms")
    peak = _mapping(profile["peak_memory"], name="profile.peak_memory")
    capacity = {
        "decoder_params": parameters["decoder"],
        "trainable_params": parameters["trainable"],
        "total_params": parameters["total"],
        "decoder_macs": macs["count"],
        "decoder_latency_ms": latency["decoder_mean"],
        "end_to_end_latency_ms": latency["end_to_end_mean"],
        "inference_peak_memory_bytes": peak["bytes"],
    }
    _validate_capacity(capacity)
    return {
        "run_id": run["run_id"],
        "phase": run["phase"],
        "seed": run["seed"],
        "condition": run["condition"],
        "metrics": metrics,
        "diagnostics": diagnostic_values,
        "capacity": capacity,
        "loss_curve": normalized_curve,
    }


def validate_query_consistency_results(
    results: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    phase: SummaryPhase,
) -> None:
    """Validate exact ready-run identity, order, schema, and numeric domains."""
    if set(results) != {"schema", "manifest_sha256", "phase", "runs"}:
        raise ValueError("Query consistency result fields changed.")
    if results["schema"] != RESULTS_SCHEMA or results["phase"] != phase:
        raise ValueError("Query consistency result schema/phase changed.")
    if results["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Query consistency results do not bind the manifest.")
    raw_results = results["runs"]
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise ValueError("Query consistency result runs must be a sequence.")
    records = tuple(_mapping(run, name="result.run") for run in raw_results)
    manifest_runs = [
        _mapping(run, name="manifest.run")
        for run in cast(Sequence[object], manifest["runs"])
        if isinstance(run, Mapping) and bool(run["queue_ready"])
    ]
    if len(records) != _PHASE_RESULT_COUNTS[phase] or len(records) != len(manifest_runs):
        raise ValueError("Query consistency results are incomplete for the selected phase.")
    if [record["run_id"] for record in records] != [
        run["run_id"] for run in manifest_runs
    ]:
        raise ValueError("Query consistency results must retain ready-run order.")
    seen: set[str] = set()
    for record, planned in zip(records, manifest_runs, strict=True):
        if set(record) != {
            "run_id",
            "phase",
            "seed",
            "condition",
            "metrics",
            "diagnostics",
            "capacity",
            "loss_curve",
        }:
            raise ValueError("Query consistency result run fields changed.")
        run_id = _nonempty_string(record["run_id"], name="result.run_id")
        if run_id in seen:
            raise ValueError("Query consistency result run IDs must be unique.")
        seen.add(run_id)
        if any(record[name] != planned[name] for name in ("phase", "seed", "condition")):
            raise ValueError("Query consistency result identity disagrees with manifest.")
        metrics = _mapping(record["metrics"], name="result.metrics")
        diagnostics = _mapping(record["diagnostics"], name="result.diagnostics")
        capacity = _mapping(record["capacity"], name="result.capacity")
        if tuple(metrics) != RESULT_METRIC_NAMES:
            raise ValueError("Result metrics must use the exact canonical #790 key order.")
        if tuple(diagnostics) != TRAIN_DIAGNOSTIC_NAMES:
            raise ValueError("Result diagnostics must use the exact #790 key order.")
        if tuple(capacity) != CAPACITY_NAMES:
            raise ValueError("Result capacity fields changed.")
        _validate_metrics(metrics)
        _validate_diagnostics(diagnostics)
        _validate_capacity(capacity)
        curve = record["loss_curve"]
        if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)):
            raise ValueError("Result loss_curve must be a sequence.")
        _validate_loss_curve(cast(Sequence[Mapping[str, object]], curve))


def _validate_metrics(metrics: Mapping[str, object]) -> None:
    if set(metrics) != set(RESULT_METRIC_NAMES):
        raise ValueError("Metrics must contain exactly the canonical #790 keys.")
    values = {
        name: _finite_number(metrics[name], name=f"metrics.{name}")
        for name in RESULT_METRIC_NAMES
    }
    if any(value < 0.0 for value in values.values()):
        raise ValueError("Query consistency metrics must be non-negative.")
    for name in ("line_dice", "seg_miou", "invalid_depth_rate"):
        if values[name] > 1.0:
            raise ValueError(f"Metric {name!r} must be in [0,1].")
    if values["visible_point_count"] <= 0.0:
        raise ValueError("Every successful #790 run requires visible point evidence.")


def _validate_diagnostics(diagnostics: Mapping[str, object]) -> None:
    if set(diagnostics) != set(TRAIN_DIAGNOSTIC_NAMES):
        raise ValueError("Diagnostics must contain exactly the canonical #790 keys.")
    values = {
        name: _finite_number(diagnostics[name], name=f"diagnostics.{name}")
        for name in TRAIN_DIAGNOSTIC_NAMES
    }
    for name in TRAIN_DIAGNOSTIC_NAMES[:4]:
        if values[name] not in {0.0, 1.0}:
            raise ValueError(f"Gradient diagnostic {name!r} must be exactly 0/1.")
    if values["train_step_time_ms"] <= 0.0 or values["cuda_peak_memory_bytes"] <= 0.0:
        raise ValueError("Training time and CUDA peak-memory diagnostics must be positive.")


def _validate_capacity(capacity: Mapping[str, object]) -> None:
    if set(capacity) != set(CAPACITY_NAMES):
        raise ValueError("Capacity record fields changed.")
    values = {
        name: _finite_number(capacity[name], name=f"capacity.{name}")
        for name in CAPACITY_NAMES
    }
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("Every capacity value must be finite and positive.")


def _validate_loss_curve(curve: Sequence[Mapping[str, object]]) -> None:
    if not curve:
        raise ValueError("Every #790 run requires a non-empty train loss curve.")
    steps: list[int] = []
    for point in curve:
        if not isinstance(point, Mapping) or set(point) != {"step", "loss"}:
            raise ValueError("Loss curve points require exactly step/loss.")
        step = point["step"]
        if type(step) is not int or step < 0:
            raise ValueError("Loss curve steps must be non-negative integers.")
        loss = _finite_number(point["loss"], name="loss_curve.loss")
        if loss < 0.0:
            raise ValueError("Loss curve values must be non-negative.")
        steps.append(step)
    if steps != sorted(set(steps)):
        raise ValueError("Loss curve steps must be unique and increasing.")


def summarize_query_consistency(
    manifest: Mapping[str, object],
    results: Mapping[str, object],
    *,
    phase: SummaryPhase,
) -> dict[str, JsonValue]:
    """Aggregate one explicitly requested completed phase and apply frozen rules."""
    validate_query_consistency_manifest(manifest, require_resolved=False)
    _validate_manifest_phase_state(manifest, phase=phase)
    validate_query_consistency_results(results, manifest=manifest, phase=phase)
    records = cast(Sequence[Mapping[str, object]], results["runs"])
    aggregates = _aggregate(records, phase=phase)
    encoder = _select_encoder(aggregates["encoder_scaling"])
    selected = _mapping(manifest["selected"], name="manifest.selected")
    summary: dict[str, JsonValue] = {
        "schema": SUMMARY_SCHEMA,
        "source_manifest_schema": MANIFEST_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "phase": phase,
        "run_count": len(records),
        "seed_count": len(SEEDS),
        "aggregation": {
            "mean": "arithmetic_mean_over_seeds",
            "variance": "population_variance_over_seeds",
            "gradient_diagnostic": "per_run_min_over_logged_steps",
            "train_step_time": "per_run_mean_over_logged_steps",
            "cuda_peak_memory": "per_run_max_over_logged_steps",
        },
        "groups": aggregates,
        "encoder_selection": encoder,
        "decoder_selection": None,
        "adoption_decision": None,
        "gradient_direction": None,
        "production_default": {
            "changed": False,
            "status": "ablation_only_until_complete_formal_adoption",
        },
    }
    if phase == "encoder_scaling":
        return summary
    if selected["encoder_depth"] != encoder["selected_depth"]:
        raise ValueError(
            "Manifest selected encoder disagrees with the frozen phase-one rule."
        )
    decoder = _select_decoder(aggregates["decoder_scaling"])
    summary["decoder_selection"] = decoder
    if phase == "decoder_scaling":
        return summary
    if (
        selected["decoder_family"] != decoder["selected_family"]
        or selected["decoder_size"] != decoder["selected_size"]
    ):
        raise ValueError(
            "Manifest selected decoder disagrees with the frozen phase-two rule."
        )
    formal = aggregates["consistency_ablation"]
    summary["adoption_decision"] = _adoption_decision(formal)
    summary["gradient_direction"] = _gradient_direction(formal)
    return summary


def _validate_manifest_phase_state(
    manifest: Mapping[str, object], *, phase: SummaryPhase
) -> None:
    selected = _mapping(manifest["selected"], name="manifest.selected")
    depth = selected["encoder_depth"]
    decoder_resolved = (
        selected["decoder_family"] is not None and selected["decoder_size"] is not None
    )
    if phase == "encoder_scaling" and (depth is not None or decoder_resolved):
        raise ValueError("Encoder summary requires an unresolved selection manifest.")
    if phase == "decoder_scaling" and (depth is None or decoder_resolved):
        raise ValueError("Decoder summary requires only the encoder selection.")
    if phase == "consistency_ablation" and (depth is None or not decoder_resolved):
        raise ValueError("Formal summary requires the fully selected architecture.")


def _aggregate(
    records: Sequence[Mapping[str, object]],
    *,
    phase: SummaryPhase,
) -> dict[str, list[dict[str, JsonValue]]]:
    included_phases = PHASE_ORDER[: PHASE_ORDER.index(phase) + 1]
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    order: list[tuple[str, str]] = []
    for record in records:
        phase_name = cast(str, record["phase"])
        candidate = _candidate_identity(record)
        key = (phase_name, candidate)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(record)
    aggregates: dict[str, list[dict[str, JsonValue]]] = {
        name: [] for name in included_phases
    }
    for phase_name, candidate in order:
        members = groups[(phase_name, candidate)]
        seeds = [cast(int, member["seed"]) for member in members]
        if seeds != list(SEEDS):
            raise ValueError(f"Candidate {phase_name}/{candidate} lacks exact seed order.")
        metrics_mean, metrics_variance = _aggregate_mapping(members, "metrics")
        diagnostics_mean, diagnostics_variance = _aggregate_mapping(
            members, "diagnostics"
        )
        capacity_mean, capacity_variance = _aggregate_mapping(members, "capacity")
        for name in _STATIC_CAPACITY_NAMES:
            values = [
                _mapping(member["capacity"], name="capacity")[name]
                for member in members
            ]
            if len(set(values)) != 1:
                raise ValueError(
                    f"Candidate {phase_name}/{candidate} changed static {name} by seed."
                )
        aggregates[phase_name].append(
            {
                "candidate": candidate,
                "seeds": seeds,
                "metrics_mean": metrics_mean,
                "metrics_variance": metrics_variance,
                "diagnostics_mean": diagnostics_mean,
                "diagnostics_variance": diagnostics_variance,
                "capacity_mean": capacity_mean,
                "capacity_variance": capacity_variance,
            }
        )
    for phase_name in included_phases:
        if len(aggregates[phase_name]) != _PHASE_GROUP_COUNTS[phase_name]:
            raise ValueError(f"Aggregate phase {phase_name!r} is incomplete.")
    return aggregates


def _aggregate_mapping(
    members: Sequence[Mapping[str, object]], field: str
) -> tuple[dict[str, float], dict[str, float]]:
    first = _mapping(members[0][field], name=field)
    means: dict[str, float] = {}
    variances: dict[str, float] = {}
    for name in first:
        values = [
            _finite_number(
                _mapping(member[field], name=field)[name], name=f"{field}.{name}"
            )
            for member in members
        ]
        means[name] = statistics.fmean(values)
        variances[name] = statistics.pvariance(values)
    return means, variances


def _candidate_identity(record: Mapping[str, object]) -> str:
    run_id = cast(str, record["run_id"])
    if record["phase"] == "encoder_scaling":
        return run_id.removeprefix("encoder-").rsplit("-seed-", maxsplit=1)[0]
    if record["phase"] == "decoder_scaling":
        return run_id.removeprefix("decoder-").rsplit("-seed-", maxsplit=1)[0]
    return cast(str, record["condition"])


def _select_encoder(rows: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    by_depth = {
        int(cast(str, row["candidate"]).removeprefix("depth-")): row for row in rows
    }
    reference = by_depth[8]
    reference_metrics = cast(Mapping[str, object], reference["metrics_mean"])
    required = (
        "kp_mean_distance_px",
        "pose_reprojection_mean_distance_px",
        "pose_translation_l2_m",
        "pose_rotation_geodesic_deg",
        "pose_focal_relative_error",
    )
    candidates: list[dict[str, JsonValue]] = []
    sufficient: list[int] = []
    for depth in sorted(by_depth):
        row = by_depth[depth]
        metrics = cast(Mapping[str, object], row["metrics_mean"])
        within = all(
            float(cast(float, metrics[name]))
            <= float(cast(float, reference_metrics[name])) * 1.05
            for name in required
        )
        if within:
            sufficient.append(depth)
        candidates.append({**row, "within_reference_tolerance": within})
    selected = min(sufficient) if sufficient else 8
    return {
        "selected_depth": selected,
        "reference_depth": 8,
        "decoder_phase_compatible": selected >= 2,
        "decoder_phase_blocker": (
            None
            if selected >= 2
            else (
                "Frozen selection chose depth 1, but the required DPT matrix needs "
                "at least two unique taps; manifest regeneration fails closed."
            )
        ),
        "candidates": [
            {
                **row,
                "adopted": row["candidate"] == f"depth-{selected}",
                "non_adoption_reason": (
                    None
                    if row["candidate"] == f"depth-{selected}"
                    else (
                        "Failed one or more frozen 5% depth-8 thresholds."
                        if not row["within_reference_tolerance"]
                        else "A smaller sufficient encoder was selected."
                    )
                ),
            }
            for row in candidates
        ],
    }


def _select_decoder(rows: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    by_name = {cast(str, row["candidate"]): row for row in rows}
    reference = by_name["dpt-base"]
    ref_metrics = cast(Mapping[str, object], reference["metrics_mean"])
    sufficient: list[dict[str, JsonValue]] = []
    decisions: list[dict[str, JsonValue]] = []
    for row in rows:
        metrics = cast(Mapping[str, object], row["metrics_mean"])
        within = all(
            float(cast(float, metrics[name]))
            <= float(cast(float, ref_metrics[name])) * 1.05
            for name in (
                "kp_mean_distance_px",
                "pose_reprojection_mean_distance_px",
                "pose_translation_l2_m",
                "pose_rotation_geodesic_deg",
                "pose_focal_relative_error",
            )
        ) and all(
            float(cast(float, metrics[name]))
            >= float(cast(float, ref_metrics[name])) - 0.01
            for name in ("line_dice", "seg_miou")
        )
        candidate = {**row, "within_reference_tolerance": within}
        decisions.append(candidate)
        if within:
            sufficient.append(candidate)
    selected = min(
        sufficient or [reference],
        key=lambda row: (
            _finite_number(
                cast(Mapping[str, object], row["capacity_mean"])["decoder_macs"],
                name="capacity.decoder_macs",
            ),
            _finite_number(
                cast(Mapping[str, object], row["capacity_mean"])["decoder_params"],
                name="capacity.decoder_params",
            ),
            cast(str, row["candidate"]),
        ),
    )
    selected_name = cast(str, selected["candidate"])
    marked = _mark_pareto(
        [
            {
                **row,
                "adopted": row["candidate"] == selected_name,
                "non_adoption_reason": (
                    None
                    if row["candidate"] == selected_name
                    else (
                        "Failed one or more frozen DPT-base accuracy thresholds."
                        if not row["within_reference_tolerance"]
                        else "A sufficient candidate with lower MACs/parameters was selected."
                    )
                ),
            }
            for row in decisions
        ]
    )
    family, size = selected_name.split("-", maxsplit=1)
    return {
        "selected_family": family,
        "selected_size": size,
        "reference": "dpt-base",
        "candidates": marked,
    }


def _mark_pareto(rows: Sequence[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    result: list[dict[str, JsonValue]] = []
    for row in rows:
        axes = _pareto_axes(row)
        dominated = any(
            _dominates(_pareto_axes(other), axes) for other in rows if other is not row
        )
        result.append({**row, "pareto_front": not dominated})
    return result


def _pareto_axes(row: Mapping[str, JsonValue]) -> tuple[float, ...]:
    metrics = cast(Mapping[str, object], row["metrics_mean"])
    diagnostics = cast(Mapping[str, object], row["diagnostics_mean"])
    capacity = cast(Mapping[str, object], row["capacity_mean"])
    return (
        float(cast(float, metrics["kp_mean_distance_px"])),
        float(cast(float, metrics["pose_reprojection_mean_distance_px"])),
        float(cast(float, capacity["decoder_macs"])),
        float(cast(float, capacity["decoder_params"])),
        float(cast(float, diagnostics["train_step_time_ms"])),
        float(cast(float, diagnostics["cuda_peak_memory_bytes"])),
    )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def _adoption_decision(rows: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    by_condition = {cast(str, row["candidate"]): row for row in rows}
    baseline = by_condition["direct-all"]
    candidate = by_condition["joint-both"]
    base_metrics = cast(Mapping[str, object], baseline["metrics_mean"])
    joint_metrics = cast(Mapping[str, object], candidate["metrics_mean"])
    base_diagnostics = cast(Mapping[str, object], baseline["diagnostics_mean"])
    joint_diagnostics = cast(Mapping[str, object], candidate["diagnostics_mean"])
    gt_names = ("kp_mean_distance_px", "pose_reprojection_mean_distance_px")
    gt_improvement = {
        name: float(cast(float, joint_metrics[name]))
        <= float(cast(float, base_metrics[name])) * 0.95
        for name in gt_names
    }
    checks = {
        "gt_improvement": any(gt_improvement.values()),
        "gt_and_direct_pose_degradation": all(
            float(cast(float, joint_metrics[name]))
            <= float(cast(float, base_metrics[name])) * 1.05
            for name in (
                *gt_names,
                "pose_translation_l2_m",
                "pose_rotation_geodesic_deg",
                "pose_focal_relative_error",
            )
        ),
        "line_seg_degradation": all(
            float(cast(float, joint_metrics[name]))
            >= float(cast(float, base_metrics[name])) - 0.01
            for name in ("line_dice", "seg_miou")
        ),
        "finite_gradients": all(
            float(
                cast(
                    float,
                    cast(Mapping[str, object], row["diagnostics_mean"])[name],
                )
            )
            == 1.0
            for row in rows
            for name in TRAIN_DIAGNOSTIC_NAMES[:4]
        ),
        "visible_points": float(cast(float, joint_metrics["visible_point_count"]))
        > 0.0,
        "train_step_overhead": float(
            cast(float, joint_diagnostics["train_step_time_ms"])
        )
        <= float(cast(float, base_diagnostics["train_step_time_ms"])) * 1.10,
        "cuda_memory_overhead": float(
            cast(float, joint_diagnostics["cuda_peak_memory_bytes"])
        )
        <= float(cast(float, base_diagnostics["cuda_peak_memory_bytes"])) * 1.10,
    }
    adopted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    consistency_only = (
        float(cast(float, joint_metrics["kp_pose_consistency_distance_px"]))
        < float(cast(float, base_metrics["kp_pose_consistency_distance_px"]))
        and not checks["gt_improvement"]
    )
    if consistency_only and "gt_improvement" not in failed:  # pragma: no cover
        failed.append("gt_improvement")
    return {
        "status": "adopted" if adopted else "not_adopted",
        "baseline": "direct-all",
        "candidate": "joint-both",
        "gt_improvement_by_metric": gt_improvement,
        "checks": checks,
        "consistency_only_improvement": consistency_only,
        "failed_requirements": failed,
        "rationale": (
            "All frozen Issue #790 adoption requirements passed."
            if adopted
            else "Frozen Issue #790 adoption requirements failed: " + ", ".join(failed)
        ),
    }


def _gradient_direction(rows: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    by_condition = {cast(str, row["candidate"]): row for row in rows}
    metrics = (
        "kp_mean_distance_px",
        "pose_reprojection_mean_distance_px",
        "kp_pose_consistency_distance_px",
    )
    best = {
        metric: min(
            by_condition,
            key=lambda condition: _finite_number(
                cast(
                    Mapping[str, object], by_condition[condition]["metrics_mean"]
                )[metric],
                name=f"metrics.{metric}",
            ),
        )
        for metric in metrics
    }
    return {
        "conditions": [
            "joint-both",
            "joint-stopgrad-pose",
            "joint-stopgrad-dense",
        ],
        "best_condition_by_lower_is_better_metric": best,
        "interpretation_contract": (
            "Stop-gradient rows diagnose update direction only; they cannot replace "
            "the joint-both versus direct-all adoption decision."
        ),
    }


def write_query_consistency_summary_artifacts(
    summary: Mapping[str, JsonValue],
    results: Mapping[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write complete JSON/CSV tables and phase-appropriate PNG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = [
        save_json_atomic(summary, output_dir / "summary.json")
    ]
    all_runs_path = output_dir / "all_runs.csv"
    scaling_path = output_dir / "scaling_table.csv"
    _write_all_runs(results, all_runs_path)
    _write_scaling(summary, scaling_path)
    artifacts.extend((all_runs_path, scaling_path))
    if summary["decoder_selection"] is not None:
        pareto_path = output_dir / "pareto_table.csv"
        _write_pareto(summary, pareto_path)
        artifacts.append(pareto_path)
    artifacts.extend(_write_plots(summary, output_dir=output_dir))
    return tuple(artifacts)


def _write_all_runs(results: Mapping[str, object], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for raw_record in cast(Sequence[Mapping[str, object]], results["runs"]):
        record = _mapping(raw_record, name="result.run")
        row: dict[str, object] = {
            "run_id": record["run_id"],
            "phase": record["phase"],
            "condition": record["condition"],
            "seed": record["seed"],
        }
        for field in ("metrics", "diagnostics", "capacity"):
            row.update(_mapping(record[field], name=field))
        rows.append(row)
    _write_csv(path, rows)


def _write_scaling(summary: Mapping[str, JsonValue], path: Path) -> None:
    rows: list[dict[str, object]] = []
    groups = cast(Mapping[str, Sequence[Mapping[str, object]]], summary["groups"])
    for phase_name in PHASE_ORDER:
        for group in groups.get(phase_name, ()):
            row: dict[str, object] = {
                "phase": phase_name,
                "candidate": group["candidate"],
            }
            for field in ("metrics", "diagnostics", "capacity"):
                means = _mapping(group[f"{field}_mean"], name=f"{field}_mean")
                variances = _mapping(
                    group[f"{field}_variance"], name=f"{field}_variance"
                )
                for name, value in means.items():
                    row[f"{name}_mean"] = value
                    row[f"{name}_variance"] = variances[name]
            rows.append(row)
    _write_csv(path, rows)


def _write_pareto(summary: Mapping[str, JsonValue], path: Path) -> None:
    decoder = cast(Mapping[str, object], summary["decoder_selection"])
    rows: list[dict[str, object]] = []
    for raw_candidate in cast(
        Sequence[Mapping[str, object]], decoder["candidates"]
    ):
        candidate = _mapping(raw_candidate, name="decoder.candidate")
        metrics = _mapping(candidate["metrics_mean"], name="metrics_mean")
        diagnostics = _mapping(candidate["diagnostics_mean"], name="diagnostics_mean")
        capacity = _mapping(candidate["capacity_mean"], name="capacity_mean")
        rows.append(
            {
                "candidate": candidate["candidate"],
                "kp_mean_distance_px": metrics["kp_mean_distance_px"],
                "pose_reprojection_mean_distance_px": metrics[
                    "pose_reprojection_mean_distance_px"
                ],
                "decoder_macs": capacity["decoder_macs"],
                "decoder_params": capacity["decoder_params"],
                "train_step_time_ms": diagnostics["train_step_time_ms"],
                "cuda_peak_memory_bytes": diagnostics["cuda_peak_memory_bytes"],
                "pareto_front": candidate["pareto_front"],
                "adopted": candidate["adopted"],
                "non_adoption_reason": candidate["non_adoption_reason"],
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Summary CSV rows cannot be empty.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    summary: Mapping[str, JsonValue], *, output_dir: Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = cast(Mapping[str, Sequence[Mapping[str, object]]], summary["groups"])
    encoder = groups["encoder_scaling"]
    depths = [
        int(cast(str, row["candidate"]).removeprefix("depth-")) for row in encoder
    ]
    kp = [
        _finite_number(
            _mapping(row["metrics_mean"], name="metrics")["kp_mean_distance_px"],
            name="metrics.kp_mean_distance_px",
        )
        for row in encoder
    ]
    pose = [
        _finite_number(
            _mapping(row["metrics_mean"], name="metrics")[
                "pose_reprojection_mean_distance_px"
            ],
            name="metrics.pose_reprojection_mean_distance_px",
        )
        for row in encoder
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(depths, kp, marker="o", label="GT KP mean")
    axis.plot(depths, pose, marker="s", label="Pose reprojection mean")
    axis.set(xlabel="Encoder depth", ylabel="Distance (px)", title="Encoder scaling")
    axis.legend()
    fig.tight_layout()
    encoder_path = output_dir / "encoder_scaling.png"
    fig.savefig(encoder_path, dpi=160)
    plt.close(fig)
    paths = [encoder_path]
    decoder = summary["decoder_selection"]
    if decoder is not None:
        candidates = cast(
            Sequence[Mapping[str, object]],
            cast(Mapping[str, object], decoder)["candidates"],
        )
        fig, axis = plt.subplots(figsize=(8, 5))
        for row in candidates:
            capacity = _mapping(row["capacity_mean"], name="capacity")
            metrics = _mapping(row["metrics_mean"], name="metrics")
            decoder_macs = _finite_number(
                capacity["decoder_macs"], name="capacity.decoder_macs"
            )
            kp_mean = _finite_number(
                metrics["kp_mean_distance_px"], name="metrics.kp_mean_distance_px"
            )
            axis.scatter(
                decoder_macs,
                kp_mean,
                marker="o" if row["pareto_front"] else "x",
            )
            axis.annotate(str(row["candidate"]), (decoder_macs, kp_mean))
        axis.set(
            xlabel="Decoder MACs",
            ylabel="GT KP mean distance (px)",
            title="Decoder scaling Pareto view",
        )
        fig.tight_layout()
        decoder_path = output_dir / "decoder_pareto.png"
        fig.savefig(decoder_path, dpi=160)
        plt.close(fig)
        paths.append(decoder_path)
    if summary["adoption_decision"] is not None:
        formal = groups["consistency_ablation"]
        names = [str(row["candidate"]) for row in formal]
        values = [
            _finite_number(
                _mapping(row["metrics_mean"], name="metrics")[
                    "kp_mean_distance_px"
                ],
                name="metrics.kp_mean_distance_px",
            )
            for row in formal
        ]
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(names, values)
        axis.set(ylabel="GT KP mean distance (px)", title="Consistency conditions")
        axis.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        formal_path = output_dir / "consistency_comparison.png"
        fig.savefig(formal_path, dpi=160)
        plt.close(fig)
        paths.append(formal_path)
    return tuple(paths)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be numeric.")
    number = float(cast(float | int, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


__all__ = [
    "CAPACITY_NAMES",
    "RESULTS_SCHEMA",
    "SUMMARY_SCHEMA",
    "QueryConsistencySummaryConfig",
    "SummaryPhase",
    "build_query_consistency_result_record",
    "collect_query_consistency_results",
    "collect_tensorboard_training_evidence",
    "load_json_mapping",
    "summarize_query_consistency",
    "validate_query_consistency_results",
    "validate_query_consistency_summary_boundary",
    "write_query_consistency_summary_artifacts",
]
