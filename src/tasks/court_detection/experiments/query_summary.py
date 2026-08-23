"""Fail-closed aggregation, selection, Pareto, table, and plot generation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

from src.tasks.court_detection.experiments.configuration import SupervisionName
from src.tasks.court_detection.experiments.query_ablation import (
    MANIFEST_SCHEMA,
    PHASE_ORDER,
    validate_ablation_manifest,
)
from src.tasks.court_detection.models.query_encoder.profiling import (
    validate_profile_record,
)
from src.utils.io import save_json_atomic

JsonValue: TypeAlias = Any

RESULTS_SCHEMA = "court_query_ablation_results_v1"
SUMMARY_SCHEMA = "court_query_ablation_summary_v1"
_POSE_METRICS = (
    "pose_translation_l2_m",
    "pose_rotation_geodesic_deg",
    "pose_focal_relative_error",
    "pose_log_focal_abs_error",
)
_DENSE_METRICS = {
    "kp": ("kp_mean_distance_px",),
    "kp+pose": ("kp_mean_distance_px", *_POSE_METRICS),
    "all": ("kp_mean_distance_px", "seg_miou", "line_iou"),
    "all+pose": (
        "kp_mean_distance_px",
        "seg_miou",
        "line_iou",
        *_POSE_METRICS,
    ),
}
_PROFILE_SCALARS = (
    "decoder_params",
    "trainable_params",
    "total_params",
    "decoder_macs",
    "decoder_latency_ms",
    "end_to_end_latency_ms",
    "peak_memory_bytes",
)


def summarize_ablation(
    manifest: Mapping[str, object],
    results: Mapping[str, object],
    *,
    adopted_supervision: SupervisionName,
    adoption_rationale: str,
    require_gpu_profiles: bool,
) -> dict[str, JsonValue]:
    """Validate complete evidence and derive frozen encoder/decoder decisions."""
    validate_ablation_manifest(manifest, require_resolved=True)
    if adopted_supervision not in _DENSE_METRICS:
        raise ValueError("Adopted supervision is outside the frozen matrix.")
    if not adoption_rationale or adoption_rationale != adoption_rationale.strip():
        raise ValueError("Adoption rationale must be a non-empty trimmed string.")
    manifest_runs = _manifest_runs(manifest)
    result_runs = _validate_results(
        results,
        manifest=manifest,
        manifest_runs=manifest_runs,
        require_gpu_profiles=require_gpu_profiles,
    )
    aggregates = _aggregate(result_runs, manifest_runs=manifest_runs)
    encoder_rows = aggregates["encoder_first"]
    decoder_rows = aggregates["decoder_second"]
    supervision_rows = aggregates["supervision_third"]
    encoder_decision = _select_encoder(encoder_rows)
    decoder_decision = _select_decoder(decoder_rows)
    selected = _mapping(manifest["selected"], name="manifest.selected")
    if selected["encoder_depth"] != encoder_decision["selected_depth"]:
        raise ValueError(
            "Resolved manifest encoder selection disagrees with the frozen "
            "three-seed selection rule."
        )
    if (
        selected["decoder_family"] != decoder_decision["selected_family"]
        or selected["decoder_size"] != decoder_decision["selected_size"]
    ):
        raise ValueError(
            "Resolved manifest decoder selection disagrees with the frozen "
            "sufficiency/MAC/parameter rule."
        )
    supervision_decisions = [
        {
            **row,
            "adopted": row["candidate"] == adopted_supervision,
            "non_adoption_reason": (
                None
                if row["candidate"] == adopted_supervision
                else f"Explicit research decision adopted {adopted_supervision}."
            ),
        }
        for row in supervision_rows
    ]
    decoder_with_pareto = _mark_pareto(decoder_decision["candidates"])
    summary: dict[str, JsonValue] = {
        "schema": SUMMARY_SCHEMA,
        "source_manifest_schema": MANIFEST_SCHEMA,
        "manifest_sha256": cast(str, manifest["manifest_sha256"]),
        "phase_order": list(PHASE_ORDER),
        "run_count": len(result_runs),
        "seed_count": 3,
        "aggregation": {
            "mean": "arithmetic_mean_over_seeds",
            "variance": "population_variance_over_seeds",
        },
        "encoder_selection": encoder_decision,
        "decoder_selection": {
            **decoder_decision,
            "candidates": decoder_with_pareto,
        },
        "supervision_selection": {
            "adopted": adopted_supervision,
            "rationale": adoption_rationale,
            "candidates": supervision_decisions,
            "seg_line_semantics": "all_courts",
        },
        "production_default": {
            "changed": False,
            "adoption_status": "ablation_only_pending_separate_production_decision",
        },
    }
    return summary


def _manifest_runs(
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = cast(Sequence[object], manifest["runs"])
    runs = {
        cast(str, run["run_id"]): run
        for item in raw
        if isinstance(item, Mapping)
        for run in (cast(Mapping[str, object], item),)
    }
    if len(runs) != len(raw):
        raise ValueError("Manifest run records must all be mappings with unique IDs.")
    return runs


def _validate_results(
    results: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_runs: Mapping[str, Mapping[str, object]],
    require_gpu_profiles: bool,
) -> tuple[Mapping[str, object], ...]:
    if set(results) != {"schema", "manifest_sha256", "runs"}:
        raise ValueError("Ablation results fields changed.")
    if results["schema"] != RESULTS_SCHEMA:
        raise ValueError("Ablation results schema changed.")
    if results["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Ablation results do not bind the supplied manifest.")
    raw = results["runs"]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Ablation result runs must be a sequence.")
    records = tuple(_mapping(item, name="result.run") for item in raw)
    by_id: dict[str, Mapping[str, object]] = {}
    profile_contract: Mapping[str, object] | None = None
    for record in records:
        if set(record) != {"run_id", "phase", "seed", "metrics", "profile"}:
            raise ValueError("Ablation result run fields changed.")
        run_id = _string(record["run_id"], name="result.run_id")
        if run_id in by_id:
            raise ValueError("Ablation results contain a duplicate run ID.")
        try:
            planned = manifest_runs[run_id]
        except KeyError as error:
            raise ValueError(
                f"Ablation result has unknown run ID {run_id!r}."
            ) from error
        if record["phase"] != planned["phase"] or record["seed"] != planned["seed"]:
            raise ValueError(
                "Ablation result phase/seed disagrees with its manifest run."
            )
        supervision = cast(str, planned["supervision"])
        metrics = _mapping(record["metrics"], name=f"result.{run_id}.metrics")
        required_metrics = set(_DENSE_METRICS[supervision])
        if set(metrics) != required_metrics:
            raise ValueError(
                f"Ablation result {run_id!r} metrics must be exactly "
                f"{sorted(required_metrics)}."
            )
        for metric, value in metrics.items():
            number = _finite_number(value, name=f"result.{run_id}.{metric}")
            if number < 0.0:
                raise ValueError("Ablation metrics must be non-negative.")
            if metric in {"seg_miou", "line_iou"} and number > 1.0:
                raise ValueError("IoU metrics must be in [0, 1].")
        profile = _mapping(record["profile"], name=f"result.{run_id}.profile")
        validate_profile_record(profile, require_gpu_evidence=require_gpu_profiles)
        contract = _mapping(profile["input_contract"], name="profile.input_contract")
        expected_device = "cuda" if require_gpu_profiles else contract["device"]
        if (
            contract["batch_size"] != 1
            or contract["channels"] != 3
            or contract["height"] != 256
            or contract["width"] != 256
            or contract["dtype"] != "float32"
            or contract["device"] != expected_device
        ):
            raise ValueError(
                "Ablation profile input must be fixed batch-1 256x256 float32."
            )
        candidate = _mapping(profile["candidate"], name="profile.candidate")
        architecture = _mapping(planned["architecture"], name="run.architecture")
        if (
            candidate["family"] != architecture["decoder_family"]
            or candidate["size"] != architecture["decoder_size"]
        ):
            raise ValueError(
                "Ablation profile candidate disagrees with its manifest run."
            )
        if profile_contract is None:
            profile_contract = contract
        elif dict(contract) != dict(profile_contract):
            raise ValueError(
                "All ablation profiles must share one fixed input contract."
            )
        by_id[run_id] = record
    missing = set(manifest_runs) - set(by_id)
    extra = set(by_id) - set(manifest_runs)
    if missing or extra or len(records) != len(manifest_runs):
        raise ValueError(
            "Ablation results are incomplete: "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    ordered_ids = [cast(str, run["run_id"]) for run in manifest_runs.values()]
    if [cast(str, record["run_id"]) for record in records] != ordered_ids:
        raise ValueError(
            "Ablation result records must retain manifest phase/run order."
        )
    return records


def _aggregate(
    results: Sequence[Mapping[str, object]],
    *,
    manifest_runs: Mapping[str, Mapping[str, object]],
) -> dict[str, list[dict[str, JsonValue]]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for result in results:
        run_id = cast(str, result["run_id"])
        planned = manifest_runs[run_id]
        phase = cast(str, planned["phase"])
        candidate = _candidate_identity(planned)
        groups.setdefault((phase, candidate), []).append(result)
    aggregated: dict[str, list[dict[str, JsonValue]]] = {
        phase: [] for phase in PHASE_ORDER
    }
    for (phase, candidate), records in groups.items():
        seeds = sorted(cast(int, record["seed"]) for record in records)
        if seeds != [42, 43, 44]:
            raise ValueError(
                f"Ablation candidate {phase}/{candidate} lacks exactly three seeds."
            )
        metric_names = tuple(sorted(_mapping(records[0]["metrics"], name="metrics")))
        metric_means: dict[str, JsonValue] = {}
        metric_variances: dict[str, JsonValue] = {}
        for name in metric_names:
            values = [
                _finite_number(
                    _mapping(record["metrics"], name="metrics")[name],
                    name=name,
                )
                for record in records
            ]
            metric_means[name] = statistics.fmean(values)
            metric_variances[name] = statistics.pvariance(values)
        profile_rows = [_profile_scalars(record["profile"]) for record in records]
        for name in (
            "decoder_params",
            "trainable_params",
            "total_params",
            "decoder_macs",
        ):
            if len({row[name] for row in profile_rows}) != 1:
                raise ValueError(
                    f"Ablation candidate {phase}/{candidate} changed {name} across seeds."
                )
        profile_mean = {
            name: statistics.fmean(row[name] for row in profile_rows)
            for name in _PROFILE_SCALARS
        }
        profile_variance = {
            name: statistics.pvariance(row[name] for row in profile_rows)
            for name in _PROFILE_SCALARS
        }
        aggregated[phase].append(
            {
                "candidate": candidate,
                "seeds": seeds,
                "metrics_mean": metric_means,
                "metrics_variance": metric_variances,
                "profile_mean": profile_mean,
                "profile_variance": profile_variance,
            }
        )
    expected_counts = {"encoder_first": 4, "decoder_second": 9, "supervision_third": 4}
    for phase, count in expected_counts.items():
        if len(aggregated[phase]) != count:
            raise ValueError(f"Ablation phase {phase!r} aggregate is incomplete.")
    return aggregated


def _candidate_identity(run: Mapping[str, object]) -> str:
    phase = run["phase"]
    if phase == "encoder_first":
        architecture = _mapping(run["architecture"], name="run.architecture")
        return f"depth-{architecture['encoder_depth']}"
    if phase == "decoder_second":
        architecture = _mapping(run["architecture"], name="run.architecture")
        return f"{architecture['decoder_family']}-{architecture['decoder_size']}"
    return cast(str, run["supervision"])


def _profile_scalars(value: object) -> dict[str, float]:
    profile = _mapping(value, name="profile")
    parameters = _mapping(profile["parameters"], name="profile.parameters")
    macs = _mapping(profile["decoder_macs"], name="profile.decoder_macs")
    latency = _mapping(profile["latency_ms"], name="profile.latency_ms")
    peak = _mapping(profile["peak_memory"], name="profile.peak_memory")
    peak_bytes = peak["bytes"]
    return {
        "decoder_params": float(cast(int, parameters["decoder"])),
        "trainable_params": float(cast(int, parameters["trainable"])),
        "total_params": float(cast(int, parameters["total"])),
        "decoder_macs": float(cast(int, macs["count"])),
        "decoder_latency_ms": _finite_number(
            latency["decoder_mean"], name="decoder_latency_ms"
        ),
        "end_to_end_latency_ms": _finite_number(
            latency["end_to_end_mean"], name="end_to_end_latency_ms"
        ),
        "peak_memory_bytes": float(cast(int, peak_bytes or 0)),
    }


def _select_encoder(
    rows: Sequence[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    by_depth = {
        int(cast(str, row["candidate"]).removeprefix("depth-")): row for row in rows
    }
    reference = by_depth[8]
    metrics = (
        "kp_mean_distance_px",
        "pose_translation_l2_m",
        "pose_rotation_geodesic_deg",
        "pose_focal_relative_error",
    )
    reference_means = cast(dict[str, JsonValue], reference["metrics_mean"])
    candidates: list[dict[str, JsonValue]] = []
    sufficient_depths: list[int] = []
    for depth in sorted(by_depth):
        row = by_depth[depth]
        means = cast(dict[str, JsonValue], row["metrics_mean"])
        within = all(
            float(cast(float, means[name]))
            <= float(cast(float, reference_means[name])) * 1.05
            for name in metrics
        )
        if within:
            sufficient_depths.append(depth)
        candidates.append(
            {
                **row,
                "within_reference_tolerance": within,
            }
        )
    selected = min(sufficient_depths) if sufficient_depths else 8
    decisions = [
        {
            **row,
            "adopted": row["candidate"] == f"depth-{selected}",
            "non_adoption_reason": (
                None
                if row["candidate"] == f"depth-{selected}"
                else (
                    "Failed one or more 5% depth-8 metric thresholds."
                    if not row["within_reference_tolerance"]
                    else "A smaller sufficient encoder depth was selected."
                )
            ),
        }
        for row in candidates
    ]
    return {
        "selected_depth": selected,
        "reference_depth": 8,
        "tolerance_ratio": 0.05,
        "candidates": decisions,
    }


def _select_decoder(
    rows: Sequence[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    by_candidate = {cast(str, row["candidate"]): row for row in rows}
    reference = by_candidate["dpt-base"]
    reference_kp = float(
        cast(dict[str, JsonValue], reference["metrics_mean"])["kp_mean_distance_px"]
    )
    sufficient = [
        row
        for row in rows
        if float(cast(dict[str, JsonValue], row["metrics_mean"])["kp_mean_distance_px"])
        <= reference_kp * 1.05
    ]
    if not sufficient:
        raise ValueError(
            "Decoder sufficiency set cannot be empty because DPT-base is reference."
        )
    selected = min(
        sufficient,
        key=lambda row: (
            float(cast(dict[str, JsonValue], row["profile_mean"])["decoder_macs"]),
            float(cast(dict[str, JsonValue], row["profile_mean"])["decoder_params"]),
            cast(str, row["candidate"]),
        ),
    )
    selected_name = cast(str, selected["candidate"])
    candidates = []
    for row in rows:
        kp = float(
            cast(dict[str, JsonValue], row["metrics_mean"])["kp_mean_distance_px"]
        )
        within = kp <= reference_kp * 1.05
        candidates.append(
            {
                **row,
                "within_reference_tolerance": within,
                "adopted": row["candidate"] == selected_name,
                "non_adoption_reason": (
                    None
                    if row["candidate"] == selected_name
                    else (
                        "KP mean distance exceeds the 5% DPT-base threshold."
                        if not within
                        else "A sufficient candidate with lower MACs/parameters was selected."
                    )
                ),
            }
        )
    family, size = selected_name.split("-", maxsplit=1)
    return {
        "selected_family": family,
        "selected_size": size,
        "reference": "dpt-base",
        "tolerance_ratio": 0.05,
        "candidates": candidates,
    }


def _mark_pareto(
    rows: object,
) -> list[dict[str, JsonValue]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("Decoder candidates must be a sequence.")
    candidates = [cast(dict[str, JsonValue], row) for row in rows]
    result: list[dict[str, JsonValue]] = []
    for row in candidates:
        axes = _pareto_axes(row)
        dominated = any(
            _dominates(_pareto_axes(other), axes)
            for other in candidates
            if other is not row
        )
        result.append({**row, "pareto_front": not dominated})
    return result


def _pareto_axes(row: Mapping[str, JsonValue]) -> tuple[float, ...]:
    metrics = cast(dict[str, JsonValue], row["metrics_mean"])
    profile = cast(dict[str, JsonValue], row["profile_mean"])
    return (
        float(cast(float, metrics["kp_mean_distance_px"])),
        float(cast(float, profile["decoder_macs"])),
        float(cast(float, profile["decoder_params"])),
        float(cast(float, profile["decoder_latency_ms"])),
        float(cast(float, profile["peak_memory_bytes"])),
    )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def write_summary_artifacts(
    summary: Mapping[str, JsonValue],
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write deterministic JSON/CSV tables and scaling/Pareto PNG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_json_atomic(summary, output_dir / "summary.json")
    scaling_path = output_dir / "scaling_table.csv"
    pareto_path = output_dir / "pareto_table.csv"
    _write_scaling_table(summary, scaling_path)
    _write_pareto_table(summary, pareto_path)
    plot_paths = _write_plots(summary, output_dir=output_dir)
    return (summary_path, scaling_path, pareto_path, *plot_paths)


def _write_scaling_table(summary: Mapping[str, JsonValue], path: Path) -> None:
    metric_fields = (
        "kp_mean_distance_px",
        "pose_translation_l2_m",
        "pose_rotation_geodesic_deg",
        "pose_focal_relative_error",
        "pose_log_focal_abs_error",
        "seg_miou",
        "line_iou",
    )
    rows: list[dict[str, object]] = []
    for selection_name in (
        "encoder_selection",
        "decoder_selection",
        "supervision_selection",
    ):
        selection = cast(Mapping[str, object], summary[selection_name])
        candidates = cast(Sequence[Mapping[str, object]], selection["candidates"])
        for candidate in candidates:
            metrics = cast(Mapping[str, object], candidate["metrics_mean"])
            variances = cast(Mapping[str, object], candidate["metrics_variance"])
            profile = cast(Mapping[str, object], candidate["profile_mean"])
            profile_variance = cast(Mapping[str, object], candidate["profile_variance"])
            row: dict[str, object] = {
                "phase": selection_name,
                "candidate": candidate["candidate"],
            }
            for metric in metric_fields:
                row[f"{metric}_mean"] = metrics.get(metric, "")
                row[f"{metric}_variance"] = variances.get(metric, "")
            row.update(
                {
                    "decoder_params": profile["decoder_params"],
                    "trainable_params": profile["trainable_params"],
                    "total_params": profile["total_params"],
                    "decoder_macs": profile["decoder_macs"],
                    "decoder_latency_ms": profile["decoder_latency_ms"],
                    "decoder_latency_variance": profile_variance["decoder_latency_ms"],
                    "end_to_end_latency_ms": profile["end_to_end_latency_ms"],
                    "end_to_end_latency_variance": profile_variance[
                        "end_to_end_latency_ms"
                    ],
                    "peak_memory_bytes": profile["peak_memory_bytes"],
                    "peak_memory_variance": profile_variance["peak_memory_bytes"],
                    "adopted": candidate["adopted"],
                    "non_adoption_reason": candidate["non_adoption_reason"],
                }
            )
            rows.append(row)
    _write_csv(path, rows)


def _write_pareto_table(summary: Mapping[str, JsonValue], path: Path) -> None:
    decoder = cast(Mapping[str, object], summary["decoder_selection"])
    candidates = cast(Sequence[Mapping[str, object]], decoder["candidates"])
    rows = []
    for candidate in candidates:
        metrics = cast(Mapping[str, object], candidate["metrics_mean"])
        profile = cast(Mapping[str, object], candidate["profile_mean"])
        rows.append(
            {
                "candidate": candidate["candidate"],
                "kp_mean_distance_px": metrics["kp_mean_distance_px"],
                "decoder_macs": profile["decoder_macs"],
                "decoder_params": profile["decoder_params"],
                "trainable_params": profile["trainable_params"],
                "total_params": profile["total_params"],
                "decoder_latency_ms": profile["decoder_latency_ms"],
                "end_to_end_latency_ms": profile["end_to_end_latency_ms"],
                "peak_memory_bytes": profile["peak_memory_bytes"],
                "pareto_front": candidate["pareto_front"],
                "adopted": candidate["adopted"],
                "non_adoption_reason": candidate["non_adoption_reason"],
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Summary CSV cannot be empty.")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    summary: Mapping[str, JsonValue],
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    encoder = cast(Mapping[str, object], summary["encoder_selection"])
    encoder_rows = cast(Sequence[Mapping[str, object]], encoder["candidates"])
    depths = [
        int(cast(str, row["candidate"]).removeprefix("depth-")) for row in encoder_rows
    ]
    encoder_path = output_dir / "encoder_scaling.png"
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    encoder_metrics = (
        ("kp_mean_distance_px", "KP mean distance [px]"),
        ("pose_translation_l2_m", "Translation L2 [m]"),
        ("pose_rotation_geodesic_deg", "Rotation geodesic [deg]"),
        ("pose_focal_relative_error", "Focal relative error"),
    )
    for axis, (metric, label) in zip(axes.flatten(), encoder_metrics, strict=True):
        values = [
            _finite_number(
                cast(Mapping[str, object], row["metrics_mean"])[metric],
                name=f"encoder.{metric}",
            )
            for row in encoder_rows
        ]
        standard_deviations = [
            math.sqrt(
                _finite_number(
                    cast(Mapping[str, object], row["metrics_variance"])[metric],
                    name=f"encoder.{metric}.variance",
                )
            )
            for row in encoder_rows
        ]
        axis.errorbar(depths, values, yerr=standard_deviations, marker="o")
        axis.set_xlabel("Task encoder depth")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Encoder scaling (three-seed mean ± std)")
    figure.tight_layout()
    figure.savefig(encoder_path, dpi=160)
    plt.close(figure)

    decoder = cast(Mapping[str, object], summary["decoder_selection"])
    decoder_rows = cast(Sequence[Mapping[str, object]], decoder["candidates"])
    decoder_path = output_dir / "decoder_pareto.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    for row in decoder_rows:
        profile = cast(Mapping[str, object], row["profile_mean"])
        metrics = cast(Mapping[str, object], row["metrics_mean"])
        axis.scatter(
            _finite_number(profile["decoder_macs"], name="decoder.decoder_macs"),
            _finite_number(
                metrics["kp_mean_distance_px"], name="decoder.kp_mean_distance_px"
            ),
            marker="o" if row["pareto_front"] else "x",
        )
        axis.annotate(
            cast(str, row["candidate"]),
            (
                _finite_number(profile["decoder_macs"], name="decoder.decoder_macs"),
                _finite_number(
                    metrics["kp_mean_distance_px"],
                    name="decoder.kp_mean_distance_px",
                ),
            ),
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Decoder MACs")
    axis.set_ylabel("KP mean distance [px]")
    axis.set_title("Decoder accuracy/capacity Pareto")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(decoder_path, dpi=160)
    plt.close(figure)

    supervision = cast(Mapping[str, object], summary["supervision_selection"])
    supervision_rows = cast(Sequence[Mapping[str, object]], supervision["candidates"])
    supervision_path = output_dir / "supervision_comparison.png"
    labels = [cast(str, row["candidate"]) for row in supervision_rows]
    values = [
        _finite_number(
            cast(Mapping[str, object], row["metrics_mean"])["kp_mean_distance_px"],
            name="supervision.kp_mean_distance_px",
        )
        for row in supervision_rows
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values)
    axis.set_ylabel("KP mean distance [px]")
    axis.set_title("Supervision comparison (three-seed mean)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(supervision_path, dpi=160)
    plt.close(figure)
    return encoder_path, decoder_path, supervision_path


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
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


def load_json_mapping(path: Path) -> Mapping[str, object]:
    """Load one exact mapping artifact without supplying a missing fallback."""
    with path.open("r", encoding="utf-8") as handle:
        value: Any = json.load(handle)
    return _mapping(value, name=str(path))


__all__ = [
    "RESULTS_SCHEMA",
    "SUMMARY_SCHEMA",
    "load_json_mapping",
    "summarize_ablation",
    "write_summary_artifacts",
]
