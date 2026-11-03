"""Independent 2D scoring, coverage and motion diagnostics on fixed time splits."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.artifacts import json_value
from src.utils.geometry.triangulation import (
    PointTriangulationConfig,
    triangulate_points,
)

from .annotations import Ball2DAnnotations
from .contracts import Ball3DResult, MethodInput
from .runner import MethodRun


@dataclass(frozen=True)
class ScoringConfig:
    split_frame: int = 505
    gross_reprojection_px: float = 20.0
    max_speed_mps: float = 65.0
    max_acceleration_mps2: float = 500.0
    event_radius_frames: int = 3

    def __post_init__(self) -> None:
        if (
            type(self.split_frame) is not int
            or self.split_frame < 1
            or self.event_radius_frames < 0
        ):
            raise ValueError("Invalid evaluation split/event window")
        if any(
            not np.isfinite(value) or value <= 0
            for value in (
                self.gross_reprojection_px,
                self.max_speed_mps,
                self.max_acceleration_mps2,
            )
        ):
            raise ValueError("Metric thresholds must be finite and positive")


DEFAULT_SCORING = ScoringConfig()


def project_result(
    inputs: MethodInput, result: Ball3DResult, camera_id: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Apply explicit offsets only within adjacent valid 3D samples, never across gaps."""
    view = inputs.views[inputs.camera_ids.index(camera_id)]
    camera, offset = view.camera, 0.0
    if camera_id in result.camera_ids:
        i = result.camera_ids.index(camera_id)
        if result.cameras is not None:
            camera = result.cameras[i]
        if result.time_offsets_seconds is not None:
            offset = float(result.time_offsets_seconds[i])
    query = inputs.frame_indices.astype(np.float64) + offset * inputs.fps
    query = np.where(np.abs(query - np.rint(query)) < 1e-9, np.rint(query), query)
    lo, hi = np.floor(query).astype(np.int64), np.ceil(query).astype(np.int64)
    inside = (lo >= 0) & (hi < inputs.num_frames)
    left, right = lo.clip(0, inputs.num_frames - 1), hi.clip(0, inputs.num_frames - 1)
    valid = inside & result.valid[left] & result.valid[right]
    fraction = query - lo
    xyz = (
        result.positions[left].astype(np.float64) * (1 - fraction[:, None])
        + result.positions[right] * fraction[:, None]
    )
    uv, front = camera.project(xyz)
    valid &= front
    return np.where(valid[:, None], uv, 0), valid


def error_samples(
    run: MethodRun, inputs: MethodInput, annotations: Ball2DAnnotations
) -> dict[str, NDArray[Any]]:
    annotations.validate_for_inputs(inputs)
    cameras = (
        inputs.camera_ids if run.held_out_camera is None else (run.held_out_camera,)
    )
    error, frames, camera_axis, kinds = [], [], [], []
    for camera_id in cameras:
        i = inputs.camera_ids.index(camera_id)
        uv, valid = project_result(inputs, run.result, camera_id)
        used = valid & annotations.valid_for_scoring[i]
        error.extend(
            np.linalg.norm(uv[used] - annotations.uv_px[i, used], axis=-1).tolist()
        )
        frames.extend(inputs.frame_indices[used].tolist())
        camera_axis.extend([i] * int(used.sum()))
        kinds.extend(annotations.reference_kind[i, used].tolist())
    return {
        "error_px": np.asarray(error, np.float64),
        "frame_index": np.asarray(frames, np.int64),
        "camera_index": np.asarray(camera_axis, np.int64),
        "reference_kind": np.asarray(kinds, dtype="U22"),
    }


def distribution(
    values: NDArray[Any], *, threshold: float | None = None
) -> dict[str, Any]:
    if not len(values):
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "max": None,
            **(
                {"threshold": threshold, "above_threshold_rate": None}
                if threshold is not None
                else {}
            ),
        }
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        **(
            {
                "threshold": threshold,
                "above_threshold_rate": float(np.mean(values > threshold)),
            }
            if threshold is not None
            else {}
        ),
    }


def motion_samples(result: Ball3DResult, fps: float) -> dict[str, NDArray[Any]]:
    xyz = result.positions.astype(np.float64)
    pairs = result.valid[:-1] & result.valid[1:]
    triples = result.valid[:-2] & result.valid[1:-1] & result.valid[2:]
    return {
        "step_frames": np.flatnonzero(pairs).astype(np.int64) + 1,
        "step_m": np.linalg.norm(np.diff(xyz, axis=0)[pairs], axis=-1),
        "acceleration_frames": np.flatnonzero(triples).astype(np.int64) + 1,
        "acceleration_mps2": np.linalg.norm(np.diff(xyz, n=2, axis=0)[triples], axis=-1)
        * fps**2,
    }


def _segments(inputs: MethodInput, config: ScoringConfig) -> dict[str, tuple[int, int]]:
    if not 0 < config.split_frame < inputs.num_frames:
        raise ValueError("Split must have nonempty tuning and evaluation intervals")
    return {
        "tuning": (0, config.split_frame),
        "evaluation": (config.split_frame, inputs.num_frames),
        "all": (0, inputs.num_frames),
    }


def _reference_metrics(
    samples: dict[str, NDArray[Any]],
    annotations: Ball2DAnnotations,
    camera_rows: list[int],
    start: int,
    stop: int,
    config: ScoringConfig,
    *,
    near_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    selected = (samples["frame_index"] >= start) & (samples["frame_index"] < stop)
    labels = annotations.valid_for_scoring[camera_rows, start:stop].copy()
    if near_frames is not None:
        # Union of source-row windows: overlapping and duplicate camera events
        # never multiply a residual's weight. No physical event count is inferred.
        near: NDArray[np.bool_] = np.zeros(len(selected), bool)
        label_near: NDArray[np.bool_] = np.zeros(stop - start, bool)
        for frame in near_frames:
            near |= np.abs(samples["frame_index"] - frame) <= config.event_radius_frames
            label_near |= (
                np.abs(np.arange(start, stop) - frame) <= config.event_radius_frames
            )
        selected &= near
        labels &= label_near[None]
    result = {}
    for kind in ("all", "observed", "estimated"):
        chosen, eligible = selected.copy(), labels.copy()
        if kind != "all":
            chosen &= samples["reference_kind"] == kind
            eligible &= annotations.reference_kind[camera_rows, start:stop] == kind
        summary = distribution(
            samples["error_px"][chosen], threshold=config.gross_reprojection_px
        )
        count = int(eligible.sum())
        result[kind] = {
            **summary,
            "label_count": count,
            "scored_label_coverage": summary["count"] / count if count else None,
        }
    return result


def _score_segment(
    run: MethodRun,
    inputs: MethodInput,
    annotations: Ball2DAnnotations | None,
    samples: dict[str, NDArray[Any]] | None,
    start: int,
    stop: int,
    config: ScoringConfig,
) -> dict[str, Any]:
    result, motion = run.result, motion_samples(run.result, inputs.fps)
    pairs = (motion["step_frames"] >= start + 1) & (motion["step_frames"] < stop)
    triples = (motion["acceleration_frames"] >= start + 1) & (
        motion["acceleration_frames"] < stop - 1
    )
    step, acceleration = motion["step_m"][pairs], motion["acceleration_mps2"][triples]
    metrics: dict[str, Any] = {
        "frames": [start, stop],
        "valid_3d_frames": int(result.valid[start:stop].sum()),
        "valid_3d_rate": float(result.valid[start:stop].mean()),
        "adjacent_step_m": distribution(
            step, threshold=config.max_speed_mps / inputs.fps
        ),
        "speed_mps": distribution(step * inputs.fps, threshold=config.max_speed_mps),
        "acceleration_mps2": distribution(
            acceleration, threshold=config.max_acceleration_mps2
        ),
        "runtime_seconds_full_run": run.elapsed_seconds,
    }
    if annotations is None or samples is None:
        metrics.update(
            {
                "reprojection_px": None,
                "gross_reprojection_frame_rate": None,
                "reprojection_by_reference_kind": None,
                "event_reprojection_px": None,
                "scoring_status": "unavailable: user reference not supplied",
            }
        )
        return metrics
    selected = (samples["frame_index"] >= start) & (samples["frame_index"] < stop)
    error, frames = samples["error_px"][selected], samples["frame_index"][selected]
    camera_rows = (
        list(range(len(inputs.views)))
        if run.held_out_camera is None
        else [inputs.camera_ids.index(run.held_out_camera)]
    )
    by_kind = _reference_metrics(samples, annotations, camera_rows, start, stop, config)
    label_count = by_kind["all"]["label_count"]
    scored_frames = np.unique(frames)
    bad_frames = np.unique(frames[error > config.gross_reprojection_px])
    events: dict[str, Any] = {}
    for kind in ("bounce", "hit"):
        rows = [
            event
            for event in annotations.reference.events
            if event["event"] == kind and start <= event["frame_index"] < stop
        ]
        event_frames = [event["frame_index"] for event in rows]
        event_metrics = _reference_metrics(
            samples,
            annotations,
            camera_rows,
            start,
            stop,
            config,
            near_frames=event_frames,
        )
        events[kind] = {
            "source_row_count": len(rows),
            "physical_event_count": None,
            "source_rows": json_value(rows),
            "radius_frames": config.event_radius_frames,
            "window_policy": "union_of_source_row_windows; each residual scored once",
            "error_px": event_metrics["all"] if rows else None,
            "by_reference_kind": event_metrics if rows else None,
        }
    metrics.update(
        {
            "reprojection_px": distribution(
                error, threshold=config.gross_reprojection_px
            ),
            "reprojection_by_reference_kind": by_kind,
            "scoring_label_count": label_count,
            "observed_label_count": by_kind["observed"]["label_count"],
            "estimated_label_count": by_kind["estimated"]["label_count"],
            "scored_label_count": len(error),
            "scored_label_coverage": len(error) / label_count if label_count else None,
            "gross_reprojection_frame_rate": len(bad_frames) / len(scored_frames)
            if len(scored_frames)
            else None,
            "event_reprojection_px": events,
            "scoring_status": "measured_on_user_reference"
            if len(error)
            else "no_scoring_label_with_valid_projection",
        }
    )
    return metrics


def score_comparison(
    runs: Mapping[str, Sequence[MethodRun]],
    inputs: MethodInput,
    annotations: Ball2DAnnotations | None,
    *,
    config: ScoringConfig = DEFAULT_SCORING,
) -> dict[str, Any]:
    segments = _segments(inputs, config)
    report: dict[str, Any] = {
        "schema": "tennis_lab_ball3d_scores.v1",
        "config": asdict(config),
        "primary_segment": "evaluation",
        "annotations": None
        if annotations is None
        else {
            "provenance": annotations.provenance,
            "state_counts": annotations.state_counts(),
            "source_events": json_value(annotations.reference.events),
            "estimation_segments": json_value(
                annotations.reference.estimation_segments
            ),
            "event_policy": "source rows retained; union windows; no inferred physical event count",
        },
        "independent_3d_ground_truth": "unavailable; reprojection gross error is not a measured 3D error rate",
        "methods": {},
    }
    for name, method_runs in runs.items():
        if len(method_runs) != 4 or {r.held_out_camera for r in method_runs} != {
            None,
            *inputs.camera_ids,
        }:
            raise ValueError("Each method must have all-view and all three LOVO runs")
        fold_samples: list[dict[str, NDArray[Any]]] = []
        folds = {}
        for run in method_runs:
            if run.method != name or run.source_camera_ids != inputs.camera_ids:
                raise ValueError("Comparison method/camera identity mismatch")
            samples = (
                None if annotations is None else error_samples(run, inputs, annotations)
            )
            folds[run.key] = {
                segment: _score_segment(
                    run, inputs, annotations, samples, start, stop, config
                )
                for segment, (start, stop) in segments.items()
            }
            if run.held_out_camera is not None and samples is not None:
                fold_samples.append(samples)
        aggregate, aggregate_by_kind = {}, {}
        for segment, (start, stop) in segments.items():
            # Pool observations, never average fold medians or fill empty folds with zero.
            combined = (
                None
                if not fold_samples
                else {
                    key: np.concatenate([sample[key] for sample in fold_samples])
                    for key in fold_samples[0]
                }
            )
            summary = (
                None
                if annotations is None or combined is None
                else _reference_metrics(
                    combined,
                    annotations,
                    list(range(len(inputs.views))),
                    start,
                    stop,
                    config,
                )
            )
            aggregate_by_kind[segment] = summary
            aggregate[segment] = None if summary is None else summary["all"]
        report["methods"][name] = {
            "folds": folds,
            "lovo_reprojection_px": aggregate,
            "lovo_reprojection_by_reference_kind": aggregate_by_kind,
            "runtime_seconds_all_four_runs": sum(
                r.elapsed_seconds for r in method_runs
            ),
        }
    return report


def annotation_oracle(
    inputs: MethodInput,
    annotations: Ball2DAnnotations,
    *,
    bundle_sha256: str,
    code_commit: str,
    code_fingerprint: str,
    reprojection_px: float = 20.0,
) -> tuple[MethodRun, ...]:
    """Scoring-only geometric reference; labels never pass through the method runner.

    Noisy 2D labels and approximate cameras do not guarantee a mathematical
    lower bound. Each LOVO oracle uses only its two retained annotation views.
    """
    annotations.validate_for_inputs(inputs)
    runs = []
    for held_out in (None, *inputs.camera_ids):
        start = time.perf_counter()
        selected = [
            i for i, camera in enumerate(inputs.camera_ids) if camera != held_out
        ]
        cameras = tuple(inputs.views[i].camera for i in selected)
        mask = annotations.valid_for_scoring[selected]
        trajectory = triangulate_points(
            np.where(mask[..., None], annotations.uv_px[selected], 0),
            mask,
            cameras,
            config=PointTriangulationConfig(reprojection_px, (-0.2, 20.0)),
        )
        result = Ball3DResult(
            tuple(c.camera_id for c in cameras),
            inputs.frame_indices,
            trajectory.positions,
            trajectory.valid,
            trajectory.inliers,
            diagnostics={"scoring_only": True},
        )
        runs.append(
            MethodRun(
                "annotation_oracle",
                held_out,
                inputs.camera_ids,
                result,
                {
                    "scoring_only": True,
                    "reprojection_px": reprojection_px,
                    "annotation_provenance": annotations.provenance,
                },
                bundle_sha256,
                code_commit,
                code_fingerprint,
                time.perf_counter() - start,
            )
        )
    return tuple(runs)
