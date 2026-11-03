"""clip_000 scoring-only diagnostics. Never import this into reconstruction methods."""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import least_squares

from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from src.tennis_scene.evaluation.ball3d.scoring import distribution
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic
from src.utils.geometry.keypoints import denormalize_grid_keypoints
from src.utils.geometry.triangulation import (
    PointTriangulationConfig,
    triangulate_points,
)


def decode_native(inputs):
    """Use production decode, in 64-frame chunks; retain pre-gate weak peaks."""
    refined, integer, confidence = [], [], []
    for view in inputs.views:
        evidence = view.detections.evidence
        uv, cells, scores = [], [], []
        for start in range(0, inputs.num_frames, 64):
            maps = torch.from_numpy(
                np.array(evidence.heatmaps[start : start + 64], copy=True)
            )[None]
            point, score = heatmaps_to_argmax(maps)
            cells.append(point[0].numpy())
            if view.detector_config.subpixel_refine:
                point = refine_peaks_log_parabolic(maps, point)
            uv.append(point[0].numpy())
            scores.append(score[0].numpy())
        refined.append(
            denormalize_grid_keypoints(np.concatenate(uv), *inputs.source_size_wh)
        )
        integer.append(
            denormalize_grid_keypoints(np.concatenate(cells), *inputs.source_size_wh)
        )
        confidence.append(np.concatenate(scores))
    return np.stack(refined), np.stack(integer), np.stack(confidence)


def interpolate_uv(uv, valid, query, *, start, stop):
    """Adjacent samples only, with an explicit interval that forbids split leakage."""
    query = np.asarray(query, np.float64)
    query = np.where(np.abs(query - np.rint(query)) < 1e-9, np.rint(query), query)
    lo, hi = np.floor(query).astype(int), np.ceil(query).astype(int)
    inside = (lo >= start) & (hi < stop)
    left, right = lo.clip(0, len(uv) - 1), hi.clip(0, len(uv) - 1)
    accepted = inside & valid[left] & valid[right]
    weight = query - lo
    # Invalid reference coordinates are NaN: mask rather than filling gaps.
    point = (1 - weight[:, None]) * uv[left] + weight[:, None] * uv[right]
    return np.where(accepted[:, None], point, 0), accepted


def annotation_speed(uv, valid, fps, *, start, stop):
    """Backward image speed (px/s); a missing previous point stays NaN."""
    speed = np.full(valid.shape, np.nan)
    pair = valid[:, start : stop - 1] & valid[:, start + 1 : stop]
    step = np.linalg.norm(np.diff(uv[:, start:stop], axis=1), axis=-1) * fps
    speed[:, start + 1 : stop] = np.where(pair, step, np.nan)
    return speed


def signed_summary(predicted, truth, selected):
    error = predicted[selected].astype(np.float64) - truth[selected]
    return {
        **distribution(np.linalg.norm(error, axis=-1), threshold=20.0),
        "mean_signed_xy_px": np.mean(error, axis=0).tolist() if len(error) else None,
        "median_signed_xy_px": np.median(error, axis=0).tolist()
        if len(error)
        else None,
    }


def binned_errors(predicted, truth, selected, feature, edges):
    return [
        {
            "range": [low, None if np.isinf(high) else high],
            **signed_summary(
                predicted, truth, selected & (feature >= low) & (feature < high)
            ),
        }
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]


def position_regression(
    position, error, frames, *, block=25, repetitions=499, seed=1558
):
    """OLS slope per normalized image extent, with source-time cluster bootstrap.

    Whole 25-frame clusters are resampled, not independent residuals. Intervals
    remain conditional on this one clip's sampled image positions.
    """
    position, error, frames = map(np.asarray, (position, error, frames))
    if len(position) < 3 or np.ptp(position) < 1e-8:
        return {"count": len(position), "status": "insufficient_position_support"}
    design = np.column_stack((np.ones(len(position)), position))
    coefficient = np.linalg.lstsq(design, error, rcond=None)[0]
    clusters = np.unique(frames // block)
    fits = []
    generator = np.random.default_rng(seed)
    if len(clusters) >= 4:
        for _ in range(repetitions):
            chosen = generator.choice(clusters, len(clusters), replace=True)
            rows = np.concatenate(
                [np.flatnonzero(frames // block == group) for group in chosen]
            )
            if np.ptp(position[rows]) > 1e-8:
                fits.append(np.linalg.lstsq(design[rows], error[rows], rcond=None)[0])
    ci = np.percentile(fits, [2.5, 97.5], axis=0).T.tolist() if fits else None
    return {
        "count": len(position),
        "status": "measured",
        "intercept_px": float(coefficient[0]),
        "slope_px_per_extent": float(coefficient[1]),
        "coefficient_ci95": ci,
        "block_frames": block,
        "block_count": len(clusters),
        "bootstrap_repetitions": len(fits),
        "bootstrap_seed": seed,
        "position_range_centered": [float(np.min(position)), float(np.max(position))],
        "rmse_px": float(np.sqrt(np.mean((error - design @ coefficient) ** 2))),
        "slope_ci_excludes_zero": None if ci is None else not ci[1][0] <= 0 <= ci[1][1],
        "interpretation": "descriptive association; serial blocks; not causal identification",
    }


def decode_report(inputs, annotations, refined, integer, scores, *, split=505):
    truth, valid = annotations.uv_px, annotations.valid_for_scoring
    result = {}
    for segment, (start, stop) in segments(inputs.num_frames, split).items():
        speed = annotation_speed(truth, valid, inputs.fps, start=start, stop=stop)
        time_mask = (inputs.frame_indices >= start) & (inputs.frame_indices < stop)
        rows = {}
        for i, camera in enumerate(inputs.camera_ids):
            grid_h, grid_w = inputs.views[i].detections.evidence.heatmaps.shape[-2:]
            error = refined[i] - truth[i]
            distance = np.linalg.norm(error, axis=-1)
            edge = np.min(
                np.concatenate(
                    (truth[i], np.asarray(inputs.source_size_wh) - 1 - truth[i]),
                    axis=-1,
                ),
                axis=-1,
            )
            kinds = {}
            for kind in ("all", "observed", "estimated"):
                mask = valid[i] & time_mask
                if kind != "all":
                    mask &= annotations.reference_kind[i] == kind
                regression = {}
                for subset, support in (
                    ("all_peaks", mask),
                    ("within20_px", mask & (distance <= 20)),
                ):
                    axes = {}
                    for axis, dimension, grid_size in (
                        (0, inputs.source_size_wh[0], grid_w),
                        (1, inputs.source_size_wh[1], grid_h),
                    ):
                        model = position_regression(
                            truth[i, support, axis] / (dimension - 1) - 0.5,
                            error[support, axis],
                            inputs.frame_indices[support],
                        )
                        hypothesis = 0.75 * dimension / grid_size
                        model["hypothesized_edge_magnitude_px"] = hypothesis / 2
                        model["hypothesized_slope_magnitude"] = hypothesis
                        ci = model.get("coefficient_ci95")
                        model["ci_contains_positive_hypothesis"] = (
                            None if ci is None else ci[1][0] <= hypothesis <= ci[1][1]
                        )
                        model["ci_contains_negative_hypothesis"] = (
                            None if ci is None else ci[1][0] <= -hypothesis <= ci[1][1]
                        )
                        axes["xy"[axis]] = model
                    regression[subset] = axes
                grid = []
                for gy in range(4):
                    for gx in range(6):
                        norm = np.where(valid[i, :, None], truth[i], 0) / (
                            np.asarray(inputs.source_size_wh) - 1
                        )
                        cell = (
                            mask
                            & (np.minimum((norm[:, 0] * 6).astype(int), 5) == gx)
                            & (np.minimum((norm[:, 1] * 4).astype(int), 3) == gy)
                        )
                        grid.append(
                            {
                                "cell_xy": [gx, gy],
                                **signed_summary(refined[i], truth[i], cell),
                            }
                        )
                kinds[kind] = {
                    "raw_log_parabola": signed_summary(refined[i], truth[i], mask),
                    "integer_argmax": signed_summary(integer[i], truth[i], mask),
                    "score": binned_errors(
                        refined[i],
                        truth[i],
                        mask,
                        scores[i],
                        [0, 0.5, 0.75, 0.9, 1.000001],
                    ),
                    "annotation_speed_px_s": binned_errors(
                        refined[i],
                        truth[i],
                        mask,
                        speed[i],
                        [0, 300, 900, 1800, np.inf],
                    ),
                    "speed_missing_count": int((mask & ~np.isfinite(speed[i])).sum()),
                    "edge_distance_px": binned_errors(
                        refined[i], truth[i], mask, edge, [0, 30, 90, 180, 360, np.inf]
                    ),
                    "position_grid_6x4": grid,
                    "regression": regression,
                }
            rows[camera] = kinds
        result[segment] = rows
    return result


def segments(frames, split=505):
    return {"tuning": (0, split), "evaluation": (split, frames), "all": (0, frames)}


def candidate_recovery(
    argmax, truth, scoring, candidates, candidate_valid, *, threshold=20.0
):
    failed = scoring & (np.linalg.norm(argmax - truth, axis=-1) > threshold)
    distance = np.linalg.norm(candidates - truth[..., None, :], axis=-1)
    close = candidate_valid & (distance <= threshold)
    curve = np.maximum.accumulate(close, axis=-1)
    count = int(failed.sum())
    return {
        "eligible_reference_count": int(scoring.sum()),
        "failed_argmax_count": count,
        "k": list(range(1, candidates.shape[-2] + 1)),
        "recovered_count": [
            int((curve[..., k] & failed).sum()) for k in range(curve.shape[-1])
        ],
        "recovered_rate": [
            float(curve[..., k][failed].mean()) if count else None
            for k in range(curve.shape[-1])
        ],
    }


def candidate_report(inputs, annotations, refined, *, split=505):
    uv = np.stack([v.detections.evidence.candidate_uv_px for v in inputs.views])
    valid = np.stack([v.detections.evidence.candidate_valid for v in inputs.views])
    report = {}
    for segment, (start, stop) in segments(inputs.num_frames, split).items():
        rows = {}
        for name, indices in (
            ("pooled", list(range(len(inputs.views)))),
            *((c, [i]) for i, c in enumerate(inputs.camera_ids)),
        ):
            kinds = {}
            for kind in ("all", "observed", "estimated"):
                mask = annotations.valid_for_scoring[indices, start:stop].copy()
                if kind != "all":
                    mask &= annotations.reference_kind[indices, start:stop] == kind
                kinds[kind] = candidate_recovery(
                    refined[indices, start:stop],
                    annotations.uv_px[indices, start:stop],
                    mask,
                    uv[indices, start:stop],
                    valid[indices, start:stop],
                )
            rows[name] = kinds
        report[segment] = rows
    return report


def event_masks(annotations, frame_indices, *, start, stop, radius=3):
    output = {}
    for kind in ("bounce", "hit"):
        mask = np.zeros(len(frame_indices), bool)
        for event in annotations.reference.events:
            if event["event"] == kind and start <= event["frame_index"] < stop:
                mask |= np.abs(frame_indices - event["frame_index"]) <= radius
        output[kind] = mask & (frame_indices >= start) & (frame_indices < stop)
    output["event_union"] = output["bounce"] | output["hit"]
    output["flight"] = (
        ~output["event_union"] & (frame_indices >= start) & (frame_indices < stop)
    )
    return output


def smoothing_report(inputs, annotations, *, split=505):
    raw = np.stack([v.detections.uv_px for v in inputs.views])
    smooth = np.stack([v.points.uv_px for v in inputs.views])
    accepted = np.stack([v.points.observed for v in inputs.views])
    report = {}
    for segment, (start, stop) in segments(inputs.num_frames, split).items():
        masks = event_masks(annotations, inputs.frame_indices, start=start, stop=stop)
        rows = {}
        for camera, indices in (
            ("pooled", list(range(len(inputs.views)))),
            *((c, [i]) for i, c in enumerate(inputs.camera_ids)),
        ):
            kinds = {}
            for kind in ("all", "observed", "estimated"):
                base = accepted[indices] & annotations.valid_for_scoring[indices]
                if kind != "all":
                    base &= annotations.reference_kind[indices] == kind
                groups = {}
                for event, time_mask in masks.items():
                    mask = base & time_mask
                    a = np.linalg.norm(
                        raw[indices] - annotations.uv_px[indices], axis=-1
                    )[mask]
                    b = np.linalg.norm(
                        smooth[indices] - annotations.uv_px[indices], axis=-1
                    )[mask]
                    groups[event] = {
                        "before_px": distribution(a, threshold=20.0),
                        "after_px": distribution(b, threshold=20.0),
                        "paired_reduction_px": distribution(a - b),
                        "mean_reduction_px": float(np.mean(a - b)) if len(a) else None,
                        "worsened_rate": float(np.mean(b > a)) if len(a) else None,
                    }
                kinds[kind] = groups
            rows[camera] = kinds
        report[segment] = rows
    return report


def fundamental(first, second):
    rotation = second.rotation @ first.rotation.T
    translation = second.translation - rotation @ first.translation
    x, y, z = translation
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.linalg.inv(second.intrinsic).T
        @ cross
        @ rotation
        @ np.linalg.inv(first.intrinsic)
    )


def fit_offsets(inputs, annotations, *, split=505, max_offset=0.99):
    """Scoring-only relative clock fit. cam0/first is the gauge, labels <split only.

    A camera frame t observes world time t+offset (frames). Minimize signed
    Sampson distances between interpolated annotation trajectories, with fixed
    3-sample support for every possible offset in the open one-frame bound.
    No held-out evaluation labels enter the optimizer.
    """
    if not 0 < max_offset < 1:
        raise ValueError("Diagnostic offsets must be strictly less than one frame")
    uv, valid = annotations.uv_px[:, :split], annotations.valid_for_scoring[:, :split]
    support = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    pairs = []
    for a, b in combinations(range(len(inputs.views)), 2):
        frames = np.flatnonzero(support[a] & support[b]) + 1
        if len(frames) < 8:
            raise ValueError(
                "Insufficient shared consecutive tuning annotations for clock fit"
            )
        pairs.append(
            (a, b, frames, fundamental(inputs.views[a].camera, inputs.views[b].camera))
        )

    def residual(parameters):
        offset = np.r_[0.0, parameters]
        values = []
        for a, b, frames, matrix in pairs:
            ua, va = interpolate_uv(
                uv[a], valid[a], frames - offset[a], start=0, stop=split
            )
            ub, vb = interpolate_uv(
                uv[b], valid[b], frames - offset[b], start=0, stop=split
            )
            if not (va & vb).all():
                raise ValueError("Clock fit changed its fixed tuning support")
            ha, hb = np.c_[ua, np.ones(len(ua))], np.c_[ub, np.ones(len(ub))]
            fa, fb = ha @ matrix.T, hb @ matrix
            denominator = np.sqrt(
                np.sum(fa[:, :2] ** 2, axis=-1) + np.sum(fb[:, :2] ** 2, axis=-1)
            )
            if np.any(denominator <= 1e-12):
                raise ValueError("Degenerate epipolar diagnostic")
            values.append(np.sum(hb * fa, axis=-1) / denominator)
        return np.concatenate(values)

    initial = residual(np.zeros(len(inputs.views) - 1))
    fit = least_squares(
        residual,
        np.zeros(len(inputs.views) - 1),
        bounds=(-max_offset, max_offset),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=120,
        diff_step=1e-4,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not fit.success:
        raise RuntimeError(f"Clock diagnostic failed: {fit.message}")
    offsets = np.r_[0.0, fit.x]
    return offsets, {
        "diagnostic_only": True,
        "fit_frames": [0, split],
        "camera_ids": list(inputs.camera_ids),
        "anchor_camera": inputs.camera_ids[0],
        "offset_frames": offsets.tolist(),
        "offset_seconds": (offsets / inputs.fps).tolist(),
        "absolute_bound_frames": max_offset,
        "at_bound": (np.abs(offsets) > max_offset - 1e-4).tolist(),
        "definition": "camera local frame t observes world frame t+offset; cam0=0 gauge",
        "objective": "soft_l1 signed Sampson pixel distance, f_scale=1, fixed tuning support",
        "objective_initial_abs_px": distribution(np.abs(initial)),
        "objective_fitted_abs_px": distribution(np.abs(fit.fun)),
        "pair_support": [
            {
                "cameras": [inputs.camera_ids[a], inputs.camera_ids[b]],
                "count": len(frames),
            }
            for a, b, frames, _ in pairs
        ],
        "optimizer_nfev": fit.nfev,
        "optimizer_message": fit.message,
        "limitations": "relative diagnostic conflates camera/label errors and timing; all-camera tuning labels used, no independent timing ground truth",
    }


def timing_oracle(inputs, annotations, offsets, template_runs, *, split=505):
    """Annotation geometry at each held-out clock, using offsets fitted on tuning.

    All-view coordinates use the anchor clock; LOVO coordinates use the held-out
    clock. This explicit reindexing allows the unchanged foundation scorer to
    compare against the original, unshifted held-out annotations. No method input
    is constructed from annotations and no inference runner is invoked.
    """
    output = []
    for template in template_runs:
        started = time.perf_counter()
        selected = [
            i for i, c in enumerate(inputs.camera_ids) if c != template.held_out_camera
        ]
        anchor = (
            0
            if template.held_out_camera is None
            else inputs.camera_ids.index(template.held_out_camera)
        )
        uv = np.zeros((len(selected), inputs.num_frames, 2), np.float64)
        mask = np.zeros(uv.shape[:-1], bool)
        for j, i in enumerate(selected):
            for start, stop in ((0, split), (split, inputs.num_frames)):
                query = inputs.frame_indices[start:stop] + offsets[anchor] - offsets[i]
                uv[j, start:stop], mask[j, start:stop] = interpolate_uv(
                    annotations.uv_px[i],
                    annotations.valid_for_scoring[i],
                    query,
                    start=start,
                    stop=stop,
                )
        cameras = tuple(inputs.views[i].camera for i in selected)
        trajectory = triangulate_points(
            uv, mask, cameras, config=PointTriangulationConfig(20.0, (-0.2, 20.0))
        )
        if template.held_out_camera is None:
            # Break the projection interpolation stencil at the scoring split.
            trajectory.valid[split - 1] = False
            trajectory.positions[split - 1] = 0
            trajectory.inliers[:, split - 1] = False
        result = Ball3DResult(
            tuple(c.camera_id for c in cameras),
            inputs.frame_indices,
            trajectory.positions,
            trajectory.valid,
            trajectory.inliers,
            time_offsets_seconds=offsets.astype(np.float64) / inputs.fps
            if template.held_out_camera is None
            else None,
            diagnostics={
                "scoring_only": True,
                "timeline_anchor_camera": inputs.camera_ids[anchor],
                "fitted_on_tuning_only": True,
            },
        )
        output.append(
            replace(
                template,
                method="annotation_time_oracle",
                result=result,
                config={
                    "scoring_only": True,
                    "fit_frames": [0, split],
                    "offset_frames": offsets.tolist(),
                    "timeline_anchor_camera": inputs.camera_ids[anchor],
                    "split_projection_buffer_frames": 1
                    if template.held_out_camera is None
                    else 0,
                },
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return tuple(output)


def candidate_oracle(inputs, annotations, template_runs, *, k):
    """Diagnostic upper envelope: nearest annotated top-K, same accepted 2D mask.

    This intentionally uses labels, is not an implementable method, and is never
    registered with the runner. Both K=1 and K=8 use unit weights and the same
    triangulator, so their paired difference isolates candidate availability.
    """
    output = []
    for template in template_runs:
        started = time.perf_counter()
        selected = [
            i for i, c in enumerate(inputs.camera_ids) if c != template.held_out_camera
        ]
        coordinates, masks = [], []
        for i in selected:
            evidence = inputs.views[i].detections.evidence
            candidate = evidence.candidate_uv_px[:, :k]
            valid = evidence.candidate_valid[:, :k]
            distance = np.linalg.norm(
                candidate - annotations.uv_px[i, :, None], axis=-1
            )
            rank = np.argmin(
                np.where(valid & np.isfinite(distance), distance, np.inf), axis=-1
            )
            coordinates.append(candidate[np.arange(inputs.num_frames), rank])
            masks.append(
                inputs.views[i].points.observed
                & annotations.valid_for_scoring[i]
                & valid.any(-1)
            )
        mask = np.stack(masks)
        uv = np.where(mask[..., None], np.stack(coordinates), 0)
        cameras = tuple(inputs.views[i].camera for i in selected)
        trajectory = triangulate_points(
            uv, mask, cameras, config=PointTriangulationConfig(20.0, (-0.2, 20.0))
        )
        result = Ball3DResult(
            tuple(c.camera_id for c in cameras),
            inputs.frame_indices,
            trajectory.positions,
            trajectory.valid,
            trajectory.inliers,
            diagnostics={
                "scoring_only": True,
                "k": k,
                "mask_policy": "baseline accepted 2D AND scoring label AND candidate present",
            },
        )
        output.append(
            replace(
                template,
                method=f"candidate{k}_oracle",
                result=result,
                config={
                    "scoring_only": True,
                    "k": k,
                    "weights": "unit",
                    "speed_filter": "none; pair comparison shares geometry only",
                },
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return tuple(output)
