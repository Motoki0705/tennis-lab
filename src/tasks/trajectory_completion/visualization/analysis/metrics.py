"""Metrics and console reports for trajectory completion visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.trajectory_completion.visualization.types import RuntimeConfig, TrajectoryInputs


def _summarize_inputs(inputs: TrajectoryInputs) -> dict[str, float]:
    t_len = int(inputs.ball_uv_gt.shape[0])
    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    orig_vis_count = int(orig_vis.sum())
    obs_count = int(obs.sum())

    newly_masked = orig_vis & (~obs)
    newly_masked_count = int(newly_masked.sum())

    err_in = np.linalg.norm(inputs.ball_uv_in - inputs.ball_uv_gt, axis=-1)
    jitter = err_in[obs]

    return {
        "frames": float(t_len),
        "orig_visible_ratio": float(orig_vis.mean()) if t_len > 0 else 0.0,
        "observed_ratio": float(obs.mean()) if t_len > 0 else 0.0,
        "orig_visible_count": float(orig_vis_count),
        "observed_count": float(obs_count),
        "newly_masked_count": float(newly_masked_count),
        "jitter_mean": float(jitter.mean()) if jitter.size > 0 else 0.0,
        "jitter_p95": float(np.quantile(jitter, 0.95)) if jitter.size > 0 else 0.0,
        "jitter_max": float(jitter.max()) if jitter.size > 0 else 0.0,
    }


def _basic_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return _basic_stats(values[mask]) if mask.size > 0 else _basic_stats(np.empty((0,), dtype=np.float32))


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(x.std()) <= 1e-9 or float(y.std()) <= 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _motion_stats(track: np.ndarray) -> dict[str, dict[str, float]]:
    speed = np.linalg.norm(track[1:] - track[:-1], axis=-1) if track.shape[0] >= 2 else np.empty((0,), dtype=np.float32)
    accel = (
        np.linalg.norm(track[2:] - 2.0 * track[1:-1] + track[:-2], axis=-1)
        if track.shape[0] >= 3
        else np.empty((0,), dtype=np.float32)
    )
    return {"speed": _basic_stats(speed), "accel": _basic_stats(accel)}


def _boundary_jump_stats(track: np.ndarray, obs_mask: np.ndarray) -> tuple[dict[str, float], list[tuple[int, float]]]:
    if track.shape[0] < 2 or obs_mask.shape[0] < 2:
        return _basic_stats(np.empty((0,), dtype=np.float32)), []
    switch = obs_mask[1:] != obs_mask[:-1]
    idx = np.where(switch)[0] + 1
    if idx.size == 0:
        return _basic_stats(np.empty((0,), dtype=np.float32)), []
    jump = np.linalg.norm(track[idx] - track[idx - 1], axis=-1)
    pairs = [(int(i), float(j)) for i, j in zip(idx.tolist(), jump.tolist(), strict=False)]
    return _basic_stats(jump), pairs


def _directional_error_stats(pred_uv: np.ndarray, gt_uv: np.ndarray) -> dict[str, dict[str, float]]:
    if pred_uv.shape[0] < 2 or gt_uv.shape[0] < 2:
        empty = _basic_stats(np.empty((0,), dtype=np.float32))
        return {"parallel_abs": empty, "perp_abs": empty}
    err = pred_uv - gt_uv
    gt_vel = gt_uv[1:] - gt_uv[:-1]
    err_aligned = err[1:]
    vel_norm = np.linalg.norm(gt_vel, axis=-1)
    valid = vel_norm > 1e-9
    if not np.any(valid):
        empty = _basic_stats(np.empty((0,), dtype=np.float32))
        return {"parallel_abs": empty, "perp_abs": empty}

    direction = gt_vel[valid] / vel_norm[valid, None]
    e = err_aligned[valid]
    parallel = np.abs(np.sum(e * direction, axis=-1))
    perp = np.sqrt(np.clip(np.sum(e * e, axis=-1) - parallel * parallel, a_min=0.0, a_max=None))
    return {"parallel_abs": _basic_stats(parallel), "perp_abs": _basic_stats(perp)}


def _event_mask(length: int, event_frames: dict[str, torch.Tensor], window: int) -> np.ndarray:
    mask = np.zeros((length,), dtype=bool)
    if length <= 0:
        return mask
    w = max(0, int(window))
    for key in ("bounce", "shot"):
        frames = event_frames.get(key)
        if frames is None or frames.numel() == 0:
            continue
        for t in frames.detach().cpu().to(torch.long).tolist():
            start = max(0, int(t) - w)
            end = min(length, int(t) + w + 1)
            mask[start:end] = True
    return mask


def _mask_runs(masked: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_masked in enumerate(masked.tolist()):
        if is_masked and start is None:
            start = i
        elif (not is_masked) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(masked.shape[0])))
    return runs


def summarize_predictions(
    *,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray,
    completed_uv: np.ndarray,
    event_frames: dict[str, torch.Tensor],
    event_window: int,
    topk: int = 20,
) -> dict[str, Any]:
    """Summarize prediction quality with per-group statistics."""
    gt = inputs.ball_uv_gt
    obs = inputs.ball_obs_mask
    orig_vis = inputs.ball_gt_visible
    masked = ~obs
    newly_masked = orig_vis & masked
    orig_missing = ~orig_vis
    event_near = _event_mask(gt.shape[0], event_frames, event_window)

    err_pred = np.linalg.norm(pred_uv - gt, axis=-1)
    err_completed = np.linalg.norm(completed_uv - gt, axis=-1)
    err_in = np.linalg.norm(inputs.ball_uv_in - gt, axis=-1)
    merge_delta = np.linalg.norm(completed_uv - pred_uv, axis=-1)

    top_idx = np.argsort(err_pred)[::-1][: max(0, int(topk))]
    top_frames = [
        {
            "frame": int(i),
            "err_pred": float(err_pred[i]),
            "err_completed": float(err_completed[i]),
            "is_observed": bool(obs[i]),
            "is_newly_masked": bool(newly_masked[i]),
            "is_orig_missing": bool(orig_missing[i]),
            "is_event_near": bool(event_near[i]),
        }
        for i in top_idx.tolist()
    ]

    run_summaries: list[dict[str, float]] = []
    for start, end in _mask_runs(masked):
        run_err = err_pred[start:end]
        if run_err.size == 0:
            continue
        run_summaries.append(
            {
                "start": float(start),
                "end": float(end),
                "length": float(end - start),
                "err_mean": float(run_err.mean()),
                "err_p95": float(np.quantile(run_err, 0.95)),
                "err_max": float(run_err.max()),
            }
        )
    run_summaries = sorted(run_summaries, key=lambda x: x["err_mean"], reverse=True)

    pred_motion = _motion_stats(pred_uv)
    completed_motion = _motion_stats(completed_uv)
    pred_boundary, pred_boundary_pairs = _boundary_jump_stats(pred_uv, obs)
    completed_boundary, completed_boundary_pairs = _boundary_jump_stats(completed_uv, obs)

    return {
        "error": {
            "pred_all": _basic_stats(err_pred),
            "pred_observed": _masked_stats(err_pred, obs),
            "pred_masked": _masked_stats(err_pred, masked),
            "pred_newly_masked": _masked_stats(err_pred, newly_masked),
            "pred_orig_missing": _masked_stats(err_pred, orig_missing),
            "pred_event_near": _masked_stats(err_pred, event_near),
            "pred_event_far": _masked_stats(err_pred, ~event_near),
            "completed_all": _basic_stats(err_completed),
        },
        "motion": {"pred": pred_motion, "completed": completed_motion},
        "boundary_jump": {
            "pred": pred_boundary,
            "completed": completed_boundary,
            "pred_pairs": pred_boundary_pairs,
            "completed_pairs": completed_boundary_pairs,
        },
        "directional_error": _directional_error_stats(pred_uv, gt),
        "input_pred_corr_observed": _safe_corrcoef(err_in[obs], err_pred[obs]),
        "merge_delta": {
            "all": _basic_stats(merge_delta),
            "observed": _masked_stats(merge_delta, obs),
            "masked": _masked_stats(merge_delta, masked),
        },
        "mask_runs": run_summaries,
        "top_frames": top_frames,
        "summary_scores": {
            "masked_err_p95": _masked_stats(err_pred, masked)["p95"],
            "pred_accel_p95": pred_motion["accel"]["p95"],
            "boundary_jump_p95": pred_boundary["p95"],
        },
    }


def _print_stat_line(name: str, stats: dict[str, float]) -> None:
    print(
        f"  {name:<20} count={int(stats['count']):4d} "
        f"mean={stats['mean']:.4f} p95={stats['p95']:.4f} max={stats['max']:.4f}"
    )


def print_info(cfg: RuntimeConfig, inputs: TrajectoryInputs) -> None:
    """Print dataset-side summary info."""
    scene_id = inputs.meta.get("scene_id", "Unknown")
    print("=" * 60)
    print("TRAJECTORY COMPLETION VISUALIZATION")
    print("=" * 60)
    print(f"Scene:      {scene_id}")
    print(f"Path:       {cfg.scene_path}")
    print(f"Camera:     {inputs.camera_idx}")
    print(f"Start:      {inputs.start}")
    print(f"Frames:     {inputs.ball_uv_gt.shape[0]}")

    stats = _summarize_inputs(inputs)
    print("\nVisibility / masking:")
    print(f"  Original visible: {stats['orig_visible_count']:.0f} ({stats['orig_visible_ratio']:.1%})")
    print(f"  Observed (input):  {stats['observed_count']:.0f} ({stats['observed_ratio']:.1%})")
    print(f"  Newly masked by augmentation: {stats['newly_masked_count']:.0f}")

    print("\nObserved-point jitter (|input - GT|):")
    print(f"  mean={stats['jitter_mean']:.4f}  p95={stats['jitter_p95']:.4f}  max={stats['jitter_max']:.4f}")


def print_predict_info(
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray,
    completed_uv: np.ndarray,
    *,
    event_frames: dict[str, torch.Tensor],
    event_window: int,
) -> None:
    """Print prediction-side summary info."""
    print_info(cfg, inputs)
    analysis = summarize_predictions(
        inputs=inputs,
        pred_uv=pred_uv,
        completed_uv=completed_uv,
        event_frames=event_frames,
        event_window=event_window,
    )

    print("\nPrediction error (|pred - GT|):")
    _print_stat_line("all", analysis["error"]["pred_all"])
    _print_stat_line("observed", analysis["error"]["pred_observed"])
    _print_stat_line("masked", analysis["error"]["pred_masked"])
    _print_stat_line("newly_masked", analysis["error"]["pred_newly_masked"])
    _print_stat_line("orig_missing", analysis["error"]["pred_orig_missing"])
    _print_stat_line("event_near", analysis["error"]["pred_event_near"])
    _print_stat_line("event_far", analysis["error"]["pred_event_far"])

    print("\nCompleted error (|completed - GT|):")
    _print_stat_line("all", analysis["error"]["completed_all"])

    print("\nTemporal motion stats:")
    _print_stat_line("pred_speed", analysis["motion"]["pred"]["speed"])
    _print_stat_line("pred_accel", analysis["motion"]["pred"]["accel"])
    _print_stat_line("completed_speed", analysis["motion"]["completed"]["speed"])
    _print_stat_line("completed_accel", analysis["motion"]["completed"]["accel"])

    print("\nBoundary jump at observed/masked switch:")
    _print_stat_line("pred_jump", analysis["boundary_jump"]["pred"])
    _print_stat_line("completed_jump", analysis["boundary_jump"]["completed"])

    print("\nDirectional error (pred vs GT motion):")
    _print_stat_line("parallel_abs", analysis["directional_error"]["parallel_abs"])
    _print_stat_line("perp_abs", analysis["directional_error"]["perp_abs"])

    print("\nCorrelation and merge effect:")
    print(f"  corr(|input-GT|, |pred-GT|) on observed = {analysis['input_pred_corr_observed']:.4f}")
    _print_stat_line("merge_delta_all", analysis["merge_delta"]["all"])
    _print_stat_line("merge_delta_obs", analysis["merge_delta"]["observed"])
    _print_stat_line("merge_delta_masked", analysis["merge_delta"]["masked"])

    print("\nTop error frames (pred):")
    for row in analysis["top_frames"][:20]:
        print(
            f"  t={row['frame']:4d} err={row['err_pred']:.4f} "
            f"completed={row['err_completed']:.4f} "
            f"obs={int(row['is_observed'])} new_mask={int(row['is_newly_masked'])} "
            f"orig_missing={int(row['is_orig_missing'])} event_near={int(row['is_event_near'])}"
        )

    print("\nWorst masked runs by mean error:")
    for run in analysis["mask_runs"][:10]:
        print(
            f"  [{int(run['start'])},{int(run['end'])}) len={int(run['length'])} "
            f"mean={run['err_mean']:.4f} p95={run['err_p95']:.4f} max={run['err_max']:.4f}"
        )

    print("\nSummary scores:")
    print(f"  masked_err_p95={analysis['summary_scores']['masked_err_p95']:.4f}")
    print(f"  pred_accel_p95={analysis['summary_scores']['pred_accel_p95']:.4f}")
    print(f"  boundary_jump_p95={analysis['summary_scores']['boundary_jump_p95']:.4f}")
