"""Full-clip evaluation, explicit baselines and offline physical diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.base.generate_dataset import build_physical_court_provenance
from src.tasks.base.model_io import validate_model_artifact_court_keypoint_contract
from src.tasks.base.model_io.court_keypoint_contract import (
    validate_neural_court_observation_order,
)
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_multiview_trajectories,
)
from src.tasks.blcs.model_io.contracts import BLCSTrajectoryPrediction
from src.tasks.blcs.model_io.factory import (
    TrajectoryBoundModelIO,
    compose_blcs_model_io,
)
from src.tasks.blcs.physics.rollout import reconstruct_segments
from src.tasks.blcs.training.metrics import BLCSMetrics
from src.utils.geometry.planar_camera import PlanarCameraFitError, fit_planar_camera
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointTriangulationConfig,
    triangulate_points,
)
from src.utils.physics.ball.fitting import FitSettings, fit_flight_physics
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
    validate_court_coordinate_normalization,
)

from .issue1044 import experiment_config


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binding(
    checkpoint: Path, data_root: Path, *, historical: bool = False
) -> tuple[TrajectoryBoundModelIO, DictConfig]:
    """Load exact model weights. Historical evaluation is an explicit experiment arm."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_court_coordinate_normalization(raw, artifact="BLCS evaluation checkpoint")
    config = OmegaConf.create(raw["hyper_parameters"]["config"])
    contract = parse_court_keypoint_contract(config)
    validate_model_artifact_court_keypoint_contract(
        raw, contract, location="BLCS evaluation checkpoint"
    )
    validate_neural_court_observation_order(raw, contract)
    if historical:
        # Only the known pre-1044 model contract is adapted for a reference run;
        # the public training/inference checkpoint APIs remain strict.
        if "physics_head" in config.model:
            raise ValueError("Historical option was supplied for a new checkpoint")
        with open_dict(config.model):
            config.model.physics_head = False
    elif "physics_head" not in config.model:
        raise ValueError(
            "Old checkpoint requires explicit --historical reference evaluation"
        )
    config.paths.data_root = str(data_root)
    config.data.seq_len_range = [1, 10240]
    config.data.num_workers = 0
    binding = compose_blcs_model_io(config)
    state = {
        key.removeprefix("model."): value
        for key, value in raw["state_dict"].items()
        if key.startswith("model.")
    }
    binding.model.load_state_dict(state, strict=True)
    del raw, state
    return binding, config


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_windows(
    binding: Any, batch: dict[str, Any], *, device: str, width: int = 128
) -> BLCSTrajectoryPrediction:
    """Nonoverlapping, fixed-width context windows; every real frame occurs once."""
    length = batch["ball_uv"].shape[2]
    values: dict[str, list[Tensor]] = {"position": []}
    for start in range(0, length, width):
        end = min(start + width, length)
        window = {
            key: batch[key][:, :, start:end].to(device)
            for key in ("ball_uv", "ball_vis", "court_kp", "court_vis", "padding_mask")
        }
        # Pad the last window as in training; all GT fields are excluded by construction.
        for key, tensor in window.items():
            padded = tensor.new_full(
                (*tensor.shape[:2], width, *tensor.shape[3:]), key == "padding_mask"
            )
            padded[:, :, : end - start] = tensor
            window[key] = padded
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            out = binding.run(window)
        for key, value in (
            ("position", out.position),
            ("velocity", out.velocity),
            ("physics_parameters", out.physics_parameters),
            ("event_logits", out.event_logits),
        ):
            if value is not None:
                values.setdefault(key, []).append(value[:, : end - start].float().cpu())
    combined = {key: torch.cat(chunks, dim=1) for key, chunks in values.items()}
    return BLCSTrajectoryPrediction(
        position=combined["position"],
        velocity=combined.get("velocity"),
        physics_parameters=combined.get("physics_parameters"),
        event_logits=combined.get("event_logits"),
    )


def geometric_baseline(
    batch: dict[str, Any],
) -> tuple[BLCSTrajectoryPrediction, dict[str, Any]]:
    """Triangulate the same input, then explicitly interpolate/extrapolate gaps."""
    cameras, selected_views, calibration = [], [], []
    court = court_keypoints_3d(STANDARD_COURT_CONFIG).double().numpy()[:14]
    for view in range(batch["ball_uv"].shape[1]):
        scores = batch["court_vis"][0, view].double().numpy()
        frame = int(scores.sum(-1).argmax())
        size = tuple(int(batch[key][0, view]) for key in ("camera_w", "camera_h"))
        pixels = batch["court_kp"][0, view, frame].double().numpy() * np.array(size)
        try:
            fit = fit_planar_camera(
                court,
                pixels,
                scores[frame],
                size,
                min_score=0.3,
                min_points=6,
                require_in_image=True,
            )
        except PlanarCameraFitError as error:
            calibration.append({"view": view, "failed": True, "reason": str(error)})
            continue
        cameras.append(PinholeCamera(str(view), fit.K, fit.R, fit.t))
        selected_views.append(view)
        calibration.append(
            {
                "view": view,
                "frame": frame,
                "failed": False,
                "rmse_px": fit.rmse_px,
                "used_points": int(fit.used_mask.sum()),
            }
        )
    if len(cameras) < 2:
        raise ValueError(
            f"Fewer than two cameras calibrated from input court observations: {calibration}"
        )
    size = torch.stack((batch["camera_w"], batch["camera_h"]), dim=-1).unsqueeze(2)
    uv = (batch["ball_uv"] * size)[0, selected_views].numpy()
    result = triangulate_points(
        uv,
        batch["ball_vis"][0, selected_views].bool().numpy(),
        tuple(cameras),
        config=PointTriangulationConfig(
            max_reprojection_px=20.0, height_range_m=(-1.0, 30.0)
        ),
    )
    indices = np.flatnonzero(result.valid)
    if len(indices) < 2:
        raise ValueError(
            "Geometric baseline requires at least two triangulated frames; report clip as unsupported"
        )
    coordinates = np.stack(
        [
            np.interp(
                np.arange(len(result.valid)), indices, result.positions[indices, axis]
            )
            for axis in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    receipt = {
        "calibration": calibration,
        "calibration_uses_ground_truth_camera": False,
        "calibration_method": "shared planar fit from physical_v1 CourtKP14; best visible frame; image-centre principal point; fx=fy; no distortion",
        "triangulated_frames": len(indices),
        "interpolated_frames": int((~result.valid).sum()),
        "endpoint_extrapolation": "constant",
        "rejection_counts": np.bincount(result.reasons, minlength=8).tolist(),
    }
    return BLCSTrajectoryPrediction(
        torch.from_numpy(normalize_court_position(coordinates))[None], None
    ), receipt


def fit_diagnostic(
    position_m: np.ndarray,
    segment: np.ndarray,
    clock: np.ndarray,
    gravity: float,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Bounded diagnostic on the first 32 frames of each flight (same GT support).

    The nonlinear fit has no guaranteed global optimum. Its settings and failed
    supports are saved, so a small residual is never used as a 3D accuracy proxy.
    """
    boundaries = [
        0,
        *(np.flatnonzero(np.diff(segment) != 0) + 1).tolist(),
        len(segment),
    ]
    spans = [
        (a, min(b, a + 32))
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        if b - a >= 6
    ]
    settings = FitSettings(
        gravity=gravity,
        dt=float(clock[0]),
        substeps=int(clock[3]),
        segment_iterations=4,
        iterations=8,
        minimum_segment_frames=6,
        field_starts=((0.01, 0.001, 0.0, 0.0), (0.03, 0.001, 0.0, 0.0)),
    )
    receipt: dict[str, Any] = {
        "spans": spans,
        "supported_frames": 0,
        "requested_frames": sum(b - a for a, b in spans),
        "failed": False,
        "rmse_m": None,
    }
    if not spans:
        receipt.update(failed=True, reason="no_segment_with_six_frames")
        return receipt
    try:
        fit = fit_flight_physics([position_m], [spans], settings, torch.device(device))[
            0
        ]
        residual = fit.residual[fit.fitted]
        receipt["supported_frames"] = len(residual)
        if not len(residual):
            receipt.update(failed=True, reason="no_finite_fit")
        else:
            receipt["rmse_m"] = float(np.sqrt(np.mean(np.sum(residual**2, axis=-1))))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        receipt.update(failed=True, reason=f"{type(error).__name__}: {error}")
    return receipt


def event_diagnostic(
    predicted: np.ndarray, target: np.ndarray, dt: float
) -> dict[str, Any]:
    a, b = (
        np.flatnonzero(predicted)[1:] if predicted[0] else np.flatnonzero(predicted),
        np.flatnonzero(target)[1:] if target[0] else np.flatnonzero(target),
    )
    distances = np.abs(a[:, None] - b[None, :])
    # Maximize valid one-to-one matches before minimizing total timing error.
    # Nearest-first matching can consume a target needed by a later event.
    penalty = (len(a) + len(b) + 1) * 3
    rows, columns = linear_sum_assignment(np.where(distances <= 2, distances, penalty))
    matched = distances[rows, columns]
    offsets = (matched[matched <= 2] * dt).tolist()
    tp = len(offsets)
    return {
        "true_positive": tp,
        "predicted_events": len(a),
        "target_events": len(b),
        "f1": 2 * tp / (len(a) + len(b)) if len(a) + len(b) else None,
        "timing_mean_s": np.mean(offsets) if offsets else None,
        "partition_failed": len(a) != tp or len(b) != tp,
    }


def save_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """A self-contained, non-pickle full-clip bundle with explicit padding."""
    maximum = max(row["pred_position"].shape[1] for row in rows)
    arrays: dict[str, Any] = {"scene_ids": np.array([row["scene_id"] for row in rows])}
    static = {
        "physics_field",
        "physics_clock",
        "physics_surface",
        "clip_frames",
        "bounce_count",
    }
    camera = {
        "camera_R",
        "camera_C",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    }
    view_time = {
        "ball_uv",
        "ball_vis",
        "ball_uv_target",
        "ball_vis_target",
        "court_kp",
        "court_vis",
        "padding_mask",
    }
    for key in rows[0]:
        if key == "scene_id":
            continue
        padded_rows = []
        for row in rows:
            value = row[key].detach().cpu().numpy()
            shape = list(value.shape)
            if key in camera | view_time:
                shape[1] = 6
            if key in view_time:
                shape[2] = maximum
            elif key not in camera | static:
                shape[1] = maximum
            fill = (
                True
                if key == "padding_mask"
                else (
                    -1
                    if key in {"physics_segment", "frame_index"}
                    else (1 if key in {"camera_w", "camera_h"} else 0)
                )
            )
            padded = np.full(shape, fill, dtype=value.dtype)
            padded[tuple(slice(0, n) for n in value.shape)] = value
            padded_rows.append(padded)
        arrays[key] = np.concatenate(padded_rows)
    np.savez_compressed(path, **arrays)


def evaluate(
    *,
    data_root: Path,
    output: Path,
    split: str,
    checkpoint: Path | None,
    baseline: str | None,
    historical: bool = False,
    device: str = "cuda",
    limit: int | None = None,
    fitting_clips: int = 12,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if checkpoint is not None:
        binding, config = load_binding(checkpoint, data_root, historical=historical)
        binding.model.to(device).eval()
    else:
        config = experiment_config(
            data_root=data_root, output="unused", arm="rollout_velocity", seed=42
        )
        config.data.seq_len_range = [1, 10240]
        binding = compose_blcs_model_io(config)
    dataset = BallTrajectoryDataset(
        scene_dir=data_root / str(config.data.scene_dir),
        split_file=f"{split}.txt",
        config=config,
        seed=42,
        augment=False,
    )
    tracker = BLCSMetrics()
    oracle_tracker, predicted_tracker = BLCSMetrics(), BLCSMetrics()
    rows, clips, failures = [], [], []
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        sample = dataset[index]
        batch = cast("dict[str, Any]", collate_multiview_trajectories([sample]))
        target = binding.adapter.build_training_batch(batch)
        extra: dict[str, Any] = {}
        if baseline is None:
            prediction = predict_windows(binding, batch, device=device)
        elif baseline == "geometry":
            try:
                prediction, extra["geometry"] = geometric_baseline(batch)
            except ValueError as error:
                failures.append({"scene_id": sample["scene_id"], "reason": str(error)})
                continue
        else:
            prediction = BLCSTrajectoryPrediction(
                target.position.clone(), target.velocity.clone()
            )
            if baseline == "jitter":
                prediction.position[:, ::2, 0] += 0.1 / 11.885
            elif baseline == "time_shift":
                prediction = replace(
                    prediction,
                    position=prediction.position.roll(1, 1),
                    velocity=target.velocity.roll(1, 1),
                )
            elif baseline == "collision_crossing":
                batch["physics_segment"] = torch.zeros_like(batch["physics_segment"])
                target = replace(
                    target,
                    physics=replace(
                        target.physics, segment=torch.zeros_like(target.physics.segment)
                    ),
                )
            elif baseline != "gt":
                raise ValueError(baseline)
        tracker.update(
            prediction, target, observed=batch["ball_vis"], scene_ids=batch["scene_id"]
        )
        clip_tracker = BLCSMetrics()
        clip_tracker.update(
            prediction, target, observed=batch["ball_vis"], scene_ids=batch["scene_id"]
        )
        if index < fitting_clips:
            extra["fit"] = fit_diagnostic(
                denormalize_court_position(prediction.position)[0].numpy(),
                target.physics.segment[0].numpy(),
                target.physics.clock[0].numpy(),
                float(target.physics.field[0, 0]),
            )
        if prediction.physics_parameters is not None:
            resets = torch.zeros_like(target.loss_mask)
            resets[:, 1:] = (
                target.physics.segment[:, 1:] != target.physics.segment[:, :-1]
            )
            estimated = prediction.event_logits[..., 0] >= 0
            extra["events"] = event_diagnostic(
                estimated[0].numpy(),
                resets[0].numpy(),
                float(target.physics.frame_dt()[0]),
            )
            for name, events, metric in (
                ("oracle", resets, oracle_tracker),
                ("predicted", estimated, predicted_tracker),
            ):
                p, v = reconstruct_segments(
                    denormalize_court_position(prediction.position),
                    prediction.velocity * 11.885,
                    prediction.physics_parameters,
                    events,
                    target.loss_mask,
                    target.physics.clock,
                    gravity=float(config.physics.gravity),
                )
                if not bool(torch.isfinite(p).all() & torch.isfinite(v).all()):
                    extra[f"{name}_integration_failed"] = True
                    extra[f"{name}_integrated_position"] = np.full_like(
                        p[0].numpy(), np.nan
                    )
                    continue
                integrated = replace(
                    prediction,
                    position=normalize_court_position(p),
                    velocity=v / 11.885,
                )
                metric.update(
                    integrated,
                    target,
                    observed=batch["ball_vis"],
                    scene_ids=batch["scene_id"],
                )
                extra[f"{name}_integration_failed"] = False
                integrated_clip = BLCSMetrics()
                integrated_clip.update(integrated, target, observed=batch["ball_vis"])
                extra[f"{name}_integrated_metrics"] = integrated_clip.compute()
                extra[f"{name}_integrated_position"] = p[0].numpy()
        integrated_arrays = {
            key: torch.as_tensor(extra.pop(key))[None]
            for key in tuple(extra)
            if key.endswith("_integrated_position")
        }
        clips.append(
            {"scene_id": sample["scene_id"], "metrics": clip_tracker.compute(), **extra}
        )
        keys = (
            "ball_uv",
            "ball_vis",
            "ball_uv_target",
            "ball_vis_target",
            "court_kp",
            "court_vis",
            "padding_mask",
            "position_3d",
            "velocity_3d",
            "physics_spin",
            "physics_segment",
            "physics_event",
            "physics_field",
            "physics_clock",
            "physics_surface",
            "frame_index",
            "clip_frames",
            "bounce_count",
            "camera_R",
            "camera_C",
            "camera_f",
            "camera_cx",
            "camera_cy",
            "camera_w",
            "camera_h",
        )
        row = {key: batch[key] for key in keys}
        row.update(scene_id=sample["scene_id"], pred_position=prediction.position)
        row.update(integrated_arrays)
        if prediction.velocity is not None:
            row["pred_velocity"] = prediction.velocity
        if prediction.physics_parameters is not None:
            row.update(
                physics_parameters=prediction.physics_parameters,
                event_logits=prediction.event_logits,
            )
        rows.append(row)
        if index % 10 == 0:
            print(f"evaluation {split} {index + 1}/{len(dataset)}", flush=True)
    report = {
        **tracker.report(),
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "baseline": baseline,
        "scene_count": len(rows),
        "unsupported_clips": failures,
        "clips": clips,
        "condition": {
            "window_frames": 128,
            "full_clips": True,
            "evaluation_seed": 42,
            "fit_first_clips": fitting_clips,
            "fit_max_frames_per_segment": 32,
            "fit_iterations": 8,
            "fit_segment_iterations": 4,
            "fit_field_starts": [[0.01, 0.001, 0.0, 0.0], [0.03, 0.001, 0.0, 0.0]],
            "fit_minimum_segment_frames": 6,
        },
    }
    if oracle_tracker.records:
        report["integrated_oracle"] = oracle_tracker.report()
        report["integrated_predicted"] = predicted_tracker.report()
    save_json(output / "evaluation.json", report)
    save_prediction_rows(output / f"pred_{split}.npz", rows)
    return report


def recompute(bundle: Path) -> dict[str, Any]:
    """Recompute ordinary full-clip metrics using only the portable bundle."""
    with np.load(bundle, allow_pickle=False) as data:
        config = experiment_config(
            data_root=Path("/tmp"), output="unused", arm="reprojection", seed=42
        )
        binding = compose_blcs_model_io(config)
        tracker = BLCSMetrics()
        for i, scene in enumerate(data["scene_ids"]):
            batch = {
                key: torch.from_numpy(data[key][i : i + 1].copy())
                for key in data.files
                if key != "scene_ids"
            }
            batch["court_reference_provenance"] = (build_physical_court_provenance(),)
            target = binding.adapter.build_training_batch(batch)
            prediction = BLCSTrajectoryPrediction(
                batch["pred_position"],
                batch.get("pred_velocity"),
                batch.get("physics_parameters"),
                batch.get("event_logits"),
            )
            tracker.update(
                prediction, target, observed=batch["ball_vis"], scene_ids=[str(scene)]
            )
        return cast("dict[str, Any]", tracker.report())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument(
        "--baseline",
        choices=("gt", "geometry", "jitter", "time_shift", "collision_crossing"),
    )
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fitting-clips", type=int, default=12)
    args = parser.parse_args()
    if not os.environ.get("TENNIS_RUN_ID"):
        raise RuntimeError("Evaluation must run through the shared training queue")
    torch.set_num_threads(2)
    result = evaluate(
        data_root=args.data_root,
        output=args.output,
        split=args.split,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        historical=args.historical,
        limit=args.limit,
        fitting_clips=args.fitting_clips,
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "trajectory_rmse_m": result["metrics"]["trajectory_rmse_m"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
