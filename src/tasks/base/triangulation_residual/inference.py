"""Strict checkpoint inference and before/after real-clip diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.base.triangulation_residual.contracts import GeometryInput
from src.tasks.base.triangulation_residual.geometry import prepare_geometry
from src.tasks.base.triangulation_residual.model import GeometricResidualModel
from src.tasks.base.triangulation_residual.training import ResidualLightningModule
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.geometry.triangulation import project_multiview
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import load_and_validate_checkpoint
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES
from src.utils.video.windows import build_window_starts

PATH_BOUNDARY = NonHydraPathBoundary(
    name="base.triangulation_residual.inference",
    fields=(
        BoundaryPathField(
            "source",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.ANY,
            must_exist=True,
        ),
        BoundaryPathField(
            "clip",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
        ),
        BoundaryPathField(
            "output", PathRole.ARTIFACT, PathDirection.OUTPUT, PathKind.DIRECTORY
        ),
    ),
)


def statistics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def predict_geometry(
    model: GeometricResidualModel,
    geometry: GeometryInput,
    *,
    window_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if window_size < 2:
        raise ValueError("Window must have at least two frames")
    total = len(geometry.root_init_m)
    width = min(window_size, total)
    stride = max(1, width // 2)
    starts = build_window_starts(
        frame_count=total, sequence_length=width, stride=stride
    )
    weights = 1 - np.abs((np.arange(width) - (width - 1) / 2) / ((width + 1) / 2))
    denominator = np.zeros(total)
    accumulated: dict[str, np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for start in starts:
            sl = slice(start, start + width)
            x = torch.from_numpy(geometry.features[:, sl].copy())[None].to(device)
            valid = torch.from_numpy(geometry.view_valid[:, sl].copy())[None].to(device)
            time = torch.from_numpy(geometry.time_positions[sl].copy())[None].to(device)
            output = model(x, valid, time)
            for key, tensor in output.items():
                value = tensor[0].float().cpu().numpy()
                if key not in accumulated:
                    accumulated[key] = np.zeros((total, *value.shape[1:]))
                accumulated[key][sl] += value * weights.reshape(
                    (width,) + (1,) * (value.ndim - 1)
                )
            denominator[sl] += weights
    if (denominator <= 0).any():
        raise RuntimeError("Sliding windows left uncovered frames")
    return {
        key: (value / denominator.reshape((total,) + (1,) * (value.ndim - 1))).astype(
            np.float32
        )
        for key, value in accumulated.items()
    }


def compare_reconstruction(
    initial: np.ndarray,
    predicted: np.ndarray,
    observations: np.ndarray,
    scores: np.ndarray,
    matrices: np.ndarray,
    initial_valid: np.ndarray,
    task: str,
) -> dict[str, Any]:
    # P,T,J,3 -> P,T,J,V,2 -> P,V,T,J,2
    before, before_depth = project_multiview(initial, matrices)
    after, after_depth = project_multiview(predicted, matrices)
    before, after = before.transpose(0, 3, 1, 2, 4), after.transpose(0, 3, 1, 2, 4)
    valid = np.isfinite(observations).all(-1) & (scores >= 0.3)
    eligible_count = int(valid.sum())
    negative_before = int((valid & (before_depth.transpose(0, 3, 1, 2) <= 0)).sum())
    negative_after = int((valid & (after_depth.transpose(0, 3, 1, 2) <= 0)).sum())
    valid &= (before_depth.transpose(0, 3, 1, 2) > 0) & (
        after_depth.transpose(0, 3, 1, 2) > 0
    )
    eb, ea = (
        np.linalg.norm(before - observations, axis=-1),
        np.linalg.norm(after - observations, axis=-1),
    )
    measured = valid & initial_valid[:, None]
    output: dict[str, Any] = {
        "independent_3d_ground_truth": False,
        "metric_definition": "Euclidean pixel residual to supplied 2D observations, same positive-depth mask for before/after; not 3D accuracy",
        "observation_count": int(valid.sum()),
        "eligible_observation_count": eligible_count,
        "before_nonpositive_depth_count": negative_before,
        "after_nonpositive_depth_count": negative_after,
        "initial_raw_valid_percent": float(initial_valid.mean() * 100),
        "before_reprojection_px": statistics(eb[valid]),
        "after_reprojection_px": statistics(ea[valid]),
        "before_raw_valid_reprojection_px": statistics(eb[measured]),
        "after_raw_valid_reprojection_px": statistics(ea[measured]),
        "correction_m": statistics(np.linalg.norm(predicted - initial, axis=-1)),
        "per_camera": [
            {
                "camera_index": v,
                "before_px": statistics(eb[:, v][valid[:, v]]),
                "after_px": statistics(ea[:, v][valid[:, v]]),
            }
            for v in range(len(matrices))
        ],
    }
    if task == "plcs":
        edges = np.array(COCO17_BONE_LENGTH_EDGES)
        for label, points in (("before", initial), ("after", predicted)):
            lengths = np.linalg.norm(
                points[:, :, edges[:, 0]] - points[:, :, edges[:, 1]], axis=-1
            )
            limbs = lengths[:, :, 4:]
            output[f"{label}_limb_instances_over_1m"] = int((limbs > 1).sum())
            output[f"{label}_frames_with_limb_over_1m"] = int((limbs > 1).any(-1).sum())
            output[f"{label}_max_limb_m"] = float(limbs.max())
            deviation = np.abs(limbs - np.median(limbs, axis=1, keepdims=True))
            output[f"{label}_limb_temporal_deviation_m"] = statistics(deviation)
    else:
        for label, points in (("before", initial), ("after", predicted)):
            output[f"{label}_z_m"] = statistics(points[..., 2])
            output[f"{label}_below_ground_percent"] = float(
                np.mean(points[..., 2] < -0.1) * 100
            )
        for label, status_mask in (
            ("observed", scores >= 0.99),
            ("interpolated", (scores > 0.49) & (scores < 0.51)),
        ):
            output[f"{label}_before_px"] = statistics(eb[valid & status_mask])
            output[f"{label}_after_px"] = statistics(ea[valid & status_mask])
    return output


def evaluate_clip(
    task: str,
    checkpoint: Path,
    clip_dir: Path,
    output_dir: Path,
    *,
    device: str,
    render: bool = True,
) -> dict[str, Any]:
    raw = load_and_validate_checkpoint(checkpoint)
    config = OmegaConf.create(raw["hyper_parameters"]["config"])
    module = ResidualLightningModule(config)
    module.on_load_checkpoint(dict(raw))
    if module.residual_config.task != task:
        raise ValueError("Checkpoint task and requested inference task disagree")
    module.load_state_dict(raw["state_dict"], strict=True)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA inference requested but CUDA is unavailable")
    model = module.model.to(target_device).eval()
    if task == "plcs":
        from src.tasks.plcs.triangulation_residual.real_clip import (
            load_real_clip as load_plcs_clip,
        )

        scenes = load_plcs_clip(clip_dir)
    elif task == "blcs":
        from src.tasks.blcs.triangulation_residual.real_clip import (
            load_real_clip as load_blcs_clip,
        )

        scenes = [load_blcs_clip(clip_dir)]
    else:
        raise ValueError("Unknown inference task")
    cfg = module.residual_config
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "initial_world_m",
            "raw_initial_world_m",
            "pred_world_m",
            "initial_valid",
            "observations_px",
            "scores",
            "root_residual_m",
            "relative_residual_m",
            "preprojected_uv",
            "reprojection_residual_uv",
        )
    }
    for scene in scenes:
        geometry = prepare_geometry(
            scene.observations_px,
            scene.scores,
            scene.court_px,
            scene.court_scores,
            scene.rig,
            root_indices=cfg.root_indices,
            fps=scene.fps,
            min_score=cfg.initializer.min_score,
            refinement_steps=cfg.initializer.refinement_steps,
        )
        # Preserve native observations and cover the same number of seconds as training.
        window = max(
            4, round(cfg.data.sequence_length * scene.fps / cfg.data.target_fps)
        )
        result = predict_geometry(
            model, geometry, window_size=window, device=target_device
        )
        root_delta = result["root_residual" if task == "plcs" else "position_residual"]
        relative_delta = (
            result["relative_residual"]
            if task == "plcs"
            else np.zeros_like(geometry.relative_init_m)
        )
        world = (
            (geometry.root_init_m + root_delta)[:, None]
            + geometry.relative_init_m
            + relative_delta
        )
        if not np.isfinite(world).all():
            raise FloatingPointError("Nonfinite residual inference")
        values = {
            "initial_world_m": geometry.init_world_m,
            "raw_initial_world_m": geometry.raw_init_world_m,
            "pred_world_m": world,
            "initial_valid": geometry.init_valid,
            "observations_px": scene.observations_px,
            "scores": scene.scores,
            "root_residual_m": root_delta,
            "relative_residual_m": relative_delta,
            "preprojected_uv": geometry.reprojected_uv,
            "reprojection_residual_uv": geometry.residual_uv,
        }
        for key, value in values.items():
            arrays[key].append(value)
    payload = {key: np.stack(value) for key, value in arrays.items()}
    payload["projection_matrices"] = scenes[0].rig.matrices
    payload["fps"] = np.array(scenes[0].fps)
    metrics = compare_reconstruction(
        payload["initial_world_m"],
        payload["pred_world_m"],
        payload["observations_px"],
        payload["scores"],
        payload["projection_matrices"],
        payload["initial_valid"],
        task,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.npz"
    cast(Callable[..., None], np.savez_compressed)(output_path, **payload)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )
    metadata = {
        "task": task,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "contract": raw["geometric_residual_contract"],
        "clip": str(clip_dir.resolve()),
        "inference_precision": "float32",
        "window_size": window,
        "native_fps": scenes[0].fps,
        "selection_uses_real_clip": False,
        "source_metadata": [s.metadata for s in scenes],
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    if render:
        from src.tasks.base.triangulation_residual.visualization import (
            render_comparison,
        )

        render_comparison(clip_dir, output_path, output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("plcs", "blcs"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    source_path = args.checkpoint if args.checkpoint is not None else args.run_dir
    if not all(path.is_absolute() for path in (source_path, args.clip, args.output)):
        raise ValueError("Source, clip and output must be explicit absolute paths")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=args.clip.parent.resolve(),
        checkpoint_root=source_path.parent.resolve(),
        artifact_root=args.output.parent.resolve(),
        output_root=args.output.parent.resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    validated = PATH_BOUNDARY.validate(
        {"source": source_path, "clip": args.clip, "output": args.output},
        resolver=PathResolver(roots),
    )
    source_path = validated.declared("source").path
    checkpoint = (
        source_path
        if args.checkpoint is not None
        else Path(
            json.loads((source_path / "evaluation.json").read_text())["checkpoint"]
        )
    )
    if args.run_dir is not None and not checkpoint.resolve().is_relative_to(
        source_path
    ):
        raise ValueError(
            "Selected best checkpoint is outside the declared run directory"
        )
    evaluate_clip(
        args.task,
        checkpoint,
        validated.declared("clip").path,
        validated.declared("output").path,
        device=args.device,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
