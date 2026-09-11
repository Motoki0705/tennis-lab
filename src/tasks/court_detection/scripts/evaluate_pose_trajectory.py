"""Evaluate one complete Synthetic Court V3 trajectory with a pose checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.court_detection.data.contracts import CourtPoseAuthority
from src.tasks.court_detection.geometry.pose import (
    CourtDecodedPose,
    build_pose_target,
    canonical_semantic_court_points_batched,
    project_predicted_canonical_points,
)
from src.tasks.court_detection.inference import CourtPosePredictor
from src.tasks.court_detection.model_io.contracts import CourtKeypointPrediction
from src.tasks.court_detection.training.losses import rotation_geodesic_radians
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

_GIB = 1024**3

PATH_BOUNDARY = NonHydraPathBoundary(
    name="court_detection.evaluate_pose_trajectory",
    fields=(
        BoundaryPathField(
            "checkpoint",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "runtime_config",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "runtime_project_root",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "dataset_root",
            PathRole.DATA,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "output_dir",
            PathRole.OUTPUT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
        ),
    ),
)


def main() -> int:
    args = _parse_args()
    checkpoint = _absolute_path(args.checkpoint, name="checkpoint")
    runtime_config_path = _absolute_path(args.runtime_config, name="runtime_config")
    runtime_project_root = _absolute_path(
        args.runtime_project_root,
        name="runtime_project_root",
    )
    dataset_root = _absolute_path(args.dataset_root, name="dataset_root")
    output_dir = _absolute_path(args.output_dir, name="output_dir")
    resolver = _runtime_resolver(
        checkpoint=checkpoint,
        runtime_project_root=runtime_project_root,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )
    paths = PATH_BOUNDARY.validate(
        {
            "checkpoint": checkpoint,
            "runtime_config": runtime_config_path,
            "runtime_project_root": runtime_project_root,
            "dataset_root": dataset_root,
            "output_dir": output_dir,
        },
        resolver=resolver,
    )
    checkpoint = paths.declared("checkpoint").path
    runtime_config_path = paths.declared("runtime_config").path
    runtime_project_root = paths.declared("runtime_project_root").path
    dataset_root = paths.declared("dataset_root").path
    output_dir = paths.declared("output_dir").path
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    memory = assess_cpu_checkpoint_memory(checkpoint)
    if memory["safe"] is not True:
        raise MemoryError(
            "Refusing CPU checkpoint load because available memory is below the "
            f"conservative threshold: {json.dumps(memory, sort_keys=True)}"
        )

    manifest = _mapping(
        json.loads((dataset_root / "dataset.json").read_text()),
        name="dataset manifest",
    )
    samples = _select_trajectory(
        manifest,
        trajectory_group_id=args.trajectory_group_id,
    )
    runtime_config = OmegaConf.load(runtime_config_path)
    runtime_config.paths.project_root = str(runtime_project_root)
    load_started = time.perf_counter()
    predictor = CourtPosePredictor.load_from_checkpoint(
        checkpoint,
        resolver=resolver,
        device="cpu",
        subpixel_refine=True,
        max_peaks=4,
        config=runtime_config,
    )
    load_seconds = time.perf_counter() - load_started

    rows: list[dict[str, object]] = []
    inference_seconds: list[float] = []
    for sample in samples:
        image_path = dataset_root / _text(sample["rgb"], name="sample.rgb")
        labels_path = dataset_root / _text(sample["labels"], name="sample.labels")
        labels = _mapping(json.loads(labels_path.read_text()), name="sample labels")
        target, target_court_id = _target_pose(labels)
        image = _load_rgb(image_path, sample=sample)
        started = time.perf_counter()
        prediction = predictor.predict(image)
        inference_seconds.append(time.perf_counter() - started)
        rows.append(
            _measure_sample(
                sample,
                labels=labels,
                prediction=prediction.pose,
                dense=prediction.dense,
                target=target,
                target_court_id=target_court_id,
            )
        )

    summary = _summarize(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "dataset_root": str(dataset_root),
            "runtime_config": str(runtime_config_path),
            "runtime_project_root": str(runtime_project_root),
            "trajectory_group_id": args.trajectory_group_id,
            "trajectory_id": rows[0]["trajectory_id"],
            "split": rows[0]["split"],
            "sample_count": len(rows),
            "checkpoint_load_seconds": load_seconds,
            "inference_seconds_total": sum(inference_seconds),
            "inference_seconds_mean": statistics.fmean(inference_seconds),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "memory_safety_before_load": memory,
        }
    )
    output_dir.mkdir(parents=True)
    (output_dir / "measurements.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n"
    )
    _plot_metrics(rows, output_dir / "trajectory_metrics.png")
    _plot_overlays(dataset_root, rows, output_dir / "trajectory_overlays.png")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def assess_cpu_checkpoint_memory(checkpoint: Path) -> dict[str, object]:
    """Conservatively gate the transient model + state-dict loading peak."""

    memory = _linux_memory_bytes()
    checkpoint_bytes = checkpoint.stat().st_size
    required = max(8 * _GIB, checkpoint_bytes * 4 + 2 * _GIB)
    return {
        "safe": memory["MemAvailable"] >= required,
        "available_bytes": memory["MemAvailable"],
        "required_bytes": required,
        "swap_free_bytes": memory["SwapFree"],
    }


def _measure_sample(
    sample: Mapping[str, object],
    *,
    labels: Mapping[str, object],
    prediction: CourtDecodedPose,
    dense: Mapping[str, object],
    target: Any,
    target_court_id: str,
) -> dict[str, object]:
    target_pose = CourtDecodedPose(
        translation_m=target.translation_m.unsqueeze(0).to(dtype=torch.float32),
        rotation=target.rotation.unsqueeze(0).to(dtype=torch.float32),
        focal_px=torch.exp(target.log_focal).reshape(1).to(dtype=torch.float32),
        log_focal=target.log_focal.reshape(1).to(dtype=torch.float32),
    )
    rotation_error = rotation_geodesic_radians(
        prediction.rotation,
        target_pose.rotation,
    ) * (180.0 / math.pi)
    translation_error = torch.linalg.vector_norm(
        prediction.translation_m - target_pose.translation_m,
        dim=-1,
    )
    focal_relative_error = torch.abs(
        prediction.focal_px - target_pose.focal_px
    ) / target_pose.focal_px

    semantic = target.semantic_to_physical.unsqueeze(0)
    canonical = canonical_semantic_court_points_batched(semantic)
    principal = target.intrinsics[:2, 2].to(dtype=torch.float32).unsqueeze(0)
    predicted_projection = project_predicted_canonical_points(
        prediction,
        canonical,
        principal,
    )
    target_projection = project_predicted_canonical_points(
        target_pose,
        canonical,
        principal,
    )
    visible = _target_visibility(labels, target_court_id)
    pose_distances = torch.linalg.vector_norm(
        predicted_projection.points_xy - target_projection.points_xy,
        dim=-1,
    )[0]
    dense_error = _dense_keypoint_error(
        dense.get("kp"),
        target_projection.points_xy[0],
        visible,
    )
    visible_pose = pose_distances[visible]
    invalid_depth = predicted_projection.depth_m[0, visible] <= 0.1
    return {
        "sample_id": _text(sample["sample_id"], name="sample.sample_id"),
        "trajectory_id": _text(sample["trajectory_id"], name="sample.trajectory_id"),
        "trajectory_frame_index": _integer(
            sample["trajectory_frame_index"],
            name="sample.trajectory_frame_index",
        ),
        "height": _integer(sample["height"], name="sample.height"),
        "width": _integer(sample["width"], name="sample.width"),
        "split": _text(sample["split"], name="sample.split"),
        "rgb": _text(sample["rgb"], name="sample.rgb"),
        "translation_error_m": float(translation_error[0]),
        "rotation_error_deg": float(rotation_error[0]),
        "focal_relative_error": float(focal_relative_error[0]),
        "pose_reprojection_mean_px": float(visible_pose.mean()),
        "pose_reprojection_median_px": float(visible_pose.median()),
        "pose_invalid_depth_fraction": float(invalid_depth.float().mean()),
        "dense_kp_mean_px": dense_error,
        "visible_keypoint_count": int(visible.sum()),
        "predicted_translation_m": prediction.translation_m[0].tolist(),
        "target_translation_m": target_pose.translation_m[0].tolist(),
        "predicted_rotation": prediction.rotation[0].tolist(),
        "target_rotation": target_pose.rotation[0].tolist(),
        "predicted_focal_px": float(prediction.focal_px[0]),
        "target_focal_px": float(target_pose.focal_px[0]),
        "predicted_pose_keypoints_xy": predicted_projection.points_xy[0].tolist(),
        "target_keypoints_xy": target_projection.points_xy[0].tolist(),
        "target_visible": visible.tolist(),
        "dense_keypoints_xy": _dense_points(dense.get("kp")),
    }


def _target_pose(labels: Mapping[str, object]) -> tuple[Any, str]:
    target_court = _mapping(labels["target_court"], name="target_court")
    authority = CourtPoseAuthority(
        source_schema="canonical_court_dataset_v3",
        camera=SceneCamera.from_dict(labels["camera"]),
        target_court=TargetCourtBinding.from_dict(target_court["binding"]),
    )
    target = build_pose_target(authority)
    return target, authority.target_court.court_instance_id


def _target_visibility(labels: Mapping[str, object], court_id: str) -> torch.Tensor:
    projection = _mapping(labels["projection"], name="projection")
    courts = _sequence(projection["courts"], name="projection.courts")
    court = next(
        _mapping(value, name="projection.court")
        for value in courts
        if _mapping(value, name="projection.court")["court_instance_id"] == court_id
    )
    classes = _sequence(court["classes"], name="projection.court.classes")
    return torch.tensor(
        [bool(_mapping(value, name="projection.class")["renderer_visible"]) for value in classes],
        dtype=torch.bool,
    )


def _dense_keypoint_error(
    value: object,
    target_points: torch.Tensor,
    visible: torch.Tensor,
) -> float | None:
    if not isinstance(value, CourtKeypointPrediction):
        return None
    distances: list[torch.Tensor] = []
    for channel in range(target_points.shape[0]):
        if not bool(visible[channel]):
            continue
        accepted = value.keypoints[channel, value.valid[channel]]
        if accepted.numel() > 0:
            distances.append(
                torch.linalg.vector_norm(accepted - target_points[channel], dim=-1).min()
            )
    return float(torch.stack(distances).mean()) if distances else None


def _dense_points(value: object) -> list[list[float] | None] | None:
    if not isinstance(value, CourtKeypointPrediction):
        return None
    result: list[list[float] | None] = []
    for points, valid in zip(value.keypoints, value.valid, strict=True):
        accepted = points[valid]
        result.append(accepted[0].tolist() if accepted.numel() else None)
    return result


def _summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key in (
        "translation_error_m",
        "rotation_error_deg",
        "focal_relative_error",
        "pose_reprojection_mean_px",
        "pose_invalid_depth_fraction",
        "dense_kp_mean_px",
    ):
        values = [
            _number(row[key], name=key)
            for row in rows
            if row.get(key) is not None
        ]
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_p95"] = sorted(values)[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
    return summary


def _plot_metrics(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    frame = [
        _integer(row["trajectory_frame_index"], name="trajectory_frame_index")
        for row in rows
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    target_xy = torch.tensor([row["target_translation_m"] for row in rows])[:, :2]
    predicted_xy = torch.tensor([row["predicted_translation_m"] for row in rows])[:, :2]
    axes[0, 0].plot(target_xy[:, 0], target_xy[:, 1], label="target")
    axes[0, 0].plot(predicted_xy[:, 0], predicted_xy[:, 1], label="prediction")
    axes[0, 0].set(title="Camera center in canonical court", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    for axis, key, title in (
        (axes[0, 1], "translation_error_m", "Translation L2 [m]"),
        (axes[1, 0], "rotation_error_deg", "Rotation geodesic [deg]"),
        (axes[1, 1], "focal_relative_error", "Focal relative error"),
    ):
        axis.plot(frame, [_number(row[key], name=key) for row in rows])
        axis.set(title=title, xlabel="trajectory frame")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_overlays(
    dataset_root: Path,
    rows: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    indices = sorted({round(value) for value in torch.linspace(0, len(rows) - 1, 6).tolist()})
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, index in zip(axes.flat, indices, strict=True):
        row = rows[index]
        image_path = dataset_root / cast(str, row["rgb"])
        axis.imshow(_load_rgb(image_path, sample=row))
        visible = torch.tensor(row["target_visible"], dtype=torch.bool)
        target = torch.tensor(row["target_keypoints_xy"])[visible]
        predicted = torch.tensor(row["predicted_pose_keypoints_xy"])[visible]
        axis.scatter(target[:, 0], target[:, 1], c="lime", s=24, label="GT")
        axis.scatter(predicted[:, 0], predicted[:, 1], c="orange", s=28, marker="x", label="pose")
        dense = row["dense_keypoints_xy"]
        if isinstance(dense, list):
            points = torch.tensor([value for value in dense if value is not None])
            if points.numel():
                axis.scatter(points[:, 0], points[:, 1], facecolors="none", edgecolors="cyan", s=30, label="dense KP")
        axis.set_title(f"frame {row['trajectory_frame_index']}")
        axis.set_axis_off()
    axes.flat[0].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _select_trajectory(
    manifest: Mapping[str, object],
    *,
    trajectory_group_id: str,
) -> list[Mapping[str, object]]:
    samples = _sequence(manifest["samples"], name="dataset.samples")
    selected = [
        _mapping(value, name="dataset.sample")
        for value in samples
        if _mapping(value, name="dataset.sample").get("trajectory_group_id")
        == trajectory_group_id
    ]
    if not selected:
        raise ValueError(f"No accepted samples for trajectory {trajectory_group_id!r}.")
    selected.sort(
        key=lambda value: _integer(
            value["trajectory_frame_index"],
            name="sample.trajectory_frame_index",
        )
    )
    return selected


def _load_rgb(image_path: Path, *, sample: Mapping[str, object]) -> Image.Image:
    """Load the strict float32 RGB payload published by Synthetic Court V3."""

    if image_path.suffix != ".npy":
        raise ValueError(f"Synthetic Court V3 RGB must be a .npy file: {image_path}")
    rgb = np.load(image_path, allow_pickle=False)
    expected = (
        _integer(sample["height"], name="sample.height"),
        _integer(sample["width"], name="sample.width"),
        3,
    )
    if rgb.dtype != np.float32 or rgb.shape != expected:
        raise ValueError(
            "Synthetic Court V3 RGB must be float32 [H,W,3] matching the manifest."
        )
    if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("Synthetic Court V3 RGB must be finite and remain in [0,1].")
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def _runtime_resolver(
    *,
    checkpoint: Path,
    runtime_project_root: Path,
    dataset_root: Path,
    output_dir: Path,
) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=runtime_project_root,
            data_root=dataset_root,
            checkpoint_root=checkpoint.parent,
            artifact_root=output_dir.parent,
            output_root=output_dir.parent,
            cache_root=(runtime_project_root / ".cache").resolve(),
            external_asset_root=(runtime_project_root / "third_party").resolve(),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--runtime-project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-group-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _absolute_path(value: Path, *, name: str) -> Path:
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {value}")
    return value.resolve(strict=False)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _linux_memory_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", maxsplit=1)
        if name in {"MemAvailable", "SwapFree"}:
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise RuntimeError(f"Unexpected /proc/meminfo field: {line}")
            values[name] = int(fields[0]) * 1024
    if set(values) != {"MemAvailable", "SwapFree"}:
        raise RuntimeError("/proc/meminfo lacks MemAvailable or SwapFree.")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
