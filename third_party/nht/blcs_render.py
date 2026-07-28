#!/usr/bin/env python3
"""Render a versioned BLCS plan as one composed NHT Gaussian scene per frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from gsplat.nht.deferred_shader import DeferredShaderModule
from gsplat.rendering import rasterization
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.blcs.planner import (  # noqa: E402
    BLCSGaussianScenePlan,
    load_blcs_gaussian_plan,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    GaussianAsset,
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.composition.gaussians import (  # noqa: E402
    GaussianTensorSet,
    compose_gaussians,
    transform_gaussians,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef  # noqa: E402

RENDER_SCHEMA_V1 = "tennis_blcs_nht_render_v1"
RENDER_SCHEMA = "tennis_blcs_nht_render_v2"
FRAME_LABEL_SCHEMA = "tennis_blcs_nht_frame_labels_v2"
VISIBILITY_METHOD = "exact-eval3d-instance-contribution-aov-v1"
DEPTH_PROXY_METHOD = "projected-centre-depth-consistency-v1"
AOV_ALPHA_CONSISTENCY_TOLERANCE = 1.0e-4
DEPTH_PROXY_LIMITATION = (
    "Uses alpha and expected-depth consistency at projected Gaussian centres. "
    "It is retained only for comparison with the exact contribution AOV."
)
_TENSOR_KEYS = {
    "means",
    "quats",
    "scales",
    "opacities",
    "features",
    "instance_ids",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument(
        "--frame-indices",
        required=True,
        help="Strictly increasing comma-separated frame indices.",
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--instance-alpha-threshold", type=float, default=1.0e-4)
    parser.add_argument(
        "--aov-alpha-consistency-tolerance",
        type=float,
        default=AOV_ALPHA_CONSISTENCY_TOLERANCE,
        help=(
            "Maximum absolute alpha drift allowed between the NHT and one-hot AOV "
            "passes. The default covers measured float32 kernel-order drift."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_dirty(path: Path) -> bool:
    output = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
    )
    return bool(output.strip())


def _local_artifact_path(artifact: ArtifactRef) -> Path:
    parsed = urlparse(artifact.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise RuntimeError(
            f"NHT rendering requires a local file artifact: {artifact.uri!r}."
        )
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"Missing NHT artifact: {path}")
    return path


def _content_ref(path: Path, artifact_id: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_file_ref(root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    return {
        "relative_path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_frame_indices(value: str, frame_count: int) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise SystemExit(
            "frame-indices must contain only comma-separated integers."
        ) from error
    if not indices or any(index < 0 or index >= frame_count for index in indices):
        raise SystemExit(f"frame-indices must lie inside [0, {frame_count}).")
    if tuple(sorted(set(indices))) != indices:
        raise SystemExit("frame-indices must be unique and strictly increasing.")
    return indices


def _load_tensor_set(
    asset: GaussianAsset,
    *,
    instance_id: int,
    device: torch.device,
) -> GaussianTensorSet:
    payload = torch.load(
        _local_artifact_path(asset.tensors),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(payload, dict) or set(payload) != _TENSOR_KEYS:
        actual = (
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        )
        raise RuntimeError(
            f"Gaussian tensor keys differ for {asset.asset_id}: {actual}."
        )
    means = payload["means"].float()
    count = int(means.shape[0]) if means.ndim == 2 else -1
    tensor_set = GaussianTensorSet(
        means=means,
        quats=payload["quats"].float(),
        log_scales=payload["scales"].float(),
        opacity_logits=payload["opacities"].float(),
        features=payload["features"].float(),
        instance_ids=torch.full(
            (count,),
            instance_id,
            dtype=torch.int64,
            device=device,
        ),
        appearance_space_sha256=asset.appearance_space_sha256,
    )
    if tensor_set.gaussian_count != asset.gaussian_count:
        raise RuntimeError(
            f"Gaussian count mismatch for {asset.asset_id}: "
            f"{tensor_set.gaussian_count} != {asset.gaussian_count}."
        )
    if tensor_set.feature_dim != asset.feature_dim:
        raise RuntimeError(
            f"Feature dimension mismatch for {asset.asset_id}: "
            f"{tensor_set.feature_dim} != {asset.feature_dim}."
        )
    return tensor_set


def _load_shader(
    appearance: ArtifactRef,
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[DeferredShaderModule, dict[str, object]]:
    payload = torch.load(
        _local_artifact_path(appearance),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or set(payload) != {"config", "state_dict"}:
        raise RuntimeError("NHT appearance payload must contain config and state_dict.")
    config = payload["config"]
    state_dict = payload["state_dict"]
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("NHT appearance config/state_dict must be mappings.")
    if config.get("feature_dim") != feature_dim:
        raise RuntimeError(
            "NHT appearance feature dimension differs from Gaussian tensors."
        )
    shader = DeferredShaderModule(**config).to(device)
    shader.load_state_dict(state_dict, strict=True)
    shader.eval()
    return shader, {str(key): value for key, value in config.items()}


def _camera_tensors(
    plan: BLCSGaussianScenePlan,
    *,
    camera_index: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
    camera = plan.cameras[camera_index]
    height = max(2, round(camera.height * width / camera.width))
    scale_x = width / camera.width
    scale_y = height / camera.height
    camera_to_scene = torch.tensor(
        camera.camera_to_scene,
        dtype=torch.float32,
        device=device,
    ).reshape(4, 4)
    intrinsics = torch.tensor(
        camera.intrinsics,
        dtype=torch.float32,
        device=device,
    ).reshape(3, 3)
    intrinsics[0] *= scale_x
    intrinsics[1] *= scale_y
    return (
        camera_to_scene.unsqueeze(0),
        intrinsics.unsqueeze(0),
        height,
        scale_x,
        scale_y,
    )


def _render_visibility_labels(
    *,
    plan: BLCSGaussianScenePlan,
    frame_index: int,
    camera_index: int,
    scale_x: float,
    scale_y: float,
    composed: GaussianTensorSet,
    info: dict[str, Any],
    alpha: torch.Tensor,
    depth: torch.Tensor,
    instance_contribution: torch.Tensor,
    exact_instance_mask: torch.Tensor,
    alpha_threshold: float,
    instance_alpha_threshold: float,
    depth_relative_tolerance: float,
    depth_absolute_tolerance: float,
) -> list[dict[str, object]]:
    radii = info["radii"]
    projected_depth = info["depths"]
    means2d = info["means2d"]
    if radii.ndim not in {2, 3} or means2d.ndim != 3 or projected_depth.ndim != 2:
        raise RuntimeError(
            "Unexpected unpacked raster metadata shapes: "
            f"radii={tuple(radii.shape)}, means2d={tuple(means2d.shape)}, "
            f"depths={tuple(projected_depth.shape)}."
        )
    projected = radii[0] > 0 if radii.ndim == 2 else (radii[0] > 0).all(dim=-1)
    centres = means2d[0]
    centre_depths = projected_depth[0]
    render_height = int(alpha.shape[1])
    render_width = int(alpha.shape[2])
    alpha_map = alpha[0, ..., 0]
    depth_map = depth[0, ..., 0]

    labels: list[dict[str, object]] = []
    for object_index, assignment in enumerate(plan.assignments):
        instance_id = assignment.instance_id
        gaussian_instance_mask = composed.instance_ids == instance_id
        projected_mask = gaussian_instance_mask & projected
        projected_indices = torch.nonzero(
            projected_mask,
            as_tuple=False,
        ).flatten()
        if int(projected_indices.numel()) == 0:
            depth_consistent_count = 0
        else:
            selected_centres = centres[projected_indices]
            pixel_x = selected_centres[:, 0].round().long().clamp(0, render_width - 1)
            pixel_y = selected_centres[:, 1].round().long().clamp(0, render_height - 1)
            sampled_alpha = alpha_map[pixel_y, pixel_x]
            sampled_depth = depth_map[pixel_y, pixel_x]
            gaussian_depth = centre_depths[projected_indices]
            tolerance = (
                depth_absolute_tolerance
                + depth_relative_tolerance
                * torch.maximum(
                    sampled_depth.abs(),
                    gaussian_depth.abs(),
                )
            )
            consistent = (
                (sampled_alpha >= alpha_threshold)
                & (sampled_depth > 0.0)
                & ((gaussian_depth - sampled_depth).abs() <= tolerance)
            )
            depth_consistent_count = int(consistent.sum())
        present = bool(plan.present[frame_index, object_index])
        geometric_visible = bool(
            plan.camera_geometric_visible[camera_index, frame_index, object_index]
        )
        projected_uv = plan.camera_uv[camera_index, frame_index, object_index]
        exact_mask = exact_instance_mask[0, ..., object_index]
        exact_visible_pixel_count = int(exact_mask.sum())
        contribution = instance_contribution[0, ..., instance_id]
        exact_contribution_mass = float(contribution.sum())
        exact_max_contribution = float(contribution.max())
        depth_proxy_visible = present and depth_consistent_count > 0
        render_visible = present and exact_visible_pixel_count > 0
        labels.append(
            {
                "instance_id": instance_id,
                "asset_id": assignment.selection.entry.asset.asset_id,
                "variant_id": assignment.selection.entry.variant_id,
                "present": present,
                "position_court_m": [
                    float(value)
                    for value in plan.positions_court_m[frame_index, object_index]
                ],
                "position_scene": [
                    float(value)
                    for value in plan.positions_scene[frame_index, object_index]
                ],
                "scene_from_asset": plan.scene_from_asset[
                    frame_index, object_index
                ].tolist(),
                "projected_uv_render_pixels": [
                    float(projected_uv[0] * scale_x),
                    float(projected_uv[1] * scale_y),
                ],
                "camera_depth": float(
                    plan.camera_depth[camera_index, frame_index, object_index]
                ),
                "geometric_visible": geometric_visible,
                "gaussian_count": int(gaussian_instance_mask.sum()),
                "projected_gaussian_count": int(projected_mask.sum()),
                "depth_consistent_gaussian_count": depth_consistent_count,
                "depth_proxy_visible": depth_proxy_visible,
                "exact_instance_alpha_threshold": instance_alpha_threshold,
                "exact_visible_pixel_count": exact_visible_pixel_count,
                "exact_contribution_mass": exact_contribution_mass,
                "exact_max_contribution": exact_max_contribution,
                "render_visible": render_visible,
            }
        )
    return labels


def _render_instance_aov(
    *,
    plan: BLCSGaussianScenePlan,
    composed: GaussianTensorSet,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    primary_alpha: torch.Tensor,
    instance_alpha_threshold: float,
    alpha_consistency_tolerance: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    if int(composed.instance_ids.max()) > plan.num_objects:
        raise RuntimeError("Composed instance ID exceeds the plan object count.")
    one_hot = torch.nn.functional.one_hot(
        composed.instance_ids,
        num_classes=plan.num_objects + 1,
    ).to(dtype=composed.features.dtype)
    instance_contribution, aov_alpha, _ = rasterization(
        means=composed.means,
        quats=composed.quats,
        scales=torch.exp(composed.log_scales),
        opacities=torch.sigmoid(composed.opacity_logits),
        colors=one_hot,
        viewmats=torch.linalg.inv(camera_to_scene),
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1.0e10,
        render_mode="RGB",
        packed=False,
        tile_size=16,
        with_ut=True,
        with_eval3d=True,
        nht=False,
    )
    if instance_contribution.shape[-1] != plan.num_objects + 1:
        raise RuntimeError(
            "Instance AOV channel count differs from background plus plan objects."
        )
    summed_contribution = instance_contribution.sum(dim=-1, keepdim=True)
    contribution_alpha_error = torch.abs(summed_contribution - aov_alpha)
    primary_alpha_error = torch.abs(aov_alpha - primary_alpha)
    metrics = {
        "contribution_sum_vs_aov_alpha_max_abs": float(contribution_alpha_error.max()),
        "contribution_sum_vs_aov_alpha_mean_abs": float(
            contribution_alpha_error.mean()
        ),
        "aov_alpha_vs_nht_alpha_max_abs": float(primary_alpha_error.max()),
        "aov_alpha_vs_nht_alpha_mean_abs": float(primary_alpha_error.mean()),
    }
    if metrics["contribution_sum_vs_aov_alpha_max_abs"] > alpha_consistency_tolerance:
        raise RuntimeError(
            "Instance contributions do not sum to AOV alpha within tolerance: "
            f"{metrics['contribution_sum_vs_aov_alpha_max_abs']:.6g} > "
            f"{alpha_consistency_tolerance:.6g}."
        )
    if metrics["aov_alpha_vs_nht_alpha_max_abs"] > alpha_consistency_tolerance:
        raise RuntimeError(
            "Instance AOV alpha differs from NHT alpha beyond tolerance: "
            f"{metrics['aov_alpha_vs_nht_alpha_max_abs']:.6g} > "
            f"{alpha_consistency_tolerance:.6g}."
        )
    exact_instance_mask = instance_contribution[..., 1:] >= instance_alpha_threshold
    segmentation = torch.argmax(instance_contribution, dim=-1).to(torch.int32)
    segmentation = torch.where(
        aov_alpha[..., 0] >= instance_alpha_threshold,
        segmentation,
        torch.full_like(segmentation, -1),
    )
    return (
        instance_contribution,
        aov_alpha,
        exact_instance_mask,
        segmentation,
        metrics,
    )


def _render_frame(
    *,
    plan: BLCSGaussianScenePlan,
    frame_index: int,
    camera_index: int,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    background: GaussianTensorSet,
    asset_cache: dict[str, GaussianTensorSet],
    shader: DeferredShaderModule,
    alpha_threshold: float,
    instance_alpha_threshold: float,
    aov_alpha_consistency_tolerance: float,
    depth_relative_tolerance: float,
    depth_absolute_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    active_instances = plan.instances_at(frame_index)
    if not active_instances:
        raise RuntimeError(
            f"Frame {frame_index} contains no active object; refusing background-only "
            "fallback."
        )
    transformed = []
    for instance in active_instances:
        local = asset_cache[instance.asset.asset_id]
        with_identity = GaussianTensorSet(
            means=local.means,
            quats=local.quats,
            log_scales=local.log_scales,
            opacity_logits=local.opacity_logits,
            features=local.features,
            instance_ids=torch.full_like(
                local.instance_ids,
                instance.instance_id,
            ),
            appearance_space_sha256=local.appearance_space_sha256,
        )
        transformed.append(
            transform_gaussians(with_identity, instance.scene_from_asset)
        )
    composed = compose_gaussians(background, transformed)

    renderer_api_call_count = 0
    with torch.no_grad():
        rendered_features, alpha, info = rasterization(
            means=composed.means,
            quats=composed.quats,
            scales=torch.exp(composed.log_scales),
            opacities=torch.sigmoid(composed.opacity_logits),
            colors=composed.features,
            viewmats=torch.linalg.inv(camera_to_scene),
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=None,
            near_plane=0.01,
            far_plane=1.0e10,
            render_mode="RGB+ED",
            packed=False,
            tile_size=16,
            with_ut=True,
            with_eval3d=True,
            nht=True,
            center_ray_mode=False,
            ray_dir_scale=shader.ray_dir_scale,
        )
        renderer_api_call_count += 1
        rgb, extras = shader(rendered_features)
        if extras is None or extras.shape[-1] != 1:
            raise RuntimeError(
                f"NHT RGB+ED returned unexpected extras: "
                f"{None if extras is None else tuple(extras.shape)}."
            )
        depth = extras
        rgb = (rgb[..., :3] + torch.ones_like(rgb[..., :3]) * (1.0 - alpha)).clamp(
            0.0, 1.0
        )
        (
            instance_contribution,
            aov_alpha,
            exact_instance_mask,
            instance_segmentation,
            aov_consistency,
        ) = _render_instance_aov(
            plan=plan,
            composed=composed,
            camera_to_scene=camera_to_scene,
            intrinsics=intrinsics,
            width=width,
            height=height,
            primary_alpha=alpha,
            instance_alpha_threshold=instance_alpha_threshold,
            alpha_consistency_tolerance=aov_alpha_consistency_tolerance,
        )
        renderer_api_call_count += 1
        torch.cuda.synchronize()
    if renderer_api_call_count != 2:
        raise RuntimeError(
            "Every v2 frame must use exactly two rasterization API calls."
        )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (rgb, alpha, depth, instance_contribution, aov_alpha)
    ):
        raise RuntimeError(f"Frame {frame_index} contains non-finite render values.")

    labels = _render_visibility_labels(
        plan=plan,
        frame_index=frame_index,
        camera_index=camera_index,
        scale_x=scale_x,
        scale_y=scale_y,
        composed=composed,
        info=info,
        alpha=alpha,
        depth=depth,
        instance_contribution=instance_contribution,
        exact_instance_mask=exact_instance_mask,
        alpha_threshold=alpha_threshold,
        instance_alpha_threshold=instance_alpha_threshold,
        depth_relative_tolerance=depth_relative_tolerance,
        depth_absolute_tolerance=depth_absolute_tolerance,
    )
    rgb_array = rgb[0].mul(255.0).round().to(torch.uint8).cpu().numpy()
    alpha_array = alpha[0, ..., 0].float().cpu().numpy()
    depth_array = depth[0, ..., 0].float().cpu().numpy()
    instance_contribution_array = instance_contribution[0].float().cpu().numpy()
    instance_mask_array = exact_instance_mask[0].cpu().numpy()
    instance_segmentation_array = instance_segmentation[0].cpu().numpy()
    frame_labels: dict[str, object] = {
        "schema": FRAME_LABEL_SCHEMA,
        "plan_fingerprint": plan.plan_fingerprint,
        "scene_id": plan.scene_id,
        "frame_index": frame_index,
        "camera_index": camera_index,
        "camera_id": plan.cameras[camera_index].camera_id,
        "resolution": [width, height],
        "active_instance_ids": [instance.instance_id for instance in active_instances],
        "composed_gaussian_count": composed.gaussian_count,
        "renderer_api_call_count": renderer_api_call_count,
        "rgb": {
            "min": float(rgb.min()),
            "max": float(rgb.max()),
            "mean": float(rgb.mean()),
        },
        "alpha": {
            "min": float(alpha.min()),
            "max": float(alpha.max()),
            "mean": float(alpha.mean()),
        },
        "depth": {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
        },
        "instance_aov": {
            "channel_instance_ids": list(range(plan.num_objects + 1)),
            "background_channel": 0,
            "instance_alpha_threshold": instance_alpha_threshold,
            "alpha_consistency_tolerance": aov_alpha_consistency_tolerance,
            **aov_consistency,
        },
        "visibility_method": VISIBILITY_METHOD,
        "depth_proxy_method": DEPTH_PROXY_METHOD,
        "depth_proxy_limitation": DEPTH_PROXY_LIMITATION,
        "instances": labels,
        "all_finite": True,
    }
    return (
        rgb_array,
        alpha_array,
        depth_array,
        instance_contribution_array,
        instance_mask_array,
        instance_segmentation_array,
        frame_labels,
    )


def _write_frame(
    root: Path,
    *,
    frame_index: int,
    rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    instance_contribution: np.ndarray,
    instance_mask: np.ndarray,
    instance_segmentation: np.ndarray,
    labels: dict[str, object],
) -> dict[str, object]:
    frame_dir = root / "frames" / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True)
    rgb_path = frame_dir / "rgb.png"
    alpha_path = frame_dir / "alpha.npy"
    depth_path = frame_dir / "depth.npy"
    instance_contribution_path = frame_dir / "instance_contribution.npy"
    instance_mask_path = frame_dir / "instance_mask.npy"
    instance_segmentation_path = frame_dir / "instance_segmentation.npy"
    labels_path = frame_dir / "labels.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(alpha_path, alpha, allow_pickle=False)
    np.save(depth_path, depth, allow_pickle=False)
    np.save(
        instance_contribution_path,
        instance_contribution,
        allow_pickle=False,
    )
    np.save(instance_mask_path, instance_mask, allow_pickle=False)
    np.save(
        instance_segmentation_path,
        instance_segmentation,
        allow_pickle=False,
    )
    labels_path.write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "frame_index": frame_index,
        "rgb": _relative_file_ref(root, rgb_path),
        "alpha": _relative_file_ref(root, alpha_path),
        "depth": _relative_file_ref(root, depth_path),
        "instance_contribution": _relative_file_ref(
            root,
            instance_contribution_path,
        ),
        "instance_mask": _relative_file_ref(root, instance_mask_path),
        "instance_segmentation": _relative_file_ref(
            root,
            instance_segmentation_path,
        ),
        "labels": _relative_file_ref(root, labels_path),
    }


def _verify_output(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {RENDER_SCHEMA_V1, RENDER_SCHEMA}:
        raise RuntimeError("Unsupported BLCS NHT render schema.")
    declared = payload.get("render_fingerprint")
    content = dict(payload)
    content.pop("render_fingerprint", None)
    computed = _canonical_sha256(content)
    if declared != computed:
        raise RuntimeError(
            f"Render fingerprint mismatch: declared {declared}, computed {computed}."
        )
    frame_file_names = ["rgb", "alpha", "depth", "labels"]
    if schema == RENDER_SCHEMA:
        frame_file_names.extend(
            [
                "instance_contribution",
                "instance_mask",
                "instance_segmentation",
            ]
        )
    for frame in payload["frames"]:
        for name in frame_file_names:
            record = frame[name]
            relative = PurePosixPath(record["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe render output path: {relative}.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"Missing render output: {relative}.")
            if path.stat().st_size != record["size_bytes"]:
                raise RuntimeError(f"Render output size mismatch: {relative}.")
            if _sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Render output hash mismatch: {relative}.")
    return {
        "render_fingerprint": declared,
        "frame_count": len(payload["frames"]),
        "renderer_api_call_count": payload["renderer"]["api_call_count"],
        "all_finite": payload["all_finite"],
    }


def main() -> None:
    args = _parse_args()
    plan_dir = args.plan_dir.resolve()
    composition_path = args.background_composition.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1:
        raise SystemExit("width must be greater than one.")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise SystemExit("alpha-threshold must lie in [0, 1].")
    if not 0.0 < args.instance_alpha_threshold <= 1.0:
        raise SystemExit("instance-alpha-threshold must lie in (0, 1].")
    if args.aov_alpha_consistency_tolerance < 0.0:
        raise SystemExit("aov-alpha-consistency-tolerance must be non-negative.")
    if args.depth_relative_tolerance < 0.0 or args.depth_absolute_tolerance < 0.0:
        raise SystemExit("Depth tolerances must be non-negative.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    plan = load_blcs_gaussian_plan(plan_dir)
    camera_indices = [
        index
        for index, camera in enumerate(plan.cameras)
        if camera.camera_id == args.camera_id
    ]
    if len(camera_indices) != 1:
        raise SystemExit(f"Expected one plan camera {args.camera_id!r}.")
    camera_index = camera_indices[0]
    frame_indices = _parse_frame_indices(args.frame_indices, plan.num_frames)
    composition = load_gaussian_scene_manifest(composition_path)
    background_asset = composition.background
    verify_local_gaussian_asset(background_asset)
    if plan.registry.appearance_space_sha256 != (
        background_asset.appearance_space_sha256
    ):
        raise SystemExit(
            "BLCS plan and background use different NHT appearance spaces."
        )
    if any(
        assignment.selection.entry.asset.feature_dim != background_asset.feature_dim
        for assignment in plan.assignments
    ):
        raise SystemExit("BLCS plan and background use different feature dimensions.")

    gsplat_path = Path(__file__).resolve().parent / "upstream" / "gsplat"
    renderer_commit = _git_head(gsplat_path)
    if renderer_commit != composition.renderer_commit:
        raise SystemExit(
            f"Renderer commit differs from composition: "
            f"{renderer_commit} != {composition.renderer_commit}."
        )
    if _git_dirty(gsplat_path):
        raise SystemExit("Refusing a modified gsplat renderer checkout.")

    device = torch.device("cuda:0")
    background = _load_tensor_set(
        background_asset,
        instance_id=0,
        device=device,
    )
    asset_cache = {
        entry.asset.asset_id: _load_tensor_set(
            entry.asset,
            instance_id=1,
            device=device,
        )
        for entry in plan.registry.entries
    }
    shader, shader_config = _load_shader(
        background_asset.appearance_payload,
        feature_dim=background.feature_dim,
        device=device,
    )
    camera_to_scene, intrinsics, height, scale_x, scale_y = _camera_tensors(
        plan,
        camera_index=camera_index,
        width=args.width,
        device=device,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        frame_records = []
        render_visible_counts = {
            assignment.instance_id: 0 for assignment in plan.assignments
        }
        depth_proxy_visible_counts = {
            assignment.instance_id: 0 for assignment in plan.assignments
        }
        proxy_exact_disagreement_instance_frames = 0
        aov_alpha_max_errors: list[float] = []
        contribution_sum_max_errors: list[float] = []
        for frame_index in frame_indices:
            (
                rgb,
                alpha,
                depth,
                instance_contribution,
                instance_mask,
                instance_segmentation,
                labels,
            ) = _render_frame(
                plan=plan,
                frame_index=frame_index,
                camera_index=camera_index,
                camera_to_scene=camera_to_scene,
                intrinsics=intrinsics,
                width=args.width,
                height=height,
                scale_x=scale_x,
                scale_y=scale_y,
                background=background,
                asset_cache=asset_cache,
                shader=shader,
                alpha_threshold=args.alpha_threshold,
                instance_alpha_threshold=args.instance_alpha_threshold,
                aov_alpha_consistency_tolerance=(args.aov_alpha_consistency_tolerance),
                depth_relative_tolerance=args.depth_relative_tolerance,
                depth_absolute_tolerance=args.depth_absolute_tolerance,
            )
            for instance in labels["instances"]:
                if instance["render_visible"]:
                    render_visible_counts[int(instance["instance_id"])] += 1
                if instance["depth_proxy_visible"]:
                    depth_proxy_visible_counts[int(instance["instance_id"])] += 1
                if instance["render_visible"] != instance["depth_proxy_visible"]:
                    proxy_exact_disagreement_instance_frames += 1
            aov_alpha_max_errors.append(
                labels["instance_aov"]["aov_alpha_vs_nht_alpha_max_abs"]
            )
            contribution_sum_max_errors.append(
                labels["instance_aov"]["contribution_sum_vs_aov_alpha_max_abs"]
            )
            frame_records.append(
                _write_frame(
                    temporary,
                    frame_index=frame_index,
                    rgb=rgb,
                    alpha=alpha,
                    depth=depth,
                    instance_contribution=instance_contribution,
                    instance_mask=instance_mask,
                    instance_segmentation=instance_segmentation,
                    labels=labels,
                )
            )
        plan_manifest = plan_dir / "manifest.json"
        manifest: dict[str, object] = {
            "schema": RENDER_SCHEMA,
            "plan": {
                **_content_ref(plan_manifest, "blcs-plan-manifest"),
                "plan_fingerprint": plan.plan_fingerprint,
                "scene_id": plan.scene_id,
            },
            "background_composition": {
                **_content_ref(composition_path, "background-composition"),
                "composition_fingerprint": composition.composition_fingerprint,
            },
            "appearance": {
                "artifact_id": background_asset.appearance_payload.artifact_id,
                "sha256": background_asset.appearance_payload.sha256,
                "size_bytes": background_asset.appearance_payload.size_bytes,
            },
            "camera_index": camera_index,
            "camera_id": plan.cameras[camera_index].camera_id,
            "resolution": [args.width, height],
            "frame_indices": list(frame_indices),
            "frames": frame_records,
            "renderer": {
                "backend": "nht-gsplat",
                "commit": renderer_commit,
                "api_call_count": 2 * len(frame_indices),
                "api_calls_per_frame": 2,
                "nht_rgb_ed_api_calls_per_frame": 1,
                "instance_eval3d_aov_api_calls_per_frame": 1,
                "nht_render_mode": "RGB+ED",
                "instance_aov_render_mode": "RGB",
                "rgb_ed_internal_depth_auxiliary_pass": True,
                "shader_config": shader_config,
            },
            "visibility": {
                "method": VISIBILITY_METHOD,
                "instance_aov_channel_instance_ids": list(range(plan.num_objects + 1)),
                "background_channel": 0,
                "instance_alpha_threshold": args.instance_alpha_threshold,
                "aov_alpha_consistency_tolerance": (
                    args.aov_alpha_consistency_tolerance
                ),
                "aov_alpha_vs_nht_alpha_max_abs": max(aov_alpha_max_errors),
                "contribution_sum_vs_aov_alpha_max_abs": max(
                    contribution_sum_max_errors
                ),
                "alpha_threshold": args.alpha_threshold,
                "depth_relative_tolerance": args.depth_relative_tolerance,
                "depth_absolute_tolerance": args.depth_absolute_tolerance,
                "depth_proxy_method": DEPTH_PROXY_METHOD,
                "depth_proxy_limitation": DEPTH_PROXY_LIMITATION,
                "exact_per_pixel_instance_mask": True,
                "render_visible_frame_counts": {
                    str(key): value for key, value in render_visible_counts.items()
                },
                "depth_proxy_visible_frame_counts": {
                    str(key): value for key, value in depth_proxy_visible_counts.items()
                },
                "proxy_exact_disagreement_instance_frames": (
                    proxy_exact_disagreement_instance_frames
                ),
            },
            "rgb_overlay_used": False,
            "all_finite": True,
            "acceptance_scope": (
                "native-composition RGB and exact instance contribution masks; "
                "asset provenance and alignment acceptance are verified upstream"
            ),
        }
        manifest["render_fingerprint"] = _canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        report = _verify_output(output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
