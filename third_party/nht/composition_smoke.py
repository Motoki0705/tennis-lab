#!/usr/bin/env python3
"""Render one real NHT background plus a moved Gaussian asset in one CUDA call."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.nht.deferred_shader import DeferredShaderModule
from gsplat.rendering import rasterization
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.alignment.scene_provider.bundle import (  # noqa: E402
    load_scene_provider_bundle,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    ASSET_COORDINATE_FRAME,
    GAUSSIAN_ASSET_SCHEMA,
    METRE_UNIT,
    NHT_APPEARANCE_MODEL,
    NHT_TENSOR_ENCODING,
    SCENE_COORDINATE_FRAME,
    SCENE_UNIT,
    GaussianAsset,
    GaussianInstance,
    GaussianSceneComposition,
    write_gaussian_scene_manifest,
)
from src.synthetic_data_generation.composition.gaussians import (  # noqa: E402
    GaussianTensorSet,
    compose_gaussians,
    transform_gaussians,
)
from src.synthetic_data_generation.scene_contract import (  # noqa: E402
    ArtifactRef,
    SimilarityTransform,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provider-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", default="frame_000000")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--asset-count", type=int, default=512)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _artifact(
    artifact_id: str,
    path: Path,
    *,
    published_path: Path | None = None,
) -> ArtifactRef:
    uri_path = published_path if published_path is not None else path
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=uri_path.resolve().as_uri(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _raw_tensor_payload(gaussians: GaussianTensorSet) -> dict[str, torch.Tensor]:
    return {
        "means": gaussians.means.detach().cpu(),
        "quats": gaussians.quats.detach().cpu(),
        "scales": gaussians.log_scales.detach().cpu(),
        "opacities": gaussians.opacity_logits.detach().cpu(),
        "features": gaussians.features.detach().cpu(),
        "instance_ids": gaussians.instance_ids.detach().cpu(),
    }


def _subset(
    source: GaussianTensorSet,
    indices: torch.Tensor,
    *,
    instance_id: int,
) -> GaussianTensorSet:
    count = int(indices.numel())
    return GaussianTensorSet(
        means=source.means[indices].clone(),
        quats=source.quats[indices].clone(),
        log_scales=source.log_scales[indices].clone(),
        opacity_logits=source.opacity_logits[indices].clone(),
        features=source.features[indices].clone(),
        instance_ids=torch.full(
            (count,),
            instance_id,
            dtype=torch.int64,
            device=source.means.device,
        ),
        appearance_space_sha256=source.appearance_space_sha256,
    )


def _select_visible_cluster(
    source: GaussianTensorSet,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    width: int,
    height: int,
    asset_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    scene_to_camera = torch.linalg.inv(camera_to_scene)
    camera_points = source.means @ scene_to_camera[:3, :3].T + scene_to_camera[:3, 3]
    depth = camera_points[:, 2]
    pixels = torch.empty(
        (source.gaussian_count, 2),
        dtype=source.means.dtype,
        device=source.means.device,
    )
    pixels[:, 0] = intrinsics[0, 0] * camera_points[:, 0] / depth + intrinsics[0, 2]
    pixels[:, 1] = intrinsics[1, 1] * camera_points[:, 1] / depth + intrinsics[1, 2]
    visible = (
        (depth > 0.05)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
    if int(visible_indices.numel()) < asset_count:
        raise RuntimeError(
            f"Only {visible_indices.numel()} projected Gaussians; "
            f"cannot select asset_count={asset_count}."
        )

    center = pixels.new_tensor((width / 2.0, height / 2.0))
    center_distance = torch.linalg.vector_norm(
        (pixels[visible_indices] - center) / pixels.new_tensor((width, height)),
        dim=-1,
    )
    opacity_penalty = 0.05 * (
        1.0 - torch.sigmoid(source.opacity_logits[visible_indices])
    )
    anchor_index = visible_indices[torch.argmin(center_distance + opacity_penalty)]
    anchor = source.means[anchor_index]
    distances = torch.linalg.vector_norm(source.means - anchor, dim=-1)
    asset_indices = torch.topk(
        distances,
        k=asset_count,
        largest=False,
        sorted=True,
    ).indices
    radius = float(torch.quantile(distances[asset_indices], 0.95))
    if not math.isfinite(radius) or radius <= 1.0e-6:
        raise RuntimeError(f"Selected asset radius is invalid: {radius}.")
    return asset_indices, anchor, radius


def _asset_local_gaussians(
    selected: GaussianTensorSet,
    *,
    anchor: torch.Tensor,
    radius: float,
) -> GaussianTensorSet:
    radius_tensor = selected.means.new_tensor(radius)
    return GaussianTensorSet(
        means=(selected.means - anchor) / radius_tensor,
        quats=selected.quats,
        log_scales=selected.log_scales - torch.log(radius_tensor),
        opacity_logits=selected.opacity_logits,
        features=selected.features,
        instance_ids=selected.instance_ids,
        appearance_space_sha256=selected.appearance_space_sha256,
    )


def _rotation_z(angle_degrees: float) -> tuple[float, ...]:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine,
        -sine,
        0.0,
        sine,
        cosine,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _visibility_labels(
    *,
    info: dict[str, Any],
    instance_ids: torch.Tensor,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    radii = info["radii"]
    if radii.ndim == 2:
        visible = radii[0] > 0
    elif radii.ndim == 3:
        visible = (radii[0] > 0).all(dim=-1)
    else:
        raise RuntimeError(f"Unexpected radii shape: {tuple(radii.shape)}.")
    means2d = info["means2d"]
    if means2d.ndim != 3 or means2d.shape[0] != 1:
        raise RuntimeError(f"Unexpected means2d shape: {tuple(means2d.shape)}.")
    projected = means2d[0]

    labels: list[dict[str, object]] = []
    for instance_id in sorted(int(value) for value in torch.unique(instance_ids)):
        instance_mask = instance_ids == instance_id
        instance_visible = instance_mask & visible
        visible_count = int(instance_visible.sum())
        if visible_count == 0:
            bbox = None
            centroid = None
        else:
            points = projected[instance_visible]
            minimum = points.min(dim=0).values
            maximum = points.max(dim=0).values
            bbox = [
                float(minimum[0].clamp(0, width - 1)),
                float(minimum[1].clamp(0, height - 1)),
                float(maximum[0].clamp(0, width - 1)),
                float(maximum[1].clamp(0, height - 1)),
            ]
            centroid_values = points.mean(dim=0)
            centroid = [float(centroid_values[0]), float(centroid_values[1])]
        labels.append(
            {
                "instance_id": instance_id,
                "class": "background" if instance_id == 0 else "smoke-patch",
                "gaussian_count": int(instance_mask.sum()),
                "visible_gaussian_count": visible_count,
                "projected_center_bbox_xyxy": bbox,
                "projected_center_centroid_xy": centroid,
            }
        )
    return labels


def _load_shader(
    checkpoint: dict[str, Any],
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[DeferredShaderModule, dict[str, object]]:
    config: dict[str, object] = {
        "feature_dim": feature_dim,
        "enable_view_encoding": True,
        "view_encoding_type": "sh",
        "mlp_hidden_dim": 128,
        "mlp_num_layers": 3,
        "sh_degree": 3,
        "sh_scale": 3.0,
        "fourier_num_freqs": 4,
        "center_ray_encoding": False,
    }
    shader = DeferredShaderModule(**config).to(device)
    shader.load_state_dict(checkpoint["deferred_module"])
    state = checkpoint.get("deferred_ema")
    if state is not None:
        for name, parameter in shader.named_parameters():
            if name not in state:
                raise RuntimeError(f"EMA state is missing deferred parameter {name}.")
            parameter.data.copy_(state[name])
    shader.eval()
    return shader, config


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.resolve()
    provider_path = args.provider_bundle.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"NHT checkpoint does not exist: {checkpoint_path}")
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1 or args.asset_count <= 0:
        raise SystemExit("width and asset-count must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    provider = load_scene_provider_bundle(provider_path)
    camera_by_id = {camera.camera_id: camera for camera in provider.manifest.cameras}
    if args.camera_id not in camera_by_id:
        raise SystemExit(f"Unknown provider camera id: {args.camera_id}")
    camera = camera_by_id[args.camera_id]
    height = max(2, round(camera.height * args.width / camera.width))

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    splats = checkpoint["splats"]
    deferred_source = checkpoint.get("deferred_ema", checkpoint["deferred_module"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        appearance_path = temporary_dir / "appearance.pt"
        shader_config = {
            "feature_dim": int(splats["features"].shape[1]),
            "enable_view_encoding": True,
            "view_encoding_type": "sh",
            "mlp_hidden_dim": 128,
            "mlp_num_layers": 3,
            "sh_degree": 3,
            "sh_scale": 3.0,
            "fourier_num_freqs": 4,
            "center_ray_encoding": False,
        }
        torch.save(
            {
                "config": shader_config,
                "state_dict": deferred_source,
            },
            appearance_path,
        )
        appearance_space_sha256 = _sha256(appearance_path)

        base = GaussianTensorSet(
            means=splats["means"].float(),
            quats=F.normalize(splats["quats"].float(), dim=-1),
            log_scales=splats["scales"].float(),
            opacity_logits=splats["opacities"].float(),
            features=splats["features"].float(),
            instance_ids=torch.zeros(
                len(splats["means"]),
                dtype=torch.int64,
            ),
            appearance_space_sha256=appearance_space_sha256,
        )
        camera_to_scene = torch.tensor(
            camera.camera_to_scene,
            dtype=base.means.dtype,
        ).reshape(4, 4)
        intrinsics = torch.tensor(
            camera.intrinsics,
            dtype=base.means.dtype,
        ).reshape(3, 3)
        intrinsics[0] *= args.width / camera.width
        intrinsics[1] *= height / camera.height
        asset_indices, anchor, radius = _select_visible_cluster(
            base,
            camera_to_scene,
            intrinsics,
            width=args.width,
            height=height,
            asset_count=args.asset_count,
        )
        background_mask = torch.ones(base.gaussian_count, dtype=torch.bool)
        background_mask[asset_indices] = False
        background_indices = torch.nonzero(
            background_mask,
            as_tuple=False,
        ).flatten()
        background = _subset(base, background_indices, instance_id=0)
        selected = _subset(base, asset_indices, instance_id=1)
        asset_local = _asset_local_gaussians(
            selected,
            anchor=anchor,
            radius=radius,
        )
        camera_right = camera_to_scene[:3, 0]
        scene_from_asset = SimilarityTransform(
            scale=radius,
            rotation=_rotation_z(15.0),
            translation=tuple(
                float(value) for value in anchor + camera_right * (0.6 * radius)
            ),
        )
        moved_asset = transform_gaussians(asset_local, scene_from_asset)
        composed = compose_gaussians(background, (moved_asset,))

        background_path = temporary_dir / "background.pt"
        asset_path = temporary_dir / "movable-asset.pt"
        torch.save(_raw_tensor_payload(background), background_path)
        torch.save(_raw_tensor_payload(asset_local), asset_path)
        final_background_path = output_dir / background_path.name
        final_asset_path = output_dir / asset_path.name
        final_appearance_path = output_dir / appearance_path.name
        source_checkpoint = _artifact("source-nht-checkpoint", checkpoint_path)
        appearance_artifact = _artifact(
            "shared-nht-appearance",
            appearance_path,
            published_path=final_appearance_path,
        )
        background_asset = GaussianAsset(
            schema=GAUSSIAN_ASSET_SCHEMA,
            asset_id="b00-background-smoke",
            asset_class="court-scene",
            role="background",
            coordinate_frame=SCENE_COORDINATE_FRAME,
            unit=SCENE_UNIT,
            metres_per_unit=None,
            gaussian_count=background.gaussian_count,
            feature_dim=background.feature_dim,
            tensor_encoding=NHT_TENSOR_ENCODING,
            tensors=_artifact(
                "background-gaussians",
                background_path,
                published_path=final_background_path,
            ),
            appearance_model=NHT_APPEARANCE_MODEL,
            appearance_space_sha256=appearance_space_sha256,
            appearance_payload=appearance_artifact,
            provenance=(source_checkpoint,),
        )
        movable_asset = GaussianAsset(
            schema=GAUSSIAN_ASSET_SCHEMA,
            asset_id="metric-smoke-patch",
            asset_class="smoke-patch",
            role="movable",
            coordinate_frame=ASSET_COORDINATE_FRAME,
            unit=METRE_UNIT,
            metres_per_unit=1.0,
            gaussian_count=asset_local.gaussian_count,
            feature_dim=asset_local.feature_dim,
            tensor_encoding=NHT_TENSOR_ENCODING,
            tensors=_artifact(
                "movable-gaussians",
                asset_path,
                published_path=final_asset_path,
            ),
            appearance_model=NHT_APPEARANCE_MODEL,
            appearance_space_sha256=appearance_space_sha256,
            appearance_payload=appearance_artifact,
            provenance=(source_checkpoint,),
        )
        provider_manifest_path = provider.root / "provider.json"
        composition = GaussianSceneComposition.create(
            composition_id="nht-composition-smoke-c02",
            scene_source=_artifact("provider-bundle", provider_manifest_path),
            background=background_asset,
            instances=(
                GaussianInstance(
                    instance_id=1,
                    asset=movable_asset,
                    scene_from_asset=scene_from_asset,
                ),
            ),
            renderer_backend="nht-gsplat",
            renderer_commit=_git_head(
                Path(__file__).resolve().parent / "upstream" / "gsplat"
            ),
        )
        write_gaussian_scene_manifest(
            temporary_dir / "composition.json",
            composition,
        )

        device = torch.device("cuda:0")
        shader, realized_shader_config = _load_shader(
            checkpoint,
            feature_dim=composed.feature_dim,
            device=device,
        )
        composed_cuda = GaussianTensorSet(
            means=composed.means.to(device),
            quats=composed.quats.to(device),
            log_scales=composed.log_scales.to(device),
            opacity_logits=composed.opacity_logits.to(device),
            features=composed.features.to(device),
            instance_ids=composed.instance_ids.to(device),
            appearance_space_sha256=composed.appearance_space_sha256,
        )
        camera_to_scene_cuda = camera_to_scene.to(device).unsqueeze(0)
        intrinsics_cuda = intrinsics.to(device).unsqueeze(0)
        render_call_count = 0
        with torch.no_grad():
            rendered_features, alpha, info = rasterization(
                means=composed_cuda.means,
                quats=composed_cuda.quats,
                scales=torch.exp(composed_cuda.log_scales),
                opacities=torch.sigmoid(composed_cuda.opacity_logits),
                colors=composed_cuda.features,
                viewmats=torch.linalg.inv(camera_to_scene_cuda),
                Ks=intrinsics_cuda,
                width=args.width,
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
            render_call_count += 1
            rgb, depth = shader(rendered_features)
            if depth is None or depth.shape[-1] < 1:
                raise RuntimeError("NHT RGB+ED render did not return depth.")
            rgb = (rgb[..., :3] + torch.ones_like(rgb[..., :3]) * (1.0 - alpha)).clamp(
                0.0, 1.0
            )
            torch.cuda.synchronize()
        if render_call_count != 1:
            raise RuntimeError(
                f"Expected exactly one rasterization call, got {render_call_count}."
            )
        if not bool(torch.isfinite(rgb).all()):
            raise RuntimeError("Composed RGB contains non-finite values.")
        if not bool(torch.isfinite(alpha).all()):
            raise RuntimeError("Composed alpha contains non-finite values.")
        depth_channel = depth[..., :1]
        if not bool(torch.isfinite(depth_channel).all()):
            raise RuntimeError("Composed depth contains non-finite values.")

        rgb_array = rgb[0].mul(255.0).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(rgb_array, mode="RGB").save(temporary_dir / "rgb.png")
        np.save(
            temporary_dir / "alpha.npy",
            alpha[0, ..., 0].float().cpu().numpy(),
        )
        np.save(
            temporary_dir / "depth.npy",
            depth_channel[0, ..., 0].float().cpu().numpy(),
        )
        labels = _visibility_labels(
            info=info,
            instance_ids=composed_cuda.instance_ids,
            width=args.width,
            height=height,
        )
        _write_json(
            temporary_dir / "labels.json",
            {
                "schema": "tennis_gaussian_instance_labels_v1",
                "camera_id": camera.camera_id,
                "width": args.width,
                "height": height,
                "instances": labels,
            },
        )
        report = {
            "schema": "tennis_nht_composition_smoke_v1",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "status": "passed",
            "checkpoint": source_checkpoint.to_dict(),
            "provider_bundle": composition.scene_source.to_dict(),
            "composition_fingerprint": composition.composition_fingerprint,
            "appearance_space_sha256": appearance_space_sha256,
            "renderer_commit": composition.renderer_commit,
            "camera_id": camera.camera_id,
            "resolution": [args.width, height],
            "gaussians": {
                "background": background.gaussian_count,
                "movable_asset": asset_local.gaussian_count,
                "composed": composed.gaussian_count,
            },
            "scene_from_asset": scene_from_asset.to_dict(),
            "asset_selection": {
                "anchor_scene": [float(value) for value in anchor],
                "radius_scene_units": radius,
            },
            "renderer": {
                "rasterization_call_count": render_call_count,
                "shader_config": realized_shader_config,
            },
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
                "min": float(depth_channel.min()),
                "max": float(depth_channel.max()),
                "mean": float(depth_channel.mean()),
            },
            "instance_labels": labels,
            "all_finite": True,
        }
        _write_json(temporary_dir / "report.json", report)
        temporary_dir.replace(output_dir)
        print(json.dumps(report, indent=2))
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
