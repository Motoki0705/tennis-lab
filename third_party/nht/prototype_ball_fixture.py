#!/usr/bin/env python3
"""Construct a metric prototype tennis-ball Gaussian and calibration capture."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from ball_feature_fit import (
    SourceGeometry,
    _absolute_file_ref,
    _canonical_sha256,
    _file_ref,
    _load_shader,
    _render,
    _runtime_revisions,
    _sha256_file,
    _write_json,
)
from PIL import Image
from prepare_ball_assets import ASSET_PREPARATION_ENTRY_SCHEMA

from src.synthetic_data_generation.blcs.calibration import (
    BALL_CALIBRATION_CAPTURE_SCHEMA,
)
from src.synthetic_data_generation.blcs.prototype import (
    build_prototype_ball_geometry,
)

PROTOTYPE_SCHEMA = "tennis_ball_generated_prototype_v1"
PROTOTYPE_ASSET_ID = "codex-prototype-tennis-ball-v1"
PROTOTYPE_VARIANT_ID = "generated-prototype-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-appearance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nominal-diameter-m", type=float, default=0.067)
    parser.add_argument("--gaussian-count", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--validation-views", type=int, default=2)
    parser.add_argument("--camera-radius-m", type=float, default=0.24)
    parser.add_argument("--focal-length-px", type=float, default=150.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _look_at_camera(
    *,
    azimuth_radians: float,
    elevation_radians: float,
    radius: float,
) -> np.ndarray:
    position = np.array(
        [
            radius * math.cos(elevation_radians) * math.cos(azimuth_radians),
            radius * math.cos(elevation_radians) * math.sin(azimuth_radians),
            radius * math.sin(elevation_radians),
        ],
        dtype=np.float32,
    )
    forward = -position / np.linalg.norm(position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, :3] = np.stack((right, down, forward), axis=1)
    camera[:3, 3] = position
    return camera


def _target_features(
    means: torch.Tensor,
    *,
    feature_dim: int,
) -> torch.Tensor:
    """Create a smooth asymmetric latent pattern for view-distinct captures."""
    directions = torch.nn.functional.normalize(means, dim=-1)
    index = torch.arange(
        means.shape[0],
        dtype=torch.float32,
        device=means.device,
    )
    features = torch.zeros(
        (means.shape[0], feature_dim),
        dtype=torch.float32,
        device=means.device,
    )
    basis = (
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        directions[:, 0] * directions[:, 1],
        directions[:, 1] * directions[:, 2],
        directions[:, 2] * directions[:, 0],
        torch.sin(index * 0.071),
        torch.cos(index * 0.113),
    )
    for feature_index in range(feature_dim):
        component = basis[feature_index % len(basis)]
        amplitude = 0.12 / (1.0 + feature_index // len(basis))
        features[:, feature_index] = amplitude * component
    return features


def _write_source(
    path: Path,
    *,
    geometry: SourceGeometry,
    feature_dim: int,
) -> None:
    torch.save(
        {
            "means": geometry.means.detach().cpu().contiguous(),
            "quats": geometry.quats.detach().cpu().contiguous(),
            "scales": geometry.log_scales.detach().cpu().contiguous(),
            "opacities": geometry.opacity_logits.detach().cpu().contiguous(),
            "features": torch.zeros(
                (geometry.gaussian_count, feature_dim),
                dtype=torch.float32,
            ),
            "instance_ids": torch.zeros(
                (geometry.gaussian_count,),
                dtype=torch.int64,
            ),
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1 or args.height <= 1:
        raise SystemExit("width and height must be greater than one.")
    if args.view_count < 3:
        raise SystemExit("view-count must be at least three.")
    if not 0 < args.validation_views <= args.view_count - 2:
        raise SystemExit("validation-views must leave at least two training views.")
    if args.camera_radius_m <= 0.0 or args.focal_length_px <= 0.0:
        raise SystemExit("camera radius and focal length must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    appearance_path = args.target_appearance.resolve()
    if not appearance_path.is_file():
        raise SystemExit(f"Target appearance is missing: {appearance_path}")

    prototype = build_prototype_ball_geometry(
        nominal_diameter_m=args.nominal_diameter_m,
        gaussian_count=args.gaussian_count,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Prototype capture construction requires a CUDA device.")
    runtime = _runtime_revisions()
    shader, feature_dim, shader_config = _load_shader(
        appearance_path,
        device=device,
    )
    geometry = SourceGeometry(
        means=torch.from_numpy(prototype.means.copy()).to(device),
        quats=torch.from_numpy(prototype.quats.copy()).to(device),
        log_scales=torch.from_numpy(prototype.log_scales.copy()).to(device),
        opacity_logits=torch.from_numpy(prototype.opacity_logits.copy()).to(device),
        source_feature_dim=feature_dim,
    )
    target_features = _target_features(geometry.means, feature_dim=feature_dim)
    cameras = np.stack(
        [
            _look_at_camera(
                azimuth_radians=2.0 * math.pi * index / args.view_count,
                elevation_radians=math.radians(14.0 if index % 2 == 0 else -8.0),
                radius=args.camera_radius_m,
            )
            for index in range(args.view_count)
        ]
    )
    intrinsics = np.broadcast_to(
        np.array(
            [
                [args.focal_length_px, 0.0, args.width / 2.0],
                [0.0, args.focal_length_px, args.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        (args.view_count, 3, 3),
    ).copy()
    rgb_views: list[np.ndarray] = []
    mask_views: list[np.ndarray] = []
    alpha_coverages: list[float] = []
    with torch.no_grad():
        for camera, intrinsic in zip(cameras, intrinsics, strict=True):
            rgb, alpha = _render(
                geometry,
                target_features,
                shader,
                torch.from_numpy(camera).to(device),
                torch.from_numpy(intrinsic).to(device),
                width=args.width,
                height=args.height,
            )
            mask = alpha > 0.01
            if not bool(mask.any()) or bool(mask.all()):
                raise RuntimeError("Prototype camera produced an invalid foreground mask.")
            rgb_views.append(
                (rgb.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            )
            mask_views.append(mask.detach().cpu().numpy().astype(np.bool_))
            alpha_coverages.append(float(mask.float().mean()))
    torch.cuda.synchronize(device)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        capture_root = temporary / "capture"
        rgb_root = capture_root / "rgb"
        mask_root = capture_root / "mask"
        rgb_root.mkdir(parents=True)
        mask_root.mkdir()
        split_start = args.view_count - args.validation_views
        capture_views: list[dict[str, object]] = []
        for index, (rgb, mask) in enumerate(
            zip(rgb_views, mask_views, strict=True)
        ):
            view_id = f"view-{index:03d}"
            rgb_path = rgb_root / f"{view_id}.png"
            mask_path = mask_root / f"{view_id}.png"
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            capture_views.append(
                {
                    "view_id": view_id,
                    "split": "validation" if index >= split_start else "train",
                    "width": args.width,
                    "height": args.height,
                    "rgb": _file_ref(capture_root, rgb_path),
                    "mask": _file_ref(capture_root, mask_path),
                    "camera_to_asset": cameras[index].ravel().tolist(),
                    "intrinsics": intrinsics[index].ravel().tolist(),
                }
            )
        capture_manifest = capture_root / "capture.json"
        _write_json(
            capture_manifest,
            {
                "schema": BALL_CALIBRATION_CAPTURE_SCHEMA,
                "capture_id": "codex-prototype-tennis-ball-capture-v1",
                "views": capture_views,
            },
        )
        source_path = temporary / "prototype-source.pt"
        _write_source(
            source_path,
            geometry=geometry,
            feature_dim=feature_dim,
        )
        spec_path = temporary / "asset-spec.json"
        final_source_path = output_dir / source_path.relative_to(temporary)
        _write_json(
            spec_path,
            {
                "schema": ASSET_PREPARATION_ENTRY_SCHEMA,
                "variant_id": PROTOTYPE_VARIANT_ID,
                "asset_id": PROTOTYPE_ASSET_ID,
                "nominal_diameter_m": prototype.nominal_diameter_m,
                "source_format": "independent_nht_tensor_pack_v1",
                "source": {
                    "artifact_id": "codex-generated-prototype-source-v1",
                    "uri": final_source_path.as_uri(),
                    "sha256": _sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                },
                "asset_from_prepared": {
                    "scale": 1.0,
                    "rotation": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "translation": [0.0, 0.0, 0.0],
                },
                "source_is_user_asset": False,
            },
        )
        unsigned: dict[str, object] = {
            "schema": PROTOTYPE_SCHEMA,
            "asset_origin": "codex-generated-prototype",
            "asset_id": PROTOTYPE_ASSET_ID,
            "variant_id": PROTOTYPE_VARIANT_ID,
            "construction_method": (
                "antipodal-fibonacci-shell-isotropic-three-sigma-envelope-v1"
            ),
            "geometry": prototype.metric_summary(),
            "target_appearance": _absolute_file_ref(appearance_path),
            "target_appearance_space_sha256": _sha256_file(appearance_path),
            "renderer": runtime,
            "shader_config": shader_config,
            "target_feature_pattern": "smooth-asymmetric-direction-basis-v1",
            "camera_model": {
                "coordinate_convention": "opencv-camera-to-asset-v1",
                "width": args.width,
                "height": args.height,
                "radius_m": args.camera_radius_m,
                "focal_length_px": args.focal_length_px,
                "view_count": args.view_count,
                "validation_views": args.validation_views,
            },
            "foreground_mask": {
                "method": "prototype-alpha-greater-than-0.01",
                "coverage_fraction_per_view": alpha_coverages,
            },
            "files": {
                "capture_manifest": _file_ref(temporary, capture_manifest),
                "source": _file_ref(temporary, source_path),
                "asset_spec": _file_ref(temporary, spec_path),
            },
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "prototype.json", manifest)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
