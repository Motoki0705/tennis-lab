#!/usr/bin/env python3
"""Build a non-user fixture for the frozen-target ball feature-fit worker."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from plyfile import PlyData, PlyElement
from prepare_ball_assets import ASSET_PREPARATION_ENTRY_SCHEMA

from src.synthetic_data_generation.blcs.calibration import (
    BALL_CALIBRATION_CAPTURE_SCHEMA,
)

FIXTURE_SCHEMA = "tennis_ball_nht_feature_fit_fixture_v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-tensors", type=Path, required=True)
    parser.add_argument("--target-appearance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--validation-views", type=int, default=2)
    parser.add_argument("--camera-radius-m", type=float, default=0.24)
    parser.add_argument("--focal-length-px", type=float, default=150.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_teacher(path: Path) -> tuple[SourceGeometry, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "means",
        "quats",
        "scales",
        "opacities",
        "features",
        "instance_ids",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Teacher tensor pack does not have the exact NHT tensor keys.")
    for name in ("means", "quats", "scales", "opacities", "features"):
        value = payload[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Teacher {name} must be finite float32.")
    count = int(payload["means"].shape[0])
    if count <= 0:
        raise ValueError("Teacher tensor pack is empty.")
    if tuple(payload["means"].shape) != (count, 3):
        raise ValueError("Teacher means shape differs.")
    if tuple(payload["quats"].shape) != (count, 4):
        raise ValueError("Teacher quats shape differs.")
    if tuple(payload["scales"].shape) != (count, 3):
        raise ValueError("Teacher scales shape differs.")
    if tuple(payload["opacities"].shape) != (count,):
        raise ValueError("Teacher opacities shape differs.")
    if payload["features"].ndim != 2 or payload["features"].shape[0] != count:
        raise ValueError("Teacher features shape differs.")
    if not torch.allclose(
        torch.linalg.vector_norm(payload["quats"], dim=-1),
        torch.ones(count),
        atol=1.0e-3,
    ):
        raise ValueError("Teacher quaternions are not normalized.")
    return (
        SourceGeometry(
            means=payload["means"].contiguous(),
            quats=payload["quats"].contiguous(),
            log_scales=payload["scales"].contiguous(),
            opacity_logits=payload["opacities"].contiguous(),
            source_feature_dim=int(payload["features"].shape[1]),
        ),
        payload["features"].contiguous(),
    )


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


def _write_independent_source(
    path: Path,
    geometry: SourceGeometry,
    teacher_features: torch.Tensor,
) -> None:
    torch.save(
        {
            "means": geometry.means.clone(),
            "quats": geometry.quats.clone(),
            "scales": geometry.log_scales.clone(),
            "opacities": geometry.opacity_logits.clone(),
            "features": -teacher_features.clone(),
            "instance_ids": torch.zeros(
                (geometry.gaussian_count,),
                dtype=torch.int64,
            ),
        },
        path,
    )


def _write_vanilla_ply(path: Path, geometry: SourceGeometry) -> None:
    property_names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    dtype = [(name, "f4") for name in property_names]
    vertices = np.empty(geometry.gaussian_count, dtype=dtype)
    means = geometry.means.numpy()
    scales = geometry.log_scales.numpy()
    quats = geometry.quats.numpy()
    opacities = geometry.opacity_logits.numpy()
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = means[:, axis]
    for name in ("nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"):
        vertices[name] = 0.0
    vertices["opacity"] = opacities
    for axis, name in enumerate(("scale_0", "scale_1", "scale_2")):
        vertices[name] = scales[:, axis]
    for axis, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        vertices[name] = quats[:, axis]
    PlyData((PlyElement.describe(vertices, "vertex"),), text=False).write(path)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1 or args.height <= 1:
        raise SystemExit("width and height must be greater than one.")
    if args.view_count < 3:
        raise SystemExit("view-count must be at least three.")
    if not 0 < args.validation_views < args.view_count:
        raise SystemExit("validation-views must leave at least one training view.")
    if args.camera_radius_m <= 0.0 or args.focal_length_px <= 0.0:
        raise SystemExit("camera radius and focal length must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    teacher_path = args.teacher_tensors.resolve()
    appearance_path = args.target_appearance.resolve()
    if not teacher_path.is_file() or not appearance_path.is_file():
        raise SystemExit("Teacher tensors and target appearance must both exist.")
    runtime = _runtime_revisions()
    geometry_cpu, teacher_features_cpu = _load_teacher(teacher_path)
    device = torch.device(args.device)
    shader, target_feature_dim, shader_config = _load_shader(
        appearance_path,
        device=device,
    )
    if target_feature_dim != int(teacher_features_cpu.shape[1]):
        raise ValueError("Teacher feature dimension differs from target appearance.")
    geometry = SourceGeometry(
        means=geometry_cpu.means.to(device),
        quats=F.normalize(geometry_cpu.quats.to(device), dim=-1),
        log_scales=geometry_cpu.log_scales.to(device),
        opacity_logits=geometry_cpu.opacity_logits.to(device),
        source_feature_dim=geometry_cpu.source_feature_dim,
    )
    teacher_features = teacher_features_cpu.to(device)
    cameras = np.stack(
        [
            _look_at_camera(
                azimuth_radians=2.0 * math.pi * index / args.view_count,
                elevation_radians=math.radians(12.0 if index % 2 == 0 else -6.0),
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
                teacher_features,
                shader,
                torch.from_numpy(camera).to(device),
                torch.from_numpy(intrinsic).to(device),
                width=args.width,
                height=args.height,
            )
            mask = alpha > 0.01
            if not bool(mask.any()):
                raise RuntimeError("Fixture camera produced no foreground pixels.")
            rgb_views.append(
                (rgb.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            )
            mask_views.append(mask.detach().cpu().numpy().astype(np.bool_))
            alpha_coverages.append(float(mask.float().mean()))
    rgb = np.stack(rgb_views)
    mask = np.stack(mask_views)
    split = np.zeros((args.view_count,), dtype=np.uint8)
    split[-args.validation_views :] = 1

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
        capture_views: list[dict[str, object]] = []
        for index in range(args.view_count):
            view_id = f"view-{index:03d}"
            rgb_path = rgb_root / f"{view_id}.png"
            mask_path = mask_root / f"{view_id}.png"
            Image.fromarray(rgb[index]).save(rgb_path)
            Image.fromarray(mask[index].astype(np.uint8) * 255).save(mask_path)
            capture_views.append(
                {
                    "view_id": view_id,
                    "split": "train" if split[index] == 0 else "validation",
                    "width": args.width,
                    "height": args.height,
                    "rgb": _file_ref(capture_root, rgb_path),
                    "mask": _file_ref(capture_root, mask_path),
                    "camera_to_asset": cameras[index]
                    .astype(np.float32)
                    .ravel()
                    .tolist(),
                    "intrinsics": intrinsics[index].astype(np.float32).ravel().tolist(),
                }
            )
        capture_manifest = capture_root / "capture.json"
        _write_json(
            capture_manifest,
            {
                "schema": BALL_CALIBRATION_CAPTURE_SCHEMA,
                "capture_id": "cycle08-frozen-target-capture-fixture",
                "views": capture_views,
            },
        )
        independent_path = temporary / "independent-source.pt"
        vanilla_path = temporary / "vanilla-source.ply"
        _write_independent_source(
            independent_path,
            geometry_cpu,
            teacher_features_cpu,
        )
        _write_vanilla_ply(vanilla_path, geometry_cpu)
        asset_specs_root = temporary / "asset-specs"
        asset_specs_root.mkdir()
        asset_spec_paths: list[Path] = []
        identity = {
            "scale": 1.0,
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }
        for tag, source_format, source_path in (
            ("independent", "independent_nht_tensor_pack_v1", independent_path),
            ("vanilla", "vanilla_3dgs_ply_v1", vanilla_path),
        ):
            spec_path = asset_specs_root / f"{tag}.json"
            final_source_path = output_dir / source_path.relative_to(temporary)
            _write_json(
                spec_path,
                {
                    "schema": ASSET_PREPARATION_ENTRY_SCHEMA,
                    "variant_id": f"{tag}-capture-fixture",
                    "asset_id": f"cycle08-{tag}-capture-fixture",
                    "nominal_diameter_m": 0.067,
                    "source_format": source_format,
                    "source": {
                        "artifact_id": f"cycle08-{tag}-fixture-source",
                        "uri": final_source_path.as_uri(),
                        "sha256": _sha256_file(source_path),
                        "size_bytes": source_path.stat().st_size,
                    },
                    "asset_from_prepared": identity,
                    "source_is_user_asset": False,
                },
            )
            asset_spec_paths.append(spec_path)
        unsigned: dict[str, object] = {
            "schema": FIXTURE_SCHEMA,
            "teacher_tensors": _absolute_file_ref(teacher_path),
            "target_appearance": _absolute_file_ref(appearance_path),
            "target_appearance_space_sha256": _sha256_file(appearance_path),
            "renderer": runtime,
            "shader_config": shader_config,
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
                "method": "teacher-alpha-greater-than-0.01",
                "coverage_fraction_per_view": alpha_coverages,
            },
            "files": {
                "capture_manifest": _file_ref(temporary, capture_manifest),
                "independent_source": _file_ref(temporary, independent_path),
                "vanilla_source": _file_ref(temporary, vanilla_path),
                "asset_specs": [
                    _file_ref(temporary, path) for path in asset_spec_paths
                ],
            },
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "fixture.json", manifest)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
