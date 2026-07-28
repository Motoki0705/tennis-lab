#!/usr/bin/env python3
"""Fit an SMPL-X Gaussian avatar into NHT and render controlled poses natively."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from ball_feature_fit import (
    INDEPENDENT_NHT_SOURCE,
    SourceGeometry,
    _load_shader,
    _masked_mse,
    _render,
    _runtime_revisions,
    load_source_geometry,
)
from PIL import Image

SCHEMA = "plcs_avatar_nht_fit_and_pose_render_v1"
FIXTURE_SCHEMA = "plcs_smplx_gaussian_asset_fixture_v1"
SEED = 20260728


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _file_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _local_uri(uri: object, *, name: str) -> Path:
    if not isinstance(uri, str):
        raise ValueError(f"{name} URI must be a string.")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"{name} must use a local file URI.")
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def _load_fixture(path: Path) -> tuple[Path, dict[str, object]]:
    root = path.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("Unsupported PLCS avatar fixture schema.")
    expected_fingerprint = manifest.get("content_fingerprint")
    if not isinstance(expected_fingerprint, str):
        raise ValueError("Fixture content fingerprint is missing.")
    unsigned = dict(manifest)
    del unsigned["content_fingerprint"]
    if _canonical_sha256(unsigned) != expected_fingerprint:
        raise ValueError("Fixture content fingerprint differs.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Fixture file inventory is missing.")
    for relative, reference in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(reference, dict)
            or set(reference) != {"sha256", "size_bytes"}
        ):
            raise ValueError("Fixture file inventory has an invalid entry.")
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Unsafe or missing fixture file: {relative}")
        if (
            _sha256(candidate) != reference["sha256"]
            or candidate.stat().st_size != reference["size_bytes"]
        ):
            raise ValueError(f"Fixture file changed: {relative}")
    model = manifest.get("smplx_model")
    if not isinstance(model, dict):
        raise ValueError("Fixture SMPL-X provenance is missing.")
    model_path = _local_uri(model.get("uri"), name="SMPL-X model")
    if _sha256(model_path) != model.get("sha256"):
        raise ValueError("Licensed SMPL-X model changed after fixture publication.")
    return root, manifest


def _look_at(
    center: np.ndarray,
    *,
    azimuth: float,
    elevation: float,
    radius: float,
) -> np.ndarray:
    direction = np.asarray(
        [
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ],
        dtype=np.float32,
    )
    position = center.astype(np.float32) + radius * direction
    forward = center.astype(np.float32) - position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, :3] = np.stack((right, down, forward), axis=1)
    camera[:3, 3] = position
    return camera


def _target_features(means: torch.Tensor, feature_dim: int) -> torch.Tensor:
    center = (means.amin(dim=0) + means.amax(dim=0)) * 0.5
    scale = torch.clamp(means.amax(dim=0) - means.amin(dim=0), min=1.0e-6)
    normalized = (means - center) / scale
    index = torch.arange(means.shape[0], device=means.device, dtype=torch.float32)
    basis = (
        normalized[:, 0],
        normalized[:, 1],
        normalized[:, 2],
        torch.sin(normalized[:, 0] * 5.0),
        torch.cos(normalized[:, 1] * 4.0),
        normalized[:, 0] * normalized[:, 1],
        torch.sin(index * 0.037),
        torch.cos(index * 0.061),
    )
    features = torch.zeros(
        (means.shape[0], feature_dim),
        device=means.device,
        dtype=torch.float32,
    )
    for feature_index in range(feature_dim):
        features[:, feature_index] = (
            0.14
            / (1.0 + feature_index // len(basis))
            * basis[feature_index % len(basis)]
        )
    return features


def _image(array: torch.Tensor) -> np.ndarray:
    return (
        array.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)


def _save_tensor_set(
    geometry: SourceGeometry,
    features: torch.Tensor,
    *,
    path: Path,
) -> None:
    torch.save(
        {
            "means": geometry.means.detach().cpu().contiguous(),
            "quats": geometry.quats.detach().cpu().contiguous(),
            "scales": geometry.log_scales.detach().cpu().contiguous(),
            "opacities": geometry.opacity_logits.detach().cpu().contiguous(),
            "features": features.detach().cpu().contiguous(),
            "instance_ids": torch.zeros(
                geometry.gaussian_count,
                dtype=torch.int64,
            ),
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--target-appearance", type=Path, required=True)
    parser.add_argument("--appearance-space-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    appearance_path = args.target_appearance.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    if not appearance_path.is_file():
        raise SystemExit(f"Target appearance is missing: {appearance_path}")
    if (
        len(args.appearance_space_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.appearance_space_sha256
        )
    ):
        raise SystemExit("appearance-space-sha256 must be a lowercase SHA-256.")
    if args.steps <= 0 or args.width <= 1 or args.height <= 1:
        raise SystemExit("steps and image dimensions must be positive.")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise SystemExit("PLCS NHT fitting requires CUDA.")

    fixture_root, fixture = _load_fixture(args.fixture)
    target_reference = fixture.get("target_nht_appearance")
    if not isinstance(target_reference, dict):
        raise SystemExit("Fixture target NHT appearance provenance is missing.")
    if (
        appearance_path.as_uri() != target_reference.get("uri")
        or _sha256(appearance_path) != target_reference.get("sha256")
    ):
        raise SystemExit("Target appearance differs from the fixture declaration.")
    pose_ids = fixture.get("pose_ids")
    if (
        not isinstance(pose_ids, list)
        or not pose_ids
        or any(not isinstance(value, str) for value in pose_ids)
    ):
        raise SystemExit("Fixture pose_ids are invalid.")

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    runtime = _runtime_revisions()
    runtime["plcs_fit_worker_sha256"] = _sha256(Path(__file__).resolve())
    runtime["shared_runtime_module"] = "third_party/nht/ball_feature_fit.py"
    shader, feature_dim, shader_config = _load_shader(
        appearance_path,
        device=device,
    )
    if feature_dim != target_reference.get("feature_dim"):
        raise RuntimeError("Fixture and NHT shader feature dimensions differ.")
    source_cpu = load_source_geometry(
        fixture_root / "canonical-source.pt",
        INDEPENDENT_NHT_SOURCE,
    )
    geometry = SourceGeometry(
        means=source_cpu.means.to(device),
        quats=source_cpu.quats.to(device),
        log_scales=source_cpu.log_scales.to(device),
        opacity_logits=source_cpu.opacity_logits.to(device),
        source_feature_dim=source_cpu.source_feature_dim,
    )

    means_np = geometry.means.detach().cpu().numpy()
    center = (means_np.min(axis=0) + means_np.max(axis=0)) * 0.5
    view_count = 10
    cameras = np.stack(
        [
            _look_at(
                center,
                azimuth=2.0 * math.pi * index / view_count,
                elevation=math.radians(4.0 if index % 2 == 0 else -3.0),
                radius=3.0,
            )
            for index in range(view_count)
        ]
    )
    focal = 240.0
    intrinsic = np.asarray(
        [
            [focal, 0.0, args.width / 2.0],
            [0.0, focal, args.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_tensors = torch.from_numpy(cameras).to(device)
    intrinsic_tensor = torch.from_numpy(intrinsic).to(device)
    target_features = _target_features(geometry.means, feature_dim)
    target_rgb: list[torch.Tensor] = []
    target_mask: list[torch.Tensor] = []
    with torch.no_grad():
        for camera in camera_tensors:
            rgb, alpha = _render(
                geometry,
                target_features,
                shader,
                camera,
                intrinsic_tensor,
                width=args.width,
                height=args.height,
            )
            mask = alpha > 0.02
            coverage = float(mask.float().mean())
            if not 0.01 < coverage < 0.85:
                raise RuntimeError(f"Invalid target coverage: {coverage}.")
            target_rgb.append(rgb.detach())
            target_mask.append(mask.detach())

    features = torch.nn.Parameter(
        torch.zeros(
            (geometry.gaussian_count, feature_dim),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.Adam((features,), lr=0.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=0.0015,
    )
    train_indices = tuple(range(8))
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        view_index = train_indices[step % len(train_indices)]
        prediction, _ = _render(
            geometry,
            features,
            shader,
            camera_tensors[view_index],
            intrinsic_tensor,
            width=args.width,
            height=args.height,
        )
        loss = _masked_mse(
            prediction,
            target_rgb[view_index],
            target_mask[view_index],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if features.grad is None or not bool(torch.isfinite(features.grad).all()):
            raise RuntimeError("Avatar feature fit produced invalid gradients.")
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            record = {
                "step": step + 1,
                "view_index": view_index,
                "masked_mse": float(loss.detach()),
                "feature_lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    validation: list[dict[str, float | int]] = []
    validation_predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for view_index in (8, 9):
            prediction, _ = _render(
                geometry,
                features,
                shader,
                camera_tensors[view_index],
                intrinsic_tensor,
                width=args.width,
                height=args.height,
            )
            mse = _masked_mse(
                prediction,
                target_rgb[view_index],
                target_mask[view_index],
            )
            psnr = float(-10.0 * torch.log10(mse.clamp_min(1.0e-12)))
            validation.append(
                {
                    "view_index": view_index,
                    "masked_psnr_db": psnr,
                    "mask_pixel_count": int(target_mask[view_index].sum()),
                }
            )
            validation_predictions.append(prediction.detach())
    mean_validation_psnr = float(
        np.mean([float(item["masked_psnr_db"]) for item in validation])
    )
    if mean_validation_psnr < 25.0:
        raise RuntimeError(
            f"Avatar validation PSNR {mean_validation_psnr:.6f} dB is below 25 dB."
        )

    render_camera_indices = (0, 2)
    pose_renders: list[list[torch.Tensor]] = []
    pose_alphas: list[list[torch.Tensor]] = []
    pose_metrics: list[dict[str, object]] = []
    pose_geometries: list[SourceGeometry] = []
    with torch.no_grad():
        for pose_index, pose_id in enumerate(pose_ids):
            path = fixture_root / "poses" / f"{pose_index:03d}-{pose_id}.pt"
            pose_cpu = load_source_geometry(path, INDEPENDENT_NHT_SOURCE)
            if pose_cpu.gaussian_count != geometry.gaussian_count:
                raise RuntimeError(f"Pose {pose_id} changed Gaussian count.")
            pose_geometry = SourceGeometry(
                means=pose_cpu.means.to(device),
                quats=pose_cpu.quats.to(device),
                log_scales=pose_cpu.log_scales.to(device),
                opacity_logits=pose_cpu.opacity_logits.to(device),
                source_feature_dim=pose_cpu.source_feature_dim,
            )
            pose_geometries.append(pose_geometry)
            rendered: list[torch.Tensor] = []
            alphas: list[torch.Tensor] = []
            coverage: list[float] = []
            visible_pixels: list[int] = []
            for camera_index in render_camera_indices:
                rgb, alpha = _render(
                    pose_geometry,
                    features,
                    shader,
                    camera_tensors[camera_index],
                    intrinsic_tensor,
                    width=args.width,
                    height=args.height,
                )
                mask = alpha > 0.02
                rendered.append(rgb.detach())
                alphas.append(alpha.detach())
                coverage.append(float(mask.float().mean()))
                visible_pixels.append(int(mask.sum()))
            pose_renders.append(rendered)
            pose_alphas.append(alphas)
            pose_metrics.append(
                {
                    "pose_id": pose_id,
                    "coverage_fraction": coverage,
                    "visible_pixel_count": visible_pixels,
                    "all_gaussians_preserved": (
                        pose_geometry.gaussian_count == geometry.gaussian_count
                    ),
                }
            )
    for pose_index in range(1, len(pose_ids)):
        differences = [
            float(
                torch.mean(
                    torch.abs(
                        pose_renders[pose_index][camera_index]
                        - pose_renders[0][camera_index]
                    )
                )
            )
            for camera_index in range(len(render_camera_indices))
        ]
        pose_metrics[pose_index]["mean_rgb_difference_from_canonical"] = differences
        if max(differences) <= 0.005:
            raise RuntimeError(f"Pose {pose_ids[pose_index]} did not change its render.")
    if any(
        min(metric["visible_pixel_count"]) < 500  # type: ignore[arg-type]
        for metric in pose_metrics
    ):
        raise RuntimeError("A controlled avatar pose has insufficient visible pixels.")
    torch.cuda.synchronize(device)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        poses_root = temporary / "poses"
        diagnostics = temporary / "diagnostics"
        poses_root.mkdir()
        diagnostics.mkdir()
        canonical_tensors = temporary / "avatar-nht-tensors.pt"
        _save_tensor_set(geometry, features, path=canonical_tensors)
        for pose_index, (pose_id, pose_geometry) in enumerate(
            zip(pose_ids, pose_geometries, strict=True)
        ):
            _save_tensor_set(
                pose_geometry,
                features,
                path=poses_root / f"{pose_index:03d}-{pose_id}-nht.pt",
            )
            for camera_offset, camera_index in enumerate(render_camera_indices):
                Image.fromarray(_image(pose_renders[pose_index][camera_offset])).save(
                    poses_root
                    / f"{pose_index:03d}-{pose_id}-camera-{camera_index:03d}.png"
                )
                np.save(
                    poses_root
                    / f"{pose_index:03d}-{pose_id}-camera-{camera_index:03d}-alpha.npy",
                    pose_alphas[pose_index][camera_offset]
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    allow_pickle=False,
                )
        for offset, view_index in enumerate((8, 9)):
            target_image = _image(target_rgb[view_index])
            prediction_image = _image(validation_predictions[offset])
            difference_image = (
                np.abs(
                    target_image.astype(np.int16)
                    - prediction_image.astype(np.int16)
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            panel = np.concatenate(
                (target_image, prediction_image, difference_image),
                axis=1,
            )
            Image.fromarray(panel).save(
                diagnostics / f"validation-view-{view_index:03d}.png"
            )
        contact_rows = [
            np.concatenate([_image(image) for image in pose_render], axis=1)
            for pose_render in pose_renders
        ]
        Image.fromarray(np.concatenate(contact_rows, axis=0)).save(
            diagnostics / "pose-contact-sheet.png"
        )
        _write_json(temporary / "optimization-history.json", history)
        metrics = {
            "mean_validation_psnr_db": mean_validation_psnr,
            "validation": validation,
            "pose_renders": pose_metrics,
            "native_nht_render": True,
            "rgb_overlay_used": False,
            "standard_3dgs_features_imported": False,
            "all_pose_frames_emitted": len(pose_ids),
            "dropped_pose_frames": [],
            "dropped_joint_indices": [],
            "status": "passed",
        }
        _write_json(temporary / "metrics.json", metrics)
        files = {
            path.relative_to(temporary).as_posix(): _file_ref(temporary, path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        unsigned = {
            "schema": SCHEMA,
            "status": "passed",
            "fixture": {
                "uri": (fixture_root / "manifest.json").as_uri(),
                "sha256": _sha256(fixture_root / "manifest.json"),
                "content_fingerprint": fixture["content_fingerprint"],
            },
            "target_appearance": {
                "uri": appearance_path.as_uri(),
                "sha256": _sha256(appearance_path),
                "appearance_space_sha256": args.appearance_space_sha256,
            },
            "renderer": runtime,
            "shader_config": shader_config,
            "optimization": {
                "seed": SEED,
                "steps": args.steps,
                "optimizer": "Adam",
                "initial_feature_lr": 0.015,
                "schedule": "cosine",
                "train_views": list(train_indices),
                "validation_views": [8, 9],
                "frozen": [
                    "means",
                    "quats",
                    "scales",
                    "opacities",
                    "target_deferred_shader",
                ],
                "optimized": ["nht_features"],
            },
            "camera": {
                "model": "opencv-camera-to-asset-v1",
                "width": args.width,
                "height": args.height,
                "focal_length_px": focal,
                "orbit_radius_m": 3.0,
                "view_count": view_count,
            },
            "metrics": metrics,
            "files": files,
        }
        manifest = {
            **unsigned,
            "content_fingerprint": _canonical_sha256(unsigned),
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"content_fingerprint={manifest['content_fingerprint']}")


if __name__ == "__main__":
    main()
