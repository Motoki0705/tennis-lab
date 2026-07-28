#!/usr/bin/env python3
"""Render compact production-NHT RGB videos and diagnostic label overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gsplat.rendering import rasterization
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.blcs.planner import (  # noqa: E402
    load_blcs_gaussian_plan,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.composition.gaussians import (  # noqa: E402
    GaussianTensorSet,
    compose_gaussians,
    transform_gaussians,
)
from src.synthetic_data_generation.court.labels import (  # noqa: E402
    MultiCourtProjection,
    project_multi_court,
    rescale_projection,
)
from src.synthetic_data_generation.court.layout import (  # noqa: E402
    MultiCourtLayout,
    load_multi_court_layout,
)
from src.synthetic_data_generation.scene_contract import (  # noqa: E402
    SceneCamera,
    load_scene_contract,
)
from third_party.nht.blcs_render import (  # noqa: E402
    _canonical_sha256,
    _git_dirty,
    _git_head,
    _load_shader,
    _load_tensor_set,
)
from third_party.nht.court_orbit_render import _load_verified_plan  # noqa: E402
from third_party.nht.plcs_render import (  # noqa: E402
    _load_plan as _load_plcs_plan,
)
from third_party.nht.plcs_render import (  # noqa: E402
    _load_pose_assets,
    _similarity_from_matrix,
)

SCHEMA = "tennis_production_nht_preview_v1"
COURT_COLORS = ((255, 210, 40), (60, 220, 255))
OBJECT_COLORS = ((255, 70, 70), (80, 255, 120), (210, 90, 255), (255, 150, 40))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_indices(value: str, *, frame_count: int) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("frame-indices must be comma-separated integers.") from error
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or indices[0] < 0
        or indices[-1] >= frame_count
    ):
        raise ValueError(
            f"frame-indices must be strictly increasing inside [0, {frame_count})."
        )
    return indices


def _camera_tensors(
    camera: SceneCamera,
    *,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
    # yuv420p requires even dimensions. Round to the nearest even height once,
    # then use that exact value for both intrinsics and label rescaling.
    height = max(2, 2 * round(camera.height * width / camera.width / 2.0))
    scale_x = width / camera.width
    scale_y = height / camera.height
    camera_to_scene = torch.tensor(
        camera.camera_to_scene,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 4, 4)
    intrinsics = torch.tensor(
        camera.intrinsics,
        dtype=torch.float32,
        device=device,
    ).reshape(3, 3)
    intrinsics[0] *= scale_x
    intrinsics[1] *= scale_y
    return camera_to_scene, intrinsics[None], height, scale_x, scale_y


def _render_rgb(
    scene: GaussianTensorSet,
    *,
    shader: torch.nn.Module,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        features, alpha, _ = rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=torch.exp(scene.log_scales),
            opacities=torch.sigmoid(scene.opacity_logits),
            colors=scene.features,
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
        rgb, extras = shader(features)
        if extras is None or extras.shape[-1] != 1:
            raise RuntimeError("Production NHT render returned no expected depth.")
        rgb = (rgb[..., :3] + (1.0 - alpha)).clamp(0.0, 1.0)
        torch.cuda.synchronize()
    if not bool(torch.isfinite(rgb).all()):
        raise RuntimeError("Production NHT RGB contains non-finite values.")
    array = rgb[0].mul(255.0).round().to(torch.uint8).cpu().numpy()
    return array, float(alpha.mean())


def _load_background(
    composition_path: Path,
    *,
    device: torch.device,
) -> tuple[Any, GaussianTensorSet, torch.nn.Module, dict[str, object]]:
    composition = load_gaussian_scene_manifest(composition_path)
    background_asset = composition.background
    verify_local_gaussian_asset(background_asset)
    gsplat_path = Path(__file__).resolve().parent / "upstream" / "gsplat"
    renderer_commit = _git_head(gsplat_path)
    if renderer_commit != composition.renderer_commit or _git_dirty(gsplat_path):
        raise RuntimeError("Production renderer checkout differs or is dirty.")
    background = _load_tensor_set(
        background_asset,
        instance_id=0,
        device=device,
    )
    shader, shader_config = _load_shader(
        background_asset.appearance_payload,
        feature_dim=background.feature_dim,
        device=device,
    )
    return composition, background, shader, shader_config


def _load_layout(orbit_plan_root: Path) -> tuple[dict[str, Any], MultiCourtLayout]:
    orbit_plan = _load_verified_plan(orbit_plan_root)
    source = orbit_plan["source"]
    contract = load_scene_contract(Path(source["scene_contract"]["path"]))
    layout = load_multi_court_layout(
        Path(source["court_geometry"]["path"]),
        contract,
        candidate_ids=("court-0", "court-1"),
    )
    return orbit_plan, layout


def _projection_for_camera(
    camera: SceneCamera,
    *,
    layout: MultiCourtLayout,
    width: int,
    height: int,
) -> MultiCourtProjection:
    return rescale_projection(
        project_multi_court(camera, layout),
        width=width,
        height=height,
    )


def _draw_court_points(
    draw: ImageDraw.ImageDraw,
    projection: MultiCourtProjection,
) -> None:
    for court_index, court in enumerate(projection.courts):
        color = COURT_COLORS[court_index % len(COURT_COLORS)]
        for class_record in court.classes:
            for point in class_record.points:
                if not point.in_frame:
                    continue
                x, y = point.uv
                radius = 4
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline=(0, 0, 0),
                    width=2,
                )
                draw.text(
                    (x + 5, y - 7),
                    f"C{court_index}:{class_record.class_id}",
                    fill=color,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )


def _annotate(
    rgb: np.ndarray,
    *,
    title: str,
    projection: MultiCourtProjection,
    objects: list[dict[str, object]],
) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    _draw_court_points(draw, projection)
    draw.rectangle((0, 0, image.width, 25), fill=(0, 0, 0))
    draw.text((7, 6), title, fill=(255, 255, 255))
    draw.text(
        (image.width - 270, 6),
        "overlay=diagnostic only | raw RGB unchanged",
        fill=(200, 200, 200),
    )
    for object_index, record in enumerate(objects):
        if record.get("visible") is not True:
            continue
        uv = record.get("uv")
        if not isinstance(uv, tuple):
            continue
        x, y = uv
        color = OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
        radius = 10
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=3,
        )
        draw.line((x - 14, y, x + 14, y), fill=color, width=2)
        draw.line((x, y - 14, x, y + 14), fill=color, width=2)
        draw.text(
            (x + 12, y + 3),
            str(record["label"]),
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(image)


def _encode_video(frame_dir: Path, output: Path, *, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.8g}",
            "-i",
            str(frame_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _contact_sheet(
    root: Path,
    *,
    raw_frames: list[np.ndarray],
    overlay_frames: list[np.ndarray],
) -> Path:
    picks = tuple(sorted({0, len(raw_frames) // 2, len(raw_frames) - 1}))
    width = raw_frames[0].shape[1]
    height = raw_frames[0].shape[0]
    canvas = Image.new("RGB", (width * len(picks), height * 2), (20, 20, 20))
    for column, index in enumerate(picks):
        canvas.paste(Image.fromarray(raw_frames[index]), (column * width, 0))
        canvas.paste(
            Image.fromarray(overlay_frames[index]),
            (column * width, height),
        )
    path = root / "contact-sheet.jpg"
    canvas.save(path, quality=90)
    return path


def _publish(
    output_dir: Path,
    *,
    task: str,
    fps: float,
    composition_path: Path,
    source: dict[str, object],
    frame_records: list[dict[str, object]],
    raw_frames: list[np.ndarray],
    overlay_frames: list[np.ndarray],
    shader_config: dict[str, object],
    alpha_means: list[float],
) -> dict[str, object]:
    temporary = output_dir
    raw_video = temporary / "rgb.mp4"
    overlay_video = temporary / "rgb-with-diagnostic-overlay.mp4"
    _encode_video(temporary / "frames", raw_video, fps=fps)
    _encode_video(temporary / "overlays", overlay_video, fps=fps)
    contact_sheet = _contact_sheet(
        temporary,
        raw_frames=raw_frames,
        overlay_frames=overlay_frames,
    )
    raw_values = np.concatenate(
        [frame.reshape(-1, 3) for frame in raw_frames],
        axis=0,
    ).astype(np.float32)
    unsigned = {
        "schema": SCHEMA,
        "status": "passed",
        "task": task,
        "source": source,
        "background_composition": {
            "path": str(composition_path),
            "sha256": _sha256(composition_path),
        },
        "renderer": {
            "backend": "nht-gsplat",
            "mode": "RGB+ED",
            "public_rasterization_calls_per_frame": 1,
            "rgb_overlay_used_in_native_render": False,
            "shader_config": shader_config,
        },
        "video": {
            "fps": fps,
            "frame_count": len(frame_records),
            "duration_seconds": len(frame_records) / fps,
            "raw_rgb": _file_ref(temporary, raw_video),
            "diagnostic_overlay": _file_ref(temporary, overlay_video),
            "contact_sheet": _file_ref(temporary, contact_sheet),
        },
        "frames": frame_records,
        "metrics": {
            "rgb_min": float(raw_values.min() / 255.0),
            "rgb_max": float(raw_values.max() / 255.0),
            "rgb_mean": float(raw_values.mean() / 255.0),
            "rgb_std": float(raw_values.std() / 255.0),
            "alpha_mean_min": min(alpha_means),
            "alpha_mean_max": max(alpha_means),
            "all_frames_finite": True,
        },
        "overlay_policy": {
            "diagnostic_only": True,
            "raw_rgb_unchanged": True,
            "court_points": "two instances; seven near/far-symmetric classes",
            "instance_grouping_training_target": False,
        },
    }
    manifest = {
        **unsigned,
        "content_fingerprint": _canonical_sha256(unsigned),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _save_frame_pair(
    root: Path,
    *,
    output_index: int,
    rgb: np.ndarray,
    overlay: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    raw_path = root / "frames" / f"{output_index:06d}.png"
    overlay_path = root / "overlays" / f"{output_index:06d}.png"
    Image.fromarray(rgb, mode="RGB").save(raw_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    return (
        {
            "output_index": output_index,
            "raw_rgb": _file_ref(root, raw_path),
            "diagnostic_overlay": _file_ref(root, overlay_path),
        },
        rgb,
        overlay,
    )


def _render_blcs(
    args: argparse.Namespace,
    *,
    root: Path,
    layout: MultiCourtLayout,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray], list[float], dict[str, object], dict[str, object]]:
    plan = load_blcs_gaussian_plan(args.plan_dir)
    indices = _parse_indices(args.frame_indices, frame_count=plan.num_frames)
    camera_matches = [
        (index, camera)
        for index, camera in enumerate(plan.cameras)
        if camera.camera_id == args.camera_id
    ]
    if len(camera_matches) != 1:
        raise ValueError(f"Expected one BLCS camera {args.camera_id!r}.")
    camera_index, camera = camera_matches[0]
    composition, background, shader, shader_config = _load_background(
        args.background_composition,
        device=device,
    )
    if plan.registry.appearance_space_sha256 != background.appearance_space_sha256:
        raise RuntimeError("BLCS plan and production background appearance differ.")
    asset_cache = {
        entry.asset.asset_id: _load_tensor_set(
            entry.asset,
            instance_id=1,
            device=device,
        )
        for entry in plan.registry.entries
    }
    camera_to_scene, intrinsics, height, scale_x, scale_y = _camera_tensors(
        camera,
        width=args.width,
        device=device,
    )
    projection = _projection_for_camera(
        camera,
        layout=layout,
        width=args.width,
        height=height,
    )
    records: list[dict[str, object]] = []
    raw_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    alpha_means: list[float] = []
    for output_index, frame_index in enumerate(indices):
        movable = []
        for instance in plan.instances_at(frame_index):
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
            movable.append(transform_gaussians(with_identity, instance.scene_from_asset))
        scene = compose_gaussians(background, movable)
        rgb, alpha_mean = _render_rgb(
            scene,
            shader=shader,
            camera_to_scene=camera_to_scene,
            intrinsics=intrinsics,
            width=args.width,
            height=height,
        )
        objects = []
        for object_index, instance_id in enumerate(plan.instance_ids):
            objects.append(
                {
                    "label": f"B{int(instance_id)}",
                    "visible": bool(
                        plan.camera_geometric_visible[
                            camera_index,
                            frame_index,
                            object_index,
                        ]
                    ),
                    "uv": (
                        float(
                            plan.camera_uv[
                                camera_index,
                                frame_index,
                                object_index,
                                0,
                            ]
                            * scale_x
                        ),
                        float(
                            plan.camera_uv[
                                camera_index,
                                frame_index,
                                object_index,
                                1,
                            ]
                            * scale_y
                        ),
                    ),
                }
            )
        overlay = _annotate(
            rgb,
            title=(
                f"BLCS multi-ball | seed={plan.seed} | camera={camera.camera_id} "
                f"| frame={frame_index}"
            ),
            projection=projection,
            objects=objects,
        )
        record, raw, diagnostic = _save_frame_pair(
            root,
            output_index=output_index,
            rgb=rgb,
            overlay=overlay,
        )
        records.append(
            {
                **record,
                "source_frame_index": frame_index,
                "active_instance_ids": [
                    instance.instance_id for instance in plan.instances_at(frame_index)
                ],
            }
        )
        raw_frames.append(raw)
        overlay_frames.append(diagnostic)
        alpha_means.append(alpha_mean)
    source = {
        "plan": str(args.plan_dir),
        "plan_fingerprint": plan.plan_fingerprint,
        "seed": plan.seed,
        "camera_id": camera.camera_id,
        "sampled_frame_indices": list(indices),
        "object_count": plan.num_objects,
        "production_composition_fingerprint": composition.composition_fingerprint,
    }
    return records, raw_frames, overlay_frames, alpha_means, source, shader_config


def _render_plcs(
    args: argparse.Namespace,
    *,
    root: Path,
    layout: MultiCourtLayout,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray], list[float], dict[str, object], dict[str, object]]:
    manifest, arrays, camera = _load_plcs_plan(args.plan_dir)
    indices = _parse_indices(
        args.frame_indices,
        frame_count=int(manifest["frame_count"]),
    )
    composition, background, shader, shader_config = _load_background(
        args.background_composition,
        device=device,
    )
    if _sha256(args.background_composition) != manifest["background_composition"]["sha256"]:
        raise RuntimeError("PLCS plan references another background composition.")
    appearance_space = manifest["avatar_nht_manifest"]["appearance_space_sha256"]
    if background.appearance_space_sha256 != appearance_space:
        raise RuntimeError("PLCS avatar and production background appearance differ.")
    pose_assets = _load_pose_assets(
        manifest,
        appearance_space_sha256=appearance_space,
        device=device,
    )
    camera_to_scene, intrinsics, height, scale_x, scale_y = _camera_tensors(
        camera,
        width=args.width,
        device=device,
    )
    projection = _projection_for_camera(
        camera,
        layout=layout,
        width=args.width,
        height=height,
    )
    records: list[dict[str, object]] = []
    raw_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    alpha_means: list[float] = []
    for output_index, frame_index in enumerate(indices):
        movable = []
        objects = []
        for person_index, instance_id in enumerate(arrays["instance_ids"]):
            pose_index = int(arrays["pose_indices"][frame_index, person_index])
            pose = pose_assets[pose_index]
            with_identity = GaussianTensorSet(
                means=pose.means,
                quats=pose.quats,
                log_scales=pose.log_scales,
                opacity_logits=pose.opacity_logits,
                features=pose.features,
                instance_ids=torch.full_like(pose.instance_ids, int(instance_id)),
                appearance_space_sha256=pose.appearance_space_sha256,
            )
            movable.append(
                transform_gaussians(
                    with_identity,
                    _similarity_from_matrix(
                        arrays["scene_from_asset"][frame_index, person_index]
                    ),
                )
            )
            objects.append(
                {
                    "label": (
                        f"P{int(instance_id)}:"
                        f"{manifest['pose_assets'][pose_index]['pose_id']}"
                    ),
                    "visible": bool(
                        arrays["camera_geometric_visible"][
                            frame_index,
                            person_index,
                        ]
                    ),
                    "uv": (
                        float(
                            arrays["camera_uv"][frame_index, person_index, 0]
                            * scale_x
                        ),
                        float(
                            arrays["camera_uv"][frame_index, person_index, 1]
                            * scale_y
                        ),
                    ),
                }
            )
        scene = compose_gaussians(background, movable)
        rgb, alpha_mean = _render_rgb(
            scene,
            shader=shader,
            camera_to_scene=camera_to_scene,
            intrinsics=intrinsics,
            width=args.width,
            height=height,
        )
        overlay = _annotate(
            rgb,
            title=(
                f"PLCS multi-person | seed={manifest['seed']} "
                f"| camera={camera.camera_id} | frame={frame_index}"
            ),
            projection=projection,
            objects=objects,
        )
        record, raw, diagnostic = _save_frame_pair(
            root,
            output_index=output_index,
            rgb=rgb,
            overlay=overlay,
        )
        records.append({**record, "source_frame_index": frame_index})
        raw_frames.append(raw)
        overlay_frames.append(diagnostic)
        alpha_means.append(alpha_mean)
    source = {
        "plan": str(args.plan_dir),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "seed": manifest["seed"],
        "camera_id": camera.camera_id,
        "sampled_frame_indices": list(indices),
        "person_count": manifest["person_count"],
        "production_composition_fingerprint": composition.composition_fingerprint,
    }
    return records, raw_frames, overlay_frames, alpha_means, source, shader_config


def _render_court_or_alignment(
    args: argparse.Namespace,
    *,
    root: Path,
    orbit_plan: dict[str, Any],
    layout: MultiCourtLayout,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray], list[float], dict[str, object], dict[str, object]]:
    composition, background, shader, shader_config = _load_background(
        args.background_composition,
        device=device,
    )
    if args.task == "court":
        frames = [
            frame
            for frame in orbit_plan["frames"]
            if frame["family_id"] == args.family_id
        ]
        if not frames:
            raise ValueError(f"Unknown court orbit family: {args.family_id!r}.")
        selected = frames[:: args.stride]
    else:
        contract = load_scene_contract(Path(args.scene_contract))
        camera_matches = [
            camera
            for camera in contract.cameras
            if camera.camera_id == args.camera_id
        ]
        if len(camera_matches) != 1:
            raise ValueError(f"Expected one alignment camera {args.camera_id!r}.")
        selected = [
            {
                "camera": camera_matches[0].to_dict(),
                "family_id": "captured-sfm-alignment",
                "family_frame_index": 0,
                "projection": asdict(project_multi_court(camera_matches[0], layout)),
            }
        ]
    records: list[dict[str, object]] = []
    raw_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []
    alpha_means: list[float] = []
    for output_index, plan_frame in enumerate(selected):
        camera = SceneCamera.from_dict(plan_frame["camera"])
        projection = project_multi_court(camera, layout)
        if _canonical_sha256(asdict(projection)) != _canonical_sha256(
            plan_frame["projection"]
        ):
            raise RuntimeError("Stored court projection changed.")
        camera_to_scene, intrinsics, height, _, _ = _camera_tensors(
            camera,
            width=args.width,
            device=device,
        )
        resized = rescale_projection(
            projection,
            width=args.width,
            height=height,
        )
        rgb, alpha_mean = _render_rgb(
            background,
            shader=shader,
            camera_to_scene=camera_to_scene,
            intrinsics=intrinsics,
            width=args.width,
            height=height,
        )
        overlay = _annotate(
            rgb,
            title=(
                f"court alignment | family={plan_frame['family_id']} "
                f"| orbit-frame={plan_frame['family_frame_index']}"
            ),
            projection=resized,
            objects=[],
        )
        record, raw, diagnostic = _save_frame_pair(
            root,
            output_index=output_index,
            rgb=rgb,
            overlay=overlay,
        )
        records.append(
            {
                **record,
                "camera_id": camera.camera_id,
                "family_frame_index": plan_frame["family_frame_index"],
                "coverage": {
                    court.court_instance_id: court.coverage_bucket
                    for court in resized.courts
                },
            }
        )
        raw_frames.append(raw)
        overlay_frames.append(diagnostic)
        alpha_means.append(alpha_mean)
    source = {
        "orbit_plan": str(args.court_orbit_plan),
        "orbit_content_fingerprint": orbit_plan["content_fingerprint"],
        "family_id": selected[0]["family_id"],
        "sampled_family_frame_indices": [
            frame["family_frame_index"] for frame in selected
        ],
        "production_composition_fingerprint": composition.composition_fingerprint,
    }
    return records, raw_frames, overlay_frames, alpha_means, source, shader_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--court-orbit-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    blcs = subparsers.add_parser("blcs")
    _add_common(blcs)
    blcs.add_argument("--plan-dir", type=Path, required=True)
    blcs.add_argument("--camera-id", required=True)
    blcs.add_argument("--frame-indices", required=True)

    plcs = subparsers.add_parser("plcs")
    _add_common(plcs)
    plcs.add_argument("--plan-dir", type=Path, required=True)
    plcs.add_argument("--frame-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")

    court = subparsers.add_parser("court")
    _add_common(court)
    court.add_argument("--family-id", required=True)
    court.add_argument("--stride", type=int, default=1)

    alignment = subparsers.add_parser("alignment")
    _add_common(alignment)
    alignment.add_argument("--scene-contract", type=Path, required=True)
    alignment.add_argument("--camera-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output: {args.output_dir}")
    if args.width <= 1 or not np.isfinite(args.fps) or args.fps <= 0.0:
        raise SystemExit("width and fps must be positive.")
    if args.task == "court" and args.stride <= 0:
        raise SystemExit("stride must be positive.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    args.background_composition = args.background_composition.resolve()
    args.court_orbit_plan = args.court_orbit_plan.resolve()
    if not args.background_composition.is_file():
        raise SystemExit("Background composition is missing.")
    orbit_plan, layout = _load_layout(args.court_orbit_plan)
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    (temporary / "frames").mkdir()
    (temporary / "overlays").mkdir()
    try:
        device = torch.device("cuda:0")
        if args.task == "blcs":
            result = _render_blcs(args, root=temporary, layout=layout, device=device)
        elif args.task == "plcs":
            result = _render_plcs(args, root=temporary, layout=layout, device=device)
        else:
            result = _render_court_or_alignment(
                args,
                root=temporary,
                orbit_plan=orbit_plan,
                layout=layout,
                device=device,
            )
        records, raw, overlays, alpha, source, shader_config = result
        manifest = _publish(
            temporary,
            task=args.task,
            fps=args.fps,
            composition_path=args.background_composition,
            source=source,
            frame_records=records,
            raw_frames=raw,
            overlay_frames=overlays,
            shader_config=shader_config,
            alpha_means=alpha,
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "task": manifest["task"],
                "content_fingerprint": manifest["content_fingerprint"],
                "frame_count": manifest["video"]["frame_count"],
                "duration_seconds": manifest["video"]["duration_seconds"],
                "raw_rgb": manifest["video"]["raw_rgb"],
                "diagnostic_overlay": manifest["video"]["diagnostic_overlay"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
