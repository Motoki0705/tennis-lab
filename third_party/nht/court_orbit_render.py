#!/usr/bin/env python3
"""Render representative multi-court orbit views with native NHT Gaussians."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib
import numpy as np
import torch
from gsplat.rendering import rasterization
from PIL import Image, ImageDraw

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.court.labels import (  # noqa: E402
    PHYSICAL_INDICES_BY_SYMMETRIC_CLASS,
    SYMMETRIC_KEYPOINT_CLASS_NAMES,
    attach_visibility,
    build_seven_channel_heatmaps,
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
    _relative_file_ref,
    _sha256_file,
)

PLAN_SCHEMA = "tennis_multicourt_orbit_plan_v1"
RENDER_SCHEMA = "tennis_multicourt_orbit_render_v1"
FRAME_LABEL_SCHEMA = "tennis_multicourt_seven_class_frame_v1"
VISIBILITY_METHOD = "nht-expected-depth-local-consistency-v1"
_CLASS_COLOURS = (
    (239, 71, 111),
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
    (131, 56, 236),
    (255, 127, 80),
    (255, 255, 255),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--frames-per-family", type=int, default=1)
    parser.add_argument("--heatmap-sigma-px", type=float, default=2.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--visibility-sample-radius-px", type=int, default=2)
    return parser.parse_args()


def _load_verified_plan(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PLAN_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("Orbit plan must be a passed v1 multi-court plan.")
    declared = payload.get("content_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("content_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("Orbit plan content fingerprint mismatch.")
    expected_policy = {
        "model_heatmap_channel_count": 7,
        "near_far_symmetry_removed": True,
        "court_instance_retained_in_annotation": True,
        "court_instance_grouping_is_training_target": False,
        "court_instance_grouping_stage": "postprocess",
        "multi_peak_composition": "pixelwise-maximum",
        "maximum_physical_peaks_per_channel": 4,
    }
    if payload.get("label_schema") != expected_policy:
        raise RuntimeError("Orbit plan label policy differs from the approved schema.")
    for record in payload["source"].values():
        source_path = Path(record["path"])
        if not source_path.is_file() or _sha256_file(source_path) != record["sha256"]:
            raise RuntimeError(f"Orbit plan source changed: {source_path}")
    return payload


def _coverage_counts(frame: dict[str, Any]) -> dict[str, int]:
    return {
        court["court_instance_id"]: sum(
            point["in_frame"]
            for class_record in court["classes"]
            for point in class_record["points"]
        )
        for court in frame["projection"]["courts"]
    }


def _representative_frames(
    plan: dict[str, Any],
    *,
    frames_per_family: int,
) -> list[dict[str, Any]]:
    if frames_per_family != 1:
        raise ValueError("The visual-probe v1 contract requires one frame per family.")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in plan["frames"]:
        by_family[frame["family_id"]].append(frame)
    selected = []
    for family in plan["sampling"]["families"]:
        candidates = by_family[family["family_id"]]
        target_id = family["target_court_instance_id"]
        midpoint = (family["sample_count"] - 1) / 2.0

        def score(
            frame: dict[str, Any],
            *,
            midpoint: float = midpoint,
            target_id: str | None = target_id,
        ) -> tuple[float, ...]:
            counts = _coverage_counts(frame)
            phase_distance = abs(frame["family_frame_index"] - midpoint)
            if target_id is None:
                values = tuple(counts.values())
                return (min(values), sum(values), -phase_distance)
            other_counts = [
                count for court_id, count in counts.items() if court_id != target_id
            ]
            other = max(other_counts, default=0)
            return (
                counts[target_id],
                -abs(other - 7),
                float(other > 0),
                -phase_distance,
            )

        selected.append(max(candidates, key=score))
    return selected


def _camera_tensors(
    camera: SceneCamera,
    *,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    height = max(2, round(camera.height * width / camera.width))
    intrinsics = torch.tensor(
        camera.intrinsics,
        dtype=torch.float32,
        device=device,
    ).reshape(3, 3)
    intrinsics[0] *= width / camera.width
    intrinsics[1] *= height / camera.height
    camera_to_scene = torch.tensor(
        camera.camera_to_scene,
        dtype=torch.float32,
        device=device,
    ).reshape(4, 4)
    return camera_to_scene.unsqueeze(0), intrinsics.unsqueeze(0), height


def _render_background(
    *,
    background: Any,
    shader: Any,
    camera_to_scene: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        features, alpha, _ = rasterization(
            means=background.means,
            quats=background.quats,
            scales=torch.exp(background.log_scales),
            opacities=torch.sigmoid(background.opacity_logits),
            colors=background.features,
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
            raise RuntimeError("NHT RGB+ED did not return one expected-depth channel.")
        depth = extras
        rgb = (rgb[..., :3] + (1.0 - alpha)).clamp(0.0, 1.0)
        torch.cuda.synchronize()
    if not all(bool(torch.isfinite(value).all()) for value in (rgb, alpha, depth)):
        raise RuntimeError("NHT orbit render contains non-finite values.")
    return (
        rgb[0].mul(255.0).round().to(torch.uint8).cpu().numpy(),
        alpha[0, ..., 0].float().cpu().numpy(),
        depth[0, ..., 0].float().cpu().numpy(),
    )


def _point_visibility(
    *,
    uv: tuple[float, float],
    point_depth: float,
    in_frame: bool,
    alpha: np.ndarray,
    depth: np.ndarray,
    radius: int,
    alpha_threshold: float,
    depth_absolute_tolerance: float,
    depth_relative_tolerance: float,
) -> tuple[bool, dict[str, object]]:
    if not in_frame:
        return False, {
            "evaluated": False,
            "reason": "outside-render-frame",
        }
    x = int(round(uv[0]))
    y = int(round(uv[1]))
    x0, x1 = max(0, x - radius), min(alpha.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(alpha.shape[0], y + radius + 1)
    alpha_patch = alpha[y0:y1, x0:x1]
    depth_patch = depth[y0:y1, x0:x1]
    valid = (
        np.isfinite(depth_patch)
        & (depth_patch > 0.0)
        & (alpha_patch >= alpha_threshold)
    )
    if not np.any(valid):
        return False, {
            "evaluated": True,
            "reason": "no-valid-depth-sample",
            "sample_alpha_max": float(alpha_patch.max(initial=0.0)),
        }
    errors = np.abs(depth_patch[valid] - point_depth)
    closest_index = int(np.argmin(errors))
    closest_depth = float(depth_patch[valid][closest_index])
    closest_alpha = float(alpha_patch[valid][closest_index])
    error = float(errors[closest_index])
    tolerance = depth_absolute_tolerance + depth_relative_tolerance * abs(point_depth)
    visible = error <= tolerance
    return visible, {
        "evaluated": True,
        "reason": "depth-consistent" if visible else "depth-occluded-or-unsupported",
        "sample_alpha": closest_alpha,
        "sample_depth_scene": closest_depth,
        "point_depth_scene": point_depth,
        "depth_error_scene": error,
        "depth_tolerance_scene": tolerance,
    }


def _attach_render_visibility(
    projection: Any,
    *,
    alpha: np.ndarray,
    depth: np.ndarray,
    sample_radius_px: int,
    alpha_threshold: float,
    depth_absolute_tolerance: float,
    depth_relative_tolerance: float,
) -> tuple[Any, dict[str, list[dict[str, object]]]]:
    visibility_by_court = {}
    samples_by_court = {}
    for court in projection.courts:
        physical_visibility = [False] * 14
        physical_samples: list[dict[str, object]] = [{} for _ in range(14)]
        for class_record in court.classes:
            physical_indices = PHYSICAL_INDICES_BY_SYMMETRIC_CLASS[
                class_record.class_id
            ]
            for point, physical_index in zip(
                class_record.points,
                physical_indices,
                strict=True,
            ):
                visible, sample = _point_visibility(
                    uv=point.uv,
                    point_depth=point.depth_scene,
                    in_frame=point.in_frame,
                    alpha=alpha,
                    depth=depth,
                    radius=sample_radius_px,
                    alpha_threshold=alpha_threshold,
                    depth_absolute_tolerance=depth_absolute_tolerance,
                    depth_relative_tolerance=depth_relative_tolerance,
                )
                physical_visibility[physical_index] = visible
                physical_samples[physical_index] = {
                    "physical_index": physical_index,
                    "class_id": class_record.class_id,
                    "uv_render_pixels": list(point.uv),
                    **sample,
                }
        visibility_by_court[court.court_instance_id] = tuple(physical_visibility)
        samples_by_court[court.court_instance_id] = physical_samples
    return attach_visibility(projection, visibility_by_court), samples_by_court


def _write_frame(
    root: Path,
    *,
    selected_index: int,
    plan_frame: dict[str, Any],
    rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    heatmaps: np.ndarray,
    labels: dict[str, object],
) -> dict[str, object]:
    frame_dir = root / "frames" / f"frame_{selected_index:03d}"
    frame_dir.mkdir(parents=True)
    paths = {
        "rgb": frame_dir / "rgb.png",
        "alpha": frame_dir / "alpha.npy",
        "depth": frame_dir / "depth.npy",
        "heatmaps": frame_dir / "heatmaps_7ch.npy",
        "labels": frame_dir / "labels.json",
    }
    Image.fromarray(rgb, mode="RGB").save(paths["rgb"])
    np.save(paths["alpha"], alpha, allow_pickle=False)
    np.save(paths["depth"], depth, allow_pickle=False)
    np.save(paths["heatmaps"], heatmaps, allow_pickle=False)
    paths["labels"].write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_index": selected_index,
        "family_id": plan_frame["family_id"],
        "family_frame_index": plan_frame["family_frame_index"],
        "camera_id": plan_frame["camera"]["camera_id"],
        **{
            name: _relative_file_ref(root, path)
            for name, path in paths.items()
        },
    }


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    colour: tuple[int, int, int],
    court_index: int,
    visible: bool,
) -> None:
    x, y = xy
    radius = 4
    box = (x - radius, y - radius, x + radius, y + radius)
    if court_index == 0:
        if visible:
            draw.ellipse(box, fill=colour, outline=(0, 0, 0), width=1)
        else:
            draw.ellipse(box, outline=colour, width=2)
    elif visible:
        draw.rectangle(box, fill=colour, outline=(0, 0, 0), width=1)
    else:
        draw.rectangle(box, outline=colour, width=2)


def _contact_sheet(
    root: Path,
    visual_frames: list[tuple[np.ndarray, dict[str, object]]],
) -> Path:
    columns = 3
    tile_width = visual_frames[0][0].shape[1]
    image_height = visual_frames[0][0].shape[0]
    caption_height = 48
    rows = (len(visual_frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + caption_height)),
        (24, 24, 24),
    )
    for index, (rgb, labels) in enumerate(visual_frames):
        tile = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(tile)
        projection = labels["projection"]
        for court_index, court in enumerate(projection["courts"]):
            for class_record in court["classes"]:
                colour = _CLASS_COLOURS[class_record["class_id"]]
                for point in class_record["points"]:
                    if point["in_frame"]:
                        _draw_marker(
                            draw,
                            xy=tuple(point["uv"]),
                            colour=colour,
                            court_index=court_index,
                            visible=point["visible"],
                        )
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + caption_height)
        sheet.paste(tile, (x, y))
        sheet_draw = ImageDraw.Draw(sheet)
        caption = (
            f"{labels['family_id']}\n"
            f"coverage={labels['geometric_coverage']} "
            f"visible={labels['renderer_visible_points']} "
            f"d={labels['nearest_captured_translation_m']:.1f}m "
            f"r={labels['nearest_captured_rotation_deg']:.1f}deg"
        )
        sheet_draw.multiline_text(
            (x + 4, y + image_height + 3),
            caption,
            fill=(240, 240, 240),
            spacing=2,
        )
    path = root / "diagnostics" / "representative-contact-sheet.png"
    path.parent.mkdir(parents=True)
    sheet.save(path)
    return path


def _trajectory_plot(
    root: Path,
    *,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    layout: MultiCourtLayout,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 8))
    by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame in plan["frames"]:
        camera = SceneCamera.from_dict(frame["camera"])
        center_scene = np.asarray(camera.camera_to_scene).reshape(4, 4)[:3, 3]
        center_reference = layout.reference.court_from_scene.apply(
            center_scene[None]
        )[0]
        by_family[frame["family_id"]].append(center_reference)
    for family_id, centers in by_family.items():
        array = np.stack(centers)
        colour = "#4477AA" if family_id.startswith("circle") else "#66CCEE"
        axis.plot(array[:, 0], array[:, 1], color=colour, alpha=0.35, linewidth=1)
    selected_centers = []
    for frame in selected:
        camera = SceneCamera.from_dict(frame["camera"])
        center_scene = np.asarray(camera.camera_to_scene).reshape(4, 4)[:3, 3]
        selected_centers.append(
            layout.reference.court_from_scene.apply(center_scene[None])[0]
        )
    selected_array = np.stack(selected_centers)
    axis.scatter(
        selected_array[:, 0],
        selected_array[:, 1],
        c="#EE3377",
        s=35,
        label="rendered representatives",
        zorder=4,
    )
    for court in layout.courts:
        points_scene = court.keypoints_scene()[:14]
        points = layout.reference.court_from_scene.apply(points_scene)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="x",
            s=30,
            label=court.court_instance_id,
        )
    axis.set(
        title="SfM-envelope circle/ellipse families around the two-court complex",
        xlabel="reference-court x (m)",
        ylabel="reference-court y (m)",
        aspect="equal",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = root / "diagnostics" / "orbit-trajectories.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _verify_output(root: Path) -> dict[str, object]:
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != RENDER_SCHEMA:
        raise RuntimeError("Unsupported multi-court orbit render schema.")
    declared = payload.get("render_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("render_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("Multi-court orbit render fingerprint mismatch.")
    records = [
        *(record[name] for record in payload["frames"] for name in (
            "rgb",
            "alpha",
            "depth",
            "heatmaps",
            "labels",
        )),
        *payload["diagnostics"].values(),
    ]
    for record in records:
        relative = PurePosixPath(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe output path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Missing output: {relative}")
        if (
            path.stat().st_size != record["size_bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Output integrity mismatch: {relative}")
    return {
        "render_fingerprint": declared,
        "frame_count": len(payload["frames"]),
        "family_count": payload["metrics"]["family_count"],
        "renderer_visible_point_count": payload["metrics"][
            "renderer_visible_point_count"
        ],
        "status": payload["status"],
    }


def main() -> None:
    args = _parse_args()
    plan_dir = args.plan_dir.resolve()
    composition_path = args.background_composition.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1 or args.frames_per_family <= 0:
        raise SystemExit("Render dimensions and frame count must be positive.")
    if args.visibility_sample_radius_px < 0:
        raise SystemExit("visibility-sample-radius-px must be non-negative.")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise SystemExit("alpha-threshold must lie in [0, 1].")
    if (
        args.depth_absolute_tolerance < 0.0
        or args.depth_relative_tolerance < 0.0
    ):
        raise SystemExit("Depth tolerances must be non-negative.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    plan = _load_verified_plan(plan_dir)
    selected = _representative_frames(
        plan,
        frames_per_family=args.frames_per_family,
    )
    source = plan["source"]
    contract = load_scene_contract(Path(source["scene_contract"]["path"]))
    layout = load_multi_court_layout(
        Path(source["court_geometry"]["path"]),
        contract,
        candidate_ids=("court-0", "court-1"),
    )
    composition = load_gaussian_scene_manifest(composition_path)
    background_asset = composition.background
    verify_local_gaussian_asset(background_asset)

    gsplat_path = Path(__file__).resolve().parent / "upstream" / "gsplat"
    renderer_commit = _git_head(gsplat_path)
    if renderer_commit != composition.renderer_commit:
        raise SystemExit("Renderer commit differs from background composition.")
    if _git_dirty(gsplat_path):
        raise SystemExit("Refusing a modified gsplat renderer checkout.")

    device = torch.device("cuda:0")
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
        visual_frames = []
        coverage_pairs: Counter[str] = Counter()
        renderer_visible_points = 0
        renderer_visible_by_class: Counter[int] = Counter()
        alpha_coverage = []
        rgb_standard_deviation = []
        maximum_visible_peaks = 0
        for selected_index, plan_frame in enumerate(selected):
            camera = SceneCamera.from_dict(plan_frame["camera"])
            projection = project_multi_court(camera, layout)
            if _canonical_sha256(asdict(projection)) != _canonical_sha256(
                plan_frame["projection"]
            ):
                raise RuntimeError(
                    f"Stored projection changed for {camera.camera_id}."
                )
            camera_to_scene, intrinsics, height = _camera_tensors(
                camera,
                width=args.width,
                device=device,
            )
            resized = rescale_projection(
                projection,
                width=args.width,
                height=height,
            )
            rgb, alpha, depth = _render_background(
                background=background,
                shader=shader,
                camera_to_scene=camera_to_scene,
                intrinsics=intrinsics,
                width=args.width,
                height=height,
            )
            visible_projection, visibility_samples = _attach_render_visibility(
                resized,
                alpha=alpha,
                depth=depth,
                sample_radius_px=args.visibility_sample_radius_px,
                alpha_threshold=args.alpha_threshold,
                depth_absolute_tolerance=args.depth_absolute_tolerance,
                depth_relative_tolerance=args.depth_relative_tolerance,
            )
            heatmaps = build_seven_channel_heatmaps(
                visible_projection,
                sigma_px=args.heatmap_sigma_px,
                require_renderer_visibility=True,
            )
            geometric_coverage = [
                court.coverage_bucket for court in visible_projection.courts
            ]
            coverage_pairs["|".join(geometric_coverage)] += 1
            visible_counts_by_class = []
            for class_id in range(len(SYMMETRIC_KEYPOINT_CLASS_NAMES)):
                count = sum(
                    point.visible is True
                    for court in visible_projection.courts
                    for point in court.classes[class_id].points
                )
                visible_counts_by_class.append(count)
                renderer_visible_by_class[class_id] += count
                maximum_visible_peaks = max(maximum_visible_peaks, count)
            visible_count = sum(visible_counts_by_class)
            renderer_visible_points += visible_count
            frame_alpha_coverage = float((alpha >= args.alpha_threshold).mean())
            frame_rgb_std = float(rgb.astype(np.float32).std() / 255.0)
            alpha_coverage.append(frame_alpha_coverage)
            rgb_standard_deviation.append(frame_rgb_std)
            labels: dict[str, object] = {
                "schema": FRAME_LABEL_SCHEMA,
                "plan_content_fingerprint": plan["content_fingerprint"],
                "family_id": plan_frame["family_id"],
                "family_frame_index": plan_frame["family_frame_index"],
                "camera": camera.to_dict(),
                "resolution": [args.width, height],
                "projection": asdict(visible_projection),
                "visibility_samples_by_court": visibility_samples,
                "visibility_method": VISIBILITY_METHOD,
                "geometric_coverage": geometric_coverage,
                "renderer_visible_points": visible_count,
                "renderer_visible_peaks_by_class": visible_counts_by_class,
                "nearest_captured_translation_m": plan_frame[
                    "nearest_captured_translation_m"
                ],
                "nearest_captured_rotation_deg": plan_frame[
                    "nearest_captured_rotation_deg"
                ],
                "collision_clearance_m": plan_frame["collision_clearance_m"],
                "alpha_coverage": frame_alpha_coverage,
                "rgb_standard_deviation": frame_rgb_std,
                "training_target": {
                    "heatmap_channels": 7,
                    "court_instance_grouping": False,
                    "multi_peak_composition": "pixelwise-maximum",
                },
                "rgb_overlay_used": False,
                "all_finite": True,
            }
            frame_records.append(
                _write_frame(
                    temporary,
                    selected_index=selected_index,
                    plan_frame=plan_frame,
                    rgb=rgb,
                    alpha=alpha,
                    depth=depth,
                    heatmaps=heatmaps,
                    labels=labels,
                )
            )
            visual_frames.append((rgb, labels))

        contact_sheet = _contact_sheet(temporary, visual_frames)
        trajectory_plot = _trajectory_plot(
            temporary,
            plan=plan,
            selected=selected,
            layout=layout,
        )
        metrics = {
            "family_count": len(plan["sampling"]["families"]),
            "rendered_frame_count": len(frame_records),
            "coverage_pair_counts": dict(sorted(coverage_pairs.items())),
            "renderer_visible_point_count": renderer_visible_points,
            "renderer_visible_points_by_class": {
                SYMMETRIC_KEYPOINT_CLASS_NAMES[class_id]: (
                    renderer_visible_by_class[class_id]
                )
                for class_id in range(len(SYMMETRIC_KEYPOINT_CLASS_NAMES))
            },
            "maximum_visible_peaks_in_one_channel": maximum_visible_peaks,
            "alpha_coverage": {
                "minimum": min(alpha_coverage),
                "median": float(np.median(alpha_coverage)),
                "maximum": max(alpha_coverage),
            },
            "rgb_standard_deviation": {
                "minimum": min(rgb_standard_deviation),
                "median": float(np.median(rgb_standard_deviation)),
                "maximum": max(rgb_standard_deviation),
            },
        }
        automatic_gate = (
            len(frame_records) == len(plan["sampling"]["families"])
            and renderer_visible_points > 0
            and metrics["maximum_visible_peaks_in_one_channel"] >= 2
            and metrics["alpha_coverage"]["minimum"] > 0.01
            and metrics["rgb_standard_deviation"]["minimum"] > 0.01
        )
        manifest: dict[str, object] = {
            "schema": RENDER_SCHEMA,
            "status": (
                "passed-automatic-awaiting-visual-review"
                if automatic_gate
                else "failed-automatic-awaiting-analysis"
            ),
            "plan": {
                "manifest_sha256": _sha256_file(plan_dir / "manifest.json"),
                "content_fingerprint": plan["content_fingerprint"],
            },
            "background_composition": {
                "manifest_sha256": _sha256_file(composition_path),
                "composition_fingerprint": composition.composition_fingerprint,
                "background_asset_id": background_asset.asset_id,
                "composition_instances_used": False,
            },
            "renderer": {
                "backend": "nht-gsplat",
                "commit": renderer_commit,
                "api_calls_per_frame": 1,
                "render_mode": "RGB+ED",
                "nht": True,
                "with_ut": True,
                "with_eval3d": True,
                "shader_config": shader_config,
            },
            "visibility": {
                "method": VISIBILITY_METHOD,
                "sample_radius_px": args.visibility_sample_radius_px,
                "alpha_threshold": args.alpha_threshold,
                "depth_absolute_tolerance": args.depth_absolute_tolerance,
                "depth_relative_tolerance": args.depth_relative_tolerance,
            },
            "training_target": {
                "heatmap_channels": 7,
                "near_far_symmetry_removed": True,
                "court_instance_grouping": False,
                "court_instance_grouping_stage": "postprocess",
                "heatmap_sigma_px": args.heatmap_sigma_px,
                "multi_peak_composition": "pixelwise-maximum",
            },
            "resolution": [args.width, visual_frames[0][0].shape[0]],
            "frames": frame_records,
            "diagnostics": {
                "representative_contact_sheet": _relative_file_ref(
                    temporary,
                    contact_sheet,
                ),
                "orbit_trajectories": _relative_file_ref(
                    temporary,
                    trajectory_plot,
                ),
            },
            "metrics": metrics,
            "automatic_gate_passed": automatic_gate,
            "visual_review": {
                "status": "pending",
                "reviewer": "Codex visual inspection",
            },
            "rgb_overlay_used": False,
            "all_finite": True,
        }
        manifest["render_fingerprint"] = _canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
        print(json.dumps(_verify_output(output_dir), indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
