#!/usr/bin/env python3
"""Publish the full family-disjoint multi-court NHT dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.alignment.scene_provider.bundle import (  # noqa: E402
    load_scene_provider_bundle,
)
from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.court.labels import (  # noqa: E402
    SYMMETRIC_KEYPOINT_CLASS_NAMES,
    build_seven_channel_heatmaps,
    encode_heatmap_atlas_u16,
    project_multi_court,
    rescale_projection,
)
from src.synthetic_data_generation.court.layout import (  # noqa: E402
    load_multi_court_layout,
)
from src.synthetic_data_generation.court.orbits import OrbitFamilySpec  # noqa: E402
from src.synthetic_data_generation.court.release import (  # noqa: E402
    DatasetSplit,
    assign_family_disjoint_splits,
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
    _local_artifact_path,
    _relative_file_ref,
    _sha256_file,
)
from third_party.nht.court_orbit_render import (  # noqa: E402
    VISIBILITY_METHOD,
    _attach_render_visibility,
    _camera_tensors,
    _load_verified_plan,
    _render_background,
)

DATASET_SCHEMA = "tennis_multicourt_nht_dataset_v1"
FRAME_SCHEMA = "tennis_multicourt_nht_dataset_frame_v1"
HEATMAP_ENCODING = "horizontal-seven-channel-u16-png-v1"
SPLIT_NAMES: tuple[DatasetSplit, ...] = ("train", "validation", "test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--split-seed", type=int, default=26072814)
    parser.add_argument("--heatmap-sigma-px", type=float, default=2.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--depth-absolute-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--visibility-sample-radius-px", type=int, default=2)
    return parser.parse_args()


def _family_specs(plan: dict[str, Any]) -> tuple[OrbitFamilySpec, ...]:
    return tuple(
        OrbitFamilySpec(
            family_id=raw["family_id"],
            shape=raw["shape"],
            radius_x_m=raw["radius_x_m"],
            radius_y_m=raw["radius_y_m"],
            height_m=raw["height_m"],
            target_court_instance_id=raw["target_court_instance_id"],
            phase_radians=raw["phase_radians"],
            sample_count=raw["sample_count"],
        )
        for raw in plan["sampling"]["families"]
    )


def _projection_with_occlusion(projection: Any) -> dict[str, object]:
    payload = asdict(projection)
    for court in payload["courts"]:
        for class_record in court["classes"]:
            for point in class_record["points"]:
                visible = point["visible"]
                if not isinstance(visible, bool):
                    raise RuntimeError("Published point visibility must be boolean.")
                point["occluded"] = bool(point["in_frame"] and not visible)
    return payload


def _write_sample(
    root: Path,
    *,
    sample_index: int,
    split: DatasetSplit,
    plan_frame: dict[str, Any],
    rgb: np.ndarray,
    heatmap_atlas: np.ndarray,
    labels: dict[str, object],
) -> dict[str, object]:
    family_id = plan_frame["family_id"]
    sample_dir = (
        root
        / "samples"
        / split
        / family_id
        / f"frame_{plan_frame['family_frame_index']:03d}"
    )
    sample_dir.mkdir(parents=True)
    rgb_path = sample_dir / "rgb.png"
    heatmap_path = sample_dir / "heatmaps_7ch_u16.png"
    labels_path = sample_dir / "labels.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, compress_level=6)
    Image.fromarray(heatmap_atlas).save(
        heatmap_path,
        compress_level=9,
    )
    labels_path.write_text(
        json.dumps(labels, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_index": sample_index,
        "split": split,
        "family_id": family_id,
        "family_frame_index": plan_frame["family_frame_index"],
        "camera_id": plan_frame["camera"]["camera_id"],
        "rgb": _relative_file_ref(root, rgb_path),
        "heatmap_atlas": _relative_file_ref(root, heatmap_path),
        "labels": _relative_file_ref(root, labels_path),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    result = np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0))
    return {
        key: float(value)
        for key, value in zip(
            ("minimum", "p10", "median", "p90", "maximum"),
            result,
            strict=True,
        )
    }


def _verify_dataset(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != DATASET_SCHEMA:
        raise RuntimeError("Unsupported multi-court dataset schema.")
    declared = payload.get("dataset_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("dataset_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise RuntimeError("Multi-court dataset fingerprint mismatch.")
    for frame in payload["frames"]:
        for name in ("rgb", "heatmap_atlas", "labels"):
            record = frame[name]
            relative = PurePosixPath(record["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe dataset path: {relative}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"Missing dataset file: {relative}")
            if (
                path.stat().st_size != record["size_bytes"]
                or _sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError(f"Dataset file integrity mismatch: {relative}")
    return {
        "dataset_fingerprint": declared,
        "frame_count": len(payload["frames"]),
        "family_count": payload["metrics"]["family_count"],
        "split_frame_counts": payload["metrics"]["split_frame_counts"],
        "status": payload["status"],
    }


def main() -> None:
    args = _parse_args()
    plan_dir = args.plan_dir.resolve()
    composition_path = args.background_composition.resolve()
    visual_review_path = args.visual_review.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    if args.width <= 1:
        raise SystemExit("width must be greater than one.")
    if args.split_seed < 0:
        raise SystemExit("split-seed must be non-negative.")
    if args.heatmap_sigma_px <= 0.0:
        raise SystemExit("heatmap-sigma-px must be positive.")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise SystemExit("alpha-threshold must lie in [0, 1].")
    if args.visibility_sample_radius_px < 0:
        raise SystemExit("visibility-sample-radius-px must be non-negative.")
    if (
        args.depth_absolute_tolerance < 0.0
        or args.depth_relative_tolerance < 0.0
    ):
        raise SystemExit("Depth tolerances must be non-negative.")
    if not visual_review_path.is_file():
        raise FileNotFoundError(visual_review_path)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    plan = _load_verified_plan(plan_dir)
    family_specs = _family_specs(plan)
    splits = assign_family_disjoint_splits(
        family_specs,
        seed=args.split_seed,
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
    provider_path = _local_artifact_path(composition.scene_source)
    provider = load_scene_provider_bundle(provider_path)
    provider_manifest = provider.manifest
    if provider_manifest.scene_fingerprint != contract.scene_fingerprint:
        raise RuntimeError("Export provider and accepted scene contract differ.")
    if tuple(provider_manifest.cameras) != tuple(contract.cameras):
        raise RuntimeError("Export provider cameras and scene contract differ.")

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
        split_frame_counts: Counter[str] = Counter()
        split_coverage_counts: dict[str, Counter[str]] = {
            split: Counter() for split in SPLIT_NAMES
        }
        renderer_visible_by_class: Counter[int] = Counter()
        renderer_visible_by_split: Counter[str] = Counter()
        alpha_coverages = []
        rgb_standard_deviations = []
        maximum_visible_peaks = 0
        for sample_index, plan_frame in enumerate(plan["frames"]):
            split = splits.split_for_family(plan_frame["family_id"])
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
            atlas = encode_heatmap_atlas_u16(heatmaps)
            geometric_coverage = [
                court.coverage_bucket for court in visible_projection.courts
            ]
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
            frame_alpha_coverage = float((alpha >= args.alpha_threshold).mean())
            frame_rgb_std = float(rgb.astype(np.float32).std() / 255.0)
            split_frame_counts[split] += 1
            renderer_visible_by_split[split] += visible_count
            for bucket in geometric_coverage:
                split_coverage_counts[split][bucket] += 1
            alpha_coverages.append(frame_alpha_coverage)
            rgb_standard_deviations.append(frame_rgb_std)
            labels: dict[str, object] = {
                "schema": FRAME_SCHEMA,
                "sample_index": sample_index,
                "split": split,
                "family_id": plan_frame["family_id"],
                "family_frame_index": plan_frame["family_frame_index"],
                "plan_content_fingerprint": plan["content_fingerprint"],
                "scene_fingerprint": contract.scene_fingerprint,
                "provider_bundle_fingerprint": (
                    provider_manifest.bundle_fingerprint
                ),
                "camera": camera.to_dict(),
                "resolution": [args.width, height],
                "projection": _projection_with_occlusion(visible_projection),
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
                    "heatmap_encoding": HEATMAP_ENCODING,
                    "heatmap_atlas_shape": list(atlas.shape),
                    "heatmap_channel_names": list(
                        SYMMETRIC_KEYPOINT_CLASS_NAMES
                    ),
                    "court_instance_grouping": False,
                    "court_instance_grouping_stage": "postprocess",
                    "multi_peak_composition": "pixelwise-maximum",
                },
                "rgb_overlay_used": False,
                "all_finite": True,
            }
            frame_records.append(
                _write_sample(
                    temporary,
                    sample_index=sample_index,
                    split=split,
                    plan_frame=plan_frame,
                    rgb=rgb,
                    heatmap_atlas=atlas,
                    labels=labels,
                )
            )
            if (sample_index + 1) % 50 == 0 or sample_index + 1 == len(
                plan["frames"]
            ):
                print(
                    json.dumps(
                        {
                            "rendered": sample_index + 1,
                            "total": len(plan["frames"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        split_family_counts = {
            split: len(splits.families_for_split(split))
            for split in SPLIT_NAMES
        }
        metrics = {
            "frame_count": len(frame_records),
            "family_count": len(family_specs),
            "court_instance_count": len(layout.courts),
            "split_family_counts": split_family_counts,
            "split_frame_counts": {
                split: split_frame_counts[split] for split in SPLIT_NAMES
            },
            "split_coverage_counts": {
                split: dict(sorted(split_coverage_counts[split].items()))
                for split in SPLIT_NAMES
            },
            "renderer_visible_point_count": sum(
                renderer_visible_by_class.values()
            ),
            "renderer_visible_points_by_split": {
                split: renderer_visible_by_split[split]
                for split in SPLIT_NAMES
            },
            "renderer_visible_points_by_class": {
                SYMMETRIC_KEYPOINT_CLASS_NAMES[class_id]: (
                    renderer_visible_by_class[class_id]
                )
                for class_id in range(len(SYMMETRIC_KEYPOINT_CLASS_NAMES))
            },
            "maximum_visible_peaks_in_one_channel": maximum_visible_peaks,
            "alpha_coverage": _quantiles(alpha_coverages),
            "rgb_standard_deviation": _quantiles(rgb_standard_deviations),
        }
        automatic_gate = (
            len(frame_records) == len(plan["frames"]) == 428
            and split_family_counts
            == {"train": 12, "validation": 3, "test": 3}
            and all(split_frame_counts[split] > 0 for split in SPLIT_NAMES)
            and all(
                renderer_visible_by_class[class_id] > 0
                for class_id in range(len(SYMMETRIC_KEYPOINT_CLASS_NAMES))
            )
            and maximum_visible_peaks == 4
            and metrics["alpha_coverage"]["minimum"] > 0.01
            and metrics["rgb_standard_deviation"]["minimum"] > 0.01
        )
        manifest: dict[str, object] = {
            "schema": DATASET_SCHEMA,
            "status": (
                "passed-mechanics-only"
                if automatic_gate
                else "failed-automatic-awaiting-analysis"
            ),
            "appearance_scope": (
                "one-step NHT mechanics prototype; not production photorealism"
            ),
            "plan": {
                "manifest_sha256": _sha256_file(plan_dir / "manifest.json"),
                "content_fingerprint": plan["content_fingerprint"],
            },
            "export_first_source": {
                "provider_manifest_sha256": _sha256_file(provider_path),
                "provider_bundle_fingerprint": (
                    provider_manifest.bundle_fingerprint
                ),
                "scene_fingerprint": provider_manifest.scene_fingerprint,
                "camera_count": len(provider_manifest.cameras),
            },
            "scene_contract": {
                "manifest_sha256": source["scene_contract"]["sha256"],
                "scene_fingerprint": contract.scene_fingerprint,
            },
            "background_composition": {
                "manifest_sha256": _sha256_file(composition_path),
                "composition_fingerprint": composition.composition_fingerprint,
                "background_asset_id": background_asset.asset_id,
                "composition_instances_used": False,
            },
            "visual_review": {
                "sha256": _sha256_file(visual_review_path),
                "decision": "passed-mechanics-only",
            },
            "split_assignment": splits.to_dict(),
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
                "heatmap_encoding": HEATMAP_ENCODING,
                "near_far_symmetry_removed": True,
                "court_instance_retained_in_annotation": True,
                "court_instance_grouping": False,
                "court_instance_grouping_stage": "postprocess",
                "heatmap_sigma_px": args.heatmap_sigma_px,
                "multi_peak_composition": "pixelwise-maximum",
                "maximum_physical_peaks_per_channel": 4,
            },
            "resolution": [args.width, round(539 * args.width / 959)],
            "frames": frame_records,
            "metrics": metrics,
            "automatic_gate_passed": automatic_gate,
            "rgb_overlay_used": False,
            "all_finite": True,
        }
        manifest["dataset_fingerprint"] = _canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
        print(json.dumps(_verify_dataset(output_dir), indent=2, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
