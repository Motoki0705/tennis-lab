#!/usr/bin/env python3
"""Verify the export-first BLCS prototype evidence and publish the P3 gate."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.assets import (
    load_ball_asset_registry,  # noqa: E402
)
from src.synthetic_data_generation.blcs.planner import (  # noqa: E402
    BLCSGaussianScenePlan,
    load_blcs_gaussian_plan,
)
from third_party.nht.prototype_blcs_plan import _load_approved_scene  # noqa: E402

PROTOTYPE_SCHEMA = "tennis_ball_generated_prototype_v1"
RENDER_SCHEMA = "tennis_blcs_nht_render_v2"
REPORT_SCHEMA = "tennis_blcs_prototype_p3_acceptance_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-bundle", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--scene-contract-root", type=Path, required=True)
    parser.add_argument("--prototype-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--single-plan", type=Path, required=True)
    parser.add_argument("--single-plan-repeat", type=Path, required=True)
    parser.add_argument("--multi-plan", type=Path, required=True)
    parser.add_argument("--single-render", type=Path, required=True)
    parser.add_argument("--multi-render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _content_ref(path: Path) -> dict[str, object]:
    return {
        "uri": path.resolve().as_uri(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("relative_path must be a string.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    path = (root / Path(*relative.parts)).resolve()
    path.relative_to(root.resolve())
    return path


def _verify_ref(root: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise TypeError("Artifact reference must be an object.")
    path = _relative_path(root, value.get("relative_path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != value.get("size_bytes"):
        raise ValueError(f"Artifact size differs: {path}")
    if _sha256_file(path) != value.get("sha256"):
        raise ValueError(f"Artifact digest differs: {path}")
    return path


def _verify_prototype(root: Path) -> dict[str, object]:
    manifest_path = root / "prototype.json"
    manifest = _json(manifest_path)
    declared = manifest.pop("content_fingerprint", None)
    if manifest.get("schema") != PROTOTYPE_SCHEMA:
        raise ValueError("Prototype schema differs.")
    computed = _canonical_sha256(manifest)
    if declared != computed:
        raise ValueError("Prototype content fingerprint differs.")
    if manifest.get("asset_origin") != "codex-generated-prototype":
        raise ValueError("Prototype origin is not explicit.")
    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise TypeError("Prototype geometry must be an object.")
    diameter = float(geometry["nominal_diameter_m"])
    radius = float(geometry["maximum_three_sigma_radius_m"])
    if abs(2.0 * radius - diameter) > 2.0e-8:
        raise ValueError("Prototype three-sigma envelope is not metric.")
    if float(geometry["mean_offset_m"]) > 1.0e-7:
        raise ValueError("Prototype is not centred.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("Prototype files must be an object.")
    for value in files.values():
        _verify_ref(root, value)
    spec = _json(root / "asset-spec.json")
    if spec.get("source_is_user_asset") is not False:
        raise ValueError("Generated prototype is incorrectly marked as a user asset.")
    return {
        "content_fingerprint": declared,
        "asset_id": manifest["asset_id"],
        "variant_id": manifest["variant_id"],
        "asset_origin": manifest["asset_origin"],
        "source_is_user_asset": False,
        "gaussian_count": int(geometry["gaussian_count"]),
        "nominal_diameter_m": diameter,
        "maximum_three_sigma_radius_m": radius,
        "manifest": _content_ref(manifest_path),
    }


def _tree_digests(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _verify_plan_repeat(first: Path, repeat: Path) -> dict[str, object]:
    first_digests = _tree_digests(first)
    repeat_digests = _tree_digests(repeat)
    if first_digests != repeat_digests:
        raise ValueError("Same-seed BLCS plan regeneration is not byte-identical.")
    return {
        "byte_identical": True,
        "file_count": len(first_digests),
        "tree_fingerprint": _canonical_sha256(first_digests),
    }


def _verify_render(
    root: Path,
    *,
    plan_root: Path,
    expected_objects: int,
) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    declared_fingerprint = manifest.pop("render_fingerprint", None)
    if manifest.get("schema") != RENDER_SCHEMA:
        raise ValueError("Render schema differs.")
    computed_fingerprint = _canonical_sha256(manifest)
    if declared_fingerprint != computed_fingerprint:
        raise ValueError("Render fingerprint differs.")
    if manifest.get("rgb_overlay_used") is not False:
        raise ValueError("RGB overlay is forbidden.")
    if manifest.get("all_finite") is not True:
        raise ValueError("Renderer did not declare finite outputs.")
    scope = manifest.get("acceptance_scope")
    expected_scope = (
        "native-composition RGB and exact instance contribution masks; "
        "asset provenance and alignment acceptance are verified upstream"
    )
    if scope != expected_scope:
        raise ValueError("Renderer acceptance scope is stale or ambiguous.")

    plan = load_blcs_gaussian_plan(plan_root)
    plan_ref = manifest.get("plan")
    if not isinstance(plan_ref, dict):
        raise TypeError("Render plan reference must be an object.")
    plan_manifest = plan_root / "manifest.json"
    if plan_ref.get("sha256") != _sha256_file(plan_manifest):
        raise ValueError("Render references different plan bytes.")
    if plan_ref.get("plan_fingerprint") != plan.plan_fingerprint:
        raise ValueError("Render references different plan content.")
    if plan.num_objects != expected_objects:
        raise ValueError("Plan object count differs from the P3 case.")

    resolution = manifest.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("Render resolution is invalid.")
    width, height = (int(value) for value in resolution)
    camera_index = int(manifest["camera_index"])
    threshold = float(manifest["visibility"]["instance_alpha_threshold"])
    per_instance_visible = {instance_id: 0 for instance_id in plan.instance_ids.tolist()}
    maximum_pixels = {instance_id: 0 for instance_id in plan.instance_ids.tolist()}

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Render has no frames.")
    for frame_record in frames:
        if not isinstance(frame_record, dict):
            raise TypeError("Render frame record must be an object.")
        frame_index = int(frame_record["frame_index"])
        resolved = {
            key: _verify_ref(root, value)
            for key, value in frame_record.items()
            if key != "frame_index"
        }
        labels = _json(resolved["labels"])
        if labels.get("all_finite") is not True:
            raise ValueError("Frame labels are not finite.")
        if labels.get("plan_fingerprint") != plan.plan_fingerprint:
            raise ValueError("Frame labels reference different plan content.")
        contribution = np.load(resolved["instance_contribution"], allow_pickle=False)
        mask = np.load(resolved["instance_mask"], allow_pickle=False)
        segmentation = np.load(
            resolved["instance_segmentation"],
            allow_pickle=False,
        )
        alpha = np.load(resolved["alpha"], allow_pickle=False)
        depth = np.load(resolved["depth"], allow_pickle=False)
        if contribution.shape != (height, width, expected_objects + 1):
            raise ValueError("Instance contribution shape differs.")
        if mask.shape != (height, width, expected_objects):
            raise ValueError("Instance mask shape differs.")
        if segmentation.shape != (height, width):
            raise ValueError("Instance segmentation shape differs.")
        if alpha.shape != (height, width) or depth.shape != (height, width):
            raise ValueError("Alpha/depth shape differs.")
        for array in (contribution, alpha, depth):
            if not np.isfinite(array).all():
                raise ValueError("Render array contains NaN or infinity.")
        active_ids = [
            int(value)
            for value in plan.instance_ids[plan.present[frame_index]].tolist()
        ]
        if labels.get("active_instance_ids") != active_ids:
            raise ValueError("Frame active-instance labels differ from the plan.")
        label_instances = labels.get("instances")
        if not isinstance(label_instances, list):
            raise TypeError("Frame instances must be an array.")
        for instance in label_instances:
            instance_id = int(instance["instance_id"])
            object_index = instance_id - 1
            expected_mask = contribution[..., instance_id] >= threshold
            if not np.array_equal(mask[..., object_index], expected_mask):
                raise ValueError("Exact contribution mask differs.")
            pixel_count = int(expected_mask.sum())
            if pixel_count != int(instance["exact_visible_pixel_count"]):
                raise ValueError("Visible-pixel label differs from exact AOV.")
            expected_position = plan.positions_court_m[frame_index, object_index]
            if not np.allclose(
                instance["position_court_m"],
                expected_position,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError("Court-space position label differs from the plan.")
            camera = plan.cameras[camera_index]
            scale = np.asarray([width / camera.width, height / camera.height])
            expected_uv = plan.camera_uv[camera_index, frame_index, object_index] * scale
            if not np.allclose(
                instance["projected_uv_render_pixels"],
                expected_uv,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError("Projected 2D position differs from the plan.")
            if pixel_count > 0:
                per_instance_visible[instance_id] += 1
                maximum_pixels[instance_id] = max(
                    maximum_pixels[instance_id],
                    pixel_count,
                )
    if any(value == 0 for value in per_instance_visible.values()):
        raise ValueError("At least one planned object was never rendered visibly.")
    visibility = manifest.get("visibility")
    if not isinstance(visibility, dict):
        raise TypeError("Render visibility must be an object.")
    if visibility.get("exact_per_pixel_instance_mask") is not True:
        raise ValueError("Render lacks exact per-pixel instance masks.")
    return {
        "render_fingerprint": declared_fingerprint,
        "frame_count": len(frames),
        "camera_id": manifest["camera_id"],
        "resolution": resolution,
        "renderer_api_call_count": manifest["renderer"]["api_call_count"],
        "maximum_aov_alpha_drift": visibility[
            "aov_alpha_vs_nht_alpha_max_abs"
        ],
        "per_instance_visible_frame_count": per_instance_visible,
        "per_instance_maximum_visible_pixels": maximum_pixels,
        "manifest": _content_ref(manifest_path),
    }


def _verify_simulation(
    root: Path,
    *,
    expected_mode: str,
    expected_objects: int,
) -> tuple[dict[str, Any], BLCSGaussianScenePlan]:
    manifest_path = root / "simulation.json"
    manifest = _json(manifest_path)
    declared = manifest.pop("content_fingerprint", None)
    if declared != _canonical_sha256(manifest):
        raise ValueError("Simulation content fingerprint differs.")
    if manifest.get("status") != "passed" or manifest.get("mode") != expected_mode:
        raise ValueError("Simulation mode/status differs.")
    simulator = manifest.get("simulator")
    if not isinstance(simulator, dict):
        raise TypeError("Simulation provenance must be an object.")
    if simulator.get("physics") != "BallPhysics":
        raise ValueError("Trajectory is not backed by BallPhysics.")
    if simulator.get("rally") != "RallySimulator":
        raise ValueError("Trajectory is not backed by RallySimulator.")
    metrics = manifest.get("trajectory_metrics")
    if not isinstance(metrics, dict) or metrics.get("object_count") != expected_objects:
        raise ValueError("Simulation object count differs.")
    plan = load_blcs_gaussian_plan(root / "plan")
    if plan.num_objects != expected_objects:
        raise ValueError("Simulation and plan object counts differ.")
    return {
        "content_fingerprint": declared,
        "seed": manifest["seed"],
        "simulator": simulator,
        "scene": manifest["scene"],
        "trajectory_metrics": metrics,
        "manifest": _content_ref(manifest_path),
        "plan_fingerprint": plan.plan_fingerprint,
    }, plan


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite acceptance report: {output}")
    provider, contract, decision_path = _load_approved_scene(
        provider_path=args.provider_bundle.resolve(),
        contract_path=args.scene_contract.resolve(),
        contract_root=args.scene_contract_root.resolve(),
    )
    prototype = _verify_prototype(args.prototype_dir.resolve())
    registry = load_ball_asset_registry(
        args.registry.resolve(),
        verify_local_artifacts=True,
    )
    if len(registry.entries) != 1:
        raise ValueError("Prototype P3 registry must contain exactly one asset.")

    single_simulation, _ = _verify_simulation(
        args.single_plan.resolve(),
        expected_mode="single",
        expected_objects=1,
    )
    multi_simulation, multi_plan = _verify_simulation(
        args.multi_plan.resolve(),
        expected_mode="multi",
        expected_objects=2,
    )
    repeat = _verify_plan_repeat(
        args.single_plan.resolve(),
        args.single_plan_repeat.resolve(),
    )
    single_render = _verify_render(
        args.single_render.resolve(),
        plan_root=args.single_plan.resolve() / "plan",
        expected_objects=1,
    )
    multi_render = _verify_render(
        args.multi_render.resolve(),
        plan_root=args.multi_plan.resolve() / "plan",
        expected_objects=2,
    )
    if not np.any(multi_plan.present.sum(axis=1) >= 2):
        raise ValueError("Multi-object physical plan has no concurrent balls.")

    unsigned: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "gate_revision": (
            "User-authorized prototype ball and user-approved current court "
            "alignment; real user ball assets are not required for P3."
        ),
        "export_first": {
            "provider_bundle_fingerprint": provider.manifest.bundle_fingerprint,
            "scene_fingerprint": provider.manifest.scene_fingerprint,
            "camera_count": len(provider.manifest.cameras),
            "scene_contract": _content_ref(args.scene_contract.resolve()),
        },
        "alignment": {
            "status": "accepted",
            "alignment_id": contract.alignment.alignment_id,
            "decision": _content_ref(decision_path),
            "selected_court_cluster": contract.alignment.selected_court_cluster,
            "selected_symmetry": contract.alignment.selected_symmetry,
            "exact_export_scene_match": True,
        },
        "prototype_asset": prototype,
        "registry": {
            "registry_id": registry.registry_id,
            "registry_fingerprint": registry.registry_fingerprint,
            "entry_count": len(registry.entries),
            "manifest": _content_ref(args.registry.resolve()),
        },
        "single_object": {
            "simulation": single_simulation,
            "same_seed_regeneration": repeat,
            "render": single_render,
        },
        "multi_object": {
            "simulation": multi_simulation,
            "has_concurrent_objects": True,
            "render": multi_render,
        },
        "native_composition": {
            "rgb_overlay_used": False,
            "exact_instance_contribution_masks": True,
            "labels_derived_from_verified_plan": True,
        },
        "p3_complete": True,
    }
    report = {
        **unsigned,
        "content_fingerprint": _canonical_sha256(unsigned),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
