"""Validate deterministic PLCS single/multi-person native renders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

PLAN_SCHEMA = "tennis_plcs_gaussian_scene_plan_v1"
RENDER_SCHEMA = "tennis_plcs_nht_render_v1"
REPORT_SCHEMA = "tennis_plcs_p5_acceptance_report_v1"
REQUIRED_LABEL_FIELDS = {
    "identity_id",
    "instance_id",
    "present",
    "pose_index",
    "pose_id",
    "pose_asset_sha256",
    "position_court_m",
    "velocity_court_mps",
    "yaw_radians",
    "scene_from_asset",
    "projected_root_uv_render_pixels",
    "camera_depth",
    "geometric_visible",
    "exact_visible_pixel_count",
    "exact_bbox_xyxy_exclusive",
    "exact_contribution_mass",
    "render_visible",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-plan", type=Path, required=True)
    parser.add_argument("--single-plan-repeat", type=Path, required=True)
    parser.add_argument("--multi-plan", type=Path, required=True)
    parser.add_argument("--multi-plan-repeat", type=Path, required=True)
    parser.add_argument("--single-render", type=Path, required=True)
    parser.add_argument("--single-render-repeat", type=Path, required=True)
    parser.add_argument("--multi-render", type=Path, required=True)
    parser.add_argument("--multi-render-repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-visible-pixels", type=int, default=50)
    parser.add_argument("--maximum-root-centroid-error-px", type=float, default=3.0)
    parser.add_argument("--maximum-nht-aov-alpha-error", type=float, default=0.005)
    parser.add_argument("--maximum-contribution-alpha-error", type=float, default=1e-5)
    parser.add_argument("--maximum-velocity-error-mps", type=float, default=1e-10)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_plan(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != PLAN_SCHEMA:
        raise ValueError("Unsupported PLCS plan.")
    unsigned = dict(manifest)
    declared = unsigned.pop("plan_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise ValueError("PLCS plan fingerprint differs.")
    arrays = {}
    for name, reference in manifest["files"].items():
        relative = PurePosixPath(reference["relative_path"])
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or not path.is_file()
            or _sha256(path) != reference["sha256"]
            or path.stat().st_size != reference["size_bytes"]
        ):
            raise ValueError(f"PLCS plan file changed: {name}.")
        arrays[relative.stem] = np.load(path, allow_pickle=False)
    return manifest, arrays


def _load_render(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != RENDER_SCHEMA:
        raise ValueError("Unsupported PLCS render.")
    unsigned = dict(manifest)
    declared = unsigned.pop("render_fingerprint", None)
    if declared != _canonical_sha256(unsigned):
        raise ValueError("PLCS render fingerprint differs.")
    for frame in manifest["frames"]:
        for name, reference in frame.items():
            if name == "frame_index":
                continue
            relative = PurePosixPath(reference["relative_path"])
            path = (root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(root)
                or not path.is_file()
                or _sha256(path) != reference["sha256"]
                or path.stat().st_size != reference["size_bytes"]
            ):
                raise ValueError(f"PLCS render file changed: {relative}.")
    return cast(dict[str, Any], manifest)


def _bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def _evaluate_mode(
    *,
    mode: str,
    plan_root: Path,
    plan_repeat_root: Path,
    render_root: Path,
    render_repeat_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    plan, arrays = _load_plan(plan_root)
    repeat_plan, _ = _load_plan(plan_repeat_root)
    render = _load_render(render_root)
    repeat_render = _load_render(render_repeat_root)
    plan_trees_equal = _tree_inventory(plan_root) == _tree_inventory(plan_repeat_root)
    render_trees_equal = _tree_inventory(render_root) == _tree_inventory(
        render_repeat_root
    )
    if plan["mode"] != mode or render["mode"] != mode:
        raise ValueError(f"PLCS {mode} mode metadata differs.")
    if plan["plan_fingerprint"] != repeat_plan["plan_fingerprint"]:
        raise ValueError(f"PLCS {mode} repeated plan fingerprint differs.")
    if render["render_fingerprint"] != repeat_render["render_fingerprint"]:
        raise ValueError(f"PLCS {mode} repeated render fingerprint differs.")

    fps = float(plan["fps"])
    velocity_recomputed = np.gradient(
        arrays["positions_court_m"],
        1.0 / fps,
        axis=0,
        edge_order=2,
    )
    velocity_error = float(
        np.abs(velocity_recomputed - arrays["velocities_court_mps"]).max()
    )
    path_lengths = np.linalg.norm(
        np.diff(arrays["positions_court_m"][..., :2], axis=0),
        axis=-1,
    ).sum(axis=0)
    transitions = np.count_nonzero(
        np.diff(arrays["pose_indices"], axis=0),
        axis=0,
    )

    visible_pixels = []
    root_errors = []
    labels_complete = True
    masks_exact = True
    bboxes_exact = True
    identities_stable = True
    expected_identities = {
        int(record["instance_id"]): record["identity_id"]
        for record in plan["identities"]
    }
    threshold = float(render["visibility"]["instance_alpha_threshold"])
    for frame in render["frames"]:
        frame_index = int(frame["frame_index"])
        labels = json.loads(
            (render_root / frame["labels"]["relative_path"]).read_text()
        )
        contribution = np.load(
            render_root / frame["instance_contribution"]["relative_path"],
            allow_pickle=False,
        )
        masks = np.load(
            render_root / frame["instance_mask"]["relative_path"],
            allow_pickle=False,
        )
        if masks.shape[-1] != int(plan["person_count"]):
            raise ValueError("PLCS instance mask channel count differs.")
        for person_index, instance in enumerate(labels["instances"]):
            instance_id = int(instance["instance_id"])
            mask = masks[..., person_index]
            expected_mask = contribution[..., instance_id] >= threshold
            masks_exact &= bool(np.array_equal(mask, expected_mask))
            pixel_count = int(mask.sum())
            visible_pixels.append(pixel_count)
            bboxes_exact &= instance["exact_bbox_xyxy_exclusive"] == _bbox(mask)
            labels_complete &= set(instance) == REQUIRED_LABEL_FIELDS
            identities_stable &= (
                instance["identity_id"] == expected_identities[instance_id]
                and instance_id == int(arrays["instance_ids"][person_index])
                and instance["pose_index"]
                == int(arrays["pose_indices"][frame_index, person_index])
            )
            y, x = np.nonzero(mask)
            if x.size:
                centroid = np.asarray([x.mean(), y.mean()])
                root = np.asarray(instance["projected_root_uv_render_pixels"])
                root_errors.append(float(np.linalg.norm(centroid - root)))

    minimum_pixels = min(visible_pixels)
    maximum_root_error = max(root_errors)
    nht_alpha_error = float(render["visibility"]["aov_alpha_vs_nht_alpha_max_abs"])
    contribution_error = float(
        render["visibility"]["contribution_sum_vs_aov_alpha_max_abs"]
    )
    checks = {
        "plan_tree_byte_identical": plan_trees_equal,
        "render_tree_byte_identical": render_trees_equal,
        "labels_complete": labels_complete,
        "identities_stable": identities_stable,
        "all_frames_present": bool(arrays["present"].all()),
        "all_people_render_visible": bool(render["visibility"]["all_people_visible"]),
        "minimum_visible_pixels_pass": (minimum_pixels >= args.minimum_visible_pixels),
        "root_centroid_projection_pass": (
            maximum_root_error <= args.maximum_root_centroid_error_px
        ),
        "instance_masks_exact": masks_exact,
        "instance_bboxes_exact": bboxes_exact,
        "velocity_consistency_pass": (
            velocity_error <= args.maximum_velocity_error_mps
        ),
        "motion_nontrivial": bool((path_lengths > 0.5).all()),
        "pose_control_nontrivial": bool((transitions >= 2).all()),
        "nht_aov_alpha_pass": (nht_alpha_error <= args.maximum_nht_aov_alpha_error),
        "contribution_alpha_pass": (
            contribution_error <= args.maximum_contribution_alpha_error
        ),
        "native_composition_no_overlay": (
            render["rgb_overlay_used"] is False
            and render["renderer"]["api_calls_per_frame"] == 2
        ),
    }
    return {
        "mode": mode,
        "plan_fingerprint": plan["plan_fingerprint"],
        "render_fingerprint": render["render_fingerprint"],
        "person_count": plan["person_count"],
        "rendered_frame_count": len(render["frame_indices"]),
        "minimum_visible_pixels": minimum_pixels,
        "maximum_visible_pixels": max(visible_pixels),
        "maximum_root_centroid_error_px": maximum_root_error,
        "velocity_reconstruction_max_abs_mps": velocity_error,
        "minimum_path_length_m": float(path_lengths.min()),
        "minimum_pose_transition_count": int(transitions.min()),
        "aov_alpha_vs_nht_alpha_max_abs": nht_alpha_error,
        "contribution_sum_vs_aov_alpha_max_abs": contribution_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    roots = (
        args.single_plan,
        args.single_plan_repeat,
        args.multi_plan,
        args.multi_plan_repeat,
        args.single_render,
        args.single_render_repeat,
        args.multi_render,
        args.multi_render_repeat,
    )
    if any(not root.resolve().is_dir() for root in roots):
        raise SystemExit("Every P5 plan/render root must exist.")
    single = _evaluate_mode(
        mode="single",
        plan_root=args.single_plan.resolve(),
        plan_repeat_root=args.single_plan_repeat.resolve(),
        render_root=args.single_render.resolve(),
        render_repeat_root=args.single_render_repeat.resolve(),
        args=args,
    )
    multi = _evaluate_mode(
        mode="multi",
        plan_root=args.multi_plan.resolve(),
        plan_repeat_root=args.multi_plan_repeat.resolve(),
        render_root=args.multi_render.resolve(),
        render_repeat_root=args.multi_render_repeat.resolve(),
        args=args,
    )
    checks = {
        "single_passed": single["passed"],
        "multi_passed": multi["passed"],
        "single_and_multi_distinct": (
            single["plan_fingerprint"] != multi["plan_fingerprint"]
            and single["render_fingerprint"] != multi["render_fingerprint"]
        ),
    }
    unsigned: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "p5_complete": all(checks.values()),
        "thresholds": {
            "minimum_visible_pixels": args.minimum_visible_pixels,
            "maximum_root_centroid_error_px": (args.maximum_root_centroid_error_px),
            "maximum_nht_aov_alpha_error": args.maximum_nht_aov_alpha_error,
            "maximum_contribution_alpha_error": (args.maximum_contribution_alpha_error),
            "maximum_velocity_error_mps": args.maximum_velocity_error_mps,
            "minimum_path_length_m": 0.5,
            "minimum_pose_transition_count": 2,
        },
        "single": single,
        "multi": multi,
        "checks": checks,
    }
    report = {
        **unsigned,
        "content_fingerprint": _canonical_sha256(unsigned),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
