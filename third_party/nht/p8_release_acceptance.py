#!/usr/bin/env python3
"""Audit the immutable BLCS/PLCS/court releases through their shared export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.alignment.scene_provider.bundle import (
    load_scene_provider_bundle,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract

SCHEMA = "tennis_3dgs_native_integrated_release_acceptance_v1"
EXPECTED_SCENE_FINGERPRINT = (
    "2c16d09503118b08a30b3819d01c23b2bc0e575f00b4f30a931c8447d4d3e160"
)
EXPECTED_PROVIDER_FINGERPRINT = (
    "4c013df9623422c036e9984710295c39491133f3479c056bb9f8dd53a243732b"
)
EXPECTED_COMPOSITION_FINGERPRINT = (
    "7a83e40ca75b139e5de1996652cd4015423e0e3e00801a6ba91eda063c20ed37"
)
EXPECTED_RENDERER_COMMIT = "20bc323d613258e5d169fdbc962c9ef27d55ca69"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_content_fingerprint(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    fingerprint = value.get("content_fingerprint")
    unsigned = dict(value)
    unsigned.pop("content_fingerprint", None)
    if fingerprint != _canonical_sha256(unsigned):
        raise ValueError(f"Content fingerprint differs: {path}")
    return value


def _verify_relative_references(root: Path, value: object) -> int:
    verified = 0
    if isinstance(value, dict):
        relative = value.get("relative_path")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if (
            isinstance(relative, str)
            and isinstance(sha256, str)
            and isinstance(size_bytes, int)
        ):
            candidate = (root / relative).resolve()
            candidate.relative_to(root.resolve())
            if (
                not candidate.is_file()
                or candidate.stat().st_size != size_bytes
                or _sha256(candidate) != sha256
            ):
                raise ValueError(f"Referenced artifact differs: {candidate}")
            verified += 1
        for child in value.values():
            verified += _verify_relative_references(root, child)
    elif isinstance(value, list):
        for child in value:
            verified += _verify_relative_references(root, child)
    return verified


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _require_identical_trees(left: Path, right: Path) -> int:
    left_hashes = _tree_hashes(left)
    right_hashes = _tree_hashes(right)
    if left_hashes != right_hashes:
        differing = sorted(
            key
            for key in left_hashes.keys() | right_hashes.keys()
            if left_hashes.get(key) != right_hashes.get(key)
        )
        raise ValueError(
            f"Repeat trees differ: {left} vs {right}; first={differing[:3]}"
        )
    return len(left_hashes)


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        length = min(left.shape[0], right.shape[0])
        left = left[:length]
        right = right[:length]
    return float(np.sqrt(np.mean(np.square(left.astype(float) - right.astype(float)))))


def _blcs_diversity(canonical: Path, distinct: Path) -> dict[str, object]:
    canonical_report = _verify_content_fingerprint(canonical / "simulation.json")
    distinct_report = _verify_content_fingerprint(distinct / "simulation.json")
    canonical_plan = _read_json(canonical / "plan" / "manifest.json")
    distinct_plan = _read_json(distinct / "plan" / "manifest.json")
    canonical_positions = np.load(
        canonical / "plan" / "positions_court_m.npy", allow_pickle=False
    )
    distinct_positions = np.load(
        distinct / "plan" / "positions_court_m.npy", allow_pickle=False
    )
    position_rms_m = _rms(canonical_positions, distinct_positions)
    physics = canonical_report["simulator"]["physics_config"]
    other_physics = distinct_report["simulator"]["physics_config"]
    changed_parameters = sorted(
        key for key in physics if physics[key] != other_physics.get(key)
    )
    passed = (
        canonical_report["seed"] != distinct_report["seed"]
        and canonical_plan["plan_fingerprint"]
        != distinct_plan["plan_fingerprint"]
        and position_rms_m > 0.25
        and len(changed_parameters) >= 3
    )
    return {
        "canonical_seed": canonical_report["seed"],
        "distinct_seed": distinct_report["seed"],
        "canonical_plan_fingerprint": canonical_plan["plan_fingerprint"],
        "distinct_plan_fingerprint": distinct_plan["plan_fingerprint"],
        "position_rms_m": position_rms_m,
        "changed_physics_parameters": changed_parameters,
        "passed": passed,
    }


def _plcs_diversity(canonical: Path, distinct: Path) -> dict[str, object]:
    canonical_manifest = _read_json(canonical / "manifest.json")
    distinct_manifest = _read_json(distinct / "manifest.json")
    canonical_positions = np.load(
        canonical / "positions_court_m.npy", allow_pickle=False
    )
    distinct_positions = np.load(distinct / "positions_court_m.npy", allow_pickle=False)
    canonical_yaw = np.load(canonical / "yaw_radians.npy", allow_pickle=False)
    distinct_yaw = np.load(distinct / "yaw_radians.npy", allow_pickle=False)
    canonical_pose = np.load(canonical / "pose_indices.npy", allow_pickle=False)
    distinct_pose = np.load(distinct / "pose_indices.npy", allow_pickle=False)
    position_rms_m = _rms(canonical_positions, distinct_positions)
    yaw_rms_degrees = math.degrees(_rms(canonical_yaw, distinct_yaw))
    pose_difference_fraction = float(np.mean(canonical_pose != distinct_pose))
    maximum_speed_delta_mps = abs(
        canonical_manifest["metrics"]["maximum_speed_mps"]
        - distinct_manifest["metrics"]["maximum_speed_mps"]
    )
    passed = (
        canonical_manifest["seed"] != distinct_manifest["seed"]
        and canonical_manifest["plan_fingerprint"]
        != distinct_manifest["plan_fingerprint"]
        and position_rms_m > 0.25
        and maximum_speed_delta_mps > 0.1
    )
    return {
        "canonical_seed": canonical_manifest["seed"],
        "distinct_seed": distinct_manifest["seed"],
        "canonical_plan_fingerprint": canonical_manifest["plan_fingerprint"],
        "distinct_plan_fingerprint": distinct_manifest["plan_fingerprint"],
        "position_rms_m": position_rms_m,
        "maximum_speed_delta_mps": maximum_speed_delta_mps,
        "yaw_rms_degrees": yaw_rms_degrees,
        "pose_difference_fraction": pose_difference_fraction,
        "passed": passed,
    }


def _camera_centers_by_frame(manifest: dict[str, Any]) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for frame in manifest["frames"]:
        matrix = np.asarray(frame["camera"]["camera_to_scene"], dtype=float).reshape(4, 4)
        result[(frame["family_id"], frame["family_frame_index"])] = matrix[:3, 3]
    return result


def _court_diversity(canonical: Path, distinct: Path) -> dict[str, object]:
    canonical_manifest = _verify_content_fingerprint(canonical / "manifest.json")
    distinct_manifest = _verify_content_fingerprint(distinct / "manifest.json")
    canonical_centers = _camera_centers_by_frame(canonical_manifest)
    distinct_centers = _camera_centers_by_frame(distinct_manifest)
    shared = sorted(canonical_centers.keys() & distinct_centers.keys())
    center_rms_scene = _rms(
        np.stack([canonical_centers[key] for key in shared]),
        np.stack([distinct_centers[key] for key in shared]),
    )
    canonical_phases = np.asarray(
        [family["phase_radians"] for family in canonical_manifest["sampling"]["families"]]
    )
    distinct_phases = np.asarray(
        [family["phase_radians"] for family in distinct_manifest["sampling"]["families"]]
    )
    phase_delta = np.angle(np.exp(1j * (distinct_phases - canonical_phases)))
    phase_rms_degrees = math.degrees(float(np.sqrt(np.mean(np.square(phase_delta)))))
    passed = (
        canonical_manifest["sampling"]["seed"]
        != distinct_manifest["sampling"]["seed"]
        and canonical_manifest["content_fingerprint"]
        != distinct_manifest["content_fingerprint"]
        and len(shared) >= 420
        and center_rms_scene > 0.01
        and phase_rms_degrees > 1.0
    )
    return {
        "canonical_seed": canonical_manifest["sampling"]["seed"],
        "distinct_seed": distinct_manifest["sampling"]["seed"],
        "canonical_plan_fingerprint": canonical_manifest["content_fingerprint"],
        "distinct_plan_fingerprint": distinct_manifest["content_fingerprint"],
        "shared_family_frames": len(shared),
        "camera_center_rms_scene": center_rms_scene,
        "orbit_phase_rms_degrees": phase_rms_degrees,
        "canonical_frame_count": len(canonical_manifest["frames"]),
        "distinct_frame_count": len(distinct_manifest["frames"]),
        "passed": passed,
    }


def _verify_render(root: Path) -> tuple[dict[str, Any], int]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("rgb_overlay_used") is not False:
        raise ValueError(f"RGB overlay is not disabled: {root}")
    if manifest.get("renderer", {}).get("commit") != EXPECTED_RENDERER_COMMIT:
        raise ValueError(f"Renderer commit differs: {root}")
    if (
        manifest.get("background_composition", {}).get("composition_fingerprint")
        != EXPECTED_COMPOSITION_FINGERPRINT
    ):
        raise ValueError(f"Composition fingerprint differs: {root}")
    return manifest, _verify_relative_references(root, manifest)


def _phase_report(path: Path, completion_key: str) -> dict[str, Any]:
    report = _verify_content_fingerprint(path)
    if report.get("status") != "passed" or report.get(completion_key) is not True:
        raise ValueError(f"Phase report is not accepted: {path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    artifacts = repo / ".codex-loop/3dgs-synthetic-data/artifacts"
    data_root = Path("/home/kamimura/projects/tennis-lab/data/tennis")
    provider_root = artifacts / "cycle-01/b00-provider-export"
    contract_path = (
        data_root
        / "3dgs_scenes/b00-default-v1/"
        "scene-contract-ground-line-user-override-v2.json"
    )

    smoke = _read_json(artifacts / "cycle-01/nht-smoke.json")
    training_root = Path(
        "/home/kamimura/projects/tennis-lab/third_party/nht/artifacts/"
        "train-smoke-c01-tempfix"
    )
    training = _read_json(training_root / "nht-run.json")
    provider = load_scene_provider_bundle(provider_root, verify_files=True)
    contract = load_scene_contract(contract_path)
    p2 = _read_json(artifacts / "cycle-02/nht-composition-smoke-v4/report.json")
    p3 = _phase_report(
        artifacts / "cycle-09/p3-acceptance-report-v1.json", "p3_complete"
    )
    p4 = _phase_report(
        artifacts / "cycle-11/p4-acceptance-report-v2.json", "p4_complete"
    )
    p5 = _phase_report(
        artifacts / "cycle-12/p5-acceptance-report-v2.json", "p5_complete"
    )
    p6 = _phase_report(
        artifacts / "cycle-13/p6-acceptance-report-v2/report.json", "p6_complete"
    )
    p7 = _phase_report(
        artifacts / "cycle-15/p7-acceptance-v2/report.json", "p7_complete"
    )

    blcs_single_render, blcs_single_files = _verify_render(
        artifacts / "cycle-09/prototype-single-render-v3"
    )
    blcs_multi_render, blcs_multi_files = _verify_render(
        artifacts / "cycle-09/prototype-multi-render-v2"
    )
    plcs_single_render, plcs_single_files = _verify_render(
        artifacts / "cycle-12/plcs-single-render-v1"
    )
    plcs_multi_render, plcs_multi_files = _verify_render(
        artifacts / "cycle-12/plcs-multi-render-v1"
    )
    court_manifest = _read_json(artifacts / "cycle-15/court-dataset-v1/manifest.json")
    court_file_count = _verify_relative_references(
        artifacts / "cycle-15/court-dataset-v1", court_manifest
    )

    same_seed = {
        "blcs_single": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-09/prototype-single-plan-v1",
                artifacts / "cycle-09/prototype-single-plan-repeat-v1",
            )
        },
        "blcs_multi": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-09/prototype-multi-plan-v1",
                artifacts / "cycle-16/blcs-multi-same-seed-repeat-v1",
            )
        },
        "plcs_single_plan": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-12/plcs-single-plan-v1",
                artifacts / "cycle-12/plcs-single-plan-repeat-v1",
            )
        },
        "plcs_multi_plan": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-12/plcs-multi-plan-v1",
                artifacts / "cycle-12/plcs-multi-plan-repeat-v1",
            )
        },
        "plcs_single_render": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-12/plcs-single-render-v1",
                artifacts / "cycle-12/plcs-single-render-repeat-v1",
            )
        },
        "plcs_multi_render": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-12/plcs-multi-render-v1",
                artifacts / "cycle-12/plcs-multi-render-repeat-v1",
            )
        },
        "court_dataset": {
            "file_count": _require_identical_trees(
                artifacts / "cycle-15/court-dataset-v1",
                artifacts / "cycle-15/court-dataset-repeat-v1",
            )
        },
    }
    diversity = {
        "blcs_single": _blcs_diversity(
            artifacts / "cycle-09/prototype-single-plan-v1",
            artifacts / "cycle-16/blcs-single-distinct-seed-v1",
        ),
        "blcs_multi": _blcs_diversity(
            artifacts / "cycle-09/prototype-multi-plan-v1",
            artifacts / "cycle-16/blcs-multi-distinct-seed-v1",
        ),
        "plcs_single": _plcs_diversity(
            artifacts / "cycle-12/plcs-single-plan-v1",
            artifacts / "cycle-16/plcs-single-distinct-seed-v1",
        ),
        "plcs_multi": _plcs_diversity(
            artifacts / "cycle-12/plcs-multi-plan-v1",
            artifacts / "cycle-16/plcs-multi-distinct-seed-v1",
        ),
        "court_orbits": _court_diversity(
            artifacts / "cycle-14/multicourt-orbit-plan-v1",
            artifacts / "cycle-16/court-orbit-distinct-seed-v1",
        ),
    }

    removed_overlay_namespaces = all(
        not (repo / relative).exists()
        for relative in (
            "src/synthetic_data_generation/dataset",
            "src/synthetic_data_generation/provider",
            "src/synthetic_data_generation/rendering",
        )
    )
    shared_scene = (
        provider.manifest.scene_fingerprint
        == contract.scene_fingerprint
        == p3["export_first"]["scene_fingerprint"]
        == court_manifest["scene_contract"]["scene_fingerprint"]
        == EXPECTED_SCENE_FINGERPRINT
        and all(
            _read_json(path / "manifest.json")["scene"]["scene_fingerprint"]
            == EXPECTED_SCENE_FINGERPRINT
            for path in (
                artifacts / "cycle-12/plcs-single-plan-v1",
                artifacts / "cycle-12/plcs-multi-plan-v1",
            )
        )
    )
    shared_provider = (
        provider.manifest.bundle_fingerprint
        == p3["export_first"]["provider_bundle_fingerprint"]
        == court_manifest["export_first_source"]["provider_bundle_fingerprint"]
        == EXPECTED_PROVIDER_FINGERPRINT
    )
    shared_composition = (
        p2["composition_fingerprint"] == EXPECTED_COMPOSITION_FINGERPRINT
    )
    native_no_overlay = (
        p2["status"] == "passed"
        and p3["native_composition"]["rgb_overlay_used"] is False
        and p4["nht"]["rgb_overlay_used"] is False
        and all(
            render["rgb_overlay_used"] is False
            for render in (
                blcs_single_render,
                blcs_multi_render,
                plcs_single_render,
                plcs_multi_render,
                court_manifest,
            )
        )
    )
    gates = {
        "p0_isolated_runtime_and_training": (
            smoke.get("all_finite") is True
            and training.get("status") == "completed"
            and (training_root / "ckpts/ckpt_0_rank0.pt").is_file()
            and (training_root / "ply/point_cloud_0.ply").is_file()
            and (training_root / "videos/traj_0.mp4").is_file()
        ),
        "p1_overlay_namespaces_removed_alignment_preserved": (
            removed_overlay_namespaces
            and (repo / "src/synthetic_data_generation/alignment").is_dir()
            and len(contract.cameras) == 491
        ),
        "p2_native_composition": shared_composition and p2["status"] == "passed",
        "p3_blcs": p3["p3_complete"] is True,
        "p4_avatar": p4["p4_complete"] is True,
        "p5_plcs": p5["p5_complete"] is True,
        "p6_camera_sampling": p6["p6_complete"] is True,
        "p7_court_dataset": p7["p7_complete"] is True,
        "shared_scene_boundary": shared_scene,
        "shared_provider_boundary": shared_provider,
        "native_no_rgb_overlay": native_no_overlay,
        "same_seed_reproducibility": all(
            value["file_count"] > 0 for value in same_seed.values()
        ),
        "distinct_seed_diversity": all(
            value["passed"] is True for value in diversity.values()
        ),
        "artifact_integrity": (
            sum(
                (
                    blcs_single_files,
                    blcs_multi_files,
                    plcs_single_files,
                    plcs_multi_files,
                    court_file_count,
                )
            )
            >= 1200
        ),
        "court_training_semantics": (
            court_manifest["training_target"]["heatmap_channels"] == 7
            and court_manifest["training_target"]["maximum_physical_peaks_per_channel"]
            == 4
            and court_manifest["training_target"]["court_instance_grouping"] is False
            and court_manifest["metrics"]["court_instance_count"] == 2
        ),
    }
    status = "passed" if all(gates.values()) else "failed"
    unsigned: dict[str, object] = {
        "schema": SCHEMA,
        "status": status,
        "p8_complete": status == "passed",
        "appearance_scope": (
            "one-step NHT mechanics prototype; native composition, labels, and "
            "dataset contracts are accepted, but photorealism is not claimed"
        ),
        "shared_boundary": {
            "camera_count": len(contract.cameras),
            "scene_fingerprint": contract.scene_fingerprint,
            "provider_bundle_fingerprint": provider.manifest.bundle_fingerprint,
            "composition_fingerprint": EXPECTED_COMPOSITION_FINGERPRINT,
            "renderer_commit": EXPECTED_RENDERER_COMMIT,
        },
        "phase_reports": {
            "p2": _sha256(artifacts / "cycle-02/nht-composition-smoke-v4/report.json"),
            "p3": _sha256(artifacts / "cycle-09/p3-acceptance-report-v1.json"),
            "p4": _sha256(artifacts / "cycle-11/p4-acceptance-report-v2.json"),
            "p5": _sha256(artifacts / "cycle-12/p5-acceptance-report-v2.json"),
            "p6": _sha256(
                artifacts / "cycle-13/p6-acceptance-report-v2/report.json"
            ),
            "p7": _sha256(artifacts / "cycle-15/p7-acceptance-v2/report.json"),
        },
        "same_seed": same_seed,
        "distinct_seed": diversity,
        "integrity": {
            "verified_render_and_dataset_references": sum(
                (
                    blcs_single_files,
                    blcs_multi_files,
                    plcs_single_files,
                    plcs_multi_files,
                    court_file_count,
                )
            ),
            "provider_files_rehashed": True,
        },
        "gates": gates,
    }
    report = dict(unsigned)
    report["content_fingerprint"] = _canonical_sha256(unsigned)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            temporary.rmdir()
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit("P8 integrated release acceptance failed.")


if __name__ == "__main__":
    main()
