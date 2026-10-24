"""Validate diagnostic boundaries, weighted QA arithmetic and every frozen probe camera."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from render_probe import load_selection, reconstruct, weighted_summary
from select_probe import select
from support import count_points, framing, load_inputs, write

from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    _look_at_opencv,
)
from src.synthetic_data_generation.dataset.court.scale.core import digest
from src.utils.geometry.planar_camera import fit_planar_camera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def check_boundaries() -> None:
    points = [{"uv": [0.0, 0.0], "in_front": True} for _ in range(14)]
    assert count_points(points, 640, 360) == 14
    points[0] = {"uv": [-0.001, 0.0], "in_front": True}
    points[1] = {"uv": [640.0, 0.0], "in_front": True}
    points[2] = {"uv": [0.0, 360.0], "in_front": True}
    points[3] = {"uv": [20.0, 20.0], "in_front": False}
    assert count_points(points, 640, 360) == 10
    assert [framing(c) for c in (0, 11, 12, 13, 14)] == [
        "partial",
        "partial",
        "near_full",
        "near_full",
        "full",
    ]
    xyz = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(np.float64)
    center = np.array([4.0, -37.0, 10.0])
    camera_to_world = _look_at_opencv(
        center, np.zeros(3), vertical_scene=np.array([0.0, 0.0, 1.0])
    )
    k = np.array([[1100.0, 0.0, 640.0], [0.0, 1100.0, 360.0], [0.0, 0.0, 1.0]])
    camera_xyz = (xyz - center) @ camera_to_world[:3, :3]
    uv = camera_xyz @ k.T
    uv = uv[:, :2] / uv[:, 2:]
    fitted = fit_planar_camera(xyz, uv, np.ones(14), (1280, 720), require_in_image=True)
    assert fitted.rmse_px < 1e-5
    np.testing.assert_allclose(fitted.center, center, atol=1e-3)
    np.testing.assert_allclose(fitted.K, k, atol=1e-2)


def check_weighting() -> None:
    scenario = {
        "strata": {
            "full": {"geometry_population": 90, "sampled": 4},
            "near_full": {"geometry_population": 9, "sampled": 4},
            "partial": {"geometry_population": 1, "sampled": 1},
        }
    }
    results = [
        {
            "framing": g,
            "accepted": g != "full",
            "reasons": [] if g != "full" else ["qa-reject"],
        }
        for g, n in (("full", 4), ("near_full", 4), ("partial", 1))
        for _ in range(n)
    ]
    result = weighted_summary(scenario, results)
    assert result["raw_sample_accepted"] == 5
    assert result["raw_sample_size"] == 9
    assert result["weighted_acceptance_given_geometry"] == 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    check_boundaries()
    check_weighting()
    selection = load_selection(args.selection)
    changed = copy.deepcopy(selection)
    changed["maximum_total_render_attempts"] = 201
    changed["selection_sha256"] = digest(
        {k: v for k, v in changed.items() if k != "selection_sha256"}
    )
    write(args.scratch / "invalid-budget.json", changed)
    try:
        load_selection(args.scratch / "invalid-budget.json")
    except AssertionError:
        pass
    else:
        raise AssertionError("A budget above 200 must fail")
    for scenario in selection["scenarios"]:
        rows = json.loads((args.plans / scenario["variant"] / "rows.json").read_text())
        assert select(rows, scenario["scene"]) == select(
            list(reversed(rows)), scenario["scene"]
        )
        assert select(rows, scenario["scene"])["samples"] == scenario["samples"]
    inputs, verification = load_inputs(args.scenes_root)
    frozen = reconstruct(selection, inputs)
    assert len(frozen) == 16
    assert sum(len(records) for _, records in frozen) == 192
    audit = json.loads(args.audit.read_text())
    assert audit["previous_counts_and_histograms_match"]
    assert audit["images_verified"] == 1102
    assert audit["camera_records_matched"] == 1209
    result: dict[str, Any] = {
        "schema": "court_d2_cpu_verification_v1",
        "framing_boundaries": "passed",
        "known_camera_fixture": "passed",
        "stratified_weighting": "passed",
        "budget_201_rejected": True,
        "selection_order_invariance": "passed",
        "reconstructed_scenarios": len(frozen),
        "reconstructed_frames": 192,
        "all_selected_camera_record_hashes_match": True,
        "original_bundle_file_checks": len(verification["verified_files"]),
        "gpu_used": False,
        "test_geometry_processed": False,
    }
    write(args.scratch / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
