"""Apply the user-approved teacher/RGB contract to saved CPU comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCENE_PSNR_MIN_DB = 57.0
VIEW_PSNR_MIN_DB = 55.0


def _psnr(metrics: dict[str, Any]) -> float:
    rmse = metrics["rmse"]
    if type(rmse) not in (int, float) or not math.isfinite(rmse) or rmse < 0:
        raise ValueError("RGB RMSE must be finite and nonnegative.")
    psnr = metrics["psnr_db"]
    if rmse == 0:
        if psnr is not None:
            raise ValueError("Exact RGB must encode infinite PSNR as null.")
        return math.inf
    expected = -20 * math.log10(rmse)
    if (
        type(psnr) not in (int, float)
        or not math.isfinite(psnr)
        or not math.isclose(psnr, expected, abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError("RGB PSNR disagrees with finite RMSE.")
    return expected


def _views(document: dict[str, Any], repeat: str) -> dict[tuple[str, str], Any]:
    if (
        document["schema"] != "court_probe_visibility_v1"
        or document["alpha_threshold"] != 0.01
        or document["sample_radius_px"] != 1
    ):
        raise ValueError("Unsupported visibility evidence or policy.")
    views = {}
    for view in document["views"]:
        if view["repeat"] != repeat:
            continue
        key = view["scene_id"], view["camera_id"]
        if key in views or not view["points"]:
            raise ValueError("Duplicate or empty visibility evidence.")
        views[key] = view["points"]
    if not views:
        raise ValueError("Missing visibility repeat.")
    return views


def evaluate_comparison(
    comparison: dict[str, Any],
    reference_visibility: dict[str, Any],
    candidate_visibility: dict[str, Any],
) -> dict[str, Any]:
    """Require exact teacher inventories and full-frame RGB floors, per repeat.

    Alpha/depth distributions remain in the original report as diagnostics.
    They do not impose cross-GPU raster equality or replace generation QA.
    """
    if comparison["schema"] != "court_render_comparison_v1":
        raise ValueError("Unsupported comparison schema.")
    reference = _views(reference_visibility, comparison["reference_repeat"])
    candidate = _views(candidate_visibility, comparison["candidate_repeat"])
    if set(reference) != set(candidate):
        raise ValueError("Reference/candidate camera inventories differ.")
    expected_scenes = {sid for sid, _ in reference}
    if set(comparison["scenes"]) != expected_scenes:
        raise ValueError("Comparison scene inventory differs from visibility evidence.")
    failures = []
    scenes = {}
    for sid, scene in comparison["scenes"].items():
        cameras = [view["camera_id"] for view in scene["views"]]
        expected_cameras = {cid for scene_id, cid in reference if scene_id == sid}
        if len(cameras) != len(set(cameras)) or set(cameras) != expected_cameras:
            raise ValueError("Comparison camera inventory differs from visibility evidence.")
        scene_psnr = _psnr(scene["summary"]["rgb"])
        if scene_psnr < SCENE_PSNR_MIN_DB:
            failures.append(f"{sid}: scene RGB PSNR below {SCENE_PSNR_MIN_DB} dB")
        point_count = 0
        min_view_psnr = math.inf
        for view in scene["views"]:
            cid = view["camera_id"]
            points = reference[sid, cid]
            if points != candidate[sid, cid]:
                failures.append(f"{sid}/{cid}: teacher inventory differs")
            visibility = view["visibility"]
            if (
                visibility["points"] != len(points)
                or any(visibility[key] != 0 for key in (
                    "geometry_disagreements", "renderer_disagreements",
                    "max_uv_delta_px", "max_kp_depth_delta_m",
                ))
            ):
                failures.append(f"{sid}/{cid}: comparison teacher evidence differs")
            point_count += len(points)
            psnr = _psnr(view["metrics"]["rgb"])
            min_view_psnr = min(min_view_psnr, psnr)
            if psnr < VIEW_PSNR_MIN_DB:
                failures.append(f"{sid}/{cid}: view RGB PSNR below {VIEW_PSNR_MIN_DB} dB")
        scenes[sid] = {
            "views": len(cameras),
            "points": point_count,
            "scene_psnr_db": scene_psnr if math.isfinite(scene_psnr) else None,
            "min_view_psnr_db": min_view_psnr if math.isfinite(min_view_psnr) else None,
        }
    return {
        "accepted": not failures,
        "scene_psnr_min_db": SCENE_PSNR_MIN_DB,
        "view_psnr_min_db": VIEW_PSNR_MIN_DB,
        "scenes": scenes,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", nargs="+", type=Path, required=True)
    parser.add_argument("--reference-visibility", type=Path, required=True)
    parser.add_argument("--candidate-visibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load(path: Path) -> Any:
        def reject_constant(value: str) -> None:
            raise ValueError(f"Non-finite JSON number: {value}")
        return json.loads(path.read_text(), parse_constant=reject_constant)

    def source(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    reference = load(args.reference_visibility)
    candidate = load(args.candidate_visibility)
    decisions = [
        {**source(path), **evaluate_comparison(load(path), reference, candidate)}
        for path in args.comparisons
    ]
    result = {
        "schema": "court_render_equivalence_acceptance_v1",
        "accepted": all(decision["accepted"] for decision in decisions),
        "reference_visibility": source(args.reference_visibility),
        "candidate_visibility": source(args.candidate_visibility),
        "comparisons": decisions,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted": result["accepted"], "output": str(args.output)}))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
