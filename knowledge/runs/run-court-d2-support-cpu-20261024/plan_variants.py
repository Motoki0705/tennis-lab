"""Compare five explicit in-memory D2 recipes using the unchanged CPU planner."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from support import (
    PREVIOUS,
    describe,
    hull_for,
    load_inputs,
    partitions,
    recipe,
    summary,
    write,
)

from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    TrajectoryMethod,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    iter_trajectory_plans,
)


def variants() -> dict[str, GeometryRecipe]:
    base = recipe()
    return {
        "current": base,
        "wide125": replace(base, hfov_degrees=(92.5, 110.0, 125.0)),
        "hull5": replace(base, hull_expansion_percent=5.0),
        "hull25_margin0": replace(base, hull_expansion_percent=25.0, hull_margin_m=0.0),
        "hull25_margin0_wide125": replace(
            base,
            hull_expansion_percent=25.0,
            hull_margin_m=0.0,
            hfov_degrees=(92.5, 110.0, 125.0),
        ),
    }


def estimate(per_scene: dict[str, dict[str, Any]]) -> dict[str, float]:
    model = json.loads((PREVIOUS / "cost-scenarios.json").read_text())
    quality = json.loads((PREVIOUS / "evidence/dataset-version.json").read_text())
    receipts = json.loads((PREVIOUS / "evidence/durability-receipts.json").read_text())
    archive_rows = receipts["caches"]
    total_bytes = json.loads(
        (PREVIOUS / "evidence/drive-dataset-inspect.json").read_text()
    )["size_bytes"]
    original_inputs = receipts["inputs"]["archive_bytes"]
    original_cache = sum(c["archive_bytes"] for c in archive_rows)
    rates = {
        s["scene_id"]: s["counts"]["qa_accepted"] / s["counts"]["geometry_pass"]
        for s in quality["scenes"]
    }
    bytes_per_image = {
        s["scene_id"]: sum(
            c["archive_bytes"]
            for c in archive_rows
            if c["cache"]["scene_id"] == s["scene_id"]
        )
        / s["counts"]["qa_accepted"]
        for s in quality["scenes"]
    }
    c = sum(v["candidates"] for v in per_scene.values())
    g = sum(v["groups"] for v in per_scene.values())
    k = sum(v["caches"] for v in per_scene.values())
    r = sum(v["geometry_pass"] for v in per_scene.values())
    n = sum(v["geometry_pass"] * rates[s] for s, v in per_scene.items())
    archive = float(
        np.dot(
            model["input_archive_fit_bytes_coefficients_scenes_groups_candidates"],
            [len(per_scene), g, c],
        )
    )
    cached = sum(
        v["geometry_pass"] * rates[s] * bytes_per_image[s] for s, v in per_scene.items()
    )
    render = (
        sum(
            model["render_fit"]["per_scene_seconds_per_cache"][s] * v["caches"]
            for s, v in per_scene.items()
        )
        + model["render_fit"]["seconds_per_geometry_pass"] * r
    )
    durable = (
        archive + cached + (total_bytes - original_inputs - original_cache) * k / 96
    )
    generation = (
        render
        + model["calibrated_put_control_seconds_per_cache"] * k
        + (archive + cached)
        / model["effective_upload_bytes_per_second_checkpoint_copyto"]
        + model["calibrated_generation_fixed_seconds_per_request"]
    )
    fetch = (
        model["calibrated_get_control_seconds_per_cache"] * k
        + durable / model["effective_download_bytes_per_second_backbone_command"]
    )
    planning = float(
        np.dot(
            model[
                "planning_fit_seconds_coefficients_constant_scenes_groups_candidates"
            ],
            [1, len(per_scene), g, c],
        )
    )
    return {
        "accepted_if_old_scene_qa_rates": n,
        "render_qa_publish_seconds": render,
        "generation_with_persistence_seconds": generation,
        "fetch_seconds": fetch,
        "planning_seconds": planning,
        "durable_bytes_if_old_jpeg_sizes": durable,
        "render_seconds_per_geometry_pass": render / r,
        "generation_seconds_per_accepted_if_old_qa": generation / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-count", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, verification = load_inputs(args.scenes_root)
    all_summaries = {}
    for name, geom in variants().items():
        started = time.monotonic()
        rows, groups_meta = [], {}
        trajectory_reasons: Counter[str] = Counter()
        for scene, data in inputs.items():
            hull = hull_for(data)
            meta = []
            for group in iter_trajectory_plans(
                scene_id=scene,
                method=TrajectoryMethod.STRATIFIED_COUNTER,
                recipe=geom,
                seed=42,
                trajectory_count=args.trajectory_count,
                density_level=0,
                cameras=data["cameras"],
                layout=data["layout"],
            ):
                trajectory_reasons.update(group["reasons"])
                selected = [
                    {
                        **describe(s, group, scene, data["layout"], hull),
                        "planned_record_sha256": digest(s),
                    }
                    for s in group["samples"]
                ]
                rows.extend(selected)
                meta.append(
                    {
                        "ordinal": group["ordinal"],
                        "trajectory_id": group["trajectory_id"],
                        "base_count": None
                        if group["arc_map"] is None
                        else group["arc_map"]["base_count"],
                        "candidates": len(selected),
                        "geometry_pass": sum(r["geometry_pass"] for r in selected),
                        "caches": math.ceil(len(selected) / 64),
                    }
                )
            groups_meta[scene] = meta
        passed = [r for r in rows if r["geometry_pass"]]
        per_scene = {
            s: {
                "groups": args.trajectory_count,
                "candidates": sum(v["candidates"] for v in gs),
                "geometry_pass": sum(v["geometry_pass"] for v in gs),
                "caches": sum(v["caches"] for v in gs),
            }
            for s, gs in groups_meta.items()
        }
        result = {
            "schema": "court_d2_cpu_variant_v1",
            "variant": name,
            "recipe": geom.to_dict(),
            "generation_seed": 42,
            "trajectory_count_per_scene": args.trajectory_count,
            "density_level": 0,
            "all_candidates": summary(rows),
            "geometry_pass": summary(passed),
            "per_scene": per_scene,
            "geometry_pass_by_scene": partitions(passed, "scene"),
            "all_candidates_by_scene": partitions(rows, "scene"),
            "geometry_pass_by_hfov": partitions(passed, "hfov_deg"),
            "geometry_pass_by_region": partitions(passed, "region"),
            "trajectory_rejection_reasons": dict(trajectory_reasons),
            "cost_estimate_using_previous_model": estimate(per_scene),
            "wall_seconds_cpu": time.monotonic() - started,
            "rendered": False,
            "qa_acceptance_measured": False,
        }
        if name == "current" and args.trajectory_count == 128:
            assert len(rows) == 6668 and len(passed) == 6166
        write(args.output / name / "rows.json", rows)
        write(args.output / name / "groups.json", groups_meta)
        write(args.output / name / "summary.json", result)
        all_summaries[name] = result
        print(
            json.dumps(
                {
                    "variant": name,
                    "candidate": len(rows),
                    "geometry_pass": len(passed),
                    "framing_passed": result["geometry_pass"]["framing"],
                    "per_scene_framing": {
                        s: v["framing"]
                        for s, v in result["geometry_pass_by_scene"].items()
                    },
                    "seconds_cpu": result["wall_seconds_cpu"],
                }
            ),
            flush=True,
        )
    write(
        args.output / "comparison.json",
        {
            "schema": "court_d2_cpu_comparison_v1",
            "asset_verification": verification,
            "variants": all_summaries,
            "production_recipe_changed": False,
            "candidate_rationale": {
                "wide125": "Fixed-pose probes show modest optical gains; replace three layers, retaining the fixed three-layer contract.",
                "hull5": "Previously discussed small expansion: compare its limited displacement against the current near-court positions.",
                "hull25_margin0": "Widen the position support after 854/1209 current candidates lie inside the court; 25% is an exploratory distance scale, not a certified safe 3DGS boundary.",
                "hull25_margin0_wide125": "Combine exterior-position and optical changes after measuring each direction; retain the same camera-height distribution and D5/D6.",
                "not_varied": "Look-height-only probes leave full count unchanged; camera heights are hard-coded strategy parameters and are outside this source-preserving comparison.",
            },
        },
    )


if __name__ == "__main__":
    main()
