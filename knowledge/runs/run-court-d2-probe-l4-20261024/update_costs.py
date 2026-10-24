"""Reprice existing campaign graphs using the small weighted QA/JPEG probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_probe import FROZEN, read, write

PILOT = FROZEN.parent / "run-court-scale-pilot-l4-20261023"


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analysis = read(args.analysis)
    prior = read(FROZEN / "cost-scenarios.json")
    model = read(PILOT / "cost-scenarios.json")
    evidence = PILOT / "evidence"
    receipts = read(evidence / "durability-receipts.json")
    counts = {
        s["scene_id"]: s["counts"]
        for s in read(evidence / "dataset-version.json")["scenes"]
    }
    metrics = read(evidence / "cache-metrics.json")
    original_cache = sum(r["archive_bytes"] for r in receipts["caches"])
    original_inputs = receipts["inputs"]["archive_bytes"]
    original_total = read(evidence / "drive-dataset-inspect.json")["size_bytes"]
    overhead = {}
    for scene in counts:
        archive = sum(
            r["archive_bytes"]
            for r in receipts["caches"]
            if r["cache"]["scene_id"] == scene
        )
        jpeg = sum(
            r["storage"]["jpeg_shard_bytes"]
            for r in metrics
            if r["cache"]["scene_id"] == scene
        )
        overhead[scene] = (archive - jpeg) / counts[scene]["geometry_pass"]
        assert overhead[scene] > 0
    old_conditions = {r["name"]: r for r in model["conditions"]}
    old_campaigns = [r for r in model["campaigns"] if r["common_updates"] == 10000]
    variants = {}
    for variant, prior_variant in prior["variants"].items():
        if variant not in analysis["variants"]:
            continue
        probes = {
            r["scene"]: r for r in analysis["scenarios"] if r["variant"] == variant
        }
        base_g = prior_variant["baseline_trajectories_per_scene_proposal"]
        baseline = prior_variant["baseline_estimate"]["per_scene"]

        def condition(
            name: str,
            per_scene: dict[str, Any],
            *,
            probes: dict[str, Any] = probes,
        ) -> dict[str, Any]:
            c = sum(s["candidates"] for s in per_scene.values())
            g = sum(s["groups"] for s in per_scene.values())
            k = sum(s["caches"] for s in per_scene.values())
            r = sum(s["geometry_pass"] for s in per_scene.values())
            n = sum(
                s["geometry_pass"] * probes[scene]["weighted_acceptance_given_geometry"]
                for scene, s in per_scene.items()
            )
            jpeg = sum(
                s["geometry_pass"]
                * probes[scene]["weighted_acceptance_given_geometry"]
                * probes[scene]["weighted_bytes_per_accepted_jpeg"]
                for scene, s in per_scene.items()
            )
            cache_bytes = jpeg + sum(
                s["geometry_pass"] * overhead[scene] for scene, s in per_scene.items()
            )
            inputs = dot(
                model["input_archive_fit_bytes_coefficients_scenes_groups_candidates"],
                [len(per_scene), g, c],
            )
            durable = (
                inputs
                + cache_bytes
                + (original_total - original_inputs - original_cache) * k / 96
            )
            render = (
                sum(
                    model["render_fit"]["per_scene_seconds_per_cache"][scene]
                    * s["caches"]
                    for scene, s in per_scene.items()
                )
                + model["render_fit"]["seconds_per_geometry_pass"] * r
            )
            generation = (
                render
                + model["calibrated_put_control_seconds_per_cache"] * k
                + (inputs + cache_bytes)
                / model["effective_upload_bytes_per_second_checkpoint_copyto"]
                + model["calibrated_generation_fixed_seconds_per_request"]
            )
            fetch = (
                model["calibrated_get_control_seconds_per_cache"] * k
                + durable
                / model["effective_download_bytes_per_second_backbone_command"]
            )
            planning = dot(
                model[
                    "planning_fit_seconds_coefficients_constant_scenes_groups_candidates"
                ],
                [1, len(per_scene), g, c],
            )
            return {
                "name": name,
                "per_scene": per_scene,
                "accepted_estimate": n,
                "geometry_pass_estimate": r,
                "caches_estimate": k,
                "accepted_jpeg_bytes": jpeg,
                "legacy_non_jpeg_overhead_bytes": cache_bytes - jpeg,
                "input_archive_bytes": inputs,
                "durable_bytes_estimate": durable,
                "render_qa_publish_seconds_legacy_model": render,
                "planning_seconds": planning,
                "generation_with_persistence_seconds": generation,
                "fetch_seconds": fetch,
            }

        conditions = [
            condition(c["name"], c["per_scene"])
            for c in prior_variant["conditions"]
            if c["name"].startswith("A")
        ]

        def scene_condition(
            name: str,
            scenes: list[str],
            *,
            baseline: dict[str, Any] = baseline,
            probes: dict[str, Any] = probes,
            base_g: int = base_g,
            variant: str = variant,
        ) -> dict[str, Any]:
            per_scene = {}
            for scene in scenes:
                expected_per_group = (
                    baseline[scene]["geometry_pass"]
                    * probes[scene]["weighted_acceptance_given_geometry"]
                    / base_g
                )
                if expected_per_group <= 0:
                    raise ValueError(
                        f"Cannot estimate a positive scene condition for {variant}/{scene}"
                    )
                groups = math.ceil(24000 / len(scenes) / expected_per_group)
                factor = groups / baseline[scene]["groups"]
                per_scene[scene] = {
                    key: value * factor for key, value in baseline[scene].items()
                }
            return condition(name, per_scene)

        scenes = list(baseline)
        conditions.extend(
            scene_condition(f"B-{count}", scenes[:count]) for count in (1, 2, 3)
        )
        supplementary = scene_condition("B-pair-B02-B03", ["B02", "B03"])
        by_name = {c["name"]: c for c in conditions}
        budgets = []
        for old in old_campaigns:
            names = old["main_conditions"]
            extra_names = [n for n in old["extra_seed_conditions"] for _ in range(2)]
            all_names = names + extra_names + old["long_conditions"]
            all_conditions = [by_name[n] for n in all_names]
            main = [by_name[n] for n in names]
            old_weight = sum(
                max(0.0, old_conditions[n]["accepted_estimate"] / 1102 - 1)
                for n in all_names
            )
            slope = (
                (
                    old["training_evaluation_hours_linear_preflight_scenario"]
                    - old["training_evaluation_hours_fixed_preflight"]
                )
                * 3600
                / old_weight
            )
            setup = model["cold_setup_seconds_four_scenes"]
            old_main_training = (
                old["main_only_l4_hold_hours_fixed_preflight"]
                - old["generation_hours_separate_roots"]
                - sum(old_conditions[n]["fetch_seconds"] for n in names) / 3600
                - setup * len(names) / 3600
            )
            gen = (
                sum(
                    c["planning_seconds"] + c["generation_with_persistence_seconds"]
                    for c in main
                )
                / 3600
            )
            fixed_main = (
                gen
                + sum(c["fetch_seconds"] for c in main) / 3600
                + setup * len(names) / 3600
                + old_main_training
            )
            fixed = (
                gen
                + sum(c["fetch_seconds"] for c in all_conditions) / 3600
                + old["setup_hours_cold_per_training_run"]
                + old["training_evaluation_hours_fixed_preflight"]
            )
            linear_main = (
                fixed_main
                + slope
                * sum(max(0.0, c["accepted_estimate"] / 1102 - 1) for c in main)
                / 3600
            )
            linear = (
                fixed
                + slope
                * sum(
                    max(0.0, c["accepted_estimate"] / 1102 - 1) for c in all_conditions
                )
                / 3600
            )
            level = {24000: 2, 48000: 3, 96000: 4}[old["nominal_max_images"]]
            datasets = sum(c["durable_bytes_estimate"] for c in main)
            supplement_fixed = (
                supplementary["planning_seconds"]
                + supplementary["generation_with_persistence_seconds"]
                + supplementary["fetch_seconds"]
                + setup
            ) / 3600 + old_main_training / len(names)
            supplement_linear = (
                supplement_fixed
                + slope * max(0.0, supplementary["accepted_estimate"] / 1102 - 1) / 3600
            )
            checkpoint_one_main = 5 * 2634502580
            budgets.append(
                {
                    "previous_nominal_max_images": old["nominal_max_images"],
                    "main_conditions": len(names),
                    "all_training_runs": len(all_names),
                    "A1_max_accepted_estimate": by_name[f"A1-{level}"][
                        "accepted_estimate"
                    ],
                    "A2_max_accepted_estimate": by_name[f"A2-{level}"][
                        "accepted_estimate"
                    ],
                    "A1_max_caches": by_name[f"A1-{level}"]["caches_estimate"],
                    "A2_max_caches": by_name[f"A2-{level}"]["caches_estimate"],
                    "main_only_l4_hold_hours": [fixed_main, linear_main],
                    "with_additions_l4_hold_hours": [fixed, linear],
                    "generation_hours_separate_roots": gen,
                    "drive_dataset_bytes": datasets,
                    "checkpoint_bytes_all_retained_legacy_allowance": old[
                        "checkpoint_bytes_all_retained"
                    ],
                    "drive_increment_bytes": datasets
                    + old["checkpoint_bytes_all_retained"],
                    "scene_supplement_l4_hold_hours": [
                        supplement_fixed,
                        supplement_linear,
                    ],
                    "scene_supplement_drive_bytes": supplementary[
                        "durable_bytes_estimate"
                    ]
                    + checkpoint_one_main,
                }
            )
        variants[variant] = {
            "baseline_trajectories_per_scene_proposal": base_g,
            "baseline_estimate": by_name["A1-0"],
            "conditions": conditions,
            "supplementary_condition": supplementary,
            "budgets": budgets,
        }
    write(
        args.output,
        {
            "schema": "court_d2_l4_conditional_costs_v1",
            "variants": variants,
            "legacy_non_jpeg_archive_bytes_per_geometry_pass": overhead,
            "assumptions": [
                "QA/JPEG estimates use frozen N_h/n_h strata weights from only 12 views per scene and variant; these are sensitivity estimates, not adopted production rates.",
                "Input archive, non-JPEG overhead, production rendering/QA/publication, cache control, transfer, training and evaluation coefficients are inherited from the previous pilot. Raw NPZ/cProfile probe time is not substituted for JPEG-only production throughput.",
                "Non-JPEG cache overhead per geometry pass is the previous tar minus its measured JPEG shard bytes; its scaling and denser/new-prefix QA remain unverified.",
                "10k updates, warmup500, val2k, seeds43/44, three independent40k schedules, cold setup/full fetch per training run and the prior all-checkpoint capacity allowance remain unapproved proposals. No checkpoint implementation is changed or certified.",
                "24k/48k/96k name multiplier graphs; endpoints use measured-population extrapolations without quota refill. A1 later prefixes and A2 densities extrapolate the frozen base prefix.",
                "Ranges compare fixed and linearly growing preflight cost, not confidence intervals or upper bounds. Large-folder control latency is not newly measured.",
                "Scene supplement B02+B03 is separately costed; it is excluded from campaign totals. Generation-only setup and assignment/human waiting are excluded just as in the earlier model.",
            ],
        },
    )
    print(
        json.dumps(
            {
                v: [
                    {
                        "max": b["previous_nominal_max_images"],
                        "hours": b["with_additions_l4_hold_hours"],
                        "GB": b["drive_increment_bytes"] / 1e9,
                    }
                    for b in r["budgets"]
                ]
                for v, r in variants.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
