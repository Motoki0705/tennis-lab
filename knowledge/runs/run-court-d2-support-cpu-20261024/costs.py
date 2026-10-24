"""Recompute the previous campaign budgets under explicit, unmeasured QA assumptions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from plan_variants import estimate, variants
from support import PREVIOUS, SCENES, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((PREVIOUS / "cost-scenarios.json").read_text())
    prior_conditions = {c["name"]: c for c in old["conditions"]}
    prior_campaigns = [c for c in old["campaigns"] if c["common_updates"] == 10000]
    setup = old["cold_setup_seconds_four_scenes"]
    rates_version = json.loads((PREVIOUS / "evidence/dataset-version.json").read_text())
    rates = {
        s["scene_id"]: s["counts"]["qa_accepted"] / s["counts"]["geometry_pass"]
        for s in rates_version["scenes"]
    }
    comparisons = {}
    for variant in variants():
        raw = json.loads((args.plans / variant / "groups.json").read_text())
        base_g = 96 if variant.startswith("hull25") else 128
        entries = {s: raw[s][:base_g] for s in SCENES}
        full128_est = sum(
            sum(g["geometry_pass"] for g in raw[s]) * rates[s] for s in SCENES
        )
        minimum_g = math.ceil(6000 / (full128_est / 128))

        def condition(
            name: str,
            groups: dict[str, float],
            density: int,
            *,
            baseline: dict[str, Any] = entries,
        ) -> dict[str, Any]:
            per_scene = {}
            for scene, count in groups.items():
                gs = baseline[scene]
                factor = count / len(gs)
                per_scene[scene] = {
                    "groups": count,
                    "candidates": sum(g["candidates"] for g in gs)
                    * factor
                    * 2**density,
                    "geometry_pass": sum(g["geometry_pass"] for g in gs)
                    * factor
                    * 2**density,
                    "caches": sum(
                        math.ceil(
                            g["base_count"]
                            * (1 if birth == 0 else 2 ** (birth - 1))
                            / 64
                        )
                        for g in gs
                        for birth in range(density + 1)
                    )
                    * factor,
                }
            return {"name": name, "per_scene": per_scene, **estimate(per_scene)}

        conditions = [
            condition(f"A1-{q}", dict.fromkeys(SCENES, base_g * 2**q), 0)
            for q in range(5)
        ]
        conditions.extend(
            condition(f"A2-{q}", dict.fromkeys(SCENES, base_g), q) for q in range(1, 5)
        )
        for count in (1, 2, 3):
            groups = {
                s: math.ceil(
                    24000
                    / count
                    / (sum(g["geometry_pass"] for g in entries[s]) * rates[s] / base_g)
                )
                for s in SCENES[:count]
            }
            conditions.append(condition(f"B-{count}", groups, 0))
        by_name = {c["name"]: c for c in conditions}
        if variant == "current":
            for c in conditions:
                prior = prior_conditions[c["name"]]
                assert math.isclose(
                    c["accepted_if_old_scene_qa_rates"],
                    prior["accepted_estimate"],
                    abs_tol=1e-8,
                )
                assert math.isclose(
                    c["generation_with_persistence_seconds"],
                    prior["generation_with_persistence_seconds"],
                    abs_tol=1e-8,
                )
        budgets = []
        for prior in prior_campaigns:
            names = prior["main_conditions"]
            main_conditions = [by_name[n] for n in names]
            extra = [
                by_name[n] for n in prior["extra_seed_conditions"] for _ in range(2)
            ]
            long = [by_name[n] for n in prior["long_conditions"]]
            all_runs = main_conditions + extra + long
            prior_all = [
                prior_conditions[n]
                for n in names
                + [n for n in prior["extra_seed_conditions"] for _ in range(2)]
                + prior["long_conditions"]
            ]
            old_weight = sum(
                max(0.0, c["accepted_estimate"] / 1102 - 1) for c in prior_all
            )
            preflight_seconds_per_ratio = (
                (
                    prior["training_evaluation_hours_linear_preflight_scenario"]
                    - prior["training_evaluation_hours_fixed_preflight"]
                )
                * 3600
                / old_weight
            )
            gen = (
                sum(
                    c["planning_seconds"] + c["generation_with_persistence_seconds"]
                    for c in main_conditions
                )
                / 3600
            )
            fetch_main = sum(c["fetch_seconds"] for c in main_conditions) / 3600
            fetch = sum(c["fetch_seconds"] for c in all_runs) / 3600
            main_training_fixed = (
                prior["main_only_l4_hold_hours_fixed_preflight"]
                - prior["generation_hours_separate_roots"]
                - sum(prior_conditions[n]["fetch_seconds"] for n in names) / 3600
                - setup * len(names) / 3600
            )
            main_fixed = (
                gen + fetch_main + setup * len(names) / 3600 + main_training_fixed
            )
            total_fixed = (
                gen
                + fetch
                + prior["setup_hours_cold_per_training_run"]
                + prior["training_evaluation_hours_fixed_preflight"]
            )
            total_linear = (
                total_fixed
                + preflight_seconds_per_ratio
                * sum(
                    max(0.0, c["accepted_if_old_scene_qa_rates"] / 1102 - 1)
                    for c in all_runs
                )
                / 3600
            )
            main_linear = (
                main_fixed
                + preflight_seconds_per_ratio
                * sum(
                    max(0.0, c["accepted_if_old_scene_qa_rates"] / 1102 - 1)
                    for c in main_conditions
                )
                / 3600
            )
            if variant == "current":
                assert math.isclose(
                    total_fixed,
                    prior["total_l4_hold_hours_fixed_preflight"],
                    abs_tol=1e-8,
                )
                assert math.isclose(
                    total_linear,
                    prior["total_l4_hold_hours_linear_preflight_scenario"],
                    abs_tol=1e-8,
                )
            datasets = sum(
                c["durable_bytes_if_old_jpeg_sizes"] for c in main_conditions
            )
            level = {24000: 2, 48000: 3, 96000: 4}[prior["nominal_max_images"]]
            budgets.append(
                {
                    "previous_nominal_max_images": prior["nominal_max_images"],
                    "main_conditions": len(names),
                    "A1_max_accepted_if_old_qa": by_name[f"A1-{level}"][
                        "accepted_if_old_scene_qa_rates"
                    ],
                    "A1_max_cache_estimate": sum(
                        s["caches"]
                        for s in by_name[f"A1-{level}"]["per_scene"].values()
                    ),
                    "A2_max_cache_exact_for_base_curves": sum(
                        s["caches"]
                        for s in by_name[f"A2-{level}"]["per_scene"].values()
                    ),
                    "main_only_l4_hold_hours": [main_fixed, main_linear],
                    "with_additions_l4_hold_hours": [total_fixed, total_linear],
                    "generation_hours_separate_roots": gen,
                    "drive_dataset_bytes": datasets,
                    "drive_increment_bytes": datasets
                    + prior["checkpoint_bytes_all_retained"],
                }
            )
        comparisons[variant] = {
            "baseline_trajectories_per_scene_proposal": base_g,
            "minimum_integer_G_for_6000_from_128_mean_if_old_qa": minimum_g,
            "baseline_estimate": by_name["A1-0"],
            "conditions": conditions,
            "budgets": budgets,
        }
    write(
        args.output,
        {
            "schema": "court_d2_conditional_cost_comparison_v1",
            "variants": comparisons,
            "assumptions": [
                "No new rendered acceptance/JPEG bytes measured. Apply old per-scene QA rates and archive sizes as sensitivity assumptions only.",
                "Reuse previous render/cache/Drive control coefficients, compile costs, 10k common updates, seeds 43/44 and three independent 40k runs; none are newly adopted.",
                "Current/wide125/hull5 use G128; 25% hull variants propose G96 to complete one stratum cycle. Base-prefix C/R/N0 are exact CPU plans; later A1 prefixes and denser geometric rates are extrapolations.",
                "The row labels 24k/48k/96k retain the previous multiplier/condition graph. Actual expected endpoints are shown separately; no exact-size quota or refill.",
                "Ranges are fixed versus linear preflight scenarios, not confidence limits. Larger-hull QA and high-K Drive costs can change these estimates.",
                "No test accuracy/framing is used. The previous total evaluation-runtime allowance is held fixed for cost accounting only.",
            ],
        },
    )
    for name, result in comparisons.items():
        print(
            name,
            result["baseline_trajectories_per_scene_proposal"],
            round(result["baseline_estimate"]["accepted_if_old_scene_qa_rates"]),
            [
                (
                    b["previous_nominal_max_images"],
                    [round(v, 1) for v in b["with_additions_l4_hold_hours"]],
                )
                for b in result["budgets"]
            ],
        )


if __name__ == "__main__":
    main()
