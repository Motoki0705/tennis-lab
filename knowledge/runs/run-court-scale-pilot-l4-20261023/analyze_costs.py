"""Fit pilot measurements and print explicit, unapproved campaign scenarios."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"


def read(name: str) -> Any:
    return json.loads((E / name).read_text())


def main() -> None:
    counters = {v["cache_key"]: v for v in read("cache-counters.json")["caches"]}
    metrics = read("cache-metrics.json")
    receipts = read("durability-receipts.json")
    rows = sorted(receipts["caches"], key=lambda item: item["verified_at"])
    times = {v["cache"]["cache_key"]: v["generation_seconds"] for v in metrics}
    gaps = [
        datetime.fromisoformat(current["verified_at"]).timestamp()
        - datetime.fromisoformat(previous["verified_at"]).timestamp()
        - times[current["cache_key"]]
        for previous, current in zip(rows[1:-1], rows[2:], strict=True)
    ]
    scenes = ["B00", "B01", "B02", "B03"]
    x = np.array(
        [
            [float(v["cache"]["scene_id"] == scene) for scene in scenes]
            + [counters[v["cache"]["cache_key"]]["geometry_pass"]]
            for v in metrics
        ]
    )
    y = np.array([v["generation_seconds"] for v in metrics])
    render_coef, _ = nnls(x, y)
    plans = read("final/plan-inputs.json")
    probes = read("cpu-plan-probes.json")["probes"]
    pilot = read("request.json")
    plan_rows = [
        {
            "scene_count": 4,
            "wall_seconds": read("plan-timing.json")["wall_seconds"],
            "scenes": [
                {"groups": v["trajectory_count"], "candidates": v["candidates"]}
                for v in pilot["scenes"]
            ],
        },
        *probes,
    ]
    xp = np.array(
        [
            [
                1,
                p["scene_count"],
                sum(s["groups"] for s in p["scenes"]),
                sum(s["candidates"] for s in p["scenes"]),
            ]
            for p in plan_rows
        ]
    )
    plan_coef, _ = nnls(xp, np.array([p["wall_seconds"] for p in plan_rows]))
    xi = np.array(
        [
            [
                len(p["scenes"]),
                sum(len(g) for g in p["scenes"].values()),
                sum(r["candidates"] for g in p["scenes"].values() for r in g),
            ]
            for p in plans.values()
        ]
    )
    input_coef, _ = nnls(
        xi, np.array([p["uncompressed_ustar_bytes"] for p in plans.values()])
    )
    training = read("training-summary.json")
    io = read("training-io.json")
    uploads = [
        c
        for c in io["rclone_calls"]
        if c["operation"] == "copyto" and ".ckpt" in " ".join(c["arguments"])
    ]
    upload_bps = sum(s["bytes"] for s in io["checkpoint_saves"]) / sum(
        c["wall_seconds"] for c in uploads
    )
    storage = read("final/storage.json")
    backbone = "ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    download_bps = (
        storage[backbone]["size_bytes"]
        / read("backbone-download-timing.json")["wall_seconds"]
    )
    cache_bytes = sum(r["archive_bytes"] for r in rows)
    input_bytes = receipts["inputs"]["archive_bytes"]
    total_bytes = read("drive-dataset-inspect.json")["size_bytes"]
    put_seconds = statistics.mean(gaps) - cache_bytes / 96 / upload_bps
    get_seconds = (
        read("fetch-timing.json")["wall_seconds"] - total_bytes / download_bps
    ) / 96
    generation_wall = sum(
        read(n)["wall_seconds"] for n in ("interrupt-timing.json", "resume-timing.json")
    )
    gen_fixed = (
        generation_wall
        - sum(y)
        - put_seconds * 96
        - (input_bytes + cache_bytes) / upload_bps
    )
    train_checkpoint = io["checkpoint_saves"][0]
    best_checkpoint = io["checkpoint_saves"][1]
    validation = best_checkpoint["started_unix"] - (
        train_checkpoint["started_unix"] + train_checkpoint["wall_seconds"]
    )
    sync = statistics.median(training["sync_tree_seconds"])
    step = training["steady_delta_median_seconds"]
    first = training["first_logged_update_wall_seconds"]
    preflight = io["data_preflight"][0]["wall_seconds"]
    eval_base = sum(
        read(f"evaluate-{s}-timing.json")["wall_seconds"] for s in ("val", "test")
    )
    eval_preflight = sum(
        read(f"{s}-framing.json")["timings"]["data_preflight_seconds"]
        for s in ("val", "test")
    )
    base = plans["base"]["scenes"]
    qualities = read("dataset-version.json")["scenes"]
    rates = {
        s["scene_id"]: s["counts"]["qa_accepted"] / s["counts"]["geometry_pass"]
        for s in qualities
    }
    scene_bytes = {
        scene: sum(r["archive_bytes"] for r in rows if r["cache"]["scene_id"] == scene)
        / sum(c["qa_accepted"] for c in counters.values() if c["scene_id"] == scene)
        for scene in scenes
    }

    def condition(
        name: str, group_counts: dict[str, float], density: int
    ) -> dict[str, Any]:
        candidates = rendered = accepted = caches = render = cached_bytes = 0.0
        per_scene = {}
        for scene, groups in group_counts.items():
            entries = base[scene]
            factor = groups / len(entries)
            c = sum(v["candidates"] for v in entries) * factor * 2**density
            r = sum(v["geometry_pass"] for v in entries) * factor * 2**density
            n = r * rates[scene]
            k = (
                sum(
                    math.ceil(
                        v["base_count"] * (1 if birth == 0 else 2 ** (birth - 1)) / 64
                    )
                    for v in entries
                    for birth in range(density + 1)
                )
                * factor
            )
            cost = render_coef[scenes.index(scene)] * k + render_coef[-1] * r
            candidates += c
            rendered += r
            accepted += n
            caches += k
            render += cost
            cached_bytes += scene_bytes[scene] * n
            per_scene[scene] = {
                "groups": groups,
                "candidates": c,
                "accepted_estimate": n,
                "caches_estimate": k,
            }
        g = sum(group_counts.values())
        archive = float(np.dot(input_coef, [len(group_counts), g, candidates]))
        # Preserve per-condition publication/receipt overhead observed in the pilot.
        metadata = total_bytes - input_bytes - cache_bytes
        durable = archive + cached_bytes + metadata * caches / 96
        planning = float(np.dot(plan_coef, [1, len(group_counts), g, candidates]))
        generation = (
            render
            + put_seconds * caches
            + (archive + cached_bytes) / upload_bps
            + gen_fixed
        )
        fetch = get_seconds * caches + durable / download_bps
        return {
            "name": name,
            "scenes": per_scene,
            "density_level": density,
            "candidates_estimate": candidates,
            "rendered_estimate": rendered,
            "accepted_estimate": accepted,
            "caches_estimate": caches,
            "planning_seconds": planning,
            "render_seconds": render,
            "generation_with_persistence_seconds": generation,
            "fetch_seconds": fetch,
            "durable_bytes_estimate": durable,
        }

    def train_cost(n: float, updates: int, interval: int) -> dict[str, float]:
        events = math.ceil(updates / interval)
        steady = step * (updates - 1)
        sync_seconds = math.ceil(step * updates / 60) * sync
        checks = events * (
            train_checkpoint["wall_seconds"] + best_checkpoint["wall_seconds"]
        )
        train = first + steady + sync_seconds + checks + validation * events
        extra_preflight = preflight * max(0, n / 1102 - 1)
        eval_extra = eval_preflight * max(0, n / 1102 - 1)
        return {
            "training_seconds_fixed_preflight": train,
            "training_seconds_linear_preflight_scenario": train + extra_preflight,
            "evaluation_seconds_fixed_preflight": eval_base,
            "evaluation_seconds_linear_preflight_scenario": eval_base + eval_extra,
            "checkpoint_bytes_if_all_retained": events
            * (train_checkpoint["bytes"] + best_checkpoint["bytes"]),
        }

    conditions = [
        condition(f"A1-{q}", dict.fromkeys(scenes, 128 * 2**q), 0) for q in range(5)
    ]
    conditions += [
        condition(f"A2-{q}", dict.fromkeys(scenes, 128), q) for q in range(1, 5)
    ]
    for count in (1, 2, 3):
        selected = scenes[:count]
        groups = {
            s: math.ceil(
                24000
                / count
                / (sum(v["geometry_pass"] for v in base[s]) * rates[s] / 128)
            )
            for s in selected
        }
        conditions.append(condition(f"B-{count}", groups, 0))
    by_name = {c["name"]: c for c in conditions}
    pair_groups = {
        s: math.ceil(
            12000 / (sum(v["geometry_pass"] for v in base[s]) * rates[s] / 128)
        )
        for s in ("B02", "B03")
    }
    supplementary = condition("B-pair-B02-B03", pair_groups, 0)
    campaigns = []
    setup = 941.4 + sum(
        read(n)["wall_seconds"]
        for n in (
            "asset-download-timing.json",
            "asset-verify-timing.json",
            "real-download-timing.json",
            "backbone-download-timing.json",
            "real-split-timing.json",
        )
    )
    for maximum in (2, 3, 4):
        names = (
            [f"A1-{q}" for q in range(maximum + 1)]
            + [f"A2-{q}" for q in range(1, maximum + 1)]
            + [f"B-{i}" for i in (1, 2, 3)]
        )
        main_conditions = [by_name[name] for name in names]
        endpoints = list(
            dict.fromkeys(["A1-0", f"A1-{maximum}", f"A2-{maximum}", "B-1", "A1-2"])
        )
        extra = [by_name[name] for name in endpoints for _ in range(2)]
        representatives = [
            by_name[name] for name in ("A1-0", f"A1-{maximum}", f"A2-{maximum}")
        ]
        for updates in (5000, 10000, 20000):
            interval = updates // 5
            run_costs = [
                train_cost(c["accepted_estimate"], updates, interval)
                for c in main_conditions + extra
            ]
            run_costs += [
                train_cost(c["accepted_estimate"], 4 * updates, interval)
                for c in representatives
            ]
            main_costs = run_costs[: len(main_conditions)]
            # No cross-request durable bank exists. Sum separate conditions even
            # when a local cache could avoid a repeated renderer invocation.
            generation = sum(
                c["planning_seconds"] + c["generation_with_persistence_seconds"]
                for c in main_conditions
            )
            fetch_main = sum(c["fetch_seconds"] for c in main_conditions)
            refetch = sum(c["fetch_seconds"] for c in extra + representatives)
            run_count = len(run_costs)
            setup_hours = setup * run_count / 3600
            fixed = (
                sum(
                    c["training_seconds_fixed_preflight"]
                    + c["evaluation_seconds_fixed_preflight"]
                    for c in run_costs
                )
                / 3600
            )
            linear = (
                sum(
                    c["training_seconds_linear_preflight_scenario"]
                    + c["evaluation_seconds_linear_preflight_scenario"]
                    for c in run_costs
                )
                / 3600
            )
            gen_hours = generation / 3600
            fetch_hours = (fetch_main + refetch) / 3600
            checkpoints = sum(c["checkpoint_bytes_if_all_retained"] for c in run_costs)
            datasets = sum(c["durable_bytes_estimate"] for c in main_conditions)
            generation_control = put_seconds * sum(
                c["caches_estimate"] for c in main_conditions
            )
            fetch_control = get_seconds * sum(
                c["caches_estimate"] for c in main_conditions + extra + representatives
            )
            main_fixed = (
                sum(
                    c["training_seconds_fixed_preflight"]
                    + c["evaluation_seconds_fixed_preflight"]
                    for c in main_costs
                )
                / 3600
            )
            main_linear = (
                sum(
                    c["training_seconds_linear_preflight_scenario"]
                    + c["evaluation_seconds_linear_preflight_scenario"]
                    for c in main_costs
                )
                / 3600
            )
            main_non_training = (
                gen_hours + fetch_main / 3600 + setup * len(main_conditions) / 3600
            )
            campaigns.append(
                {
                    "nominal_max_images": 6000 * 2**maximum,
                    "common_updates": updates,
                    "validation_checkpoint_interval": interval,
                    "main_conditions": names,
                    "extra_seed_conditions": endpoints,
                    "extra_training_seeds": [43, 44],
                    "long_conditions": [c["name"] for c in representatives],
                    "long_updates": 4 * updates,
                    "training_run_count": run_count,
                    "generation_hours_separate_roots": gen_hours,
                    "fetch_hours_one_per_training_run": fetch_hours,
                    "setup_hours_cold_per_training_run": setup_hours,
                    "training_evaluation_hours_fixed_preflight": fixed,
                    "training_evaluation_hours_linear_preflight_scenario": linear,
                    "total_l4_hold_hours_fixed_preflight": gen_hours
                    + fetch_hours
                    + setup_hours
                    + fixed,
                    "total_l4_hold_hours_linear_preflight_scenario": gen_hours
                    + fetch_hours
                    + setup_hours
                    + linear,
                    "dataset_bytes_logical_condition_roots": datasets,
                    "checkpoint_bytes_all_retained": checkpoints,
                    "drive_bytes_increment_estimate": datasets + checkpoints,
                    "main_only_l4_hold_hours_fixed_preflight": main_non_training
                    + main_fixed,
                    "main_only_l4_hold_hours_linear_preflight_scenario": main_non_training
                    + main_linear,
                    "threefold_control_stress_extra_hours": 2
                    * (generation_control + fetch_control)
                    / 3600,
                }
            )
    summary = {
        "schema": "court_pilot_cost_scenarios_v1",
        "production_decisions_approved": False,
        "durable_non_generation_gap_seconds": {
            "n": len(gaps),
            "mean": statistics.mean(gaps),
            "median": statistics.median(gaps),
            "p95": float(np.percentile(gaps, 95)),
        },
        "render_fit": {
            "per_scene_seconds_per_cache": dict(
                zip(scenes, render_coef[:-1].tolist(), strict=True)
            ),
            "seconds_per_geometry_pass": float(render_coef[-1]),
            "mae_seconds_per_cache": float(np.abs(x @ render_coef - y).mean()),
        },
        "planning_fit_seconds_coefficients_constant_scenes_groups_candidates": plan_coef.tolist(),
        "input_archive_fit_bytes_coefficients_scenes_groups_candidates": input_coef.tolist(),
        "effective_upload_bytes_per_second_checkpoint_copyto": upload_bps,
        "effective_download_bytes_per_second_backbone_command": download_bps,
        "calibrated_put_control_seconds_per_cache": put_seconds,
        "calibrated_get_control_seconds_per_cache": get_seconds,
        "calibrated_generation_fixed_seconds_per_request": gen_fixed,
        "validation_train_seconds_including_forced_sync": validation,
        "cold_setup_seconds_four_scenes": setup,
        "conditions": conditions,
        "supplementary_pair": supplementary,
        "campaigns": campaigns,
        "assumptions": [
            "Scenario estimates only; not confidence bounds or adopted production conditions.",
            "G=128 per scene is proposed. Base CPU plans are exact; future acceptance is extrapolated from 24 rendered trajectories per scene.",
            "Birth-level cache counts use the exact base128 N0 values. A1 extrapolates their distribution to unseen trajectories.",
            "Separate immutable Drive roots duplicate logical cache archives and acknowledgements; no global durable cache bank is assumed.",
            "Reference budgets rerender every condition. Shared local caches can reduce rendering but not each request's durable acknowledgements.",
            "One full fetch and cold setup per training run; omit fetch only when the same VM holds already-verified data. Generation-only restarts add setup/fetch costs.",
            "Linear-preflight scenario scales the entire mixed fixed/variable observed preflight with N. The true slope has not been identified.",
            "Both exact and best-val checkpoints retained at all validation events; retention bug must be fixed before these main runs.",
            "Checkpoint sizes and L4 throughput fixed to this batch8 model; cold compiled validation cost reused conservatively.",
            "Drive listing page costs at thousands of caches, VM failure/recovery, D2 redesign, supplementary pair and operator idle time are additional.",
            "A factor-three control-plane stress scenario is a sensitivity case, not a measured upper bound.",
        ],
    }
    (ROOT / "cost-scenarios.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for c in campaigns:
        if c["common_updates"] == 10000:
            print(
                {
                    k: c[k]
                    for k in (
                        "nominal_max_images",
                        "training_run_count",
                        "generation_hours_separate_roots",
                        "fetch_hours_one_per_training_run",
                        "training_evaluation_hours_fixed_preflight",
                        "training_evaluation_hours_linear_preflight_scenario",
                        "total_l4_hold_hours_fixed_preflight",
                        "total_l4_hold_hours_linear_preflight_scenario",
                        "drive_bytes_increment_estimate",
                    )
                }
            )


if __name__ == "__main__":
    main()
