"""Summarize saved failure evidence; update setup costs only, without QA guesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import pstats
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--operation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    cpu = here.parent / "run-court-d2-support-cpu-20261024"
    pilot = here.parent / "run-court-scale-pilot-l4-20261023"
    selection = read(cpu / "render-selection.json")
    assert read(args.recovered / "probe/selection.json") == selection
    state = read(args.recovered / "probe/state.json")
    assert state["outcome"] == "failed" and state["completed_frames"] == 0
    assert state["draw_attempts"] == 12
    assert (
        state["error"]
        == "ValueError: output_directory must be an absolute pathlib.Path."
    )
    assert state["renderer_release"]["renderer_context_released"]
    assert not list((args.recovered / "probe").glob("frames/**/*"))
    assert state["source_commit"] == "4916def72d0b46728572006d7fbb19076460ec02"
    source = read(args.operation / "source_changes.json")
    assert source["patch"] == "" and source["untracked"] == {}
    render_status = read(args.operation / "status.json")
    assert render_status["status"] == "failed" and render_status["returncode"] == 1
    assert render_status["drive_saved_at"]
    copies = {
        "probe/state.json": "probe-state.json",
        "render-timing.json": "render-timing.json",
        **{
            f"setup/{n}.json": f"setup-{n}.json"
            for n in ("uv", "nht", "asset-download", "asset-verify")
        },
    }
    for original, name in copies.items():
        shutil.copyfile(args.recovered / original, args.output / name)
    for original, name in (
        ("stdout.log", "render-stdout.log"),
        ("source_changes.json", "render-source-changes.json"),
        ("status.json", "remote-render-status.json"),
    ):
        shutil.copyfile(args.operation / original, args.output / name)
    inventory = [
        {
            "path": p.relative_to(args.recovered).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "retained_in_git": p.suffix != ".pstats" and p.name != "selection.json",
        }
        for p in sorted(args.recovered.rglob("*"))
        if p.is_file()
    ]
    assert len(inventory) == 8
    write(
        args.output / "recovered-files.json",
        {
            "schema": "court_d2_failure_recovery_v1",
            "files": inventory,
            "selection_matches_frozen": True,
            "raw_profile_location": read(here / "run.json")["output_root"]
            + "/profile.pstats",
        },
    )
    stats = pstats.Stats(str(args.recovered / "profile.pstats"))
    raw_stats = cast(dict[tuple[str, int, str], tuple[Any, ...]], vars(stats)["stats"])
    selected_names = {
        "load_scene_inputs",
        "reconstruct",
        "run",
        "__call__",
        "render",
        "evaluate_render",
        "encode_rgb_jpeg",
        "savez_compressed",
    }
    rows = [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "calls": value[1],
            "self_seconds": value[2],
            "cumulative_seconds": value[3],
        }
        for key, value in raw_stats.items()
        if key[2] in selected_names
        and ("/tennis-lab/" in key[0] or key[0].startswith("knowledge/"))
    ]
    render_calls = sum(
        value[1]
        for key, value in raw_stats.items()
        if "/rendering/nht/" in key[0] and key[2] == "render"
    )
    qa_calls = sum(
        value[1] for key, value in raw_stats.items() if key[2] == "evaluate_render"
    )
    assert render_calls == qa_calls == 0
    setup_seconds = sum(
        read(args.recovered / f"setup/{n}.json")["wall_seconds"]
        for n in ("uv", "nht", "asset-download", "asset-verify")
    )
    finalize = (
        datetime.fromisoformat(render_status["finished_at"])
        - datetime.fromisoformat(render_status["updated_at"])
    ).total_seconds()
    write(
        args.output / "failure-summary.json",
        {
            "schema": "court_d2_l4_failure_summary_v1",
            "completed_frames": 0,
            "nht_client_render_calls": render_calls,
            "qa_evaluate_calls": qa_calls,
            "budget_charged_frames_from_durable_state": state["draw_attempts"],
            "remaining_stage_budget": selection["maximum_total_render_attempts"]
            - state["draw_attempts"],
            "full_192_retry_within_remaining_budget": False,
            "setup_measured_seconds": setup_seconds,
            "render_attempt_process_seconds": read(
                args.recovered / "render-timing.json"
            )["wall_seconds"],
            "worker_final_save_seconds": finalize,
            "worker_periodic_save_overlap_not_isolated": True,
            "profile": rows,
            "qa_acceptance_and_image_quality": None,
            "qa_missing_reason": "No NHT call, image, QA report or overlay exists. No rate or appearance comparison can be inferred.",
        },
    )
    model = read(pilot / "cost-scenarios.json")
    legacy = read(cpu / "cost-scenarios.json")
    training_extras = sum(
        read(pilot / f"evidence/{n}-timing.json")["wall_seconds"]
        for n in ("real-download", "backbone-download", "real-split")
    )
    new_setup = setup_seconds + training_extras
    delta = new_setup - model["cold_setup_seconds_four_scenes"]
    campaign_counts = {
        c["nominal_max_images"]: len(c["main_conditions"])
        + 2 * len(c["extra_seed_conditions"])
        + len(c["long_conditions"])
        for c in model["campaigns"]
        if c["common_updates"] == 10000
    }
    variants = {}
    for name in selection["recipes"]:
        budgets = []
        for old in legacy["variants"][name]["budgets"]:
            run_count = campaign_counts[old["previous_nominal_max_images"]]
            budgets.append(
                {
                    **old,
                    "main_only_l4_hold_hours": [
                        v + delta * old["main_conditions"] / 3600
                        for v in old["main_only_l4_hold_hours"]
                    ],
                    "with_additions_l4_hold_hours": [
                        v + delta * run_count / 3600
                        for v in old["with_additions_l4_hold_hours"]
                    ],
                    "all_training_runs": run_count,
                }
            )
        variants[name] = {
            "baseline_trajectories_per_scene_proposal": legacy["variants"][name][
                "baseline_trajectories_per_scene_proposal"
            ],
            "budgets": budgets,
        }
    write(
        args.output / "setup-only-costs.json",
        {
            "schema": "court_d2_setup_only_cost_sensitivity_v1",
            "legacy_training_setup_seconds": model["cold_setup_seconds_four_scenes"],
            "new_generation_setup_measured_seconds": setup_seconds,
            "legacy_training_input_extra_seconds": training_extras,
            "conditional_training_setup_seconds": new_setup,
            "delta_setup_seconds_per_training_run": delta,
            "variants": variants,
            "assumptions": [
                "Only uv/NHT/asset acquisition and verification are replaced by this run's measurements. Real/backbone/split preparation is inherited from the old pilot.",
                "Every QA acceptance rate, JPEG size, render/QA/publication and Drive-control coefficient remains the previous conditional assumption. New D2 quality/QA was not measured.",
                "Old G, 10k/warmup500/val2k, seeds43/44, three independent40k runs, run counts and checkpoint-byte allowance are not newly approved. The scene supplement remains outside totals.",
                "Intervals are old fixed/linear preflight sensitivity cases, not bounds. Actual image endpoints are inherited estimates. This is not the requested QA-updated campaign budget.",
                "The failed probe's one-time VM usage and assignment/human waiting are outside these future campaign totals.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "setup_seconds": setup_seconds,
                "nht_render_calls": render_calls,
                "charged_frames": 12,
                "completed_frames": 0,
                "remaining_stage_budget": 188,
                "render_final_save_seconds": finalize,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
