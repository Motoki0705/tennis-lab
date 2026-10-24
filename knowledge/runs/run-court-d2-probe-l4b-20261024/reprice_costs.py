"""Update frozen QA/JPEG sensitivity estimates and this run's measured setup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--setup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parents[1]
    pilot = runs / "run-court-scale-pilot-l4-20261023"
    previous = runs / "run-court-d2-probe-l4-20261024"
    with tempfile.TemporaryDirectory(dir="tmp") as temporary:
        intermediate = Path(temporary) / "legacy-setup-costs.json"
        subprocess.run(
            [
                sys.executable,
                str(previous / "update_costs.py"),
                "--analysis",
                str(args.analysis),
                "--output",
                str(intermediate),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = read(intermediate)
    model = read(pilot / "cost-scenarios.json")
    setup_seconds = sum(
        read(args.setup / f"{name}.json")["wall_seconds"]
        for name in ("uv", "nht", "asset-download", "asset-verify")
    )
    training_extras = sum(
        read(pilot / f"evidence/{name}-timing.json")["wall_seconds"]
        for name in ("real-download", "backbone-download", "real-split")
    )
    delta = setup_seconds + training_extras - model["cold_setup_seconds_four_scenes"]
    for variant in result["variants"].values():
        for budget in variant["budgets"]:
            for key, count in (
                ("main_only_l4_hold_hours", budget["main_conditions"]),
                ("with_additions_l4_hold_hours", budget["all_training_runs"]),
                ("scene_supplement_l4_hold_hours", 1),
            ):
                budget[key] = [value + delta * count / 3600 for value in budget[key]]
    result.update(
        schema="court_d2_l4b_conditional_costs_v1",
        measured_generation_setup_seconds=setup_seconds,
        inherited_training_input_extra_seconds=training_extras,
        conditional_training_setup_seconds=setup_seconds + training_extras,
        delta_setup_seconds_per_training_run=delta,
    )
    result["assumptions"].append(
        "Only uv/NHT/adopted-asset download and validation setup is replaced by this run's measured setup. Real/backbone/split preparation remains inherited; the legacy render/Drive/training model remains separate from raw-NPZ probe throughput."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: [
                    {
                        "nominal_max": budget["previous_nominal_max_images"],
                        "hours": budget["with_additions_l4_hold_hours"],
                        "drive_GB": budget["drive_increment_bytes"] / 1e9,
                    }
                    for budget in variant["budgets"]
                ]
                for name, variant in result["variants"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
