"""Independent raw-report arithmetic and CSV/JSON cross-check (no compare import)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


def check(raw_directory: Path, bundle: Path) -> dict[str, Any]:
    output = json.loads((bundle / "comparison.json").read_text())
    raw = {
        path.stem: json.loads(path.read_text()) for path in raw_directory.glob("*.json")
    }
    aliases = {"P01": "scene-2"}
    metrics = (
        "kp_mean_distance_px",
        "kp_median_distance_px",
        "seg_miou",
        "line_dice",
        "semantic_line_miou",
    )
    comparisons: list[tuple[float, float]] = []
    exact: list[bool] = []
    for table in ("real_all", "real_pairs", "b00_addition"):
        for row in output[table]:
            name = aliases.get(row["condition"], row["condition"])
            for metric in metrics:
                exact.append(
                    row[metric]
                    == raw[name]["splits"]["tennis_court_detector-val"]["metrics"][
                        metric
                    ]
                )
    for contrast in output["contrasts"]:
        metric = contrast["metric"]
        values = {
            name: report["splits"]["tennis_court_detector-val"]["metrics"][metric]
            for name, report in raw.items()
        }
        # Fraction.from_float sums the exact binary input values, independently
        # of the production fsum implementation and Python's version of sum.
        same = float((Fraction(values["scene-2"]) + Fraction(values["P23"])) / 2)
        cross = float(
            sum(Fraction(values[p]) for p in ("P02", "P03", "P12", "P13")) / 4
        )
        expected = {
            "same_group_mean": same,
            "cross_group_mean": cross,
            "delta": same - cross,
            "additivity": (values["P02"] + values["P13"])
            - (values["P03"] + values["P12"]),
        }
        comparisons.extend((contrast[k], v) for k, v in expected.items())
    groups = {"B00": 0, "B01": 0, "B02": 1, "B03": 1}
    for row in output["synthetic_long"]:
        report = raw[aliases.get(row["condition"], row["condition"])]
        observed = report["splits"]["synthetic_court-val"]["scenes"][row["scene"]]
        exact.append(row["count"] == observed["count"])
        members = set(report["data_condition"]["scene_images"])
        expected_category = (
            "seen"
            if row["scene"] in members
            else "unseen-with-same-group"
            if any(groups[s] == groups[row["scene"]] for s in members)
            else "unseen-without-same-group"
        )
        exact.append(row["category"] == expected_category)
        exact.extend(row[m] == observed["metrics"][m] for m in metrics)
    for metric, rows in output["synthetic_matrix"].items():
        for row in rows:
            scenes = raw[aliases.get(row["condition"], row["condition"])]["splits"][
                "synthetic_court-val"
            ]["scenes"]
            expected_mean = float(
                sum(Fraction(scenes[s]["metrics"][metric]) for s in groups) / 4
            )
            comparisons.append((row["four_scene_mean"], expected_mean))
            exact.extend(row[s] == scenes[s]["metrics"][metric] for s in groups)
    csv_cells = 0
    for path in sorted(bundle.glob("*.csv")):
        key = path.stem
        expected_rows = (
            output["synthetic_matrix"][key[len("synthetic-") :]]
            if key.startswith("synthetic-")
            else output[key]
        )
        with path.open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        assert len(actual) == len(expected_rows), path
        for row, expected_row in zip(actual, expected_rows, strict=True):
            for key, value in expected_row.items():
                assert (
                    float(row[key]) == value
                    if type(value) in (float, int)
                    else row[key] == value
                ), (path, key)
                csv_cells += 1
    assert all(exact), "Extracted metrics/categories differ from raw reports"
    differences = [abs(a - b) for a, b in comparisons]
    assert all(d == 0 for d in differences), (
        "Independent contrast/macro arithmetic differs"
    )
    return {
        "schema": "court_scene_pair_comparison_crosscheck_v1",
        "verified": True,
        "raw_report_count": len(raw),
        "exact_extracted_cells": len(exact),
        "independent_arithmetic_cells": len(comparisons),
        "maximum_absolute_difference": max(differences),
        "csv_json_cells_exact": csv_cells,
        "comparison_sha256": hashlib.sha256(
            (bundle / "comparison.json").read_bytes()
        ).hexdigest(),
        "method": "Direct raw JSON extraction; Fraction exact binary means independently of compare.py; all output CSV cells checked against JSON.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = check(args.evaluations, args.bundle)
    (args.bundle / "crosscheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
