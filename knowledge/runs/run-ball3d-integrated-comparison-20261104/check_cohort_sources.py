"""Compare newly replayed cohort distributions with all three child cohort reports."""

import argparse
import json
from pathlib import Path

from common import write_json

from src.utils.checksum import dual_sha256

PINS = {
    "topk_multihypothesis_cohort_v1": "10e3ae2529b1fbe8bc354f95a5749c6e0ed91aa7da084fe850617a4142e21ea5",
    "candidate_gmm_cohort_v1": "f904ee8407efe802317166034ec6aeba16d107412fed29fe5fa11ce376507cc0",
    "dense_likelihood_cohort_v1": "c6da1d4261fa944a6acac158b4e697b1e477477fcb1c129c0d282a4fec119e97",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.output / "execution-plan.json").read_text())
    root = Path(plan["inputs"])
    actual = json.loads((args.output / "cohorts.json").read_text())
    reports = {}
    for directory, pin in PINS.items():
        path = root / directory / "cohorts.json"
        if dual_sha256(path) != pin:
            raise ValueError(f"Saved child cohort checksum differs: {directory}")
        prior = json.loads(path.read_text())
        if directory.startswith("topk"):
            groups = {prior["condition"]: prior["strata"]}
        elif directory.startswith("candidate"):
            groups = {}
            for row in prior["rows"]:
                m = row["metrics"]
                normalized = {
                    "segment": row["segment"],
                    "held_out_camera": row["camera"],
                    "reference_kind": row["reference_kind"],
                    "common": {
                        "method": m["common_method"],
                        "baseline": m["common_baseline"],
                    },
                    "method_only": {"method": m["method_only"]},
                    "baseline_only": {"baseline": m["baseline_only"]},
                    "totals": {
                        "method": m["method_all"],
                        "baseline": m["baseline_all"],
                    },
                }
                groups.setdefault(row["method"], []).append(normalized)
        else:
            groups = {
                name: [
                    {
                        **row,
                        "totals": {
                            "method": row["method_all"],
                            "baseline": row["baseline_all"],
                        },
                    }
                    for row in rows
                ]
                for name, rows in prior["methods"].items()
                if name in actual
            }
        checks = 0
        for name, rows in groups.items():
            for row in rows:
                new = next(
                    r
                    for r in actual[name]["strata"]
                    if all(
                        r[k] == row[k]
                        for k in ("segment", "held_out_camera", "reference_kind")
                    )
                )
                for cohort, sides in (
                    ("common", ("method", "baseline")),
                    ("method_only", ("method",)),
                    ("baseline_only", ("baseline",)),
                    ("totals", ("method", "baseline")),
                ):
                    for side in sides:
                        if new[cohort][side] != row[cohort][side]:
                            raise ValueError(
                                f"Child cohort mismatch: {directory}/{name}/{row['segment']}/{row['held_out_camera']}/{row['reference_kind']}/{cohort}/{side}"
                            )
                        checks += 1
        reports[directory] = {
            "sha256": pin,
            "distribution_comparisons": checks,
            "exact": True,
            "methods": list(groups),
        }
    write_json(args.output / "prior-cohort-comparison.json", reports)
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
