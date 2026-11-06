"""Write compact tables and receipts from the unmodified foundation scores."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from common import HERE, write_json

from src.utils.checksum import dual_sha256


def number(value, rate=False):
    return "—" if value is None else f"{value:.2%}" if rate else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-receipt", action="store_true")
    args = parser.parse_args()
    output = args.output
    plan = json.loads((output / "execution-plan.json").read_text())
    scores = json.loads((output / "scores.json").read_text())
    cohorts = json.loads((output / "cohorts.json").read_text())
    groups = (
        ("予測方式・組合せ（LOESSなしは参考）", plan["methods"]),
        ("正解を使う参考診断（実運用方式ではない）", plan["diagnostics"]),
    )
    lines = [
        "# LOVO 区間別主表",
        "",
        "値は median / p90 px / 厳密な>20px率 ; 採点/対象 (coverage)。後半が主評価。",
        "",
    ]
    compact = {}
    for title, rows in groups:
        lines.extend(
            [
                f"## {title}",
                "",
                "| 条件 | 後半 [505,1010) | 前半 [0,505) | 全区間 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            name = row["name"]
            method = scores["methods"][name]
            values = []
            compact[name] = {}
            for segment in ("evaluation", "tuning", "all"):
                m = method["lovo_reprojection_px"][segment]
                values.append(
                    f"{number(m['median'])}/{number(m['p90'])}/{number(m['above_threshold_rate'], True)} ; {m['count']}/{m['label_count']} ({number(m['scored_label_coverage'], True)})"
                )
                compact[name][segment] = m
            compact[name]["all_view_valid_rate_evaluation"] = method["folds"][
                "all_views"
            ]["evaluation"]["valid_3d_rate"]
            compact[name]["elapsed_seconds_four_runs"] = method[
                "runtime_seconds_all_four_runs"
            ]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
        lines.append("")
    (HERE / "metrics.md").write_text("\n".join(lines) + "\n")
    (output / "metrics.md").write_bytes((HERE / "metrics.md").read_bytes())
    write_json(HERE / "metrics-summary.json", compact)
    shutil.copyfile(output / "metrics.csv", HERE / "fold-metrics.csv")
    cohort_lines = [
        "# baselineとの共通・追加・喪失（後半主評価）",
        "",
        "cell: n; median/p90 px; >20px率。共通双方は同一camera/frame。空集合の誤差は未測定。",
        "",
        "| 条件 | 共通:方式 | 共通:baseline | 追加:方式のみ | 喪失:baselineのみ |",
        "|---|---:|---:|---:|---:|",
    ]

    def cell(m):
        return f"{m['count']}; {number(m['median'])}/{number(m['p90'])}; {number(m['above_threshold_rate'], True)}"

    for row in [*plan["methods"], *plan["diagnostics"]]:
        r = next(
            r
            for r in cohorts[row["name"]]["strata"]
            if r["segment"] == "evaluation"
            and r["held_out_camera"] == "all"
            and r["reference_kind"] == "all"
        )
        cohort_lines.append(
            f"| {row['name']} | "
            + " | ".join(
                cell(m)
                for m in (
                    r["common"]["method"],
                    r["common"]["baseline"],
                    r["method_only"]["method"],
                    r["baseline_only"]["baseline"],
                )
            )
            + " |"
        )
    (HERE / "cohort-primary.md").write_text("\n".join(cohort_lines) + "\n")
    (output / "cohort-primary.md").write_bytes(
        (HERE / "cohort-primary.md").read_bytes()
    )
    with (output / "cohorts.csv").open() as stream:
        reader = csv.DictReader(stream)
        rows = [
            r
            for r in reader
            if r["held_out_camera"] == "all" and r["reference_kind"] == "all"
        ]
    with (HERE / "cohort-pooled.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    active_names = [
        r["name"] for r in plan["methods"] if compact[r["name"]]["all"]["count"]
    ]
    sets = {n: set() for n in active_names}
    with (output / "cohort-samples.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["method"] in sets and r["cohort"] != "baseline_only":
                sets[r["method"]].add((r["held_out_camera"], int(r["frame"])))
    common = set.intersection(*sets.values())
    common_report = {
        "all_registered_count": len(scores["common_lovo_cohort"]["camera_frame_pairs"]),
        "nonempty_prediction_methods": active_names,
        "nonempty_prediction_common_by_segment": {
            name: sum(start <= frame < stop for _, frame in common)
            for name, start, stop in (
                ("evaluation", 505, 1010),
                ("tuning", 0, 505),
                ("all", 0, 1010),
            )
        },
        "nonempty_prediction_common_camera_frame_pairs": sorted(common),
        "primary_comparison": "baseline-versus-each-method pairs; common sets are conditional on availability, never fill missing errors",
    }
    write_json(HERE / "common-cohort-summary.json", common_report)
    write_json(output / "common-cohort-summary.json", common_report)
    if (output / "metrics-evaluation.png").is_file():
        shutil.copyfile(
            output / "metrics-evaluation.png", HERE / "metrics-evaluation.png"
        )
    if args.seal_receipt:
        files = (
            "final-artifacts.json",
            "index.html",
            "portable/index.html",
            "portable-comparison.zip",
            "all-methods/comparison.mp4",
            "shortlist/comparison.mp4",
            "scores.json",
            "cohorts.json",
            "execution-plan.json",
            "prior-comparison.json",
            "prior-cohort-comparison.json",
            "verification.json",
            "visual-review.json",
            "final-verification.json",
        )
        receipt = {
            "output": str(output),
            "checksums": {
                relative: {
                    "sha256": dual_sha256(output / relative),
                    "size_bytes": (output / relative).stat().st_size,
                }
                for relative in files
            },
            "code_commit": plan["code_commit"],
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "prior_comparison": json.loads(
                (output / "prior-comparison.json").read_text()
            ),
            "prior_cohort_comparison": json.loads(
                (output / "prior-cohort-comparison.json").read_text()
            ),
            "verification": json.loads((output / "verification.json").read_text()),
        }
        write_json(HERE / "artifact-receipt.json", receipt)
    print(
        json.dumps(
            {
                "methods": len(compact),
                "common": common_report["nonempty_prediction_common_by_segment"],
                "run_bytes": sum(
                    p.stat().st_size for p in HERE.rglob("*") if p.is_file()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
