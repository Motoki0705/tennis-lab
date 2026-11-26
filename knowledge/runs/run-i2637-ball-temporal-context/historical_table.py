"""Read the three prior validation comparisons into one paper-ready CSV."""

import argparse
import csv
from pathlib import Path

import yaml

FIELDS = (
    "series", "node", "condition", "frames", "parameters", "cumulative_updates",
    "query_updates", "observed_frames", "mean_error_px", "p95_error_px",
    "within_20px_rate", "negative_fpr", "observed_fnr",
)


def node(path: Path) -> dict:
    return yaml.safe_load(path.read_text().split("---", 2)[1])


def row(metrics: dict, **fields) -> dict:
    values = {key: metrics.get(key) for key in FIELDS}
    values.update(fields)
    values["within_20px_rate"] = metrics.get("within_20px_rate", metrics.get("within_distance_rate"))
    return values


def collect(root: Path) -> list[dict]:
    length_path, capacity_path, query_path = [next(root.glob(f"{n:06d}-*.md")) for n in (68, 88, 94)]
    length = node(length_path)
    capacity = node(capacity_path)
    query = node(query_path)
    rows = []
    for condition, metrics in length["metrics"]["terminal_validation"].items():
        frames = int(condition.rsplit("T", 1)[1])
        rows.append(row(
            metrics, series="long_sequence", node=length_path.name, condition=condition,
            frames=frames, observed_frames=metrics["support"]["observed"],
            query_updates=None if condition.startswith(("dpt-", "common-")) else (
                metrics["cumulative_updates"] - length["config"]["common_pretraining_updates"]
            ),
        ))
    for member in capacity["members"]:
        path = next(root.glob(f"*-{member}.md"))
        record = node(path)
        config, metrics = record["config"], record["metrics"]
        rows.append(row(
            metrics, series="capacity", node=path.name,
            condition=f"d{config['dim']}-L{config['layers']}", frames=config["frames"],
            parameters=metrics["total_parameters"], query_updates=config["query_updates_after"],
            observed_frames=query["config"]["observed_validation_frames"],
        ))
    for member in query["members"]:
        if member == length["id"]:
            continue  # The unchanged comparison baseline is already in long_sequence.
        path = next(root.glob(f"*-{member}.md"))
        metrics = node(path)["metrics"]
        rows.append(row(metrics, series="controlled_query", node=path.name,
                        condition=member.removeprefix("run-i1899-").removesuffix("-s42-12k"),
                        frames=query["config"]["final_T"]))
    if len(rows) != 14 or {r["observed_frames"] for r in rows} != {85_473}:
        raise ValueError("Unexpected historic comparison coverage")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = collect(Path("knowledge/nodes/ball_detection"))
    with args.output.open("w", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        table.writeheader()
        table.writerows(rows)


if __name__ == "__main__":
    main()
