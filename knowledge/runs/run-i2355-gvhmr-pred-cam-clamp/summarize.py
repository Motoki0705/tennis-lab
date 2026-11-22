"""Collect build outcomes from diagnosis outputs into one small tracked JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FOREIGN = re.compile(r"Computing costs for \{(ForeignNode\[[^\]]*\])\}")


def summarize(directory: Path) -> dict[str, object]:
    report = json.loads((directory / "diagnosis.json").read_text())
    if report.get("status") != "done":
        raise ValueError(f"{directory} did not finish")
    graphs = report.get("graphs", {})
    builds = {}
    for name, build in report["builds"].items():
        costs = build.get("last_computing_costs") or ""
        match = FOREIGN.search(costs)
        graph = graphs.get(name, {})
        builds[name] = {
            "exit_code": build["exit_code"],
            "plan_published": build["plan_published"],
            "options": build["options"],
            "graph_nodes": graph.get("nodes"),
            "graph_inputs": graph.get("inputs"),
            "last_foreign_node": match.group(1) if match else None,
        }
    summary = {
        "queue_job": report["queue_job"],
        "output": str(directory),
        "builds": builds,
    }
    for key in ("sources", "source", "variants", "dropped_boundaries"):
        if key in report:
            summary[key] = report[key]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = {path.name: summarize(path) for path in args.diagnosis}
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
