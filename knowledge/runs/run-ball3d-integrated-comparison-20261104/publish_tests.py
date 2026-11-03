"""Promote exact test node/phase evidence within knowledge's per-file size limits."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from common import HERE, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    collect = json.loads((args.source / "collect.json").read_text())
    run = json.loads((args.source / "run.json").read_text())
    if collect["exit_code"] or run["exit_code"] or collect["nodes"] != run["nodes"]:
        raise ValueError("Test collection/execution did not match or succeed")
    (HERE / "test-nodes.txt").write_text("\n".join(collect["nodes"]) + "\n")
    phases = run["phases"]
    nodes = list(phases)
    for i in range(0, len(nodes), 180):
        write_json(
            HERE / f"test-phases-{i // 180 + 1:02d}.json",
            {n: phases[n] for n in nodes[i : i + 180]},
        )
    raw = (args.source / "run.txt").read_bytes()
    parts = []
    remaining = raw
    while remaining:
        bound = min(200 * 1024, len(remaining))
        if bound < len(remaining):
            bound = remaining.rfind(b"\n", 0, bound) + 1
        parts.append(remaining[:bound])
        remaining = remaining[bound:]
    for i, part in enumerate(parts):
        (HERE / f"test-durations-{i + 1:02d}.txt").write_bytes(part)
    if b"".join(parts) != raw:
        raise ValueError("Split test log changed bytes")
    new = {
        n: p
        for n, p in phases.items()
        if "test_ball3d_integrated_comparison.py::" in n
        or "::test_explicit_cells_" in n
        or "::test_explicit_redecoded_pixels_" in n
        or "::test_explicit_pixels_" in n
    }
    totals = {n: sum(p["seconds"] for p in rows.values()) for n, rows in phases.items()}
    longcases = {n: s for n, s in totals.items() if s > 5}
    longsetup = {
        n: p["setup"]["seconds"]
        for n, p in phases.items()
        if p.get("setup", {}).get("seconds", 0) > 1
    }
    text = raw.decode()
    pytest_elapsed = re.findall(r"=+ (.*?) in ([0-9.]+)s", text)
    summary = {
        "collected": collect["count"],
        "executed_nodes": len(phases),
        "new_cases": len(new),
        "collection_wrapper_seconds": collect["wrapper_seconds"],
        "collection_pytest_seconds": float(
            re.findall(
                r"tests collected in ([0-9.]+)s",
                (args.source / "collect.txt").read_text(),
            )[-1]
        ),
        "execution_wrapper_seconds": run["wrapper_seconds"],
        "pytest_final_summary": pytest_elapsed[-1] if pytest_elapsed else None,
        "collection_arguments": collect["pytest_arguments"],
        "execution_arguments": run["pytest_arguments"],
        "phase_sum_seconds": sum(totals.values()),
        "new_phase_sum_seconds": sum(totals[n] for n in new),
        "max_new_case_seconds": max(totals[n] for n in new),
        "max_new_setup_seconds": max(
            p.get("setup", {}).get("seconds", 0) for p in new.values()
        ),
        "new_nodes": list(new),
        "cases_over_5_seconds": longcases,
        "setups_over_1_second": longsetup,
        "run_log_sha256": hashlib.sha256(raw).hexdigest(),
        "run_log_parts": len(parts),
        "all_test_phases": sorted(p.name for p in HERE.glob("test-phases-*.json")),
        "exit_code": run["exit_code"],
        "thread_limits": run["thread_limits"],
        "limitation": "Worker phase time sums differ from elapsed wall time. Existing cold import/source audits, real numerical solvers and subprocess CLI end-to-end tests are retained; no new case requires >5s or setup >1s unless measured above.",
    }
    write_json(HERE / "test-summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "collected",
                    "new_cases",
                    "pytest_final_summary",
                    "execution_wrapper_seconds",
                    "max_new_case_seconds",
                    "max_new_setup_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
