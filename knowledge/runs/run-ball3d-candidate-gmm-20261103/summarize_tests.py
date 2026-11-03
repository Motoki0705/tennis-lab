"""Summarize exact collected nodes and phase durations without hiding skips."""

import json
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).parent
    collection = json.loads((root/"test-collection.json").read_text())
    execution = json.loads((root/"test-results.json").read_text())
    if collection["exit_code"] or execution["exit_code"] or collection["nodes"] != execution["nodes"]:
        raise ValueError("Test collection/execution failed or nodes differ")
    cases = defaultdict(dict)
    for report in execution["reports"]:
        if report["outcome"] != "passed":
            raise ValueError("Failed/skipped tests cannot be reported as passing")
        cases[report["node"]][report["phase"]] = report["seconds"]
    if set(cases) != set(execution["nodes"]) or any(set(v) != {"setup", "call", "teardown"} for v in cases.values()):
        raise ValueError("Incomplete pytest phase evidence")
    files = defaultdict(lambda: {"count": 0, "phase_seconds": 0.})
    new = []
    for node, phases in cases.items():
        total = sum(phases.values())
        filename = node.split("::")[0]
        files[filename]["count"] += 1
        files[filename]["phase_seconds"] += total
        if ("/ball_gmm/" in node or node.endswith(".py::test_default_pipeline_matches_frozen_main_arrays")
                or "/test_kalman.py::" in node or "/test_uncertainty.py::" in node
                or "/test_ball3d_candidate_gmm.py::" in node):
            new.append({"node": node, "total_seconds": total, **phases})
    summary = {"collected": len(collection["nodes"]), "passed": len(cases),
        "collection_wrapper_seconds": collection["wall_seconds"],
        "execution_wrapper_seconds": execution["wall_seconds"],
        "max_case_seconds": max(sum(p.values()) for p in cases.values()),
        "max_setup_seconds": max(p["setup"] for p in cases.values()),
        "new_case_count": len(new), "new_cases": new, "files": dict(files)}
    (root/"test-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"new_cases", "files"}}, indent=2))


if __name__ == "__main__":
    main()
