"""Convert collected nodes, all pytest duration phases and JUnit into a receipt."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    target = Path(__file__).resolve().parent
    log = (target / "test-durations.txt").read_text()
    collection = (target / "test-collection.txt").read_text()
    phases = {}
    for seconds, stage, node in re.findall(r"([0-9.]+)s\s+(setup|call|teardown)\s+(tests/[^\n]+)", log):
        phases.setdefault(node, {})[stage] = float(seconds)
    suite = ET.parse("/tmp/court-ball-ba-tests.xml").getroot().find("testsuite")
    cases = []
    for case in suite.findall("testcase"):
        if any(case.find(kind) is not None for kind in ("failure", "error", "skipped")):
            raise ValueError("Cannot record a failed or unexecuted case as passed")
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        if set(phases[node]) != {"setup", "call", "teardown"}:
            raise ValueError("Missing duration phases")
        cases.append({"node": node, "total_seconds": float(case.attrib["time"]),
                      "phases_seconds_rounded": phases[node]})
    new = [c for c in cases if "/ball_calibration/" in c["node"] or "test_court_ball_ba.py" in c["node"]]
    collected = re.search(r"(\d+) tests collected in ([0-9.]+)s", collection)
    passed = re.search(r"(\d+) passed(?:, [^\n]+?)? in ([0-9.]+)s", log)
    if collected is None or passed is None:
        raise ValueError("Missing pytest collection/pass summary")
    if int(collected[1]) != len(cases) or int(passed[1]) != len(cases):
        raise ValueError("Test count differs between collection and execution")
    report = {"collected": len(cases), "passed": len(cases), "new_cases": len(new),
        "collection_seconds": float(collected[2]), "pytest_elapsed_seconds": float(passed[2]),
        "case_time_sum_seconds": sum(c["total_seconds"] for c in cases),
        "new_case_time_sum_seconds": sum(c["total_seconds"] for c in new), "cases": cases}
    (target / "test-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
