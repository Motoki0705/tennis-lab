"""Capture small verified receipts; prediction arrays and videos remain in outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path, required=True)
    parser.add_argument("--collection-log", type=Path, required=True)
    parser.add_argument("--resource-log", type=Path, required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parent
    output = args.comparison.resolve()
    inventory_path = output / "artifacts.json"
    if sha(inventory_path) != (output / "artifacts.sha256").read_text().strip():
        raise ValueError("Artifact manifest checksum mismatch")
    inventory = json.loads(inventory_path.read_text())
    for relative, spec in inventory["files"].items():
        path = output / relative
        if path.stat().st_size != spec["size_bytes"] or sha(path) != spec["sha256"]:
            raise ValueError(f"Output changed: {relative}")
    files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if files != set(inventory["files"]) | {"artifacts.json", "artifacts.sha256"}:
        raise ValueError("Output has undeclared files")
    for filename in (
        "artifacts.json",
        "verification.json",
        "extremes.json",
        "metrics-table.png",
    ):
        source = output / filename
        limit = 512 * 1024 if source.suffix == ".png" else 256 * 1024
        if source.stat().st_size > limit:
            raise ValueError(f"Receipt exceeds knowledge storage limit: {filename}")
        shutil.copyfile(source, target / filename)
    verification = json.loads((output / "verification.json").read_text())
    scores = json.loads((output / "scores.json").read_text())
    summary = {
        "schema": "tennis_lab_ball3d_run_summary.v1",
        "output_dir": str(output),
        "artifact_manifest_sha256": sha(inventory_path),
        "verified_output_files": len(inventory["files"]),
        "inputs": inventory["inputs"],
        "source_code_commit": verification["source_code_commit"],
        "replay_code_commit": verification["replay_code_commit"],
        "methods": {},
    }
    for name, metrics in scores["methods"].items():
        summary["methods"][name] = {
            "lovo_reprojection_by_reference_kind": metrics[
                "lovo_reprojection_by_reference_kind"
            ],
            "runtime_seconds_all_four_runs": metrics["runtime_seconds_all_four_runs"],
            "all_views": {
                segment: {
                    key: value
                    for key, value in values.items()
                    if key != "event_reprojection_px"
                }
                for segment, values in metrics["folds"]["all_views"].items()
            },
        }
    write(target / "run-summary.json", summary)
    logs = args.pytest_log.read_text()
    durations = {}
    for value, stage, node in re.findall(
        r"([0-9.]+)s\s+(setup|call|teardown)\s+(tests/[^\n]+)", logs
    ):
        durations.setdefault(node, {})[stage] = float(value)
    suite = ET.parse(args.junit).getroot().find("testsuite")
    if suite is None:
        raise ValueError("Missing pytest suite")
    cases = []
    for case in suite.findall("testcase"):
        node = (
            case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        )
        if set(durations[node]) != {"setup", "call", "teardown"}:
            raise ValueError(f"Incomplete duration stages: {node}")
        if (
            case.find("failure") is not None
            or case.find("error") is not None
            or case.find("skipped") is not None
        ):
            raise ValueError(f"Cannot record test as passed: {node}")
        cases.append(
            {
                "node": node,
                "seconds_setup_call_teardown": float(case.attrib["time"]),
                "duration_stages_seconds_rounded": durations[node],
            }
        )
    collection = re.search(
        r"(\d+) tests collected in ([0-9.]+)s", args.collection_log.read_text()
    )
    passed = re.search(r"(\d+) passed in ([0-9.]+)s", logs)
    if (
        collection is None
        or passed is None
        or int(collection[1]) != len(cases)
        or int(passed[1]) != len(cases)
    ):
        raise ValueError("Collection/pass count mismatch")
    write(
        target / "test-results.json",
        {
            "schema": "tennis_lab_ball3d_test_measurements.v1",
            "collected": len(cases),
            "passed": len(cases),
            "collection_seconds": float(collection[2]),
            "pytest_elapsed_seconds": float(passed[2]),
            "cases": cases,
        },
    )
    shutil.copyfile(args.pytest_log, target / "test-durations.txt")
    shutil.copyfile(args.resource_log, target / "resource-usage.txt")
    print(
        f"Verified {len(inventory['files'])} output files; captured {len(cases)} tests. Arrays/video remain in {output}"
    )


if __name__ == "__main__":
    main()
