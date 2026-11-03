"""Verify all artifacts and retain small metric/test receipts, without prediction arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tests(target):
    raw = (target / "test-durations.txt").read_text()
    stages = {}
    for value, stage, node in re.findall(r"([0-9.]+)s\s+(setup|call|teardown)\s+(tests/[^\n]+)", raw):
        stages.setdefault(node, {})[stage] = float(value)
    root = ET.parse(target / "test-results-junit.txt").getroot()
    rows = []
    for case in root.iter("testcase"):
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        if any(case.find(k) is not None for k in ("failure", "error", "skipped")):
            raise ValueError(f"Cannot claim passed: {node}")
        rows.append({"node": node, "seconds_setup_call_teardown": float(case.attrib["time"]), "stages_rounded": stages[node]})
    collection = re.search(r"(\d+) tests collected in ([\d.]+)s", (target / "test-collection.txt").read_text())
    execution = re.search(r"(\d+) passed in ([\d.]+)s", raw)
    if int(collection[1]) != len(rows) or int(execution[1]) != len(rows):
        raise ValueError("Test count mismatch")
    write(target / "test-results.json", {"collected": len(rows), "passed": len(rows), "collection_seconds": float(collection[2]),
                                        "pytest_elapsed_seconds": float(execution[2]), "cases": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--resource", type=Path)
    args = parser.parse_args()
    target = Path(__file__).resolve().parent
    tests(target)
    if args.comparison is None:
        return
    output = args.comparison.resolve()
    inventory = json.loads((output / "artifacts.json").read_text())
    for relative, spec in inventory["files"].items():
        path = output / relative
        if path.stat().st_size != spec["size_bytes"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Changed artifact: {relative}")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != set(inventory["files"]) | {"artifacts.json"}:
        raise ValueError("Undeclared artifact")
    scores = json.loads((output / "scores.json").read_text())
    paired = json.loads((output / "paired-scores.json").read_text())
    rows = []
    for name, method in scores["methods"].items():
        for segment in ("evaluation", "tuning", "all"):
            for fold in ("lovo_pool", *method["folds"]):
                metrics = method["lovo_reprojection_px"][segment] if fold == "lovo_pool" else method["folds"][fold][segment]["reprojection_px"]
                if fold != "lovo_pool":
                    values = method["folds"][fold][segment]
                    metrics = {**metrics, "label_count": values["scoring_label_count"], "scored_label_coverage": values["scored_label_coverage"]}
                rows.append({"method": name, "segment": segment, "fold": fold, **metrics})
    with (target / "fold-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    compact = {"output_dir": str(output), "artifact_manifest_sha256": digest(output / "artifacts.json"), "verified_output_files": len(inventory["files"]),
               "inputs": inventory["inputs"], "code_commit": inventory["code_commit"], "methods": {}}
    for name, method in scores["methods"].items():
        compact["methods"][name] = {k: method[k] for k in ("lovo_reprojection_by_reference_kind", "common_lovo_reprojection_by_reference_kind", "runtime_seconds_all_four_runs")}
        compact["methods"][name]["folds"] = {
            fold: {segment: {k: values[k] for k in ("reprojection_px", "valid_3d_rate", "valid_3d_frames", "scored_label_count", "scoring_label_count", "scored_label_coverage") if k in values} for segment, values in segments.items()}
            for fold, segments in method["folds"].items()}
    compact["paired_common"] = {name: value["common_lovo_reprojection_px"] for name, value in paired["methods"].items()}
    write(target / "run-summary.json", compact)
    for name in ("selection.json", "verification.json", "likelihood-slices.json", "extremes.json", "metrics-table.png", "likelihood-slices.png"):
        source = output / name
        limit = 512 * 1024 if source.suffix == ".png" else 256 * 1024
        if source.stat().st_size > limit:
            raise ValueError(f"Knowledge file budget exceeded: {name}")
        shutil.copyfile(source, target / name)
    checksums = {name: {"sha256": digest(output / name), "size_bytes": (output / name).stat().st_size}
                 for name in ("artifacts.json", "scores.json", "paired-scores.json", "selection.json", "index.html", "comparison.mp4", "metrics-table.png", "likelihood-slices.png")}
    write(target / "checksums.json", {"output_dir": str(output), "files": checksums, "all_manifest_entries_verified": True})
    if args.resource:
        shutil.copyfile(args.resource, target / "resource-usage.txt")
    print(f"Verified {len(inventory['files'])} output files")


if __name__ == "__main__":
    main()
