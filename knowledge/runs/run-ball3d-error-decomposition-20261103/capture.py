"""Verify immutable artifacts and retain small diagnostic/test receipts."""

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--log-prefix", type=Path, required=True)
    args = parser.parse_args()
    output, target = args.comparison, args.destination
    inventory = json.loads((output / "artifacts.json").read_text())
    digest = sha(output / "artifacts.json")
    if digest != (output / "artifacts.sha256").read_text().strip():
        raise ValueError("Manifest digest differs")
    for name, spec in inventory["files"].items():
        path = output / name
        if sha(path) != spec["sha256"] or path.stat().st_size != spec["size_bytes"]:
            raise ValueError(f"Artifact changed: {name}")
    files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if files != set(inventory["files"]) | {"artifacts.json", "artifacts.sha256"}:
        raise ValueError("Undeclared output files")
    scores = json.loads((output / "scores.json").read_text())
    verification = json.loads((output / "verification.json").read_text())

    def drop_times(value):
        if isinstance(value, dict):
            return {
                k: drop_times(v)
                for k, v in value.items()
                if not k.startswith("runtime_seconds")
            }
        if isinstance(value, list):
            return [drop_times(v) for v in value]
        return value

    prior = json.loads((output.parent / "foundation_v1/scores.json").read_text())
    unchanged = {}
    for name in ("pipeline_baseline", "without_loess", "annotation_oracle"):
        unchanged[name] = drop_times(scores["methods"][name]) == drop_times(
            prior["methods"][name]
        )
        if not unchanged[name]:
            raise ValueError(f"Foundation metrics changed: {name}")
    summary = {
        "schema": "tennis_lab_ball3d_error_decomposition_summary.v1",
        "execution_commit": verification["execution_commit"],
        "output_dir": str(output),
        "artifact_manifest_sha256": digest,
        "verified_output_files": len(inventory["files"]),
        "foundation_metrics_identical_excluding_runtime": unchanged,
        "methods": {
            name: {
                "lovo_reprojection_by_reference_kind": row[
                    "lovo_reprojection_by_reference_kind"
                ],
                "folds": {
                    fold: {
                        split: {
                            k: v
                            for k, v in values.items()
                            if k
                            in (
                                "reprojection_px",
                                "valid_3d_rate",
                                "scored_label_count",
                                "scoring_label_count",
                                "scored_label_coverage",
                            )
                        }
                        for split, values in folds.items()
                    }
                    for fold, folds in row["folds"].items()
                },
                "runtime_seconds_all_four_runs": row["runtime_seconds_all_four_runs"],
            }
            for name, row in scores["methods"].items()
        },
        "artifact_hashes": {
            name: inventory["files"][name]
            for name in (
                "index.html",
                "comparison.mp4",
                "scores.json",
                "decode.json",
                "paired-budget.json",
            )
        },
    }
    write(target / "run-summary.json", summary)
    decode = json.loads((output / "decode.json").read_text())
    write(
        target / "regression-summary.json",
        {
            split: {
                camera: {kind: data["regression"] for kind, data in kinds.items()}
                for camera, kinds in cameras.items()
            }
            for split, cameras in decode.items()
        },
    )
    write(
        target / "decode-strata.json",
        {
            split: {
                camera: {
                    kind: {
                        key: value
                        for key, value in data.items()
                        if key not in ("regression", "position_grid_6x4")
                    }
                    for kind, data in kinds.items()
                }
                for camera, kinds in cameras.items()
            }
            for split, cameras in decode.items()
        },
    )
    for name in (
        "verification.json",
        "artifacts.json",
        "candidates.json",
        "smoothing.json",
        "timing.json",
        "paired-budget.json",
        "extremes.json",
        "position-bias.png",
        "candidate-recovery.png",
        "regression.png",
        "smoothing.png",
    ):
        path = output / name
        limit = 512 * 1024 if path.suffix == ".png" else 256 * 1024
        if path.stat().st_size > limit:
            raise ValueError(f"Knowledge size limit: {name}")
        shutil.copyfile(path, target / name)
    prefix = str(args.log_prefix)
    log = Path(prefix + "-durations.txt").read_text()
    collection = Path(prefix + "-collection.txt").read_text()
    durations = {}
    for value, stage, node in re.findall(
        r"([0-9.]+)s\s+(setup|call|teardown)\s+(tests/[^\n]+)", log
    ):
        durations.setdefault(node, {})[stage] = float(value)
    suite = ET.parse(prefix + "-final.xml").getroot().find("testsuite")
    cases = []
    for case in suite.findall("testcase"):
        node = (
            case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        )
        if len(case) or set(durations[node]) != {"setup", "call", "teardown"}:
            raise ValueError(f"Incomplete/pending test: {node}")
        cases.append(
            {
                "node": node,
                "seconds_setup_call_teardown": float(case.attrib["time"]),
                "duration_stages_seconds_rounded": durations[node],
            }
        )
    collected = re.search(r"(\d+) tests collected in ([0-9.]+)s", collection)
    passed = re.search(r"(\d+) passed in ([0-9.]+)s", log)
    if int(collected[1]) != len(cases) or int(passed[1]) != len(cases):
        raise ValueError("Collected/passed counts disagree")
    write(
        target / "test-results.json",
        {
            "collected": len(cases),
            "passed": len(cases),
            "collection_seconds": float(collected[2]),
            "pytest_elapsed_seconds": float(passed[2]),
            "cases": cases,
        },
    )
    for suffix, destination in (
        ("-durations.txt", "test-durations.txt"),
        ("-collection.txt", "test-collection.txt"),
        ("-resource.txt", "resource-usage.txt"),
        ("-run.log", "execution.log"),
    ):
        shutil.copyfile(prefix + suffix, target / destination)
    for path in target.iterdir():
        if path.is_file() and path.stat().st_size > (
            512 * 1024 if path.suffix == ".png" else 256 * 1024
        ):
            raise ValueError(f"Knowledge file too large: {path}")
    size = sum(
        p.stat().st_size
        for p in target.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    if size > 2 * 1024 * 1024:
        raise ValueError(f"Knowledge run too large: {size}")
    print(
        f"Verified {len(inventory['files'])} artifacts and {len(cases)} passed tests; knowledge bytes={size}"
    )


if __name__ == "__main__":
    main()
