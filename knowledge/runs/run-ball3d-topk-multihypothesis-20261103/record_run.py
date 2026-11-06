"""Validate an immutable replay directory and promote only small evidence files."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    destination = Path(__file__).parent
    inventory = json.loads((args.output / "artifacts.json").read_text())
    for name, record in inventory["files"].items():
        path = args.output / name
        assert path.stat().st_size == record["size_bytes"], name
        assert dual_sha256(path) == record["sha256"], name
    actual = {str(p.relative_to(args.output)) for p in args.output.rglob("*") if p.is_file()}
    assert actual == set(inventory["files"]) | {"artifacts.json"}
    unchanged = []
    candidate_diagnostics = {}
    for method in ("pipeline_baseline", "topk_multihypothesis"):
        for fold in ("all_views", "without_cam0", "without_cam1", "without_cam2"):
            relative = Path("results") / method / fold
            before, after = load_run(args.previous / relative), load_run(args.output / relative)
            assert before.config == after.config
            for field in ("frame_indices", "positions", "valid", "used_views"):
                np.testing.assert_array_equal(getattr(before.result, field), getattr(after.result, field))
            for key in before.result.diagnostics:
                np.testing.assert_array_equal(before.result.diagnostics[key], after.result.diagnostics[key])
            unchanged.append(str(relative))
            if method == "topk_multihypothesis":
                diagnostic = after.result.diagnostics
                candidate_diagnostics[fold] = {}
                for segment, start, stop in (("tuning", 0, 505), ("evaluation", 505, 1010), ("all", 0, 1010)):
                    indices = diagnostic["selected_candidate_indices"][:, start:stop]
                    selected = indices[indices >= 0]
                    candidate_diagnostics[fold][segment] = {
                        "valid_frames": int(after.result.valid[start:stop].sum()),
                        "frames_with_geometric_hypotheses": int((diagnostic["hypothesis_count"][start:stop] > 0).sum()),
                        "missing_despite_geometric_hypotheses": int(((diagnostic["hypothesis_count"][start:stop] > 0) & ~after.result.valid[start:stop]).sum()),
                        "selected_candidate_rank_counts_one_based": np.bincount(selected, minlength=8).tolist(),
                        "temporal_restarts": int(diagnostic["temporal_restart"][start:stop].sum()),
                    }
    scores = json.loads((args.output / "scores.json").read_text())
    paired = json.loads((args.output / "paired-scores.json").read_text())
    metrics = {"primary_segment": scores["primary_segment"], "methods": {}}
    for name, values in scores["methods"].items():
        metrics["methods"][name] = {key: value for key, value in values.items() if key != "folds"}
        metrics["methods"][name]["folds"] = {
            fold: {segment: {key: value for key, value in row.items()
                             if key not in {"event_reprojection_px", "reprojection_by_reference_kind"}}
                   for segment, row in segments.items()}
            for fold, segments in values["folds"].items()}
    metrics["paired_two_method_common_lovo"] = {
        name: row["common_lovo_reprojection_px"] for name, row in paired["methods"].items()}
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (destination / "candidate-diagnostics.json").write_text(json.dumps(candidate_diagnostics, indent=2) + "\n")
    for name in ("artifacts.json", "verification.json", "run-summary.json", "frozen-config.json",
                 "fold-metrics.csv", "metrics-table.png", "trajectories.png"):
        shutil.copyfile(args.output / name, destination / name)
    receipt = json.loads((destination / "test-results.json").read_text())
    collection = json.loads((destination / "test-collection.json").read_text())
    cases = defaultdict(float)
    phases = defaultdict(dict)
    for report in receipt["reports"]:
        cases[report["node"]] += report["seconds"]
        phases[report["node"]][report["phase"]] = report["seconds"]
    by_file = defaultdict(lambda: {"count": 0, "case_seconds": 0.0})
    for node, seconds in cases.items():
        row = by_file[node.split("::")[0]]
        row["count"] += 1
        row["case_seconds"] += seconds
    test_summary = {"count": len(cases), "exit_code": receipt["exit_code"],
        "collection_wall_seconds": collection["wall_seconds"], "execution_wrapper_wall_seconds": receipt["wall_seconds"],
        "by_file": by_file, "cases": {node: {"seconds": seconds, "phases": phases[node]} for node, seconds in cases.items()}}
    (destination / "test-summary.json").write_text(json.dumps(test_summary, indent=2) + "\n")
    validation = {"output": str(args.output.resolve()), "verified_inventory_files": len(inventory["files"]),
        "inventory_sha256": dual_sha256(args.output / "artifacts.json"),
        "array_and_diagnostic_bitwise_equal_previous_version": unchanged,
        "previous_output": str(args.previous.resolve()),
        "code_commit": inventory["code_commit"],
        "artifact_hashes": {name: dual_sha256(args.output / name) for name in
                            ("index.html", "comparison.mp4", "scores.json", "paired-scores.json", "metrics-table.png")},
        "max_test_case_seconds": max(cases.values()),
        "max_test_setup_seconds": max(row.get("setup", 0) for row in phases.values())}
    (destination / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
