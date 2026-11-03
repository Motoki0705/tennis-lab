"""Split saved LOVO scores into shared, added and lost camera/frame cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    DEFAULT_SCORING,
    distribution,
    error_samples,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
ANNOTATIONS_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"
COMPARISON_SHA = "1907104f5ca83aecff16c06b120e66c1810c781f577648c839ba8beded02563d"
SEGMENTS = {"evaluation": (505, 1010), "tuning": (0, 505), "all": (0, 1010)}
BASELINE, METHOD = "pipeline_baseline", "topk_multihypothesis"
Key = tuple[str, int]
Scores = dict[Key, tuple[float, str]]


def index_samples(samples: Mapping[str, NDArray[Any]], camera_ids: tuple[str, ...], num_frames: int) -> Scores:
    columns = [samples[name] for name in ("camera_index", "frame_index", "error_px", "reference_kind")]
    if any(column.ndim != 1 or len(column) != len(columns[0]) for column in columns):
        raise ValueError("Scoring sample columns disagree")
    result: Scores = {}
    for camera, frame, error, kind in zip(*columns, strict=True):
        if not 0 <= camera < len(camera_ids) or not 0 <= frame < num_frames:
            raise ValueError("Scoring camera/frame outside the fixed input")
        if not np.isfinite(error) or error < 0 or kind not in {"observed", "estimated"}:
            raise ValueError("Invalid scored error or reference kind")
        key = camera_ids[int(camera)], int(frame)
        if key in result:
            raise ValueError(f"Duplicate scored camera/frame: {key}")
        result[key] = float(error), str(kind)
    return result


def validate_counts(common: int, added: int, lost: int, method_total: int, baseline_total: int) -> dict[str, bool]:
    checks = {"common_plus_added_equals_method": common + added == method_total,
              "common_plus_lost_equals_baseline": common + lost == baseline_total}
    if not all(checks.values()):
        raise ValueError(f"Cohort counts do not conserve scored points: {checks}")
    return checks


def stats(index: Scores, keys: set[Key]) -> dict[str, Any]:
    errors: NDArray[np.float64] = np.asarray([index[key][0] for key in sorted(keys)], dtype=np.float64)
    threshold = DEFAULT_SCORING.gross_reprojection_px
    return {**distribution(errors, threshold=threshold), "above_threshold_count": int((errors > threshold).sum())}


def partition(method: Scores, baseline: Scores) -> dict[str, Any]:
    common = method.keys() & baseline.keys()
    for key in common:
        if method[key][1] != baseline[key][1]:
            raise ValueError(f"Common point reference kind mismatch: {key}")
    added, lost = method.keys() - baseline.keys(), baseline.keys() - method.keys()
    checks = validate_counts(len(common), len(added), len(lost), len(method), len(baseline))
    return {"totals": {"method": stats(method, set(method)), "baseline": stats(baseline, set(baseline))},
        "common": {"count": len(common), "method": stats(method, common), "baseline": stats(baseline, common)},
        "method_only": {"count": len(added), "method": stats(method, added)},
        "baseline_only": {"count": len(lost), "baseline": stats(baseline, lost)}, "checks": checks}


def select(index: Scores, start: int, stop: int, camera: str, kind: str) -> Scores:
    return {key: value for key, value in index.items() if start <= key[1] < stop
            and (camera == "all" or key[0] == camera) and (kind == "all" or value[1] == kind)}


def summarize(method: Scores, baseline: Scores, camera_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    # Validate shared reference identities before filtering into kind strata.
    partition(method, baseline)
    rows = []
    for segment, (start, stop) in SEGMENTS.items():
        for camera in ("all", *camera_ids):
            for kind in ("all", "observed", "estimated"):
                rows.append({"segment": segment, "frames": [start, stop], "held_out_camera": camera,
                    "reference_kind": kind, **partition(select(method, start, stop, camera, kind),
                                                        select(baseline, start, stop, camera, kind))})
    return rows


def table_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report["strata"]:
        for cohort, sides in (("common", ("method", "baseline")), ("method_only", ("method",)), ("baseline_only", ("baseline",))):
            for side in sides:
                metric = row[cohort][side]
                rows.append({"condition": report["condition"], "segment": row["segment"],
                    "held_out_camera": row["held_out_camera"], "reference_kind": row["reference_kind"],
                    "cohort": cohort, "side": side, "count": metric["count"], "median_px": metric["median"],
                    "p90_px": metric["p90"], "above20_count": metric["above_threshold_count"],
                    "above20_rate": metric["above_threshold_rate"]})
    return rows


def write_tables(output: Path, report: Mapping[str, Any]) -> None:
    rows = table_rows(report)
    with (output / "cohorts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# LOVO coverage cohorts", "", "Empty cohorts have undefined error metrics (—). Threshold: strictly >20 px.", "",
             "| segment | held-out camera | reference | cohort | side | n | median px | p90 px | >20 px |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        median = "—" if row["median_px"] is None else f"{row['median_px']:.3f}"
        p90 = "—" if row["p90_px"] is None else f"{row['p90_px']:.3f}"
        rate = "—" if row["above20_rate"] is None else f"{row['above20_rate']:.2%}"
        lines.append(f"| {row['segment']} | {row['held_out_camera']} | {row['reference_kind']} | {row['cohort']} | {row['side']} | {row['count']} | {median} | {p90} | {rate} |")
    (output / "cohorts.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "comparison", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if subprocess.check_output(["git", "status", "--porcelain", "--", str(Path(__file__).relative_to(Path.cwd()))], text=True):
        raise ValueError("Commit the analysis script before recording evidence")
    if args.output.exists() or any(args.output.resolve().is_relative_to(p.resolve())
                                 for p in (args.bundle, args.annotations.parent, args.comparison)):
        raise ValueError("Use a new output directory outside immutable inputs")
    inventory_path = args.comparison / "artifacts.json"
    if dual_sha256(inventory_path) != COMPARISON_SHA:
        raise ValueError("Pinned comparison inventory mismatch")
    inventory = json.loads(inventory_path.read_text())
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=ANNOTATIONS_SHA)
    inputs = bundle.inputs
    if inputs.num_frames != 1010 or inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("This report is pinned to clip_000")
    source_files: dict[str, str] = {str(inventory_path): COMPARISON_SHA}

    def check_source(path: Path) -> str:
        entry = inventory["files"][str(path.relative_to(args.comparison))]
        digest: str = dual_sha256(path)
        if digest != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Original comparison artifact changed: {path}")
        source_files[str(path)] = digest
        return digest

    indices: dict[str, Scores] = {}
    run_provenance = []
    for name in (BASELINE, METHOD):
        scores: Scores = {}
        for camera in inputs.camera_ids:
            directory = args.comparison / "results" / name / f"without_{camera}"
            for path in sorted(directory.iterdir()):
                check_source(path)
            run = load_run(directory, expected_sha256=source_files[str(directory / "result.json")])
            if run.method != name or run.held_out_camera != camera:
                raise ValueError("Saved run method/fold identity mismatch")
            samples = error_samples(run, bundle, annotations)
            if not np.all(samples["camera_index"] == inputs.camera_ids.index(camera)):
                raise ValueError("LOVO scorer returned a different held-out camera")
            fold = index_samples(samples, inputs.camera_ids, inputs.num_frames)
            if scores.keys() & fold.keys():
                raise ValueError("Repeated camera/frame across LOVO folds")
            scores.update(fold)
            run_provenance.append({"method": name, "fold": run.key, "code_commit": run.code_commit,
                "config": run.config, "result_json_sha256": source_files[str(directory / "result.json")]})
        indices[name] = scores
    rows = summarize(indices[METHOD], indices[BASELINE], inputs.camera_ids)
    # Independent consistency check against the original two-method report.
    original_scores = args.comparison / "paired-scores.json"
    check_source(original_scores)
    original = json.loads(original_scores.read_text())
    checked = []
    for row in rows:
        if row["held_out_camera"] != "all" or row["reference_kind"] != "all":
            continue
        for side, name in (("method", METHOD), ("baseline", BASELINE)):
            for scope, metric in (("lovo_reprojection_px", row["totals"][side]),
                                  ("common_lovo_reprojection_px", row["common"][side])):
                before = original["methods"][name][scope][row["segment"]]
                for metric_name in ("count", "median", "p90", "above_threshold_rate"):
                    if metric[metric_name] != before[metric_name]:
                        raise ValueError(f"Original metric mismatch: {name}/{scope}/{row['segment']}/{metric_name}")
        checked.append(row["segment"])
    report = {"schema": "tennis_lab_ball3d_topk_coverage_cohorts.v1", "condition": METHOD,
        "baseline": BASELINE, "key": ["held_out_camera", "frame_index"], "strata": rows,
        "provenance": {"bundle_sha256": BUNDLE_SHA, "annotations_sha256": ANNOTATIONS_SHA,
            "comparison_inventory_sha256": COMPARISON_SHA, "comparison": str(args.comparison.resolve()),
            "analysis_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "analysis_script_sha256": dual_sha256(Path(__file__)), "command": sys.argv,
            "run_provenance": run_provenance, "original_metric_segments_exactly_matched": checked,
            "all_partition_count_checks_passed": all(all(row["checks"].values()) for row in rows),
            "scorer": "src.tennis_scene.evaluation.ball3d.scoring.error_samples",
            "statistics": "src.tennis_scene.evaluation.ball3d.scoring.distribution",
            "method_reexecuted": False, "src_changed": False, "gpu_used": False}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "cohorts.json", report)
    write_tables(args.output, report)
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "segment", "held_out_camera", "frame_index", "reference_kind", "cohort", "method_error_px", "baseline_error_px"])
        method, baseline = indices[METHOD], indices[BASELINE]
        for key in sorted(method.keys() | baseline.keys()):
            m, b = method.get(key), baseline.get(key)
            cohort = "common" if m is not None and b is not None else "method_only" if m is not None else "baseline_only"
            writer.writerow([METHOD, "tuning" if key[1] < 505 else "evaluation", *key,
                method[key][1] if key in method else baseline[key][1], cohort,
                m[0] if m is not None else "", b[0] if b is not None else ""])
    for path, digest in source_files.items():
        if dual_sha256(Path(path)) != digest:
            raise ValueError(f"Original input changed during analysis: {path}")
    manifest = {"schema": "tennis_lab_ball3d_cohort_inventory.v1", "source_files_sha256": source_files,
        "all_read_comparison_files_unchanged": True, "bundle_sha256": BUNDLE_SHA, "annotations_sha256": ANNOTATIONS_SHA,
        "elapsed_seconds": time.perf_counter() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "files": {p.name: {"sha256": dual_sha256(p), "size_bytes": p.stat().st_size} for p in sorted(args.output.iterdir())}}
    write_json_atomic(args.output / "artifacts.json", manifest)
    print(json.dumps({"output": str(args.output.resolve()), "artifacts_json_sha256": dual_sha256(args.output / "artifacts.json"),
        "primary": next(row for row in rows if row["segment"] == "evaluation" and row["held_out_camera"] == row["reference_kind"] == "all"),
        "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
