"""Partition saved LOVO scoring samples against the original baseline; no inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import (
    Ball2DAnnotations,
    load_annotations,
)
from src.tennis_scene.evaluation.ball3d.bundle import FixedInputBundle, load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import distribution, error_samples

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"
SOURCE_MANIFEST_SHA = "3b53a53ff5ca34bfdb6c4c764039bf1e3fe8eb3a4ea20b2a363f2fd6d97c9d6c"
SOURCE_SCORES_SHA = "630563ee095e30bf2d13f37a9c0659aebc46e83f1ecf8b0d613246eea4edc306"
BASELINE = "pipeline_baseline"
METHODS = (BASELINE, "without_loess", "dense_selected", "dense_temporal", "dense_quadratic_logit",
           "dense_quadratic_log", "dense_coarse", "annotation_oracle")
SEGMENTS = (("evaluation", 505, 1010), ("tuning", 0, 505), ("all", 0, 1010))
KINDS = ("observed", "estimated")
THRESHOLD_PX = 20.0


class Sample(NamedTuple):
    error_px: float
    reference_kind: str


SampleIndex = dict[tuple[str, int], Sample]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def merge_samples(target: SampleIndex, samples: dict[str, NDArray[Any]], held_out: str, camera_ids: tuple[str, ...]) -> None:
    expected_index = camera_ids.index(held_out)
    for camera, frame, error, kind in zip(samples["camera_index"], samples["frame_index"],
                                          samples["error_px"], samples["reference_kind"], strict=True):
        if int(camera) != expected_index:
            raise ValueError("LOVO sample camera differs from the held-out camera")
        if not 0 <= int(frame) < 1010 or not np.isfinite(error) or error < 0 or str(kind) not in KINDS:
            raise ValueError("Invalid scored sample identity/error/reference kind")
        key = (held_out, int(frame))
        if key in target:
            raise ValueError(f"Duplicate camera/frame sample: {key}")
        target[key] = Sample(float(error), str(kind))


def stats(points: SampleIndex, keys: set[tuple[str, int]]) -> dict[str, Any]:
    values = np.array([points[key].error_px for key in sorted(keys)], np.float64)
    return {**distribution(values, threshold=THRESHOLD_PX),
            "above_threshold_count": int(np.count_nonzero(values > THRESHOLD_PX))}


def partition(baseline: SampleIndex, method: SampleIndex, start: int, stop: int,
              camera: str = "all", reference_kind: str = "all") -> dict[str, Any]:
    for key in baseline.keys() & method.keys():
        if baseline[key].reference_kind != method[key].reference_kind:
            raise ValueError(f"Reference kind mismatch at {key}")

    def chosen(points: SampleIndex) -> set[tuple[str, int]]:
        return {key for key, point in points.items() if start <= key[1] < stop
                and (camera == "all" or key[0] == camera)
                and (reference_kind == "all" or point.reference_kind == reference_kind)}

    baseline_keys, method_keys = chosen(baseline), chosen(method)
    common, added, lost = baseline_keys & method_keys, method_keys - baseline_keys, baseline_keys - method_keys
    if common & added or common & lost or added & lost:
        raise ValueError("Cohorts must be disjoint")
    if len(common) + len(added) != len(method_keys) or len(common) + len(lost) != len(baseline_keys):
        raise ValueError("Cohorts do not reconstruct the full scoring counts")
    return {"counts": {"common": len(common), "method_only": len(added), "baseline_only": len(lost),
                       "method_all": len(method_keys), "baseline_all": len(baseline_keys)},
            "common": {"method": stats(method, common), "baseline": stats(baseline, common)},
            "method_only": {"method": stats(method, added)}, "baseline_only": {"baseline": stats(baseline, lost)},
            "method_all": stats(method, method_keys), "baseline_all": stats(baseline, baseline_keys),
            "counts_verified": True}


def all_rows(baseline: SampleIndex, method: SampleIndex, camera_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = [{"segment": segment, "start_frame": start, "stop_frame": stop,
             "held_out_camera": camera, "reference_kind": kind,
             **partition(baseline, method, start, stop, camera, kind)}
            for segment, start, stop in SEGMENTS for camera in ("all", *camera_ids) for kind in ("all", *KINDS)]
    lookup = {(row["segment"], row["held_out_camera"], row["reference_kind"]): row for row in rows}
    for row in rows:
        segment, camera, kind = row["segment"], row["held_out_camera"], row["reference_kind"]
        for cohort, count in row["counts"].items():
            if camera == "all" and sum(lookup[segment, c, kind]["counts"][cohort] for c in camera_ids) != count:
                raise ValueError("Camera subtotals do not reconstruct pooled counts")
            if kind == "all" and sum(lookup[segment, camera, k]["counts"][cohort] for k in KINDS) != count:
                raise ValueError("Reference-kind subtotals do not reconstruct pooled counts")
            if segment == "all" and sum(lookup[s, camera, kind]["counts"][cohort] for s in ("tuning", "evaluation")) != count:
                raise ValueError("Time splits do not reconstruct full-interval counts")
    return rows


def load_lovo(source: Path, method: str, bundle: FixedInputBundle, annotations: Ball2DAnnotations,
              manifest: dict[str, Any]) -> tuple[SampleIndex, list[dict[str, Any]]]:
    points: SampleIndex = {}
    provenance = []
    for camera in bundle.inputs.camera_ids:
        # The original report persisted its scoring-only oracle separately.
        root = source / "scoring_oracle" if method == "annotation_oracle" else source / "results" / method
        directory = root / f"without_{camera}"
        relative = str((directory / "result.json").relative_to(source))
        expected = manifest["files"][relative]["sha256"]
        run = load_run(directory, expected_sha256=expected)
        if run.method != method or run.held_out_camera != camera:
            raise ValueError("Saved run method/fold identity mismatch")
        merge_samples(points, error_samples(run, bundle, annotations), camera, bundle.inputs.camera_ids)
        provenance.append({"path": str(directory), "result_sha256": expected, "held_out_camera": camera,
                           "prediction_code_commit": run.code_commit, "saved_prediction_runtime_seconds": run.elapsed_seconds})
    return points, provenance


def validate_original_scores(rows: list[dict[str, Any]], method: str, scores: dict[str, Any]) -> None:
    for row in rows:
        if row["held_out_camera"] != "all" or row["reference_kind"] != "all":
            continue
        for name, owner in ((method, "method_all"), (BASELINE, "baseline_all")):
            original = scores["methods"][name]["lovo_reprojection_px"][row["segment"]]
            for key in ("count", "median", "p90", "max", "above_threshold_rate", "threshold"):
                if row[owner][key] != original[key]:
                    raise ValueError(f"Saved score mismatch: {name}/{row['segment']}/{key}")


def primary_markdown(methods: dict[str, list[dict[str, Any]]]) -> str:
    def values(value: dict[str, Any]) -> str:
        if not value["count"]:
            return "0; —; —; —"
        return f"{value['count']}; {value['median']:.3f}; {value['p90']:.3f}; {100 * value['above_threshold_rate']:.2f}%"

    lines = ["# LOVO共通・追加・喪失の誤差分布", "", "主評価 [505,1010)。各cellは 件数; 中央値px; p90px; >20px率。",
             "共通の両列は同じcamera/frame集合。0件の誤差と率は未測定。", "",
             "| 条件 | 共通:方式 | 共通:baseline | 追加:方式のみ | 喪失:baselineのみ |",
             "|---|---:|---:|---:|---:|"]
    for name, rows in methods.items():
        row = next(r for r in rows if r["segment"] == "evaluation" and r["held_out_camera"] == "all" and r["reference_kind"] == "all")
        lines.append(f"| {name} | {values(row['common']['method'])} | {values(row['common']['baseline'])} | "
                     f"{values(row['method_only']['method'])} | {values(row['baseline_only']['baseline'])} |")
    return "\n".join(lines) + "\n"


def write_csv(output: Path, methods: dict[str, list[dict[str, Any]]]) -> None:
    with (output / "cohorts.csv").open("w", newline="") as stream:
        columns = ("method", "segment", "start_frame", "stop_frame", "held_out_camera", "reference_kind", "cohort", "error_owner",
                   "count", "median", "p90", "above_threshold_rate", "above_threshold_count")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, rows in methods.items():
            for row in rows:
                for cohort in ("common", "method_only", "baseline_only"):
                    for owner, summary in row[cohort].items():
                        writer.writerow({"method": name, **{key: row[key] for key in columns[1:6]}, "cohort": cohort,
                                         "error_owner": owner, **{key: summary[key] for key in columns[8:]}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (source, args.bundle, args.annotations.parent)):
        raise ValueError("Use a new output directory outside all frozen inputs")
    started = time.perf_counter()
    if digest(source / "artifacts.json") != SOURCE_MANIFEST_SHA or digest(source / "scores.json") != SOURCE_SCORES_SHA:
        raise ValueError("Original comparison manifest/scores SHA-256 mismatch")
    manifest, scores = (json.loads((source / name).read_text()) for name in ("artifacts.json", "scores.json"))
    if set(scores["methods"]) != set(METHODS):
        raise ValueError("Conditions differ from those reported in PR #1676")
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=REFERENCE_SHA)
    if bundle.inputs.num_frames != 1010:
        raise ValueError("Expected the pinned 1010-frame clip")
    baseline, baseline_runs = load_lovo(source, BASELINE, bundle, annotations, manifest)
    methods, provenance = {}, {BASELINE: baseline_runs}
    raw: dict[str, SampleIndex] = {}
    for name in METHODS:
        if name == BASELINE:
            samples = baseline
        else:
            samples, provenance[name] = load_lovo(source, name, bundle, annotations, manifest)
        rows = all_rows(baseline, samples, bundle.inputs.camera_ids)
        validate_original_scores(rows, name, scores)
        methods[name], raw[name] = rows, samples
    if digest(source / "artifacts.json") != SOURCE_MANIFEST_SHA or digest(source / "scores.json") != SOURCE_SCORES_SHA:
        raise ValueError("Frozen inputs changed during cohort analysis")
    checks = {"all_row_counts_verified": True, "camera_kind_time_subtotals_verified": True,
              "original_all_point_scores_exactly_matched": True, "predictions_rerun": False, "retuning": False,
              "conditions": len(methods), "lovo_runs_loaded": sum(len(v) for v in provenance.values()),
              "partition_rows": sum(len(v) for v in methods.values())}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("method", "held_out_camera", "frame", "reference_kind", "cohort", "method_error_px", "baseline_error_px"))
        for name, samples in raw.items():
            for key in sorted(baseline.keys() | samples.keys()):
                method_sample, baseline_sample = samples.get(key), baseline.get(key)
                cohort = "common" if method_sample is not None and baseline_sample is not None else "method_only" if method_sample is not None else "baseline_only"
                sample = method_sample if method_sample is not None else baseline_sample
                assert sample is not None
                writer.writerow((name, *key, sample.reference_kind, cohort,
                                 method_sample.error_px if method_sample is not None else "", baseline_sample.error_px if baseline_sample is not None else ""))
    write_csv(output, methods)
    (output / "primary.md").write_text(primary_markdown(methods))
    report = {"schema": "tennis_lab_dense_likelihood_lovo_cohorts.v1", "threshold_px": THRESHOLD_PX,
              "bundle": {"path": str(args.bundle.resolve()), "sha256": BUNDLE_SHA},
              "annotations": {"path": str(args.annotations.resolve()), "sha256": REFERENCE_SHA},
              "source": {"path": str(source), "manifest_sha256": SOURCE_MANIFEST_SHA, "scores_sha256": SOURCE_SCORES_SHA},
              "analysis_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "analysis_script_sha256": digest(Path(__file__)), "command": [sys.executable, *sys.argv],
              "scorer_sha256": digest(Path(error_samples.__code__.co_filename)),
              "runs": provenance, "checks": checks, "methods": methods,
              "analysis_seconds": time.perf_counter() - started}
    write_json(output / "cohorts.json", report)
    write_json(output / "checksums.json", {"schema": "tennis_lab_dense_cohort_checksums.v1", "files": {
        name: {"sha256": digest(output / name), "size_bytes": (output / name).stat().st_size}
        for name in ("cohorts.json", "cohorts.csv", "samples.csv", "primary.md")}})
    print(json.dumps({"output": str(output), "checksums_json_sha256": digest(output / "checksums.json"), **checks}, indent=2))


if __name__ == "__main__":
    main()
