"""cross-1: partition saved LOVO scores without rerunning a prediction method."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.evaluation.ball3d.annotations import (
    Ball2DAnnotations,
    load_annotations,
)
from src.tennis_scene.evaluation.ball3d.bundle import FixedInputBundle, load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    DEFAULT_SCORING,
    distribution,
    error_samples,
)
from src.tennis_scene.evaluation.ball3d.storage import read_json
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"
COMPARISON_SHA = "46ecf147cb52139fb6cdf81a5da82f7fb0d94403e4d63c19e2abdfad179136b5"
BASELINE = "pipeline_baseline"
METHODS = (
    "candidate_gmm_map_none",
    "candidate_gmm_mean_none",
    "candidate_gmm_map_rts",
    "candidate_gmm_mean_rts",
)
LABELS = dict(zip(METHODS, ("MAP", "mean", "MAP-RTS", "mean-RTS"), strict=True))
SEGMENTS = {"evaluation": (505, 1010), "tuning": (0, 505), "all": (0, 1010)}
KINDS = ("all", "observed", "estimated")
THRESHOLD = DEFAULT_SCORING.gross_reprojection_px
Key = tuple[str, int]
Samples = dict[Key, tuple[float, str]]


def index_samples(
    samples: Mapping[str, NDArray[Any]],
    camera_ids: tuple[str, ...],
    held_out: str,
) -> Samples:
    """Retain scorer values and their identities; duplicates must never be pooled."""
    result: Samples = {}
    for camera, frame, error, kind in zip(
        samples["camera_index"],
        samples["frame_index"],
        samples["error_px"],
        samples["reference_kind"],
        strict=True,
    ):
        if not 0 <= camera < len(camera_ids) or camera_ids[camera] != held_out:
            raise ValueError("LOVO sample is not from its held-out camera")
        if kind not in KINDS[1:] or not np.isfinite(error) or error < 0:
            raise ValueError("Invalid scored error/reference kind")
        key = (held_out, int(frame))
        if key in result:
            raise ValueError(f"Duplicate LOVO camera/frame: {key}")
        result[key] = (float(error), str(kind))
    return result


def partitions(method: Samples, baseline: Samples) -> dict[str, set[Key]]:
    common = method.keys() & baseline.keys()
    if any(method[key][1] != baseline[key][1] for key in common):
        raise ValueError("Common camera/frame has inconsistent reference kind")
    return {
        "common": common,
        "method_only": method.keys() - baseline.keys(),
        "baseline_only": baseline.keys() - method.keys(),
    }


def cohort_metrics(
    method: Samples,
    baseline: Samples,
    *,
    start: int,
    stop: int,
    camera: str = "all",
    kind: str = "all",
) -> dict[str, Any]:
    def select(samples: Samples) -> Samples:
        return {
            key: value
            for key, value in samples.items()
            if start <= key[1] < stop
            and (camera == "all" or key[0] == camera)
            and (kind == "all" or value[1] == kind)
        }

    method, baseline = select(method), select(baseline)
    groups = partitions(method, baseline)

    def summarize(samples: Samples, keys: set[Key]) -> dict[str, Any]:
        values: NDArray[np.float64] = np.asarray(
            [samples[key][0] for key in sorted(keys)], np.float64
        )
        return {
            **distribution(values, threshold=THRESHOLD),
            "above_threshold_count": int(np.count_nonzero(values > THRESHOLD)),
        }

    metrics = {
        "common_method": summarize(method, groups["common"]),
        "common_baseline": summarize(baseline, groups["common"]),
        "method_only": summarize(method, groups["method_only"]),
        "baseline_only": summarize(baseline, groups["baseline_only"]),
        "method_all": summarize(method, set(method)),
        "baseline_all": summarize(baseline, set(baseline)),
    }
    for owner, only in (("method", "method_only"), ("baseline", "baseline_only")):
        if metrics[f"common_{owner}"]["count"] + metrics[only]["count"] != len(
            method if owner == "method" else baseline
        ):
            raise ValueError(f"Cohort count partition mismatch: {owner}")
    return metrics


def load_lovo(
    directory: Path,
    name: str,
    inventory: dict[str, Any],
    bundle: FixedInputBundle,
    annotations: Ball2DAnnotations,
) -> tuple[Samples, list[dict[str, Any]]]:
    samples: Samples = {}
    receipts = []
    for camera in bundle.inputs.camera_ids:
        relative = Path("results") / name / f"without_{camera}"
        digest = inventory["files"][str(relative / "result.json")]["sha256"]
        run = load_run(directory / relative, expected_sha256=digest)
        if run.method != name or run.held_out_camera != camera:
            raise ValueError("Saved method/fold identity mismatch")
        indexed = index_samples(
            error_samples(run, bundle, annotations), bundle.inputs.camera_ids, camera
        )
        if samples.keys() & indexed.keys():
            raise ValueError("Duplicate camera/frame across LOVO folds")
        samples.update(indexed)
        receipts.append(
            {
                "path": str(relative),
                "manifest_sha256": digest,
                "prediction_code_commit": run.code_commit,
            }
        )
    if any(not 0 <= frame < bundle.inputs.num_frames for _, frame in samples):
        raise ValueError("LOVO sample outside source timeline")
    return samples, receipts


def audit_rows(rows: list[dict[str, Any]], scores: dict[str, Any]) -> int:
    """Compare independently saved v2 totals, then audit partition additivity."""
    checks = 0
    for row in rows:
        for owner, name in (("method", row["method"]), ("baseline", BASELINE)):
            previous = scores["methods"][name]
            expected = (
                previous["lovo_reprojection_by_reference_kind"][row["segment"]]
                if row["camera"] == "all"
                else previous["folds"][f"without_{row['camera']}"][row["segment"]][
                    "reprojection_by_reference_kind"
                ]
            )[row["reference_kind"]]
            actual = row["metrics"][f"{owner}_all"]
            for key in ("count", "median", "p90", "max", "above_threshold_rate"):
                if actual[key] != expected[key]:
                    raise ValueError(
                        f"Saved v2 scoring mismatch: {row['method']} {key}"
                    )
                checks += 1
        for category in row["metrics"]:
            for axis, total, parts in (
                ("camera", "all", ("cam0", "cam1", "cam2")),
                ("reference_kind", "all", ("observed", "estimated")),
                ("segment", "all", ("tuning", "evaluation")),
            ):
                if row[axis] != total:
                    continue
                matching = [
                    other
                    for other in rows
                    if all(
                        other[key] == row[key]
                        for key in ("method", "segment", "camera", "reference_kind")
                        if key != axis
                    )
                    and other[axis] in parts
                ]
                if (
                    len(matching) != len(parts)
                    or sum(other["metrics"][category]["count"] for other in matching)
                    != row["metrics"][category]["count"]
                ):
                    raise ValueError(f"Cohort marginal count mismatch: {axis}")
                checks += 1
    return checks


def markdown_table(rows: list[dict[str, Any]], segment: str) -> str:
    lines = [
        "| 条件 | 共通n | 共通・方式 median/p90/>20% | 共通・baseline median/p90/>20% | 追加 n; median/p90/>20% | 喪失 n; baseline median/p90/>20% |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def cell(summary: dict[str, Any], count: bool = False) -> str:
        values = (
            "—"
            if not summary["count"]
            else f"{summary['median']:.3f}/{summary['p90']:.3f}/"
            f"{100 * summary['above_threshold_rate']:.2f}%"
        )
        return f"{summary['count']}; {values}" if count else values

    for row in rows:
        if (row["segment"], row["camera"], row["reference_kind"]) != (
            segment,
            "all",
            "all",
        ):
            continue
        m = row["metrics"]
        lines.append(
            f"| {LABELS[row['method']]} | {m['common_method']['count']} | "
            f"{cell(m['common_method'])} | {cell(m['common_baseline'])} | "
            f"{cell(m['method_only'], True)} | {cell(m['baseline_only'], True)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "comparison", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; existing results are immutable")
    if dual_sha256(args.comparison / "artifacts.json") != COMPARISON_SHA:
        raise ValueError("Pinned v2 comparison inventory checksum mismatch")
    inventory = read_json(args.comparison / "artifacts.json")
    scores_path = args.comparison / "scores.json"
    if dual_sha256(scores_path) != inventory["files"]["scores.json"]["sha256"]:
        raise ValueError("Saved v2 score checksum mismatch")
    scores = read_json(scores_path)
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=REFERENCE_SHA
    )
    if (bundle.inputs.camera_ids, bundle.inputs.num_frames) != (
        ("cam0", "cam1", "cam2"),
        1010,
    ):
        raise ValueError("Only the pinned three-camera 1010-frame clip is supported")
    baseline, baseline_receipts = load_lovo(
        args.comparison, BASELINE, inventory, bundle, annotations
    )
    rows, sample_rows = [], []
    run_receipts = {BASELINE: baseline_receipts}
    for name in METHODS:
        method, run_receipts[name] = load_lovo(
            args.comparison, name, inventory, bundle, annotations
        )
        # Validate shared reference kinds before filtering by kind.
        groups = partitions(method, baseline)
        for cohort, keys in groups.items():
            for key in sorted(keys):
                sample_rows.append(
                    {
                        "method": name,
                        "camera": key[0],
                        "frame": key[1],
                        "reference_kind": (
                            method[key] if key in method else baseline[key]
                        )[1],
                        "cohort": cohort,
                        "method_error_px": method[key][0] if key in method else None,
                        "baseline_error_px": baseline[key][0]
                        if key in baseline
                        else None,
                    }
                )
        for segment, (start, stop) in SEGMENTS.items():
            for camera in ("all", *bundle.inputs.camera_ids):
                for kind in KINDS:
                    rows.append(
                        {
                            "method": name,
                            "segment": segment,
                            "start": start,
                            "stop": stop,
                            "camera": camera,
                            "reference_kind": kind,
                            "metrics": cohort_metrics(
                                method,
                                baseline,
                                start=start,
                                stop=stop,
                                camera=camera,
                                kind=kind,
                            ),
                        }
                    )
    checks = audit_rows(rows, scores)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "bundle": {"path": str(args.bundle.resolve()), "sha256": BUNDLE_SHA},
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": REFERENCE_SHA,
        },
        "comparison": {
            "path": str(args.comparison.resolve()),
            "manifest_sha256": COMPARISON_SHA,
        },
        "runs": run_receipts,
        "analysis_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": dual_sha256(Path(__file__)),
        "prediction_methods_rerun": False,
    }
    write_json_atomic(
        args.output / "cohorts.json",
        {
            "schema_version": 1,
            "provenance": provenance,
            "key": ["held_out_camera", "source_frame"],
            "threshold_px_strictly_greater": THRESHOLD,
            "segments_half_open": SEGMENTS,
            "rows": rows,
            "validation": {
                "partition_rows_passed": len(rows),
                "saved_score_and_marginal_checks_passed": checks,
            },
        },
    )
    flat_rows = [
        {
            **{k: v for k, v in row.items() if k != "metrics"},
            "cohort": cohort,
            **summary,
        }
        for row in rows
        for cohort, summary in row["metrics"].items()
    ]
    for filename, values in (("cohorts.csv", flat_rows), ("samples.csv", sample_rows)):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    text = "# 保存済みGMMとbaselineのLOVO cohort\n\n単位はpx。共通集合は各条件対baselineの2方式間。>20は厳密な超過。\n\n"
    text += "\n\n".join(
        f"## {segment} {SEGMENTS[segment]}\n\n{markdown_table(rows, segment)}"
        for segment in SEGMENTS
    )
    text += "\n\ncamera/kind別はcohorts.csv/JSON、各点のidentity・誤差はsamples.csv。空集合の誤差はnull/空欄。\n"
    (args.output / "cohorts.md").write_text(text)
    files = {
        p.name: {"size_bytes": p.stat().st_size, "sha256": dual_sha256(p)}
        for p in sorted(args.output.iterdir())
        if p.is_file()
    }
    write_json_atomic(
        args.output / "artifacts.json", {"provenance": provenance, "files": files}
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifacts_sha256": dual_sha256(args.output / "artifacts.json"),
                "partition_rows_passed": len(rows),
                "checks_passed": checks,
            }
        )
    )


if __name__ == "__main__":
    main()
