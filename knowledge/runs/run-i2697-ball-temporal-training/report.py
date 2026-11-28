"""Recompute the recorded metrics and generate the paper figure on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from data import (
    ANCHOR_COUNTS,
    ANCHOR_SHA,
    ANCHORS,
    CONDITIONS,
    IDENTITY_FIELDS,
    LABEL_KINDS,
    PREVIOUS,
    SOURCE_LABELS,
    VAL_COUNTS,
    require_sha,
)
from src.tasks.ball_detection.checkpoints.temporal import json_bytes, save_json
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import (
    calibrate,
    frame_metrics,
)
from src.tasks.ball_detection.evaluation.temporal_metrics import prediction_identities
from src.utils.checksum import dual_sha256

FIXED = dict(mode="threshold", threshold=0.5)
DISTANCE = 20.0
FIELDS = (
    "dataset",
    "experiment",
    "condition",
    "frames",
    "source",
    "operating_point",
    "threshold_mode",
    "threshold",
    "scored_frames",
    "observed",
    "reviewed_empty",
    "out_of_frame",
    "mean_error_px",
    "median_error_px",
    "p95_error_px",
    "within_20px_rate",
    "fpr",
    "fnr",
    "f1",
    "fp",
    "fn",
    "tp",
    "tn",
)


def teacher(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record[k]
        for k in (
            "clip_id",
            "frame_index",
            "native_fps",
            "source",
            "kind",
            "source_size",
            "teacher_xy",
        )
    )


def read_predictions(path: Path, digest: str) -> list[dict[str, Any]]:
    require_sha(path, digest)
    records = []
    with path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            u, v = row["pred_uv"]
            width, height = row["source_size"]
            if (
                row["kind"] not in LABEL_KINDS
                or row["source"] not in SOURCE_LABELS
                or not all(0 <= x <= 1 for x in (u, v, row["score"]))
                or min(width, height) < 2
            ):
                raise ValueError(f"Unexpected prediction value or geometry: {path}")
            xy = row["teacher_xy"]
            if row["kind"] == "observed":
                if (
                    xy is None
                    or len(xy) != 2
                    or not (0 <= xy[0] <= width - 1 and 0 <= xy[1] <= height - 1)
                ):
                    raise ValueError(
                        "Observed coordinates must be in the original image grid"
                    )
                row["error_px"] = math.dist((u * (width - 1), v * (height - 1)), xy)
            else:
                if xy is not None:
                    raise ValueError(
                        "A non-observed frame cannot carry a position target"
                    )
                row["error_px"] = None
            records.append(row)
    return records


def validate_predictions(
    records: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    audit: dict[str, Any],
    anchor_plan: list[dict[str, Any]],
) -> dict[str, str]:
    support = Counter(r["kind"] for r in records)
    if support != VAL_COUNTS or Counter(r["kind"] for r in anchors) != ANCHOR_COUNTS:
        raise ValueError(
            "Predictions did not retain the fixed validation and anchor masks"
        )
    keys = [(r["clip_id"], r["frame_index"]) for r in records]
    if keys != sorted(set(keys)):
        raise ValueError(
            "Validation predictions must be ordered and have unique owner frames"
        )
    hashes = prediction_identities(records)
    if any(hashes[k] != audit["validation_identity"][k] for k in IDENTITY_FIELDS):
        raise ValueError(
            "Validation owner/teacher/scored identity differs from the frozen manifest"
        )
    for source in SOURCE_LABELS:
        counts = Counter(r["kind"] for r in records if r["source"] == source)
        if counts != Counter(audit["validation_by_source"][source]):
            raise ValueError(f"Validation source support changed: {source}")
    if [teacher(a) for a in anchors] != [teacher(a) for a in anchor_plan]:
        raise ValueError(
            "Prediction anchors do not match the #2637 plan and target pixels"
        )
    by_key = dict(zip(keys, records, strict=True))
    if any(
        teacher(by_key[a["clip_id"], a["frame_index"]]) != teacher(a) for a in anchors
    ):
        raise ValueError("Anchor and owner predictions disagree on the teacher")

    def digest(rows: list[dict[str, Any]]) -> str:
        sha = hashlib.sha256()
        for row in rows:
            sha.update(json_bytes(teacher(row)))
        return sha.hexdigest()

    return dict(
        **hashes,
        validation_targets_sha256=digest(records),
        anchor_targets_sha256=digest(anchors),
    )


def presence(fp: int, fn: int, observed: int, negatives: int) -> dict[str, Any]:
    tp, tn = observed - fn, negatives - fp
    if min(fp, fn, tp, tn) < 0:
        raise ValueError("Confusion matrix lies outside the known labels")
    f1_denominator = 2 * tp + fp + fn
    return dict(
        fp=fp,
        fn=fn,
        tp=tp,
        tn=tn,
        fpr=fp / negatives if negatives else None,
        fnr=fn / observed if observed else None,
        f1=2 * tp / f1_denominator if f1_denominator else None,
    )


def summarize(rows: list[dict[str, Any]], threshold: dict[str, Any]) -> dict[str, Any]:
    metrics = frame_metrics(rows, threshold, distance_px=DISTANCE)
    support = metrics["support"]
    return dict(
        **support,
        **{k: metrics[k] for k in ("mean_error_px", "median_error_px", "p95_error_px")},
        within_20px_rate=metrics["within_distance_rate"],
        **presence(
            metrics["false_positive"],
            metrics["false_negative"],
            support["observed"],
            support["reviewed_empty"] + support["out_of_frame"],
        ),
        negative_kinds=metrics["by_negative_kind"],
    )


def history() -> list[dict[str, Any]]:
    table = []
    with PREVIOUS.open(newline="") as stream:
        for old in csv.DictReader(stream):
            fp, fn, obs, empty, outside = (
                int(old[k])
                for k in (
                    "false_positive",
                    "false_negative",
                    "observed",
                    "reviewed_empty",
                    "out_of_frame",
                )
            )
            row = dict(
                dataset="anchors",
                experiment="#2637",
                condition=f"{old['model']}/{old['condition']}",
                frames={"x3d_t32": 32, "i935_t8": 8, "shape_baseline": None}[
                    old["model"]
                ],
                source=old["source"],
                operating_point="original",
                threshold_mode="threshold",
                threshold={
                    "x3d_t32": 0.5380124449729919,
                    "i935_t8": 0.5,
                    "shape_baseline": 0.9966928958892822,
                }[old["model"]],
                scored_frames=int(old["frames"]),
                observed=obs,
                reviewed_empty=empty,
                out_of_frame=outside,
                **{
                    k: float(old[k]) if old[k] else None
                    for k in ("mean_error_px", "median_error_px", "p95_error_px")
                },
                within_20px_rate=float(old["within_distance_rate"])
                if old["within_distance_rate"]
                else None,
                **presence(fp, fn, obs, empty + outside),
            )
            table.append(row)
    return table


def collect(training: Path, output: Path) -> None:
    finish = json.loads((training / "completed.json").read_bytes())
    plan_file = training / "plan.json"
    require_sha(plan_file, finish["plan_sha256"])
    plan = json.loads(plan_file.read_bytes())
    require_sha(ANCHORS, ANCHOR_SHA)
    anchor_plan = [
        a
        for a in json.loads(ANCHORS.read_bytes())["anchors"]
        if a["split"] == "evaluation"
    ]
    metrics: dict[str, Any] = {}
    rows = []
    identity = None
    if set(finish["conditions"]) != set(CONDITIONS):
        raise ValueError("Expected three completed conditions")
    for name, frames in CONDITIONS.items():
        paths = training / name
        evidence = finish["conditions"][name]
        subsets = {
            subset: read_predictions(
                paths / f"{subset}.jsonl", evidence[subset]["sha256"]
            )
            for subset in ("validation", "anchors")
        }
        found = validate_predictions(
            subsets["validation"], subsets["anchors"], plan["inputs"], anchor_plan
        )
        if identity is None:
            identity = found
        elif identity != found:
            raise ValueError(
                "Training conditions did not use the same target coordinates and masks"
            )
        calibration = calibrate(subsets["validation"], split="val")
        thresholds = dict(balanced=calibration, fixed_0_5=FIXED)
        metrics[name] = dict(calibration=calibration, results={})
        for subset, records in subsets.items():
            groups = {"all": records}
            groups.update(
                {
                    source: [r for r in records if r["source"] == source]
                    for source in SOURCE_LABELS
                }
            )
            metrics[name]["results"][subset] = {}
            for operating_point, threshold in thresholds.items():
                rates = {
                    source: summarize(group, threshold)
                    for source, group in groups.items()
                }
                metrics[name]["results"][subset][operating_point] = rates
                rows.extend(
                    dict(
                        values,
                        dataset=subset,
                        experiment="#2697",
                        condition=name,
                        frames=frames,
                        source=source,
                        operating_point=operating_point,
                        threshold_mode=threshold["mode"],
                        threshold=threshold["threshold"],
                    )
                    for source, values in rates.items()
                )
    rows.extend(history())
    report = dict(
        schema="ball_temporal_training_ablation_metrics.v1",
        plan=plan,
        execution=finish,
        identity=identity,
        conditions=metrics,
        distance_threshold_source_px=DISTANCE,
        historical_table_sha256=dual_sha256(PREVIOUS),
    )
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", report)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(table: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = "Noto Sans CJK JP"
    font_manager.findfont(font, fallback_to_default=False)
    matplotlib.rcParams.update(
        {
            "font.family": font,
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 3,  # Embed the CFF Japanese glyphs as vector outlines.
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    with table.open(newline="") as stream:
        rows = {
            (r["condition"], r["source"]): r
            for r in csv.DictReader(stream)
            if (r["dataset"], r["experiment"], r["operating_point"])
            == ("validation", "#2697", "balanced")
        }

    names = ["T=1 RGB", "T=8 RGB+MDD", "T=32 RGB+MDD"]
    colors = ["#6858a3", "#237dac", "#186858"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    ax_hit, ax_px, ax_presence, ax_source = axes.ravel()

    def value(condition: str, metric: str, source: str = "all") -> float:
        return float(rows[condition, source][metric])

    def grouped(
        ax: Any, labels: list[str], vectors: list[list[float]], *, percent: bool
    ) -> None:
        for i, (name, vector, color) in enumerate(
            zip(names, vectors, colors, strict=True)
        ):
            bars = ax.bar(
                [j + 0.25 * (i - 1) for j in range(len(labels))],
                vector,
                0.24,
                label=name,
                color=color,
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel("割合（%）" if percent else "元画像の距離（px）")
        ax.set_ylim(
            0, 113 if percent else 1.18 * max(x for vector in vectors for x in vector)
        )
        ax.grid(axis="y", color="#dde2e7", alpha=0.8)
        ax.set_axisbelow(True)

    first = [(100 * value(condition, "within_20px_rate")) for condition in CONDITIONS]
    bars = ax_hit.bar(names, first, color=colors, width=0.58)
    ax_hit.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax_hit.set_ylim(0, 113)
    ax_hit.set_ylabel("割合（%）")
    ax_hit.grid(axis="y", color="#dde2e7")
    ax_hit.set_axisbelow(True)
    ax_hit.set_title("位置誤差が20 px以内の割合")
    grouped(
        ax_px,
        ["平均", "中央値"],
        [
            [value(c, k) for k in ("mean_error_px", "median_error_px")]
            for c in CONDITIONS
        ],
        percent=False,
    )
    ax_px.set_title("位置誤差の平均と中央値")
    grouped(
        ax_presence,
        ["F1", "誤検出率", "見逃し率"],
        [[100 * value(c, k) for k in ("f1", "fpr", "fnr")] for c in CONDITIONS],
        percent=True,
    )
    ax_presence.set_title("存在判定（validationの最適バランス閾値）")
    grouped(
        ax_source,
        list(SOURCE_LABELS.values()),
        [
            [100 * value(c, "within_20px_rate", source) for source in SOURCE_LABELS]
            for c in CONDITIONS
        ],
        percent=True,
    )
    ax_source.set_title("収録元別の20 px以内率")
    ax_source.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("ボール検出の入力フレーム数による比較", fontsize=16, weight="bold")
    output.mkdir(parents=True, exist_ok=True)
    figure = output / "temporal_training_ablation"
    fig.savefig(
        figure.with_suffix(".png"), dpi=180, metadata={"Software": "Matplotlib"}
    )
    fig.savefig(
        figure.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    collect_args = sub.add_parser("collect")
    collect_args.add_argument("--training", required=True, type=Path)
    collect_args.add_argument("--output", required=True, type=Path)
    plot_args = sub.add_parser("plot")
    plot_args.add_argument("--table", required=True, type=Path)
    plot_args.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.training, args.output)
    else:
        plot(args.table, args.output)


if __name__ == "__main__":
    main()
