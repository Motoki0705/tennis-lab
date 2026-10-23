"""Matched data-scaling tables and log-image-count curves from evaluation receipts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.tasks.court_detection.ablation.evaluation import DENSE_METRICS
from src.tasks.court_detection.scaling.real_split import canonical_sha256
from src.utils.data.image_record_store import file_sha256


def comparison_contract(report: dict[str, Any]) -> dict[str, Any]:
    config = report["config"]
    return {
        "model": config["model"],
        "loss": config["loss"],
        "training": config["training"],
        "initialization": config["run"]["init_weights"],
        "preprocessing": {
            key: config["data"][key]
            for key in ("batch_size", "augmentation", "processing")
        },
        "target_bundle": report["target_bundle"],
        "parameters": report["parameters"],
        "global_step": report["global_step"],
        "precision": report["precision"],
        "registry": report["data_condition"]["evaluation_registry_sha256"],
        "splits": {
            key: {
                name: split[name]
                for name in (
                    "source",
                    "split",
                    "batch_size",
                    "count",
                    "ordered_sample_ids_sha256",
                    "pose_supervised",
                )
            }
            for key, split in report["splits"].items()
        },
    }


def compare_evaluations(inputs: list[Path], output: Path) -> Path:
    if not inputs:
        raise ValueError("At least one evaluation is required.")
    reports = [(path, json.loads(path.read_text())) for path in inputs]
    for _, report in reports:
        if report.get("schema") != "court_data_scaling_evaluation_v1":
            raise ValueError("Expected a data scaling evaluation receipt.")
        if type(report["global_step"]) is not int or report["global_step"] <= 0:
            raise ValueError(
                "Evaluation must identify a positive optimizer update count."
            )
        if report["data_condition"]["accepted_images"] <= 0:
            raise ValueError(
                "Scaling curves require measured positive accepted image counts."
            )
    contract = comparison_contract(reports[0][1])
    contract.pop("splits")
    split_contracts: dict[str, Any] = {}
    rows = []
    seen = set()
    for path, report in reports:
        candidate = comparison_contract(report)
        candidate_splits = candidate.pop("splits")
        if candidate != contract:
            raise ValueError(
                "Evaluation/model/update-count/preprocessing/training contracts differ."
            )
        condition = report["data_condition"]
        identity = (
            report["series"],
            condition["data_version"],
            condition["generation_seed"],
            condition["training_seed"],
        )
        if not report["splits"]:
            raise ValueError("Evaluation must contain a real val/test split.")
        for name, split in report["splits"].items():
            previous = split_contracts.setdefault(name, candidate_splits[name])
            if previous != candidate_splits[name]:
                raise ValueError(
                    "Evaluation split inventories, order or batches differ."
                )
            split_identity = (*identity, split["split"])
            if split_identity in seen:
                raise ValueError("Duplicate condition, seed and split evaluation.")
            seen.add(split_identity)
            if (
                split["source"] != "tennis_court_detector"
                or split["split"] not in {"val", "test"}
                or split["pose_supervised"]
            ):
                raise ValueError(
                    "Scaling comparison only accepts real val/test without pose scoring."
                )
            metrics = {metric: split["metrics"][metric] for metric in DENSE_METRICS}
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in metrics.values()
            ):
                raise ValueError("Required dense metrics must be finite.")
            rows.append(
                {
                    "series": report["series"],
                    "data_version": condition["data_version"],
                    "accepted_images": condition["accepted_images"],
                    "scenes": len(condition["scene_images"]),
                    "generation_seed": condition["generation_seed"],
                    "training_seed": condition["training_seed"],
                    "global_step": report["global_step"],
                    "split": split["split"],
                    **metrics,
                    "evaluation": str(path.resolve()),
                    "evaluation_sha256": file_sha256(path),
                }
            )
    contract["splits"] = split_contracts
    rows.sort(
        key=lambda row: (
            row["series"],
            row["accepted_images"],
            row["generation_seed"],
            row["training_seed"],
            row["split"],
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        "series",
        "data_version",
        "accepted_images",
        "scenes",
        "generation_seed",
        "training_seed",
        "global_step",
        "split",
        *DENSE_METRICS,
    ]
    lines = [
        "# Court data scaling",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
            for key in columns
        )
        + " |"
        for row in rows
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(DENSE_METRICS), figsize=(22, 4), squeeze=False)
    grouped: dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["series"], row["split"], row["scenes"])][
            row["accepted_images"]
        ].append(row)
    curves = []
    for (series, split, scenes), points in sorted(grouped.items()):
        x = sorted(points)
        for metric, axis in zip(DENSE_METRICS, axes[0], strict=True):
            y = [mean(point[metric] for point in points[count]) for count in x]
            error = [
                stdev(point[metric] for point in points[count])
                if len(points[count]) > 1
                else 0.0
                for count in x
            ]
            axis.errorbar(
                x, y, yerr=error, marker="o", label=f"{series}/{split}/{scenes} scenes"
            )
            axis.set(xscale="log", xlabel="Accepted training images", title=metric)
            for count, value, deviation in zip(x, y, error, strict=True):
                curves.append(
                    {
                        "series": series,
                        "split": split,
                        "scenes": scenes,
                        "images": count,
                        "metric": metric,
                        "mean": value,
                        "std": deviation,
                        "runs": len(points[count]),
                    }
                )
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "scaling.png", dpi=150)
    plt.close(figure)
    receipt = {
        "schema": "court_data_scaling_comparison_v1",
        "contract_sha256": canonical_sha256(contract),
        "contract": contract,
        "evaluations": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path, _ in reports
        ],
        "rows": rows,
        "curves": curves,
    }
    (output / "comparison.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return output / "comparison.md"
