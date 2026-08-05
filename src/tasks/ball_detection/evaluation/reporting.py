"""Machine-readable and Markdown comparison report generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.io import save_json

_CSV_FIELDS = [
    "category",
    "model_id",
    "model_name",
    "dataset_id",
    "split",
    "precision",
    "recall",
    "f1",
    "mean_distance_px",
    "negative_frame_fpr",
    "throughput_frames_per_second",
    "latency_ms_per_batch",
    "peak_vram_mb",
]


def write_comparison_reports(
    results: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    """Write summary JSON, flat CSV, and category-separated Markdown."""
    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") == "failed"]
    rows = [_flatten_result(result) for result in successful]
    save_json(
        {
            "schema": "ball_detection_evaluation_summary_v1",
            "successful_jobs": len(successful),
            "failed_jobs": len(failed),
            "results": results,
        },
        output_dir / "summary.json",
    )
    _write_csv(rows, output_dir / "comparison.csv")
    (output_dir / "comparison.md").write_text(
        _render_markdown(rows, failed),
        encoding="utf-8",
    )


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result["result"]["metrics"]["aggregate"]
    performance = result["result"]["performance"]
    return {
        "category": result["category"],
        "model_id": result["model_id"],
        "model_name": result["model_name"],
        "dataset_id": result["dataset_id"],
        "split": result["split"],
        "precision": aggregate["precision"],
        "recall": aggregate["recall"],
        "f1": aggregate["f1"],
        "mean_distance_px": aggregate["mean_distance_px"],
        "negative_frame_fpr": aggregate["negative_frame_fpr"],
        "throughput_frames_per_second": performance["throughput_frames_per_second"],
        "latency_ms_per_batch": performance["latency_ms_per_batch"],
        "peak_vram_mb": performance["peak_vram_mb"],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> str:
    lines = ["# Ball detector evaluation", ""]
    titles = {
        "architecture-controlled": "Architecture-controlled",
        "full-strategy": "Full strategy",
    }
    for category, title in titles.items():
        lines.extend([f"## {title}", ""])
        category_rows = [row for row in rows if row["category"] == category]
        if not category_rows:
            lines.extend(["No completed evaluations.", ""])
            continue
        lines.extend(
            [
                "| Model | Dataset | Split | Precision | Recall | F1 | Distance px | Negative FPR | FPS | Peak VRAM MB |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in category_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["model_id"]),
                        str(row["dataset_id"]),
                        str(row["split"]),
                        _format(row["precision"]),
                        _format(row["recall"]),
                        _format(row["f1"]),
                        _format(row["mean_distance_px"]),
                        _format(row["negative_frame_fpr"]),
                        _format(row["throughput_frames_per_second"]),
                        _format(row["peak_vram_mb"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    if failed:
        lines.extend(["## Failed jobs", ""])
        for result in failed:
            lines.append(
                f"- `{result['job_id']}`: "
                f"{result['error']['type']}: {result['error']['message']}"
            )
        lines.append("")
    return "\n".join(lines)


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


__all__ = ["write_comparison_reports"]
