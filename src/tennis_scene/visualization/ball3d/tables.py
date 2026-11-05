"""One source for the numeric cells in the HTML and standalone comparison table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricTable:
    key: str
    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str


def comparison_tables(scores: dict[str, Any]) -> tuple[MetricTable, MetricTable]:
    def number(value: Any) -> str:
        if value is None:
            return "unavailable"
        return str(value) if isinstance(value, int) else f"{value:.4g}"

    def rate(value: Any) -> str:
        return "unavailable" if value is None else f"{100 * value:.2f}%"

    primary, paired = [], []
    for name, method in scores["methods"].items():
        for segment in ("tuning", "evaluation", "all"):
            full = method["folds"]["all_views"][segment]
            lovo = method["lovo_reprojection_px"][segment]
            common = method["common_lovo_reprojection_px"][segment]
            primary.append(
                (
                    name,
                    segment,
                    number(None if lovo is None else lovo["median"]),
                    number(None if lovo is None else lovo["p90"]),
                    rate(None if lovo is None else lovo["above_threshold_rate"]),
                    number(None if lovo is None else lovo["count"]),
                    number(None if lovo is None else lovo["label_count"]),
                    rate(None if lovo is None else lovo["scored_label_coverage"]),
                    rate(full["valid_3d_rate"]),
                )
            )
            paired.append(
                (
                    name,
                    segment,
                    number(None if common is None else common["count"]),
                    number(None if common is None else common["median"]),
                    number(None if common is None else common["p90"]),
                    rate(None if common is None else common["scored_label_coverage"]),
                    number(
                        None
                        if full["reprojection_px"] is None
                        else full["reprojection_px"]["median"]
                    ),
                    rate(full["gross_reprojection_frame_rate"]),
                    rate(full["speed_mps"]["above_threshold_rate"]),
                    rate(full["acceleration_mps2"]["above_threshold_rate"]),
                )
            )
    threshold = scores["config"]["gross_reprojection_px"]
    return (
        MetricTable(
            "lovo-summary",
            "LOVO error and scoring coverage",
            (
                "Method",
                "Segment",
                "LOVO median px",
                "LOVO p90 px",
                f"LOVO >{threshold:g}px rate",
                "LOVO scored",
                "LOVO eligible",
                "LOVO coverage",
                "All-view 3D valid rate",
            ),
            tuple(primary),
            "Methods score different camera/frame pairs. Compare coverage alongside error. "
            "Missing predictions are excluded, never scored as zero error. All-view validity is a separate metric.",
        ),
        MetricTable(
            "common-lovo",
            "Common-sample LOVO and all-view diagnostics",
            (
                "Method",
                "Segment",
                "Common LOVO scored",
                "Common median px",
                "Common p90 px",
                "Common coverage",
                "All-view median px",
                "All-view gross rate",
                "All-view speed outlier rate",
                "All-view accel outlier rate",
            ),
            tuple(paired),
            "Common samples are the intersection of scored (held-out camera, source frame) pairs across: "
            + ", ".join(scores["common_lovo_cohort"]["methods"])
            + ". Coverage uses all eligible references as denominator. "
            "This comparison remains conditional on common availability; it does not recover missing predictions.",
        ),
    )
