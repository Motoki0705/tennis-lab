"""Publication artifacts: metrics JSON, flat CSV, and include-ready TeX.

The three formats are rendered from one aggregate structure so a number cannot
drift between the machine-readable and the paper-facing tables.  Any undefined
value is written as ``null`` (JSON), an empty CSV cell, or ``--`` (TeX) instead
of a silent zero.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.tasks.court_detection.evaluation.contracts import METRICS_SCHEMA
from src.tasks.court_detection.evaluation.storage import write_json_atomic

CSV_COLUMNS = (
    "domain",
    "domain_display_name",
    "model",
    "metric_group",
    "metric",
    "subkey",
    "value",
    "count",
)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_tex_value(value: object, *, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _flatten(
    value: object,
    *,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], object]]:
    if isinstance(value, Mapping):
        rows: list[tuple[tuple[str, ...], object]] = []
        for key in sorted(value):
            rows.extend(_flatten(value[key], prefix=(*prefix, str(key))))
        return rows
    return [(prefix, value)]


def csv_rows(
    aggregates: Sequence[Mapping[str, object]],
    *,
    display_names: Mapping[str, str],
) -> list[tuple[str, ...]]:
    """Flatten every aggregate into one row per scalar leaf."""
    rows: list[tuple[str, ...]] = []
    for aggregate in aggregates:
        domain = str(aggregate["domain"])
        model = str(aggregate["model"])
        for group in ("keypoints", "alignment"):
            leaves = dict(_flatten(_mapping(aggregate, group)))
            for path, leaf in _flatten(_mapping(aggregate, group)):
                rows.append(
                    (
                        domain,
                        display_names.get(domain, domain),
                        model,
                        group,
                        ".".join(path[:-1]) if len(path) > 1 else path[0],
                        path[-1] if len(path) > 1 else "",
                        _format_value(leaf),
                        _row_count(path, leaf, leaves),
                    )
                )
        for channel in _sequence(aggregate, "per_channel"):
            channel_mapping = _as_mapping(channel)
            name = str(channel_mapping["name"])
            rows.append(
                (
                    domain,
                    display_names.get(domain, domain),
                    model,
                    "per_channel",
                    name,
                    "pck",
                    _format_value(
                        "; ".join(
                            f"{key}={_format_value(item) or 'null'}"
                            for key, item in sorted(
                                _mapping(channel_mapping, "pck").items()
                            )
                        )
                    ),
                    str(channel_mapping["ground_truth_visible"]),
                )
            )
            rows.append(
                (
                    domain,
                    display_names.get(domain, domain),
                    model,
                    "per_channel",
                    name,
                    "pair_error_px.mean",
                    _format_value(_mapping(channel_mapping, "pair_error_px")["mean"]),
                    str(_mapping(channel_mapping, "pair_error_px")["count"]),
                )
            )
    return rows


def render_csv(rows: Sequence[tuple[str, ...]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


_STAT_LEAF_KEYS = frozenset({"mean", "median", "q90"})
_SELF_COUNT_LEAF_KEYS = frozenset(
    {"count", "line_samples_in_frame", "line_samples_total"}
)


def _row_count(
    path: tuple[str, ...],
    leaf: object,
    leaves: Mapping[tuple[str, ...], object],
) -> str:
    """Fill the CSV ``count`` column so a denominator never lives only in JSON.

    Distribution leaves (``mean``/``median``/``q90``) borrow the ``count`` of the
    statistics block they belong to, and the count leaves themselves report their
    own value.
    """
    if not path:
        return ""
    if path[-1] in _STAT_LEAF_KEYS and len(path) > 1:
        sibling = leaves.get((*path[:-1], "count"))
        return "" if sibling is None else _format_value(sibling)
    if path[-1] in _SELF_COUNT_LEAF_KEYS:
        return _format_value(leaf)
    return ""


def _primary_table(
    aggregates: Sequence[Mapping[str, object]],
    *,
    display_names: Mapping[str, str],
) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Domain & Model & Complete & PCK@0.01 & PCK@0.02 & Pair median (px) & "
            r"H success & IoU \\"
        ),
        r"\midrule",
    ]
    for aggregate in aggregates:
        keypoints = _mapping(aggregate, "keypoints")
        alignment = _mapping(aggregate, "alignment")
        domain = str(aggregate["domain"])
        pck = _mapping(keypoints, "pck")
        lines.append(
            " & ".join(
                (
                    _tex_escape(display_names.get(domain, domain)),
                    str(aggregate["model"]),
                    _format_tex_value(keypoints["completeness"]),
                    _format_tex_value(_first_matching(pck, "0.01")),
                    _format_tex_value(_first_matching(pck, "0.02")),
                    _format_tex_value(_mapping(keypoints, "pair_error_px")["median"]),
                    _format_tex_value(alignment["predicted_homography_success_rate"]),
                    _format_tex_value(
                        _mapping(alignment, "doubles_polygon_iou")["mean"]
                    ),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _line_table(
    aggregates: Sequence[Mapping[str, object]],
    *,
    display_names: Mapping[str, str],
) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Domain & Model & Line mean (px) & Line q90 (px) & In-frame mean (px) & "
            r"Line mean / diag \\"
        ),
        r"\midrule",
    ]
    for aggregate in aggregates:
        alignment = _mapping(aggregate, "alignment")
        line = _mapping(alignment, "line_reprojection_px")
        in_frame = _mapping(alignment, "line_reprojection_in_frame_px")
        normalized = _mapping(alignment, "line_reprojection_diagonal")
        lines.append(
            " & ".join(
                (
                    _tex_escape(
                        display_names.get(
                            str(aggregate["domain"]), str(aggregate["domain"])
                        )
                    ),
                    str(aggregate["model"]),
                    _format_tex_value(line["mean"]),
                    _format_tex_value(line["q90"]),
                    _format_tex_value(in_frame["mean"]),
                    _format_tex_value(normalized["mean"]),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def render_tex(
    aggregates: Sequence[Mapping[str, object]],
    *,
    display_names: Mapping[str, str],
    manifest_fingerprint: str,
    command: str,
) -> str:
    """Render TeX tables a paper section can ``\\input`` directly."""
    header = [
        "% Generated by src/tasks/court_detection/scripts/benchmark_alignment.py",
        f"% manifest fingerprint: {manifest_fingerprint}",
        f"% command: {command}",
        "% 'real_validation' is the held-out TennisCourtDetector validation split,",
        "% not an official test set.",
        "",
    ]
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Court keypoint and alignment benchmark. Complete is the fraction "
        r"of ground-truth-visible keypoints a model also reports; PCK thresholds are "
        r"image-diagonal fractions and count a missing keypoint as incorrect.}",
        r"\label{tab:court-alignment}",
        _primary_table(aggregates, display_names=display_names),
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Symmetric line reprojection error between each model's court "
        r"homography and the ground-truth homography, sampled along the regulation "
        r"court lines.}",
        r"\label{tab:court-alignment-lines}",
        _line_table(aggregates, display_names=display_names),
        r"\end{table}",
        "",
    ]
    return "\n".join(header + body)


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Aggregate metric groups must be mappings.")
    payload = value.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Aggregate metric {key!r} must be a mapping.")
    return payload


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Aggregate entries must be mappings.")
    return value


def _sequence(value: object, key: str) -> Sequence[object]:
    if not isinstance(value, Mapping):
        raise ValueError("Aggregate metric groups must be mappings.")
    payload = value.get(key)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"Aggregate metric {key!r} must be a sequence.")
    return payload


def _first_matching(payload: Mapping[str, object], needle: str) -> object:
    """Look up a PCK key by numeric value, not by its rendered spelling."""
    for key, value in payload.items():
        try:
            if abs(float(key) - float(needle)) < 1e-12:
                return value
        except (TypeError, ValueError):
            continue
    return None


def write_report(
    output_root: Path,
    *,
    aggregates: Sequence[Mapping[str, object]],
    display_names: Mapping[str, str],
    manifest_fingerprint: str,
    command: str,
) -> dict[str, Path]:
    """Write metrics.json, metrics.csv, and metrics.tex, returning their paths."""
    document = {
        "schema": METRICS_SCHEMA,
        "manifest_fingerprint": manifest_fingerprint,
        "command": command,
        "domains": {key: value for key, value in sorted(display_names.items())},
        "models": sorted({str(item["model"]) for item in aggregates}),
        "aggregates": list(aggregates),
        "notes": {
            "real_validation": (
                "held-out TennisCourtDetector validation split; this is not the "
                "official upstream test split"
            ),
            "missing_predictions": (
                "a keypoint a model does not report counts as a PCK miss; the "
                "valid-pair error distributions additionally publish their own "
                "pair count"
            ),
            "line_reprojection": (
                "sampled along the template court lines, so samples that project "
                "outside the frame are homography extrapolations; the *_in_frame "
                "variants restrict the same error to line samples whose ground-truth "
                "projection lands inside the image, and "
                "line_samples_in_frame_fraction reports how much of the court that is"
            ),
            "doubles_polygon_iou": (
                "measured after clipping both polygons to the image, so it is "
                "sensitive when a court corner projects far outside the frame; read "
                "it together with line_samples_in_frame_fraction"
            ),
        },
    }
    paths = {
        "json": output_root / "metrics.json",
        "csv": output_root / "metrics.csv",
        "tex": output_root / "metrics.tex",
    }
    write_json_atomic(paths["json"], document)
    output_root.mkdir(parents=True, exist_ok=True)
    paths["csv"].write_text(
        render_csv(csv_rows(aggregates, display_names=display_names)),
        encoding="utf-8",
    )
    paths["tex"].write_text(
        render_tex(
            aggregates,
            display_names=display_names,
            manifest_fingerprint=manifest_fingerprint,
            command=command,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "CSV_COLUMNS",
    "csv_rows",
    "render_csv",
    "render_tex",
    "write_report",
]
