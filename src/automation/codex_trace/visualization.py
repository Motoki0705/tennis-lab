"""Self-contained HTML/SVG and static PNG reports for trace analysis."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from src.automation.codex_trace.model import AnalysisReport, InferenceStep
from src.automation.codex_trace.output import atomic_output

_PALETTE = (
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#0891b2",
    "#ca8a04",
    "#475569",
    "#db2777",
    "#4f46e5",
)


def write_html_report(report: AnalysisReport, path: Path, *, force: bool) -> None:
    """Write a self-contained report with inline SVG charts and no raw content."""

    rendered = render_html_report(report)
    with atomic_output(path, force=force) as temporary:
        temporary.write_text(rendered, encoding="utf-8")


def write_png_report(report: AnalysisReport, path: Path, *, force: bool) -> None:
    """Write a four-panel static dashboard suitable for a GitHub PR body."""

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    flattened: list[Axes] = list(axes.flat)
    _plot_exact_tokens(flattened[0], report)
    _plot_input_attribution(flattened[1], report)
    _plot_tool_outputs(flattened[2], report)
    _plot_timeline(flattened[3], report)
    figure.suptitle(
        f"Codex rollout {report.source.rollout_id} — {report.source.status}",
        fontsize=16,
        fontweight="bold",
    )
    with atomic_output(path, force=force) as temporary:
        figure.savefig(
            temporary,
            format="png",
            dpi=160,
            metadata={"Title": f"Codex trace {report.source.trace_id}"},
        )
    plt.close(figure)


def render_html_report(report: AnalysisReport) -> str:
    """Render an offline HTML report without prompts, tool bodies, or local paths."""

    exact_series = {
        "input": [
            float(step.usage_exact.input_tokens if step.usage_exact else 0)
            for step in report.inference_steps
        ],
        "output": [
            float(step.usage_exact.output_tokens if step.usage_exact else 0)
            for step in report.inference_steps
        ],
        "reasoning": [
            float(step.usage_exact.reasoning_output_tokens if step.usage_exact else 0)
            for step in report.inference_steps
        ],
    }
    inference_labels = [
        _short_id(step.inference_call_id) for step in report.inference_steps
    ]
    input_rows = [
        step.input_tokens_by_cluster_estimated for step in report.inference_steps
    ]
    reasoning_rows = [
        step.reasoning_tokens_by_cluster_estimated for step in report.inference_steps
    ]
    tool_series = {
        "raw estimated": [
            float(tool.raw_output_tokens_estimated or 0) for tool in report.tool_calls
        ],
        "model-visible estimated": [
            float(tool.model_visible_output_tokens_estimated)
            for tool in report.tool_calls
        ],
        "pre-truncation reported": [
            float(tool.original_output_tokens or 0) for tool in report.tool_calls
        ],
    }
    tool_labels = [_short_id(tool.tool_call_id) for tool in report.tool_calls]
    warning_html = (
        "".join(f"<li>{html.escape(warning)}</li>" for warning in report.warnings)
        if report.warnings
        else "<li>None</li>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Codex trace {html.escape(report.source.trace_id)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f8fafc; --panel:#fff; --text:#0f172a; --muted:#64748b; --line:#cbd5e1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#020617; --panel:#0f172a; --text:#e2e8f0; --muted:#94a3b8; --line:#334155; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width:1280px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:0 0 12px; font-size:18px; }}
.subtitle,.note {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:24px 0; }}
.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.value {{ font-size:24px; font-weight:700; }} .label {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:16px; }}
.panel {{ overflow-x:auto; }} svg {{ width:100%; min-width:560px; height:auto; }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ padding:8px; border-bottom:1px solid var(--line); text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
code {{ color:inherit; }} .exact {{ color:#2563eb; }} .estimated {{ color:#9333ea; }}
</style>
</head>
<body><main>
<h1>Codex rollout trace</h1>
<div class="subtitle">rollout <code>{html.escape(report.source.rollout_id)}</code> · trace <code>{html.escape(report.source.trace_id)}</code> · {html.escape(report.source.status)}</div>
<div class="cards">
{_card("Exact input", report.totals_exact.input_tokens, "tokens")}
{_card("Exact cached input", report.totals_exact.cached_input_tokens, "tokens")}
{_card("Exact output", report.totals_exact.output_tokens, "tokens")}
{_card("Exact reasoning", report.totals_exact.reasoning_output_tokens, "tokens")}
{_card("Inference calls", report.totals_exact.inference_calls, "attempts")}
{_card("Tool calls", len(report.tool_calls), "runtime calls")}
</div>
<p class="note"><span class="exact">Exact</span> totals come from provider usage. <span class="estimated">Estimated</span> item and cluster attribution uses the method recorded in the analysis output; hover SVG marks for values.</p>
<div class="grid">
{_panel("Exact tokens by inference (log-scaled height)", _grouped_bar_svg(inference_labels, exact_series, log_scale=True))}
{_panel("Estimated input attribution", _stacked_bar_svg(inference_labels, input_rows))}
{_panel("Estimated reasoning-cluster allocation", _stacked_bar_svg(inference_labels, reasoning_rows))}
{_panel("Tool output volume", _grouped_bar_svg(tool_labels, tool_series, log_scale=True))}
</div>
{_panel("Execution timeline", _timeline_svg(report))}
{_panel("Inference details", _inference_table(report.inference_steps))}
{_panel("Tool details", _tool_table(report))}
<section class="panel"><h2>Warnings</h2><ul>{warning_html}</ul></section>
<p class="note">This report intentionally excludes raw prompts, raw tool bodies, and the local trace-bundle path.</p>
</main></body></html>
"""


def _card(label: str, value: int, suffix: str) -> str:
    return f'<div class="card"><div class="value">{value:,}</div><div class="label">{html.escape(label)} · {html.escape(suffix)}</div></div>'


def _panel(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{html.escape(title)}</h2>{body}</section>'


def _grouped_bar_svg(
    labels: Sequence[str], series: Mapping[str, Sequence[float]], *, log_scale: bool
) -> str:
    if not labels:
        return _empty_svg("No data")
    width, height = 900, 340
    left, right, top, bottom = 70, 24, 54, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    transformed = [
        math.log10(value + 1) if log_scale else value
        for values in series.values()
        for value in values
    ]
    maximum = max(transformed, default=0.0) or 1.0
    group_w = plot_w / len(labels)
    bar_w = min(44.0, group_w * 0.72 / max(len(series), 1))
    pieces = [_svg_frame(width, height, left, top, plot_w, plot_h, maximum)]
    for series_index, (name, values) in enumerate(series.items()):
        color = _PALETTE[series_index % len(_PALETTE)]
        pieces.append(_legend_item(left + series_index * 180, 20, color, name))
        for item_index, value in enumerate(values):
            scaled = math.log10(value + 1) if log_scale else value
            bar_h = plot_h * scaled / maximum
            x = left + item_index * group_w + group_w * 0.14 + series_index * bar_w
            y = top + plot_h - bar_h
            pieces.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"><title>{html.escape(name)} · {html.escape(labels[item_index])}: {value:,.1f}</title></rect>'
            )
    pieces.extend(_x_labels(labels, left, top + plot_h, group_w))
    pieces.append("</svg>")
    return "".join(pieces)


def _stacked_bar_svg(labels: Sequence[str], rows: Sequence[Mapping[str, float]]) -> str:
    if not labels:
        return _empty_svg("No data")
    clusters = sorted({cluster for row in rows for cluster in row})
    if not clusters:
        return _empty_svg("No attributed tokens")
    width, height = 900, 340
    left, right, top, bottom = 70, 24, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [sum(row.values()) for row in rows]
    maximum = max(totals, default=0.0) or 1.0
    group_w = plot_w / len(labels)
    bar_w = min(70.0, group_w * 0.65)
    pieces = [_svg_frame(width, height, left, top, plot_w, plot_h, maximum)]
    for cluster_index, cluster in enumerate(clusters):
        color = _PALETTE[cluster_index % len(_PALETTE)]
        legend_x = left + (cluster_index % 4) * 205
        legend_y = 18 + (cluster_index // 4) * 20
        pieces.append(_legend_item(legend_x, legend_y, color, cluster))
    for item_index, row in enumerate(rows):
        accumulated = 0.0
        x = left + item_index * group_w + (group_w - bar_w) / 2
        for cluster_index, cluster in enumerate(clusters):
            value = float(row.get(cluster, 0.0))
            bar_h = plot_h * value / maximum
            y = top + plot_h - (plot_h * accumulated / maximum) - bar_h
            if value > 0:
                color = _PALETTE[cluster_index % len(_PALETTE)]
                pieces.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"><title>{html.escape(cluster)} · {html.escape(labels[item_index])}: {value:,.1f}</title></rect>'
                )
            accumulated += value
    pieces.extend(_x_labels(labels, left, top + plot_h, group_w))
    pieces.append("</svg>")
    return "".join(pieces)


def _timeline_svg(report: AnalysisReport) -> str:
    rows: list[tuple[str, str, int, int]] = []
    for step in report.inference_steps:
        end = step.ended_at_unix_ms or step.started_at_unix_ms
        rows.append(
            (
                f"inference {_short_id(step.inference_call_id)}",
                "inference",
                step.started_at_unix_ms,
                end,
            )
        )
    for tool in report.tool_calls:
        end = tool.ended_at_unix_ms or tool.started_at_unix_ms
        rows.append(
            (
                f"{tool.kind} {_short_id(tool.tool_call_id)}",
                "tool",
                tool.started_at_unix_ms,
                end,
            )
        )
    if not rows:
        return _empty_svg("No execution windows")
    rows.sort(key=lambda row: (row[2], row[0]))
    start = min(row[2] for row in rows)
    end = max(row[3] for row in rows)
    span = max(end - start, 1)
    width = 1100
    left, right, top, row_h = 230, 24, 34, 28
    plot_w = width - left - right
    height = top + len(rows) * row_h + 36
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    pieces.append(
        f'<line x1="{left}" y1="{top - 10}" x2="{left + plot_w}" y2="{top - 10}" stroke="currentColor" opacity=".35"/>'
    )
    for index, (label, kind, row_start, row_end) in enumerate(rows):
        y = top + index * row_h
        x = left + plot_w * (row_start - start) / span
        duration = max(row_end - row_start, 0)
        bar_w = max(2.0, plot_w * duration / span)
        color = _PALETTE[0 if kind == "inference" else 1]
        pieces.append(
            f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" fill="currentColor" font-size="12">{html.escape(label)}</text>'
        )
        pieces.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="18" rx="4" fill="{color}"><title>{html.escape(label)}: {duration:,} ms</title></rect>'
        )
    pieces.append(
        f'<text x="{left}" y="{height - 10}" fill="currentColor" font-size="11">0 ms</text>'
    )
    pieces.append(
        f'<text x="{left + plot_w}" y="{height - 10}" text-anchor="end" fill="currentColor" font-size="11">{span:,} ms</text>'
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _inference_table(steps: Sequence[InferenceStep]) -> str:
    rows: list[str] = []
    for step in steps:
        usage = step.usage_exact
        duration = (
            None
            if step.ended_at_unix_ms is None
            else step.ended_at_unix_ms - step.started_at_unix_ms
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(_short_id(step.inference_call_id))}</code></td>"
            f"<td>{html.escape(step.model)}</td><td>{_number(duration)}</td>"
            f"<td>{_number(usage.input_tokens if usage else None)}</td>"
            f"<td>{_number(usage.output_tokens if usage else None)}</td>"
            f"<td>{_number(usage.reasoning_output_tokens if usage else None)}</td>"
            f"<td>{_number(step.input_residual_tokens)}</td>"
            f"<td>{html.escape(step.reasoning_evidence_mode)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Inference</th><th>Model</th><th>Duration ms</th><th>Exact input</th><th>Exact output</th><th>Exact reasoning</th><th>Input residual</th><th>Reasoning evidence</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _tool_table(report: AnalysisReport) -> str:
    if not report.tool_calls:
        return '<p class="note">No tool calls.</p>'
    rows: list[str] = []
    for tool in report.tool_calls:
        files = ", ".join(tool.files_touched) or "—"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(_short_id(tool.tool_call_id))}</code></td>"
            f"<td>{html.escape(tool.kind)}</td><td>{html.escape(tool.status)}</td>"
            f"<td>{_number(tool.raw_output_tokens_estimated)}</td>"
            f"<td>{tool.model_visible_output_tokens_estimated:,}</td>"
            f"<td>{html.escape(tool.model_visible_output_evidence)}</td>"
            f"<td>{_number(tool.original_output_tokens)}</td>"
            f"<td>{html.escape(files)}</td><td>+{tool.lines_added}/-{tool.lines_deleted}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Tool</th><th>Kind</th><th>Status</th><th>Raw est.</th><th>Visible est.</th><th>Visible evidence</th><th>Pre-trunc.</th><th>Files</th><th>Patch</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _svg_frame(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    maximum: float,
) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="currentColor" opacity=".35"/>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="currentColor" opacity=".35"/>'
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" fill="currentColor" font-size="11">{maximum:,.1f}</text>'
        f'<text x="{left - 8}" y="{top + plot_h + 4}" text-anchor="end" fill="currentColor" font-size="11">0</text>'
    )


def _x_labels(
    labels: Sequence[str], left: int, baseline: int, group_w: float
) -> list[str]:
    return [
        f'<text x="{left + index * group_w + group_w / 2:.1f}" y="{baseline + 18}" text-anchor="middle" fill="currentColor" font-size="11">{html.escape(label)}</text>'
        for index, label in enumerate(labels)
    ]


def _legend_item(x: float, y: float, color: str, label: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="11" height="11" fill="{color}"/><text x="{x + 16:.1f}" y="{y + 10:.1f}" fill="currentColor" font-size="11">{html.escape(label)}</text>'


def _empty_svg(message: str) -> str:
    return f'<svg viewBox="0 0 900 180" role="img"><text x="450" y="90" text-anchor="middle" fill="currentColor" opacity=".65">{html.escape(message)}</text></svg>'


def _plot_exact_tokens(axis: Axes, report: AnalysisReport) -> None:
    labels = [_short_id(step.inference_call_id) for step in report.inference_steps]
    x = list(range(len(labels)))
    width = 0.24
    values = (
        (
            "input",
            [
                step.usage_exact.input_tokens if step.usage_exact else 0
                for step in report.inference_steps
            ],
        ),
        (
            "output",
            [
                step.usage_exact.output_tokens if step.usage_exact else 0
                for step in report.inference_steps
            ],
        ),
        (
            "reasoning",
            [
                step.usage_exact.reasoning_output_tokens if step.usage_exact else 0
                for step in report.inference_steps
            ],
        ),
    )
    for index, (name, counts) in enumerate(values):
        offsets = [value + (index - 1) * width for value in x]
        axis.bar(offsets, counts, width=width, label=name, color=_PALETTE[index])
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("tokens (symlog)")
    axis.set_title("Exact provider token usage")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)


def _plot_input_attribution(axis: Axes, report: AnalysisReport) -> None:
    labels = [_short_id(step.inference_call_id) for step in report.inference_steps]
    clusters = sorted(
        {
            cluster
            for step in report.inference_steps
            for cluster in step.input_tokens_by_cluster_estimated
        }
    )
    bottom = [0.0] * len(labels)
    for index, cluster in enumerate(clusters):
        values = [
            float(step.input_tokens_by_cluster_estimated.get(cluster, 0))
            for step in report.inference_steps
        ]
        axis.bar(
            labels,
            values,
            bottom=bottom,
            label=cluster,
            color=_PALETTE[index % len(_PALETTE)],
        )
        bottom = [
            current + value for current, value in zip(bottom, values, strict=True)
        ]
    axis.set_title("Estimated input attribution")
    axis.set_ylabel("estimated tokens")
    axis.tick_params(axis="x", rotation=30)
    if clusters:
        axis.legend(fontsize=7)
    axis.grid(axis="y", alpha=0.2)


def _plot_tool_outputs(axis: Axes, report: AnalysisReport) -> None:
    if not report.tool_calls:
        axis.text(
            0.5,
            0.5,
            "No tool calls",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Tool output volume")
        return
    labels = [_short_id(tool.tool_call_id) for tool in report.tool_calls]
    x = list(range(len(labels)))
    width = 0.24
    series = (
        (
            "raw est.",
            [tool.raw_output_tokens_estimated or 0 for tool in report.tool_calls],
        ),
        (
            "visible est.",
            [tool.model_visible_output_tokens_estimated for tool in report.tool_calls],
        ),
        (
            "pre-trunc.",
            [tool.original_output_tokens or 0 for tool in report.tool_calls],
        ),
    )
    for index, (name, values) in enumerate(series):
        offsets = [value + (index - 1) * width for value in x]
        axis.bar(offsets, values, width=width, label=name, color=_PALETTE[index])
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_title("Tool output: raw vs model-visible")
    axis.set_ylabel("tokens (symlog)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)


def _plot_timeline(axis: Axes, report: AnalysisReport) -> None:
    rows: list[tuple[str, int, int, str]] = []
    for step in report.inference_steps:
        rows.append(
            (
                f"inf {_short_id(step.inference_call_id)}",
                step.started_at_unix_ms,
                step.ended_at_unix_ms or step.started_at_unix_ms,
                _PALETTE[0],
            )
        )
    for tool in report.tool_calls:
        rows.append(
            (
                f"{tool.kind} {_short_id(tool.tool_call_id)}",
                tool.started_at_unix_ms,
                tool.ended_at_unix_ms or tool.started_at_unix_ms,
                _PALETTE[1],
            )
        )
    rows.sort(key=lambda row: (row[1], row[0]))
    if not rows:
        axis.text(
            0.5,
            0.5,
            "No execution windows",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Execution timeline")
        return
    start = min(row[1] for row in rows)
    labels = [row[0] for row in rows]
    offsets = [(row[1] - start) / 1000 for row in rows]
    durations = [max((row[2] - row[1]) / 1000, 0.001) for row in rows]
    axis.barh(labels, durations, left=offsets, color=[row[3] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("seconds from first event")
    axis.set_title("Execution timeline")
    axis.grid(axis="x", alpha=0.2)


def _short_id(value: str) -> str:
    return value if len(value) <= 14 else value[:8]


def _number(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"
