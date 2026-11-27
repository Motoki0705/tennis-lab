"""Render labeled, light-background comparison charts in both public formats."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

from .chart_spec import Bars, Chart, Scatter, pareto_frontier, raw_number
from .sources import require
from .table_spec import Row

FONT = "Noto Sans CJK JP"
INK = "#17212d"
COLORS = (
    "#145da0",
    "#087366",
    "#954e29",
    "#66489e",
    "#9c4065",
    "#4d6818",
    "#535c6b",
    "#867000",
    "#007f8a",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "<", ">", "*")


def _category(row: Row, column: int = 0) -> str:
    category = row.cells[column]
    if not isinstance(category, str):
        raise ValueError(f"{row.key}: expected a category label")
    return category


def _style_axis(axis: Axes, direction: Literal["both", "x", "y"]) -> None:
    axis.grid(axis=direction, color="#d8e2ec", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#a9b8c8")


def _plot_scatter(
    axis: Axes, spec: Scatter, styles: dict[str, tuple[str, str]]
) -> None:
    x_column, y_column = spec.data.columns[spec.x], spec.data.columns[spec.y]
    coordinates = {
        row.key: (
            float(raw_number(row, spec.x) * x_column.scale),
            float(raw_number(row, spec.y) * y_column.scale),
        )
        for row in spec.plotted_rows
    }
    frontier = pareto_frontier(spec)
    if spec.memory is None and frontier:
        xy = [coordinates[row.key] for row in frontier]
        axis.plot(
            [point[0] for point in xy],
            [point[1] for point in xy],
            color=INK,
            linewidth=0.95,
            zorder=2,
        )
    for row in spec.plotted_rows:
        color, marker = styles[_category(row)]
        x, y = coordinates[row.key]
        if row.exclusion is not None:
            axis.scatter(x, y, c=color, marker="x", s=38, linewidths=1.15, zorder=3)
            continue
        hollow = _category(row, 1) == "TRT16"
        axis.scatter(
            x,
            y,
            marker=marker,
            s=34,
            facecolors="white" if hollow else color,
            edgecolors=color,
            linewidths=1.05,
            zorder=3,
        )
        if spec.memory is not None and row in frontier:
            axis.scatter(
                x,
                y,
                marker="o",
                s=108,
                facecolors="none",
                edgecolors=INK,
                linewidths=0.75,
                zorder=4,
            )
    axis.set_xlabel(x_column.heading)
    axis.set_ylabel(y_column.heading)
    axis.margins(x=0.10, y=0.12)
    _style_axis(axis, "both")


def _scatter(fig: Figure, spec: Scatter) -> None:
    widths = (3.1, 1.45) if spec.zoom is None else (1.2, 1.35, 0.92)
    axes = fig.subplots(1, len(widths), gridspec_kw={"width_ratios": widths})
    key = axes[-1]
    key.set_axis_off()
    rows = spec.plotted_rows
    require(bool(rows), f"{spec.data.slug}: no paired measurements")
    labels = list(dict.fromkeys(_category(row) for row in rows))
    require(
        len(labels) <= len(MARKERS),
        f"{spec.data.slug}: too many models for one scatter chart",
    )
    styles = {label: (COLORS[i], MARKERS[i]) for i, label in enumerate(labels)}
    _plot_scatter(axes[0], spec, styles)
    if spec.zoom is not None:
        axes[0].set_title("全計測点", fontsize=9, color="#425369")
        zoom = axes[1]
        _plot_scatter(zoom, spec, styles)
        zoom.set_xlim(*spec.zoom[0])
        zoom.set_ylim(*spec.zoom[1])
        zoom.set_ylabel("")
        zoom.set_title("集中域の拡大", fontsize=9, color="#425369")
    handles = [
        Line2D(
            [],
            [],
            marker=styles[label][1],
            linestyle="",
            color=styles[label][0],
            label=label,
            markersize=5,
        )
        for label in labels
    ]
    handles.extend(
        [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=INK,
                label="PT16（塗り）",
                markersize=4,
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=INK,
                markerfacecolor="white",
                label="TRT16（白抜き）",
                markersize=4,
            ),
        ]
    )
    if any(r.exclusion is not None for r in rows):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color=INK,
                label="変換診断",
                markersize=5,
            )
        )
    if spec.memory is None:
        handles.append(Line2D([], [], color=INK, label="Pareto（2 指標）", linewidth=1))
    else:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=INK,
                markerfacecolor="none",
                label="Pareto（VRAM 含む）",
                markersize=7,
            )
        )
    key.legend(
        handles=handles, loc="center left", frameon=False, handlelength=1.25, fontsize=8
    )


def _grouped_bars(axis: Axes, spec: Bars, metric: int, series_column: int) -> None:
    rows = spec.data.rows
    categories = list(dict.fromkeys(_category(row) for row in rows))
    series = list(dict.fromkeys(_category(row, series_column) for row in rows))
    require(len(series) <= len(COLORS), f"{spec.data.slug}: too many series")
    pairs = {(_category(row), _category(row, series_column)): row for row in rows}
    require(
        len(pairs) == len(rows), f"{spec.data.slug}: duplicate bar category and series"
    )
    width = 0.72 / len(series)
    for i, name in enumerate(series):
        positions = []
        values = []
        for j, category in enumerate(categories):
            row = pairs.get((category, name))
            if row is None or row.cells[metric] is None:
                continue
            positions.append(j + (i - (len(series) - 1) / 2) * width)
            values.append(
                float(raw_number(row, metric) * spec.data.columns[metric].scale)
            )
        axis.bar(
            positions,
            values,
            width=width,
            label=name,
            color=COLORS[i],
            edgecolor=INK,
            linewidth=0.55,
            hatch="/" if i % 2 else None,
        )
    axis.set_xticks(range(len(categories)), categories)
    axis.set_ylabel(spec.data.columns[metric].heading)
    axis.legend(frameon=False)
    _style_axis(axis, "y")


def _horizontal_bars(axis: Axes, spec: Bars, metric: int) -> None:
    rows = spec.data.rows
    for i, row in enumerate(rows):
        if row.cells[metric] is None:
            continue
        value = float(raw_number(row, metric) * spec.data.columns[metric].scale)
        axis.barh(
            i,
            value,
            color=COLORS[i % len(COLORS)],
            height=0.55,
            edgecolor=INK,
            linewidth=0.55,
            hatch="/" if row.exclusion is not None else None,
        )
    axis.set_yticks(range(len(rows)), [_category(row) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel(spec.data.columns[metric].heading)
    _style_axis(axis, "x")


def _bars(fig: Figure, spec: Bars) -> None:
    for i, metric in enumerate(spec.metrics, 1):
        axis = fig.add_subplot(len(spec.metrics), 1, i)
        if spec.series is None:
            _horizontal_bars(axis, spec, metric)
        else:
            _grouped_bars(axis, spec, metric, spec.series)


def render_chart(spec: Chart, target: Path, *, font: str = FONT) -> None:
    findfont(FontProperties(family=[font]), fallback_to_default=False)
    if isinstance(spec, Scatter) and spec.zoom is not None:
        size = (9.0, 4.25)
        margins = {
            "left": 0.085,
            "right": 0.98,
            "bottom": 0.24,
            "top": 0.83,
            "wspace": 0.28,
        }
    elif isinstance(spec, Scatter):
        size = (7.2, 4.25)
        margins = {"left": 0.12, "right": 0.98, "bottom": 0.24, "top": 0.88}
    elif spec.series is not None:
        size = (5.85, 3.65)
        margins = {"left": 0.15, "right": 0.97, "bottom": 0.25, "top": 0.87}
    else:
        size = (6.1, 4.4)
        margins = {
            "left": 0.35,
            "right": 0.96,
            "bottom": 0.24,
            "top": 0.87,
            "hspace": 0.64,
        }
    with (
        mpl.rc_context(
            rc={
                "font.family": [font],
                "font.size": 10,
                "text.color": INK,
                "axes.labelcolor": INK,
                "axes.formatter.useoffset": False,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.transparent": False,
                "svg.fonttype": "path",
                "svg.hashsalt": "tennis-lab-wiki-assets",
                "text.usetex": False,
            }
        ),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("error", message=r"Glyph .* missing from font")
        figure = Figure(figsize=size)
        FigureCanvasAgg(figure)
        figure.subplots_adjust(**margins)
        figure.text(
            0.03, 0.95, spec.data.title, fontsize=14, fontweight="bold", va="top"
        )
        figure.text(
            0.03, 0.035, spec.data.note, fontsize=8, color="#425369", va="bottom"
        )
        if isinstance(spec, Scatter):
            _scatter(figure, spec)
        else:
            _bars(figure, spec)
        figure.savefig(
            target.with_suffix(".svg"), metadata={"Date": None, "Creator": None}
        )
        figure.savefig(target.with_suffix(".png"), dpi=144)
