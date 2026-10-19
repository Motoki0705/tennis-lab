"""Interpolate abstract architecture states on caller-owned matplotlib axes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.tasks.base.visualization.presentation import style

if TYPE_CHECKING:
    from matplotlib.axes import Axes

BlockState = Literal["hidden", "fixed", "training", "frozen", "new", "removed"]
RGB = tuple[float, float, float]


def draw_flow_particles(
    ax: Axes,
    path_points: Sequence[tuple[float, float]],
    phase: float,
    color: str,
) -> None:
    """Draw a traveling particle and its short trail on a polyline.

    ``phase`` is in [0, 1]; both endpoints represent the start of the loop.
    Travel is uniform in path length, including across unequal segments.
    The caller owns the clock, axes limits and any supervision gates.
    """
    points = np.asarray(path_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path_points must contain at least two 2D points")
    if not np.isfinite(points).all():
        raise ValueError("path_points must be finite")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("adjacent path_points must be distinct")
    if isinstance(phase, bool) or not math.isfinite(phase) or not 0 <= phase <= 1:
        raise ValueError("phase must be finite and in [0, 1]")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    distances = ((phase - np.arange(4) * 0.018) % 1) * cumulative[-1]
    positions = np.column_stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in (0, 1)]
    )
    particles = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=[30, 19, 11, 5],
        color=[(*to_rgb(color), opacity) for opacity in (1.0, 0.6, 0.35, 0.15)],
        zorder=8,
    )
    particles.set_gid("flow-particles")


@dataclass(frozen=True)
class Block:
    """A labeled rectangle; ``xy`` is its lower-left corner in axes data units."""

    id: str
    label: str
    xy: tuple[float, float]
    wh: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.id or not self.label:
            raise ValueError("block id and label must be nonempty")
        if len(self.xy) != 2 or len(self.wh) != 2:
            raise ValueError("block xy and wh must each contain two values")
        if not all(math.isfinite(value) for value in (*self.xy, *self.wh)):
            raise ValueError("block geometry must be finite")
        if any(value <= 0 for value in self.wh):
            raise ValueError("block width and height must be positive")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str

    def __post_init__(self) -> None:
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError("edges must connect two distinct, nonempty block ids")


@dataclass(frozen=True)
class ArchitectureState:
    """Visible block states, active flow edges and short per-block badges.

    Omitted blocks are hidden. Unknown ids are rejected by ``draw_architecture``.
    A badge equal to ``style.CONCEPT_BADGE`` also makes its block and incident
    edges dashed, marking unimplemented concepts without adding a new state.
    """

    blocks: Mapping[str, BlockState]
    active_edges: frozenset[tuple[str, str]]
    badges: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for state in self.blocks.values():
            if state not in get_args(BlockState):
                raise ValueError(f"unknown block state: {state!r}")
        if not isinstance(self.active_edges, frozenset):
            raise TypeError("active_edges must be a frozenset")
        if any(
            not isinstance(badge, str) or not badge for badge in self.badges.values()
        ):
            raise ValueError("badges must be nonempty strings")


def _appearance(state: BlockState) -> tuple[RGB, float]:
    colors = {
        "hidden": style.FIXED,
        "fixed": style.FIXED,
        "training": style.TRAINING,
        "frozen": style.FROZEN,
        "new": style.NEW,
    }
    if state == "removed":
        return style.REMOVED_RGB, style.REMOVED_ALPHA
    color: RGB = to_rgb(colors[state])
    return color, 0.0 if state == "hidden" else 1.0


def _mix(before: float, after: float, t: float) -> float:
    return (1 - t) * before + t * after


def _ports(src: Block, dst: Block) -> tuple[tuple[float, float], tuple[float, float]]:
    centers = [
        (block.xy[0] + block.wh[0] / 2, block.xy[1] + block.wh[1] / 2)
        for block in (src, dst)
    ]
    dx, dy = centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]
    if dx == 0 and dy == 0:
        raise ValueError("connected blocks must have distinct centers")
    scales = [
        min(
            block.wh[0] / (2 * abs(dx)) if dx else math.inf,
            block.wh[1] / (2 * abs(dy)) if dy else math.inf,
        )
        for block in (src, dst)
    ]
    return (
        (centers[0][0] + dx * scales[0], centers[0][1] + dy * scales[0]),
        (centers[1][0] - dx * scales[1], centers[1][1] - dy * scales[1]),
    )


def _validate_state(
    state: ArchitectureState,
    block_ids: set[str],
    edge_ids: set[tuple[str, str]],
) -> None:
    unknown = (set(state.blocks) | set(state.badges)) - block_ids
    if unknown:
        raise ValueError(f"unknown block ids: {sorted(unknown)}")
    if not state.active_edges <= edge_ids:
        raise ValueError("active_edges contains an undefined edge")
    for src, dst in state.active_edges:
        if (
            state.blocks.get(src, "hidden") == "hidden"
            or state.blocks.get(dst, "hidden") == "hidden"
        ):
            raise ValueError("active edges cannot connect hidden blocks")
    if any(
        state.blocks.get(block_id, "hidden") == "hidden" for block_id in state.badges
    ):
        raise ValueError("badges cannot be attached to hidden blocks")


def _draw_block(
    ax: Axes,
    block: Block,
    before: ArchitectureState,
    after: ArchitectureState,
    t: float,
) -> None:
    old_state = before.blocks.get(block.id, "hidden")
    new_state = after.blocks.get(block.id, "hidden")
    old_color, old_alpha = _appearance(old_state)
    new_color, new_alpha = _appearance(new_state)
    alpha = _mix(old_alpha, new_alpha, t)
    if alpha == 0:
        return
    # Fade a hidden endpoint in the visible state's color, without a gray flash.
    if old_alpha == 0:
        old_color = new_color
    if new_alpha == 0:
        new_color = old_color
    color = tuple(_mix(a, b, t) for a, b in zip(old_color, new_color, strict=True))
    new_weight = _mix(float(old_state == "new"), float(new_state == "new"), t)
    rounding = min(block.wh) * 0.1

    def box(**kwargs: Any) -> FancyBboxPatch:
        return FancyBboxPatch(
            block.xy,
            *block.wh,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            **kwargs,
        )

    if new_weight:
        for width, opacity in ((9, 0.08), (5, 0.18)):
            ax.add_patch(
                box(
                    facecolor="none",
                    edgecolor=style.NEW,
                    linewidth=width,
                    alpha=alpha * new_weight * opacity,
                    zorder=2,
                )
            )
    fill = box(facecolor=color, edgecolor="none", alpha=alpha, zorder=3)
    fill.set_gid(f"block:{block.id}")
    ax.add_patch(fill)
    concept = _mix(
        float(before.badges.get(block.id) == style.CONCEPT_BADGE),
        float(after.badges.get(block.id) == style.CONCEPT_BADGE),
        t,
    )
    for weight, linestyle in ((1 - concept, "-"), (concept, style.CONCEPT_LINESTYLE)):
        if weight:
            ax.add_patch(
                box(
                    facecolor="none",
                    edgecolor=style.FOREGROUND,
                    linewidth=1.2,
                    linestyle=linestyle,
                    alpha=alpha * weight,
                    zorder=4,
                )
            )
    x, y = block.xy
    w, h = block.wh
    ax.text(
        x + w / 2,
        y + h / 2,
        block.label,
        ha="center",
        va="center",
        color=style.FOREGROUND,
        alpha=alpha,
        fontsize=11,
        fontfamily=style.FONT_FAMILY,
        zorder=5,
    )
    locked = _mix(float(old_state == "frozen"), float(new_state == "frozen"), t)
    if locked:
        ax.add_patch(
            Arc(
                (x + w * 0.87, y + h * 0.83),
                w * 0.09,
                h * 0.19,
                theta1=0,
                theta2=180,
                color=style.FOREGROUND,
                alpha=alpha * locked,
                linewidth=1.4,
                zorder=5,
            )
        )
        ax.add_patch(
            Rectangle(
                (x + w * 0.81, y + h * 0.65),
                w * 0.12,
                h * 0.18,
                facecolor=style.FOREGROUND,
                alpha=alpha * locked,
                zorder=5,
            )
        )
    old_badge, new_badge = before.badges.get(block.id), after.badges.get(block.id)
    badges = (
        [(old_badge, 1.0)]
        if old_badge == new_badge
        else [
            (old_badge, 1 - t),
            (new_badge, t),
        ]
    )
    for badge, weight in badges:
        if badge and weight:
            ax.text(
                x + w,
                y + h,
                badge,
                ha="right",
                va="bottom",
                fontsize=8,
                fontfamily=style.FONT_FAMILY,
                color=style.FOREGROUND,
                alpha=alpha * weight,
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    edgecolor=style.FIXED,
                    alpha=alpha * weight,
                    facecolor=style.FIXED
                    if badge == style.CONCEPT_BADGE
                    else style.TRAINING,
                    linestyle=style.CONCEPT_LINESTYLE
                    if badge == style.CONCEPT_BADGE
                    else "-",
                ),
            )


def draw_architecture(
    ax: Axes,
    blocks: Sequence[Block],
    edges: Sequence[Edge],
    before: ArchitectureState,
    after: ArchitectureState,
    t: float,
) -> None:
    """Append an interpolated diagram; the caller clears axes and sets limits.

    ``t`` is in [0, 1]. Active edges use traveling dashes, looping once over this
    interval, so both endpoints exactly match a standalone drawing of the state
    at phase zero. Repeated identical states still animate their active edges.
    This function does not change matplotlib's backend or create a figure.
    """
    if isinstance(t, bool) or not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be finite and in [0, 1]")
    by_id = {block.id: block for block in blocks}
    if len(by_id) != len(blocks):
        raise ValueError("block ids must be unique")
    edge_ids = {(edge.src, edge.dst) for edge in edges}
    if len(edge_ids) != len(edges):
        raise ValueError("edges must be unique")
    if any(src not in by_id or dst not in by_id for src, dst in edge_ids):
        raise ValueError("edge references an unknown block")
    for state in (before, after):
        _validate_state(state, set(by_id), edge_ids)
    for block_id in by_id:
        if after.blocks.get(block_id) == "removed" and before.blocks.get(
            block_id, "hidden"
        ) not in ("training", "removed", "hidden"):
            raise ValueError(
                "removed supports training blocks only; the API has no base color"
            )
    ports = [_ports(by_id[edge.src], by_id[edge.dst]) for edge in edges]
    ax.set_facecolor(style.BACKGROUND)
    ax.set_axis_off()
    for edge, (start, stop) in zip(edges, ports, strict=True):
        visibility = []
        for state in (before, after):
            visibility.append(
                min(
                    _appearance(state.blocks.get(block_id, "hidden"))[1]
                    for block_id in (edge.src, edge.dst)
                )
            )
        alpha = _mix(visibility[0], visibility[1], t)
        if alpha == 0:
            continue
        concept = _mix(
            float(
                any(
                    before.badges.get(block_id) == style.CONCEPT_BADGE
                    for block_id in (edge.src, edge.dst)
                )
            ),
            float(
                any(
                    after.badges.get(block_id) == style.CONCEPT_BADGE
                    for block_id in (edge.src, edge.dst)
                )
            ),
            t,
        )
        active = _mix(
            float((edge.src, edge.dst) in before.active_edges),
            float((edge.src, edge.dst) in after.active_edges),
            t,
        )
        for weight, linestyle in (
            (1 - concept, "-"),
            (concept, style.CONCEPT_LINESTYLE),
        ):
            if weight:
                arrow = FancyArrowPatch(
                    start,
                    stop,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=style.FIXED,
                    alpha=alpha * weight,
                    linewidth=1.4,
                    linestyle=linestyle,
                    zorder=1,
                )
                arrow.set_gid(f"edge:{edge.src}->{edge.dst}")
                ax.add_patch(arrow)
        if active:
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    stop,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=style.FOREGROUND,
                    alpha=alpha * active,
                    linewidth=2,
                    linestyle=((t % 1) * 12, (4, 4)),
                    zorder=1.5,
                )
            )
    for block in blocks:
        _draw_block(ax, block, before, after, t)
