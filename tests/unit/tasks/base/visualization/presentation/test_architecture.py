"""State transitions, endpoint pixel equality and invalid graph contracts."""

from __future__ import annotations

from colorsys import rgb_to_hsv
from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from src.tasks.base.visualization.presentation import style
from src.tasks.base.visualization.presentation.architecture import (
    ArchitectureState,
    Block,
    BlockState,
    Edge,
    draw_architecture,
    draw_flow_particles,
)

pytestmark = pytest.mark.unit


def test_flow_particle_moves_by_path_length_and_loops_at_endpoints() -> None:
    ax = _axes()
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 3.0)]
    draw_flow_particles(ax, path, 0.5, "#4CC9F0")
    np.testing.assert_allclose(ax.collections[-1].get_offsets()[0], [1.0, 1.0])
    draw_flow_particles(ax, path, 0.0, "#4CC9F0")
    start = ax.collections[-1].get_offsets().copy()
    draw_flow_particles(ax, path, 1.0, "#4CC9F0")
    np.testing.assert_allclose(ax.collections[-1].get_offsets(), start)


@pytest.mark.parametrize(
    "path,phase",
    [
        ([], 0.2),
        ([(0, 0)], 0.2),
        ([(0, 0), (0, 0)], 0.2),
        ([(0, 0), (float("nan"), 1)], 0.2),
        ([(0, 0), (1, 1)], -0.1),
        ([(0, 0), (1, 1)], float("inf")),
        ([(0, 0), (1, 1)], True),
    ],
)
def test_invalid_flow_particles_fail_before_drawing(
    path: list[tuple[float, float]], phase: float
) -> None:
    ax = _axes()
    with pytest.raises(ValueError):
        draw_flow_particles(ax, path, phase, style.TRAINING)
    assert not ax.collections


BLOCKS = [
    Block("input", "RGB", (0.2, 0.5), (1.5, 1)),
    Block("model", "Encoder", (2.5, 0.5), (2, 1)),
]
EDGES = [Edge("input", "model")]


def _state(state: BlockState, badge: str | None = None) -> ArchitectureState:
    return ArchitectureState(
        {"input": "fixed", "model": state},
        frozenset() if state == "hidden" else frozenset({("input", "model")}),
        {} if badge is None else {"model": badge},
    )


def _axes() -> Axes:
    figure = Figure(figsize=(5, 2.2), dpi=64, facecolor=style.BACKGROUND)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2.2)
    return ax


def _pixels(
    before: ArchitectureState, after: ArchitectureState, t: float
) -> np.ndarray:
    ax = _axes()
    draw_architecture(ax, BLOCKS, EDGES, before, after, t)
    canvas = ax.figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    pixels: np.ndarray = np.asarray(canvas.buffer_rgba()).copy()
    return pixels


@pytest.mark.parametrize(
    "before,after",
    [
        (_state("hidden"), _state("training")),
        (_state("training"), _state("frozen")),
        (_state("new"), _state("training", "×0.1")),
        (_state("training", "×0.1"), _state("hidden")),
        (_state("fixed"), _state("new", style.CONCEPT_BADGE)),
        (_state("frozen", "LoRA"), _state("training")),
        (_state("training"), _state("removed")),
        (_state("removed"), _state("hidden")),
    ],
)
def test_transition_endpoints_equal_standalone_state_pixels(
    before: ArchitectureState,
    after: ArchitectureState,
) -> None:
    np.testing.assert_array_equal(_pixels(before, after, 0), _pixels(before, before, 0))
    np.testing.assert_array_equal(_pixels(before, after, 1), _pixels(after, after, 0))


def test_hidden_to_training_fades_in_without_changing_training_color() -> None:
    ax = _axes()
    draw_architecture(ax, BLOCKS, EDGES, _state("hidden"), _state("training"), 0.5)
    model = next(patch for patch in ax.patches if patch.get_gid() == "block:model")
    assert model.get_facecolor() == (*to_rgb(style.TRAINING), 0.5)
    assert not np.array_equal(
        _pixels(_state("hidden"), _state("training"), 0.5),
        _pixels(_state("hidden"), _state("training"), 0),
    )


def test_frozen_lock_new_glow_and_concept_badge_are_visible() -> None:
    training = _pixels(_state("training"), _state("training"), 0)
    frozen = _pixels(_state("frozen"), _state("frozen"), 0)
    new = _pixels(_state("new"), _state("new"), 0)
    assert not np.array_equal(training, frozen)
    assert not np.array_equal(training, new)
    ax = _axes()
    state = _state("new", style.CONCEPT_BADGE)
    draw_architecture(ax, BLOCKS, EDGES, state, state, 0)
    assert any(text.get_text() == style.CONCEPT_BADGE for text in ax.texts)
    assert any(patch.get_linestyle() == "--" for patch in ax.patches)
    assert len(ax.patches) > 4  # edges, fills, outlines and glow


def test_removed_training_block_is_dimmed_and_desaturated() -> None:
    ax = _axes()
    draw_architecture(ax, BLOCKS, EDGES, _state("training"), _state("removed"), 1)
    model = next(patch for patch in ax.patches if patch.get_gid() == "block:model")
    color = model.get_facecolor()
    assert color[3] == 0.2
    assert rgb_to_hsv(*color[:3])[1] < rgb_to_hsv(*to_rgb(style.TRAINING))[1]


@pytest.mark.parametrize("state", ["fixed", "frozen", "new"])
def test_removal_of_other_base_colors_is_rejected(state: BlockState) -> None:
    ax = _axes()
    with pytest.raises(ValueError, match="training blocks only"):
        draw_architecture(ax, BLOCKS, EDGES, _state(state), _state("removed"), 0.5)
    assert not ax.patches


def test_active_edge_animation_changes_only_between_loop_endpoints() -> None:
    state = _state("training")
    assert not np.array_equal(_pixels(state, state, 0), _pixels(state, state, 0.4))
    np.testing.assert_array_equal(_pixels(state, state, 0), _pixels(state, state, 1))


def test_omitted_blocks_are_hidden() -> None:
    state = ArchitectureState({"input": "fixed"}, frozenset())
    np.testing.assert_array_equal(
        _pixels(state, state, 0), _pixels(_state("hidden"), _state("hidden"), 0)
    )


@pytest.mark.parametrize("t", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_interpolation_parameter_is_rejected(t: float) -> None:
    with pytest.raises(ValueError, match="t must"):
        draw_architecture(_axes(), BLOCKS, EDGES, _state("fixed"), _state("fixed"), t)


@pytest.mark.parametrize(
    "state",
    [
        ArchitectureState({"unknown": "training"}, frozenset()),
        ArchitectureState({"input": "fixed"}, frozenset({("input", "unknown")})),
        ArchitectureState({"input": "fixed"}, frozenset({("input", "model")})),
        ArchitectureState({"input": "fixed"}, frozenset(), {"model": "LoRA"}),
    ],
)
def test_invalid_state_references_fail_before_drawing(state: ArchitectureState) -> None:
    ax = _axes()
    with pytest.raises(ValueError):
        draw_architecture(ax, BLOCKS, EDGES, state, _state("training"), 0)
    assert not ax.patches


@pytest.mark.parametrize(
    "blocks,edges",
    [
        (BLOCKS * 2, EDGES),
        (BLOCKS, EDGES * 2),
        (BLOCKS, [Edge("input", "unknown")]),
        ([BLOCKS[0], Block("model", "Encoder", BLOCKS[0].xy, BLOCKS[0].wh)], EDGES),
    ],
)
def test_invalid_graph_is_rejected(blocks: list[Block], edges: list[Edge]) -> None:
    with pytest.raises(ValueError):
        draw_architecture(_axes(), blocks, edges, _state("fixed"), _state("fixed"), 0)


def test_bad_state_and_geometry_types_are_rejected() -> None:
    invalid_state: Any = "typo"
    with pytest.raises(ValueError, match="unknown block state"):
        ArchitectureState({"model": invalid_state}, frozenset())
    not_frozen: Any = set()
    with pytest.raises(TypeError, match="frozenset"):
        ArchitectureState({}, not_frozen)
    for xy, wh in [((float("inf"), 0), (1, 1)), ((0, 0), (0, 1))]:
        with pytest.raises(ValueError):
            Block("bad", "bad", xy, wh)
    with pytest.raises(ValueError, match="distinct"):
        Edge("model", "model")
