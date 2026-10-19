"""Measured pre/posttraining curves and an explicitly conceptual learning loop."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.visualization.presentation.configuration import (
    SEGMENT_NAMES,
    LearningStagesConfig,
)
from src.tasks.ball_detection.visualization.presentation.input_panel import (
    InputPanel,
    real_clip_panel,
    schematic_panel,
)
from src.tasks.ball_detection.visualization.presentation.stage_logs import (
    StageLogs,
    StageRun,
    load_stage_logs,
)
from src.tasks.base.visualization.presentation import (
    ArchitectureState,
    Block,
    BlockState,
    Edge,
    Segment,
    StepClock,
    draw_architecture,
    draw_progressive_curves,
    render_presentation,
    style,
)
from src.utils.paths import PROJECT_ROOT

__all__ = [
    "LearningStagesPresentation",
    "StageLogs",
    "ball_architecture",
    "load_stage_logs",
    "run_learning_stages",
]

_CHAIN1 = frozenset(
    {("input", "mdd"), ("mdd", "cnn"), ("cnn", "dpt"), ("dpt", "heatmap")}
)
_CHAIN2 = frozenset(
    {
        ("input", "mdd"),
        ("mdd", "cnn"),
        ("cnn", "query"),
        ("query", "coords"),
        ("aug", "input"),
    }
)
_CYCLE = ("pre_infer", "codex", "dataset", "retrain", "adopt", "pre_infer")
_CONCEPT_IDS = ("new_clips", "pre_infer", "codex", "dataset", "retrain", "adopt")


def ball_architecture() -> tuple[
    tuple[Block, ...], tuple[Edge, ...], Mapping[str, ArchitectureState]
]:
    """Abstract architecture/state vocabulary; scheduling comes from real logs."""
    blocks = (
        Block("input", "RGB 32frame", (0.6, 8.9), (4.3, 0.9)),
        Block("mdd", "MDD", (6.5, 8.9), (4.3, 0.9)),
        Block("cnn", "CNN", (4.25, 5.7), (3.0, 1.1)),
        Block("dpt", "DPT", (0.8, 3.6), (3.4, 1.1)),
        Block("heatmap", "Heatmap", (0.8, 1.5), (3.4, 1.1)),
        Block("query", "Query Transformer", (6.4, 3.6), (4.5, 1.1)),
        Block("coords", "座標", (6.9, 1.5), (3.5, 1.1)),
        Block("aug", "カメラ・遮蔽の拡張", (0.6, 6.2), (3.2, 0.75)),
        Block("model", "採用モデル", (4.25, 5.1), (3.1, 1.5)),
        Block("new_clips", "新規クリップ", (0.45, 8.4), (2.8, 1.0)),
        Block("pre_infer", "事前推論", (4.25, 8.4), (3.1, 1.0)),
        Block("codex", "Codex修正", (8.45, 6.4), (2.7, 1.0)),
        Block("dataset", "データセット", (8.45, 3.5), (2.7, 1.0)),
        Block("retrain", "追加学習", (4.25, 1.5), (3.1, 1.0)),
        Block("adopt", "採用", (0.5, 3.5), (2.7, 1.0)),
    )
    concept_edges = frozenset(zip(_CYCLE[:-1], _CYCLE[1:], strict=True)) | {
        ("new_clips", "pre_infer"),
        ("model", "pre_infer"),
    }
    edges = tuple(
        Edge(src, dst) for src, dst in sorted(_CHAIN1 | _CHAIN2 | concept_edges)
    )
    intro: dict[str, BlockState] = {"input": "fixed", "mdd": "fixed"}
    first: dict[str, BlockState] = {
        **intro,
        "cnn": "training",
        "dpt": "training",
        "heatmap": "training",
    }
    handoff: dict[str, BlockState] = {
        **first,
        "dpt": "removed",
        "heatmap": "removed",
        "query": "new",
        "coords": "new",
    }
    second: dict[str, BlockState] = {
        **handoff,
        "cnn": "training",
        "query": "training",
        "coords": "training",
        "aug": "training",
    }
    concept: dict[str, BlockState] = {
        block_id: "fixed" for block_id in ("model", *_CONCEPT_IDS)
    }
    states = {
        "intro": ArchitectureState(intro, frozenset({("input", "mdd")})),
        "phase1": ArchitectureState(first, _CHAIN1),
        "handoff12": ArchitectureState(
            handoff, frozenset({("input", "mdd"), ("mdd", "cnn")})
        ),
        "phase2": ArchitectureState(second, _CHAIN2),
        "handoff23": ArchitectureState(
            concept, frozenset(), {key: style.CONCEPT_BADGE for key in _CONCEPT_IDS}
        ),
    }
    states["phase3"] = states["outro"] = states["handoff23"]
    return blocks, edges, states


class LearningStagesPresentation:
    """Own the Agg canvas; render any frame deterministically without a GPU."""

    def __init__(
        self, config: LearningStagesConfig, logs: StageLogs, panel: InputPanel
    ) -> None:
        self.config, self.logs, self.panel = config, logs, panel
        self.blocks, self.edges, self.states = ball_architecture()
        first, second = logs.pretraining.steps, logs.posttraining.steps
        self.total_steps = first + second
        ranges = (
            (0, 0),
            (0, first),
            (logs.parent_step, logs.parent_step),
            (first, self.total_steps),
            (self.total_steps, self.total_steps),
            (self.total_steps, self.total_steps),
            (self.total_steps, self.total_steps),
        )
        self.clock = StepClock(
            [
                Segment(name, config.durations[name], *steps)
                for name, steps in zip(SEGMENT_NAMES, ranges, strict=True)
            ],
            config.fps,
        )
        self.segment_frames: dict[str, list[int]] = {name: [] for name in SEGMENT_NAMES}
        for index in range(self.clock.frame_count()):
            self.segment_frames[self.clock.at(index)[0].name].append(index)
        dpi = config.width / 12.8
        self.figure = Figure(
            figsize=(config.width / dpi, config.height / dpi),
            dpi=dpi,
            facecolor=style.BACKGROUND,
        )
        self.canvas = FigureCanvasAgg(self.figure)
        self.diagram = self.figure.add_axes((0, 0, 1150 / 1920, 1))
        self.error_axes = self.figure.add_axes(
            (1245 / 1920, 570 / 1080, 610 / 1920, 430 / 1080)
        )
        self.loss_axes = (
            self.figure.add_axes((1245 / 1920, 295 / 1080, 255 / 1920, 170 / 1080)),
            self.figure.add_axes((1600 / 1920, 295 / 1080, 255 / 1920, 170 / 1080)),
        )
        self.state_axes = self.figure.add_axes(
            (1245 / 1920, 205 / 1080, 610 / 1920, 35 / 1080)
        )
        self.lr_axes = self.figure.add_axes(
            (1245 / 1920, 80 / 1080, 610 / 1920, 70 / 1080)
        )

    def still_frames(self) -> dict[str, int]:
        return {
            "intro": self.segment_frames["intro"][-1],
            "phase1": self.segment_frames["phase1"][
                math.ceil(
                    (len(self.segment_frames["phase1"]) - 1)
                    * self.logs.parent_step
                    / self.logs.pretraining.steps
                )
            ],
            "handoff12": self.segment_frames["handoff12"][-1],
            "phase2": self.segment_frames["phase2"][-1],
            "phase3": self.segment_frames["phase3"][-1],
        }

    def architecture_at(self, frame: int) -> ArchitectureState:
        segment, _, step = self.clock.at(frame)
        state = self.states[segment.name]
        if segment.name == "phase2":
            frozen = self.logs.posttraining.frozen_at(
                step - self.logs.pretraining.steps
            )
            blocks = dict(state.blocks, cnn="frozen" if frozen else "training")
            ratio = self.logs.posttraining.recipe["encoder_lr_ratio"]
            badges = {} if frozen else {"cnn": f"×{ratio:g}"}
            return ArchitectureState(blocks, state.active_edges, badges)
        return state

    def cursor_at(self, frame: int) -> int:
        segment, _, step = self.clock.at(frame)
        if segment.name == "intro":
            return 0
        if segment.name == "handoff12":
            return self.logs.pretraining.steps
        return int(step)

    def _diagram_frame(self, frame: int) -> None:
        segment, progress, _ = self.clock.at(frame)
        state = self.architecture_at(frame)
        before, transition = state, (frame / self.config.fps) % 1
        if segment.name == "intro":
            before, transition = ArchitectureState({}, frozenset()), progress
        elif segment.name == "phase1" and progress < 0.06:
            before, transition = self.states["intro"], progress / 0.06
        elif segment.name == "handoff12":
            before, transition = self.states["phase1"], progress
        elif segment.name == "handoff23":
            before = self.architecture_at(self.segment_frames["phase2"][-1])
            transition = progress
        self.diagram.clear()
        self.diagram.set_xlim(0, 11.5)
        self.diagram.set_ylim(0, 10.8)
        self.diagram.set_aspect("equal")
        draw_architecture(
            self.diagram, self.blocks, self.edges, before, state, transition
        )
        alpha = (
            progress
            if segment.name == "intro"
            else 1 - progress
            if segment.name == "handoff23"
            else 0
            if segment.name in ("phase3", "outro")
            else 1
        )
        if alpha:
            for start, images in ((0.65, self.panel.rgb), (6.55, self.panel.mdd)):
                for index, image in enumerate(images):
                    x = start + index * 1.43
                    self.diagram.imshow(
                        image,
                        extent=(x, x + 1.33, 9.9, 10.7),
                        alpha=alpha,
                        aspect="auto",
                        zorder=6,
                    )
        if segment.name in ("handoff23", "phase3", "outro"):
            # One visible pulse makes a full circuit in 3 seconds; no metrics
            # are extrapolated into this conceptual section.
            phase = (
                (frame - self.segment_frames["handoff23"][0]) / self.config.fps / 3
            ) % 1
            position = phase * (len(_CYCLE) - 1)
            edge = min(int(position), len(_CYCLE) - 2)
            lookup = {block.id: block for block in self.blocks}
            points = [
                np.array(block.xy) + np.array(block.wh) / 2
                for block in (lookup[_CYCLE[edge]], lookup[_CYCLE[edge + 1]])
            ]
            xy = (1 - position % 1) * points[0] + (position % 1) * points[1]
            self.diagram.add_patch(
                Circle(
                    tuple(xy),
                    0.065,
                    color=style.FOREGROUND,
                    alpha=progress if segment.name == "handoff23" else 1,
                    zorder=2,
                )
            )

    def _axes(
        self,
        ax: Axes,
        *,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        ylabel: str,
    ) -> None:
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(
            ylabel, color=style.FOREGROUND, fontfamily=style.FONT_FAMILY, fontsize=10
        )
        ax.set_facecolor(style.BACKGROUND)
        ax.tick_params(colors=style.FOREGROUND, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(style.FIXED)
        ax.spines[["top", "right"]].set_visible(False)

    def _curves(self, frame: int) -> None:
        cursor = self.cursor_at(frame)
        first, second = self.logs.pretraining, self.logs.posttraining
        limit = self.total_steps * 1.1
        spans = (
            (0, first.steps, "事前学習"),
            (first.steps, self.total_steps, "事後学習"),
            (self.total_steps, int(limit), style.CONCEPT_BADGE),
        )
        self._axes(
            self.error_axes,
            xlim=(0, limit),
            ylim=(0, 60),
            ylabel="Common 平均誤差 [px]",
        )
        self.error_axes.set_prop_cycle(color=[style.TRAINING, style.NEW])
        draw_progressive_curves(
            self.error_axes,
            (first.common, second.common),
            cursor,
            spans=spans,
            markers=(first.best_point, second.best_point),
        )
        # Each validation point appears only at its logged epoch boundary.
        for line in self.error_axes.lines[:2]:
            line.set_marker("o")
            line.set_markersize(4)
        self.error_axes.set_yticks([0, 20, 40, 60])
        ticks = [0, first.steps, self.total_steps]
        labels = ["0", f"{first.steps // 1000}k", f"{self.total_steps // 1000}k"]
        self.error_axes.set_xticks(ticks, labels)
        for start, stop, label in spans:
            self.error_axes.text(
                (start + stop) / 2,
                63,
                label,
                color=style.FOREGROUND,
                fontfamily=style.FONT_FAMILY,
                fontsize=9,
                ha="center",
                va="bottom",
            )
        name, progress, _ = self.clock.at(frame)
        if name.name == "handoff12" or cursor > first.steps:
            amount = progress if name.name == "handoff12" else 1
            start, error, _ = first.best_point
            self.error_axes.add_patch(
                FancyArrowPatch(
                    (start, error),
                    (start + (first.steps - start) * max(amount, 0.001), error),
                    connectionstyle="arc3,rad=-0.6",
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linestyle=style.CONCEPT_LINESTYLE,
                    color=style.NEW,
                    linewidth=1.6,
                    zorder=5,
                )
            )
        # Independent normalized axes: heatmap and coordinate losses have
        # different objectives/units, and are never joined across the handoff.
        for ax, run, color in zip(
            self.loss_axes, (first, second), (style.TRAINING, style.NEW), strict=True
        ):
            self._axes(
                ax,
                xlim=(run.offset, run.offset + run.steps),
                ylim=(0, 1.05),
                ylabel=f"{run.loss.name} loss\n[段階内正規化]",
            )
            ax.set_prop_cycle(color=[color])
            draw_progressive_curves(ax, (run.loss,), cursor, spans=())
            ax.set_yticks([0, 1])
            ax.set_xticks(
                [run.offset, run.offset + run.steps],
                [f"{run.offset // 1000}k", f"{(run.offset + run.steps) // 1000}k"],
            )
        self._state_strip(cursor, limit)
        maximum_lr = max(
            float(run.learning_rate.values.max()) for run in (first, second)
        )
        self._axes(
            self.lr_axes, xlim=(0, limit), ylim=(0, maximum_lr * 1.15), ylabel="学習率"
        )
        self.lr_axes.set_prop_cycle(
            color=[style.TRAINING, style.TRAINING, style.FROZEN]
        )
        if second.encoder_learning_rate is None:
            raise ValueError("Posttraining encoder learning rate is required")
        draw_progressive_curves(
            self.lr_axes,
            (first.learning_rate, second.learning_rate, second.encoder_learning_rate),
            cursor,
            spans=spans,
        )
        self.lr_axes.lines[2].set_linestyle("--")
        self.lr_axes.set_xticks(ticks, labels)
        self.lr_axes.set_xlabel(
            "累積更新数",
            color=style.FOREGROUND,
            fontfamily=style.FONT_FAMILY,
            fontsize=10,
        )
        self.lr_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        self.lr_axes.yaxis.get_offset_text().set_color(style.FOREGROUND)
        self.lr_axes.yaxis.get_offset_text().set_fontsize(8)

    def _state_strip(self, cursor: int, limit: float) -> None:
        self._axes(self.state_axes, xlim=(0, limit), ylim=(0, 1), ylabel="CNN")
        self.state_axes.set_yticks([])
        self.state_axes.set_xticks([])
        for run in (self.logs.pretraining, self.logs.posttraining):
            starts = np.concatenate(([run.offset], run.learning_rate.steps[:-1]))
            # Merge adjacent summary intervals of the same state into one band.
            changes = np.flatnonzero(
                np.r_[True, run.encoder_frozen[1:] != run.encoder_frozen[:-1], True]
            )
            for start, stop in zip(changes[:-1], changes[1:], strict=True):
                left = int(starts[start])
                right = min(cursor, int(run.learning_rate.steps[stop - 1]))
                if right > left:
                    self.state_axes.axvspan(
                        left,
                        right,
                        color=style.FROZEN
                        if run.encoder_frozen[start]
                        else style.TRAINING,
                    )
        self.state_axes.axvspan(
            self.total_steps, limit, color=style.FIXED, alpha=0.22, hatch="//"
        )

    def render_frame(self, frame: int) -> NDArray[np.uint8]:
        segment, _, _ = self.clock.at(frame)
        if segment.name == "outro":
            frame = self.segment_frames["phase3"][-1]
        self._diagram_frame(frame)
        self._curves(frame)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def _stage_manifest(run: StageRun) -> dict[str, Any]:
    return {
        "origin": "measured",
        "offset": run.offset,
        "metrics_rows": len(run.metrics),
        "training_rows": len(run.training),
        "best": run.best,
        "final_common_mean_error_px": float(run.common.values[-1]),
        "final_full_mean_error_px": run.metrics[-1]["scopes"]["full"][
            "macro_mean_error_px"
        ],
    }


def run_learning_stages(
    config: LearningStagesConfig, resolved_config: DictConfig
) -> LearningStagesPresentation:
    """Read evidence, render all three formats and record reproducible provenance."""
    if config.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {config.output_dir}")
    logs = load_stage_logs(config.pretraining, config.posttraining)
    if config.input_panel == "schematic":
        panel = schematic_panel()
    else:
        if config.input_clip is None:
            raise ValueError("real_clip requires input_clip")
        panel = real_clip_panel(
            config.input_clip, logs.posttraining.recipe["parent"]["input_contract"]
        )
    presentation = LearningStagesPresentation(config, logs, panel)
    manifest = {
        "schema": "ball_learning_stages_presentation.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        ),
        "config": OmegaConf.to_container(resolved_config, resolve=True),
        "input_files": dict(logs.input_files, **panel.input_files),
        "checkpoints": {
            "phase1": logs.pretraining.best["sha256"],
            "phase2": logs.posttraining.best["sha256"],
        },
        "elements": {
            "phase1": _stage_manifest(logs.pretraining),
            "phase2": _stage_manifest(logs.posttraining),
            "architecture": {
                "origin": "schematic",
                "detail": "abstract heatmap to query transfer",
            },
            "input_panel": panel.provenance,
            "phase3": {"origin": "concept", "metrics": None},
        },
        "handoff": {
            "parent_global_step": logs.parent_step,
            "phase2_axis_start": logs.posttraining.offset,
            "initial_phase2_metric": "first measured point is epoch 0 end; no step-0 measurement",
        },
        "loss": {
            "smoothing": "trailing mean",
            "window": 20,
            "normalization": "independent maximum of each stage's smoothed loss",
        },
        "segments": [
            asdict(presentation.clock.at(frames[0])[0])
            for frames in presentation.segment_frames.values()
        ],
        "stills": presentation.still_frames(),
        "frame_count": presentation.clock.frame_count(),
        "fps": config.fps,
        "review_status": "candidate; user selection pending",
    }
    render_presentation(
        presentation.render_frame,
        frame_count=presentation.clock.frame_count(),
        fps=config.fps,
        output_dir=config.output_dir,
        name="learning_stages",
        gif_width=config.gif_width,
        gif_fps=config.gif_fps,
        stills=presentation.still_frames(),
        manifest=manifest,
    )
    return presentation
