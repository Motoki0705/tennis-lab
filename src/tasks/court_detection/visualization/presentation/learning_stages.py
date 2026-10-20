"""Measured court learning curves beside a schematic supervision diagram."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.visualization.presentation import (
    ArchitectureState,
    Block,
    BlockState,
    CurveSeries,
    Edge,
    Segment,
    StepClock,
    draw_architecture,
    draw_flow_particles,
    draw_progressive_curves,
    render_presentation,
    style,
)
from src.tasks.court_detection.visualization.presentation.configuration import (
    TAG_SCHEMA,
    CourtPresentationRequest,
    default_scalar_summary,
    default_tags,
)
from src.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class CourtRunLog:
    bundle: Path
    config: DictConfig
    series: Mapping[str, CurveSeries]
    tags: Mapping[str, str]
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int
    best_step: int
    best_epoch: int
    best_checkpoint: str
    checkpoint_sha256: str
    input_sha256: Mapping[str, str]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_b_config(config: DictConfig) -> None:
    expected = {
        "model.encoder.train_mode": "frozen",
        "model.encoder.backbone_name": "dinov3_vitb16",
        "model.encoder.out_indices": [2, 5, 8, 11],
        "model.encoder.lora.enabled": False,
        "model.transformer_encoder.dim": 1024,
        "model.transformer_encoder.depth": 8,
        "mixed.train_batch_counts.synthetic_court": 4,
        "mixed.train_batch_counts.tennis_court_detector": 4,
        "training.warmup_epochs": 3,
        "training.trainer.max_epochs": 20,
        "training.trainer.check_val_every_n_epoch": 3,
        "loss.pose.enabled": True,
    }
    for key, value in expected.items():
        actual = OmegaConf.select(config, key)
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(
                f"B run config mismatch at {key}: expected {value!r}, got {actual!r}"
            )


def _read_scalar_summary(bundle: Path, summary: Path) -> Mapping[str, Any]:
    payload = json.loads(summary.read_text())
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "kind", "source_sha256", "scalars"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "court_learning_scalars"
        or not isinstance(payload["scalars"], dict)
    ):
        raise ValueError("Invalid court learning scalar summary schema")
    sources = json.loads((bundle / "tensorboard-sources.json").read_text())
    training = [source for source in sources["sources"] if source["role"] == "training"]
    if len(training) != 1 or payload["source_sha256"] != training[0]["sha256"]:
        raise ValueError(
            "Scalar summary source SHA-256 differs from the recorded training event"
        )
    for tag, values in payload["scalars"].items():
        if (
            not isinstance(tag, str)
            or not isinstance(values, dict)
            or set(values) != {"steps", "values", "wall_times"}
            or any(not isinstance(value, list) for value in values.values())
            or len({len(value) for value in values.values()}) != 1
            or any(type(step) is not int for step in values["steps"])
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for key in ("values", "wall_times")
                for value in values[key]
            )
        ):
            raise ValueError(f"Invalid scalar arrays for {tag}")
    return cast(Mapping[str, Any], payload["scalars"])


def load_court_run(
    bundle: Path,
    *,
    tags: Mapping[str, str] | None = None,
    scalar_summary: Path | None = None,
) -> CourtRunLog:
    """Load the compact measured summary without sampling or tag fallback.

    Train/val event steps are zero-based update indices; +1 converts them to
    completed optimizer updates. LR-monitor already records epoch-start update
    counts and is left unchanged. The raw inputs are always hashed unchanged.
    When tags are omitted, their sole default authority is the task YAML.
    """
    tag_map = cast(
        Mapping[str, str], TAG_SCHEMA.validate(default_tags() if tags is None else tags)
    )
    config = cast(DictConfig, OmegaConf.load(bundle / "config.yaml"))
    _validate_b_config(config)
    selection = json.loads((bundle / "checkpoint-selection.json").read_text())
    total = selection["final_global_step"]
    epochs = int(config.training.trainer.max_epochs)
    if (
        type(total) is not int
        or total <= 0
        or total % epochs
        or selection["final_epoch"] != epochs - 1
    ):
        raise ValueError("Checkpoint selection has an inconsistent final step/epoch")
    per_epoch = total // epochs
    summary = default_scalar_summary() if scalar_summary is None else scalar_summary
    scalars = _read_scalar_summary(bundle, summary)
    available = set(scalars)
    missing = set(tag_map.values()) - available
    if missing:
        raise ValueError(f"Missing required TensorBoard tags: {sorted(missing)}")
    series: dict[str, CurveSeries] = {}
    val_every = int(config.training.trainer.check_val_every_n_epoch)
    for name, tag in tag_map.items():
        values = scalars[tag]
        offset = 0 if name == "learning_rate" else 1
        steps = np.asarray(values["steps"], dtype=np.int64) + offset
        curve = CurveSeries(
            name,
            steps,
            np.asarray(values["values"], dtype=np.float64),
            "points" if name.startswith("val_") else "line",
        )
        expected_steps: NDArray[np.int64]
        if name.startswith("val_"):
            expected_steps = np.arange(val_every, epochs + 1, val_every) * per_epoch
        elif name.startswith("train_"):
            expected_steps = np.arange(50, total + 1, 50)
        else:
            expected_steps = np.arange(epochs) * per_epoch
            if np.any(curve.values <= 0):
                raise ValueError("Learning rates must be positive")
        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"Unexpected scalar count/steps for {tag}: expected {len(expected_steps)} points"
            )
        series[name] = curve
    best_checkpoint = selection["best_checkpoint"]
    match = re.search(r"epoch=(\d+)\.ckpt$", best_checkpoint)
    if match is None:
        raise ValueError("best_checkpoint must identify its epoch")
    best_epoch = int(match[1])
    best_step = (best_epoch + 1) * per_epoch
    validation = series["val_loss"]
    index = int(np.argmin(validation.values))
    if best_step != int(validation.steps[index]) or not math.isclose(
        selection["best_score"], float(validation.values[index]), rel_tol=1e-6
    ):
        raise ValueError(
            "Checkpoint selection differs from the minimum measured val/loss"
        )
    if selection["criterion"] != {"monitor": "val/loss", "mode": "min"}:
        raise ValueError("Checkpoint selection must minimize val/loss")
    checkpoint_hash = selection["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None
    ):
        raise ValueError("Checkpoint selection requires a SHA-256")
    evidence = [
        bundle / name
        for name in (
            "config.yaml",
            "checkpoint-selection.json",
            "tensorboard-derivation.json",
            "tensorboard-sources.json",
            "artifact-manifest.json",
        )
    ]
    evidence.extend([summary, summary.with_name("derivation.json")])
    return CourtRunLog(
        bundle=bundle,
        config=config,
        series=series,
        tags=tag_map,
        steps_per_epoch=per_epoch,
        total_steps=total,
        warmup_steps=int(config.training.warmup_epochs) * per_epoch,
        best_step=best_step,
        best_epoch=best_epoch,
        best_checkpoint=best_checkpoint,
        checkpoint_sha256=checkpoint_hash,
        input_sha256={str(path.resolve()): _sha256(path) for path in evidence},
    )


def court_architecture(
    config: DictConfig,
) -> tuple[list[Block], list[Edge], Mapping[str, ArchitectureState]]:
    """Build model states and labels from the saved training configuration."""
    encoder = config.model.encoder
    transformer = config.model.transformer_encoder
    counts = config.mixed.train_batch_counts
    backbone = re.fullmatch(r"dinov3_vit([a-z]+)(\d+)", str(encoder.backbone_name))
    if backbone is None:
        raise ValueError("Court diagram requires a DINOv3 ViT backbone")
    if not config.loss.pose.enabled:
        raise ValueError("This story requires the camera-pose head")
    blocks = [
        Block("rgb_syn", f"合成 ×{counts.synthetic_court}", (0.2, 6.1), (1.8, 1.1)),
        Block(
            "rgb_real",
            f"実画像 ×{counts.tennis_court_detector}",
            (0.2, 3.6),
            (1.8, 1.1),
        ),
        Block(
            "dinov3",
            f"DINOv3\nViT-{backbone[1].upper()}/{backbone[2]}",
            (2.8, 4.65),
            (2.1, 1.65),
        ),
        Block("proj", f"1×1\n×{len(encoder.out_indices)}", (5.65, 4.9), (1.55, 1.15)),
        Block(
            "transformer",
            f"Transformer\n{transformer.dim} × {transformer.depth}\npose query",
            (7.9, 4.65),
            (2.1, 1.65),
        ),
        Block("dpt", "DPT", (10.7, 4.9), (1.2, 1.15)),
    ]
    labels = {
        "kp": "KP14",
        "seg": "SEG",
        "line": "LINE",
        "semline": "線種",
        "pose": "カメラpose",
    }
    for head, y in zip(labels, (7.5, 6.0, 4.5, 3.0, 1.1), strict=True):
        blocks.append(Block(head, labels[head], (12.6, y), (1.7, 0.85)))
        blocks.append(Block(f"loss_{head}", " ", (15.1, y + 0.25), (0.35, 0.35)))
    edges = [
        Edge("rgb_syn", "dinov3"),
        Edge("rgb_real", "dinov3"),
        Edge("dinov3", "proj"),
        Edge("proj", "transformer"),
        Edge("transformer", "dpt"),
    ]
    edges.extend(
        Edge("transformer" if head == "pose" else "dpt", head) for head in labels
    )
    edges.extend(Edge(head, f"loss_{head}") for head in labels)
    states: dict[str, BlockState] = {block.id: "training" for block in blocks}
    states.update(rgb_syn="fixed", rgb_real="fixed")
    states["dinov3"] = (
        "frozen"
        if encoder.train_mode == "frozen" or encoder.lora.enabled
        else "training"
    )
    badges = {"dinov3": "LoRA"} if encoder.lora.enabled else {}
    training = ArchitectureState(
        states, frozenset((edge.src, edge.dst) for edge in edges), badges
    )
    handoff = ArchitectureState(
        {**states, "kp": "new", "line": "new"}, frozenset(), badges
    )
    return (
        blocks,
        edges,
        {
            "intro": ArchitectureState(states, frozenset(), badges),
            "warmup": training,
            "train": training,
            "select": ArchitectureState(states, frozenset(), badges),
            "handoff": handoff,
        },
    )


def supervision_paths(
    blocks: Sequence[Block],
) -> Mapping[str, tuple[tuple[float, float], ...]]:
    """Schematic paths: real pose ends at a gate, never at the pose loss."""
    centers = {
        block.id: (block.xy[0] + block.wh[0] / 2, block.xy[1] + block.wh[1] / 2)
        for block in blocks
    }
    paths: dict[str, tuple[tuple[float, float], ...]] = {}
    for source in ("rgb_syn", "rgb_real"):
        trunk = [
            centers[source],
            centers["dinov3"],
            centers["proj"],
            centers["transformer"],
        ]
        for head in style.COURT_HEAD_COLORS:
            branch = [] if head == "pose" else [centers["dpt"]]
            destination = centers[f"loss_{head}"]
            name = f"{source}:loss_{head}"
            if source == "rgb_real" and head == "pose":
                destination = (14.55, centers["pose"][1] - 0.13)
                name = "rgb_real:pose_gate"
            paths[name] = tuple([*trunk, *branch, centers[head], destination])
    return paths


def trailing_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Causal rolling mean, using the available prefix until a full window."""
    if window <= 0 or values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "A positive window and nonempty one-dimensional values are required"
        )
    cumulative = np.r_[0.0, np.cumsum(values)]
    stops = np.arange(1, len(values) + 1)
    starts = np.maximum(0, stops - window)
    return cast(
        NDArray[np.float64], (cumulative[stops] - cumulative[starts]) / (stops - starts)
    )


def learning_clock(
    run: CourtRunLog, seconds: Mapping[str, float], fps: float
) -> StepClock:
    return StepClock(
        [
            Segment("intro", seconds["intro"], 0, 0),
            Segment("warmup", seconds["warmup"], 0, run.warmup_steps),
            Segment("train", seconds["train"], run.warmup_steps, run.total_steps),
            Segment("select", seconds["select"], run.best_step, run.best_step),
            Segment("handoff", seconds["handoff"], run.best_step, run.best_step),
        ],
        fps,
    )


class CourtLearningStory:
    """Own an Agg figure; redraw frames without changing global rcParams."""

    def __init__(
        self,
        run: CourtRunLog,
        *,
        seconds: Mapping[str, float],
        fps: float,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("Story dimensions must be positive and even")
        self.run, self.fps = run, fps
        self.clock = learning_clock(run, seconds, fps)
        self.blocks, self.edges, self.states = court_architecture(run.config)
        self.paths = supervision_paths(self.blocks)
        self.figure = Figure(
            figsize=(width / 100, height / 100), dpi=100, facecolor=style.BACKGROUND
        )
        self.canvas = FigureCanvasAgg(self.figure)
        self.diagram = self.figure.add_axes((0.015, 0.12, 0.58, 0.78))
        self.metrics = [
            self.figure.add_axes(rect)
            for rect in (
                (0.66, 0.70, 0.135, 0.19),
                (0.85, 0.70, 0.135, 0.19),
                (0.66, 0.42, 0.135, 0.19),
                (0.85, 0.42, 0.135, 0.19),
            )
        ]
        self.loss = self.figure.add_axes((0.66, 0.20, 0.325, 0.13))
        self.lr = self.figure.add_axes((0.66, 0.085, 0.325, 0.065))
        raw = run.series["train_loss"]
        self.smoothed_loss = CurveSeries(
            "train_loss_ma20", raw.steps, trailing_mean(raw.values, 20), "line"
        )
        groups: dict[str, list[int]] = {name: [] for name in self.states}
        for frame in range(self.clock.frame_count()):
            groups[self.clock.at(frame)[0].name].append(frame)
        self.segments = [self.clock.at(frames[0])[0] for frames in groups.values()]
        self.stills = {
            name: frames[-1]
            if name in {"intro", "handoff"}
            else frames[round((len(frames) - 1) * 0.65)]
            for name, frames in groups.items()
            if name != "warmup"
        }

    def _intro(
        self, progress: float
    ) -> tuple[ArchitectureState, ArchitectureState, float]:
        groups = [
            ("rgb_syn", "rgb_real"),
            ("dinov3",),
            ("proj",),
            ("transformer",),
            ("dpt",),
            tuple(style.COURT_HEAD_COLORS),
            tuple(f"loss_{head}" for head in style.COURT_HEAD_COLORS),
        ]
        position = progress * len(groups)
        completed = min(len(groups) - 1, int(position))
        before_ids = {key for group in groups[:completed] for key in group}
        after_ids = before_ids | set(groups[completed])
        full = self.states["intro"]

        def state(ids: set[str]) -> ArchitectureState:
            return ArchitectureState(
                {key: full.blocks[key] for key in ids},
                frozenset(),
                {key: badge for key, badge in full.badges.items() if key in ids},
            )

        return state(before_ids), state(after_ids), min(1.0, position - completed)

    def _decorate_diagram(self, name: str, progress: float) -> None:
        by_xy = {block.xy: block for block in self.blocks}
        by_label = {block.label: block.id for block in self.blocks}
        palette = {
            **style.COURT_HEAD_COLORS,
            "rgb_syn": style.SYNTHETIC,
            "rgb_real": style.REAL,
        }
        palette.update(
            {f"loss_{head}": color for head, color in style.COURT_HEAD_COLORS.items()}
        )
        dim = 1 - 0.82 * progress if name == "handoff" else 1.0
        for patch in self.diagram.patches:
            factor = dim
            if isinstance(patch, FancyBboxPatch):
                block = by_xy[(patch.get_x(), patch.get_y())]
                if block.id in {"kp", "line"}:
                    factor = 1.0
                if block.id.startswith("loss_"):
                    patch.set_boxstyle("circle,pad=0")
                if block.id in palette:
                    if patch.get_facecolor()[3] > 0:
                        patch.set_facecolor(palette[block.id])
                    if block.id in {"kp", "line"} and name == "handoff":
                        patch.set_edgecolor(palette[block.id])
            alpha = patch.get_alpha()
            patch.set_alpha((1.0 if alpha is None else alpha) * factor)
        for text in self.diagram.texts:
            alpha = text.get_alpha()
            block_id = by_label.get(text.get_text())
            text.set_fontsize(14)
            if block_id in palette:
                text.set_color(style.BACKGROUND)
            factor = 1.0 if block_id in {"kp", "line"} else dim
            text.set_alpha((1.0 if alpha is None else alpha) * factor)

    def _diagram_frame(self, frame: int, segment: Segment, progress: float) -> None:
        ax = self.diagram
        ax.clear()
        ax.set_xlim(-0.1, 16)
        ax.set_ylim(0, 10)
        name = segment.name
        if name == "intro":
            before, after, t = self._intro(progress)
        elif name == "handoff":
            before, after, t = self.states["select"], self.states["handoff"], progress
        else:
            before = after = self.states[name]
            t = (frame / self.fps / 1.2) % 1
        draw_architecture(ax, self.blocks, self.edges, before, after, t)
        self._decorate_diagram(name, progress)
        if name in {"warmup", "train"}:
            for route, points in self.paths.items():
                synthetic = route.startswith("rgb_syn:")
                phase = (frame / self.fps / 1.2 + (0 if synthetic else 0.16)) % 1
                draw_flow_particles(
                    ax, points, phase, style.SYNTHETIC if synthetic else style.REAL
                )
            # Two closed bars: real RGB reaches the pose prediction, but its
            # supervision terminates here, before the pose loss node.
            y = self.paths["rgb_real:pose_gate"][-1][1]
            ax.plot(
                [14.62, 14.62],
                [y - 0.18, y + 0.18],
                color=style.REAL,
                linewidth=3,
                zorder=9,
            )
            ax.plot(
                [14.72, 14.72],
                [y - 0.18, y + 0.18],
                color=style.REAL,
                linewidth=3,
                zorder=9,
            )

    def _curve_axes(
        self,
        ax: Axes,
        curve: CurveSeries,
        head: str,
        label: str,
        cursor: int,
        *,
        marker: bool,
    ) -> None:
        ax.clear()
        ax.set_prop_cycle(color=[style.COURT_HEAD_COLORS[head]])
        index = int(np.searchsorted(curve.steps, self.run.best_step))
        markers = (
            [(self.run.best_step, float(curve.values[index]), "best checkpoint")]
            if marker
            else []
        )
        draw_progressive_curves(
            ax,
            [curve],
            cursor,
            spans=[(0, self.run.warmup_steps, "warmup")],
            markers=markers,
        )
        ax.set_xlim(0, self.run.total_steps)
        low, high = float(curve.values.min()), float(curve.values.max())
        padding = max((high - low) * 0.18, high * 0.015)
        ax.set_ylim(max(0.0, low - padding), high + padding)
        ax.set_ylabel(label, fontfamily=style.FONT_FAMILY, fontsize=11)
        ax.set_xticks(
            [0, self.run.total_steps], ["0", f"{self.run.total_steps / 1000:.1f}k"]
        )
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    def _curves_frame(self, segment: Segment, progress: float, step: float) -> None:
        selected = segment.name in {"select", "handoff"}
        # Selection revisits an earlier checkpoint after all training has been
        # shown. Keep the observed history, instead of hiding its later tail.
        cursor = self.run.total_steps if selected else int(step)
        for ax, (head, label) in zip(
            self.metrics,
            (
                ("kp", "KP誤差 [px]"),
                ("line", "LINE Dice"),
                ("seg", "SEG mIoU"),
                ("pose", "pose並進誤差 [m]"),
            ),
            strict=True,
        ):
            curve = self.run.series[f"val_{head}"]
            self._curve_axes(ax, curve, head, label, cursor, marker=True)
            if selected:
                ax.lines[-1].set_xdata([self.run.best_step, self.run.best_step])
                ax.axvline(
                    self.run.best_step,
                    color=style.COURT_HEAD_COLORS[head],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.65 + 0.3 * math.sin(progress * math.pi * 4) ** 2,
                )
        ax = self.loss
        ax.clear()
        ax.set_prop_cycle(color=[style.TRAINING])
        draw_progressive_curves(
            ax,
            [self.smoothed_loss],
            cursor,
            spans=[(0, self.run.warmup_steps, "warmup")],
        )
        if selected:
            ax.lines[-1].set_xdata([self.run.best_step, self.run.best_step])
        ax.set_xlim(0, self.run.total_steps)
        ax.set_yscale("log")
        ax.set_ylim(
            float(self.smoothed_loss.values.min()) * 0.8,
            float(self.smoothed_loss.values.max()) * 1.2,
        )
        ax.set_ylabel("train loss", fontfamily=style.FONT_FAMILY, fontsize=11)
        ax.tick_params(labelsize=9, labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax = self.lr
        ax.clear()
        lr = self.run.series["learning_rate"]
        ax.set_prop_cycle(color=[style.TRAINING])
        draw_progressive_curves(ax, [lr], cursor, spans=[])
        ax.lines[0].set_drawstyle("steps-post")
        if selected:
            ax.lines[-1].set_xdata([self.run.best_step, self.run.best_step])
        count = int(np.searchsorted(lr.steps, cursor, side="right"))
        if count:
            ax.fill_between(
                np.r_[lr.steps[:count], cursor],
                np.r_[lr.values[:count], lr.values[count - 1]],
                step="post",
                color=style.TRAINING,
                alpha=0.35,
            )
        ax.set_xlim(0, self.run.total_steps)
        ax.set_ylim(0, float(lr.values.max()) * 1.12)
        ax.set_ylabel("LR", fontfamily=style.FONT_FAMILY, fontsize=11)
        ax.set_xlabel(
            "global step",
            fontfamily=style.FONT_FAMILY,
            color=style.FOREGROUND,
            fontsize=11,
        )
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    def render_frame(self, frame: int) -> NDArray[np.uint8]:
        segment, progress, step = self.clock.at(frame)
        self._diagram_frame(frame, segment, progress)
        self._curves_frame(segment, progress, step)
        self.canvas.draw()
        return cast(
            NDArray[np.uint8], np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()
        )


def render_learning_stages(request: CourtPresentationRequest) -> Path:
    """Export the fixed story and its complete input/measurement provenance."""
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise FileExistsError(f"output_dir already exists: {request.output_dir}")
    run = load_court_run(
        request.bundle, tags=request.tags, scalar_summary=request.scalar_summary
    )
    story = CourtLearningStory(
        run,
        seconds=request.seconds,
        fps=request.fps,
        width=request.width,
        height=request.height,
    )
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "git_commit": commit,
        "resolved_config": request.resolved_config,
        "inputs": {
            "bundle": str(run.bundle),
            "sha256_paths": "absolute",
            "sha256": run.input_sha256,
        },
        "checkpoint": {
            "path": run.best_checkpoint,
            "sha256": run.checkpoint_sha256,
            "hash_authority": "checkpoint-selection.json; checkpoint bytes are not loaded",
            "epoch": run.best_epoch,
            "step": run.best_step,
        },
        "provenance": {
            "run_id": run.bundle.name,
            "measured": "Compact summary preserving original TensorBoard train/val and lr-AdamW tag, step, value and wall_time",
            "schematic": "Architecture, mixed-batch particles, real-pose supervision gate, checkpoint and head highlights",
            "pipeline_checkpoint_difference": "This is the fully frozen DINOv3-B run from #983, not pipeline checkpoint court_detection/multiscale_depth3/b863df1f01f0.ckpt. Numerical slide annotations must identify the B run.",
        },
        "step_convention": {
            "train_val": "raw event.step + 1 (completed updates)",
            "learning_rate": "raw event.step (epoch-start completed updates)",
        },
        "stages": [
            {
                "name": segment.name,
                "video_seconds": segment.video_seconds,
                "step_start": segment.step_start,
                "step_stop": segment.step_stop,
            }
            for segment in story.segments
        ],
        "stills": story.stills,
        "frame_count": story.clock.frame_count(),
        "colors": style.COURT_HEAD_COLORS,
        "loss_smoothing": {"kind": "causal_mean", "window_points": 20},
        "learning_rate_display": "Measured epoch-start samples, held within each epoch; no reconstructed scheduler values",
        "scalars": {
            name: {
                "tag": run.tags[name],
                "steps": curve.steps.tolist(),
                "values": curve.values.tolist(),
                "provenance": "measured",
            }
            for name, curve in run.series.items()
        },
    }
    render_presentation(
        story.render_frame,
        frame_count=story.clock.frame_count(),
        fps=request.fps,
        output_dir=request.output_dir,
        name="learning_stages",
        gif_width=request.gif_width,
        gif_fps=request.gif_fps,
        stills=story.stills,
        manifest=manifest,
    )
    return request.output_dir
