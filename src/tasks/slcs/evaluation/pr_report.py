"""Reproducible CPU figures from paired validation bundles and TensorBoard."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.tasks.slcs.evaluation.comparison import (
    CONDITIONS,
    MATCH_KEYS,
    compare_conditions,
    compare_gap_conditions,
)
from src.tasks.slcs.evaluation.motion import _stats
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

REPORT_CONDITIONS = (*CONDITIONS, "detector_gap_no_rgb")
COLORS = ("#0072B2", "#D55E00")
CAPTION = "Pseudo-3D teacher agreement, not measured 3D accuracy. Checkpoint selection: validation only."
AGGREGATION = "Unweighted valid window occurrences; overlapping frames counted separately. No smoothing or clipping."
TAGS = (
    "val/player_position_error_m_epoch",
    "val/ball_position_error_m_epoch",
    "val/scene_position_error_m_epoch",
    "val/loss",
    "train/loss_epoch",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_curves(training: Path, selected_epoch: int) -> dict[str, list[list[float]]]:
    """Join epoch scalars by exact global step; reject conflicting resumed data."""
    logs = sorted(training.glob("logs/version_*"))
    series: dict[str, dict[int, float]] = {tag: {} for tag in TAGS}
    if not logs:
        raise ValueError(f"No TensorBoard log versions: {training}")
    for directory in logs:
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags()["scalars"]
        if "epoch" not in tags:
            raise ValueError(f"Missing TensorBoard epoch tag: {directory}")
        epochs: dict[int, int] = {}
        for event in accumulator.Scalars("epoch"):
            if (
                not np.isfinite(event.value)
                or event.value < 0
                or int(event.value) != event.value
            ):
                raise ValueError("Invalid logged epoch")
            if event.step in epochs and epochs[event.step] != int(event.value):
                raise ValueError("Conflicting epoch at one global step")
            epochs[event.step] = int(event.value)
        for tag in TAGS:
            if tag not in tags:
                continue
            for event in accumulator.Scalars(tag):
                if event.step not in epochs or not np.isfinite(event.value):
                    raise ValueError(
                        f"Nonfinite or unaligned TensorBoard scalar: {tag}"
                    )
                epoch = epochs[event.step]
                if epoch in series[tag] and series[tag][epoch] != event.value:
                    raise ValueError(f"Ambiguous repeated epoch {epoch}: {tag}")
                series[tag][epoch] = float(event.value)
    for tag, values in series.items():
        if not values or selected_epoch not in values:
            raise ValueError(f"Missing data at selected epoch {selected_epoch}: {tag}")
    return {
        tag: [[float(epoch), value] for epoch, value in sorted(values.items())]
        for tag, values in series.items()
    }


def _load_run(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    sources = [path / "selection.json", path / "val/comparison/comparison.json"]
    sources += [
        path / "val" / condition / name
        for condition in REPORT_CONDITIONS
        for name in ("metrics.json", "eval_arrays.npz")
    ]
    source_hashes = {str(source.resolve()): _digest(source) for source in sources}
    receipt = _json(path / "selection.json")
    if receipt.get("monitor") != TAGS[2] or receipt.get("mode") != "min":
        raise ValueError(
            "Selection receipt must minimize validation scene position error"
        )
    domain_source = path / "val/comparison/comparison.json"
    domains = _json(domain_source)["domain_mapping"]
    if not isinstance(domains, dict) or not domains or "all" in domains.values():
        raise ValueError(
            "Expected explicit nonempty domain mapping without reserved 'all'"
        )
    bundles = {condition: path / "val" / condition for condition in REPORT_CONDITIONS}
    comparison = compare_conditions({key: bundles[key] for key in CONDITIONS}, domains)
    gap = compare_gap_conditions(
        {key: bundles[key] for key in ("detector_gap", "detector_gap_no_rgb")}, domains
    )
    if (
        receipt["checkpoint_sha256"] != comparison["checkpoint_sha256"]
        or receipt["checkpoint_sha256"] != gap["checkpoint_sha256"]
    ):
        raise ValueError("Selection receipt and evaluation checkpoint mismatch")
    epoch = receipt["selected"]["epoch_zero_based"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Selected epoch must be a nonnegative integer")
    rows = comparison["rows"] + [
        row for row in gap["rows"] if row["condition"] == "detector_gap_no_rgb"
    ]
    full: dict[str, np.ndarray] = {}
    summaries = []
    for condition, directory in bundles.items():
        context = _json(directory / "metrics.json")["context"]
        if context.get("split") != "val":
            raise ValueError("Only explicit validation bundles are supported")
        with np.load(directory / "eval_arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if not full:
            full = arrays
        for key in MATCH_KEYS:
            if arrays[key].dtype != full[key].dtype or not np.array_equal(
                arrays[key], full[key]
            ):
                raise ValueError(f"Unmatched condition {condition}: {key}")
        videos = arrays["video_ids"]
        for row in rows:
            if row["condition"] != condition or row["group_type"] not in {
                "all",
                "domain",
            }:
                continue
            selection = np.array(
                [
                    row["group_type"] == "all" or domains[str(video)] == row["group"]
                    for video in videos
                ]
            )
            for entity in ("player", "ball"):
                values = position_errors(arrays, entity, selection)
                if not values.size:
                    raise ValueError(
                        f"Empty {entity} targets: {condition}/{row['group']}"
                    )
                stats = _stats(values)
                mean = row[f"{entity}_position_error_m"]
                if not np.isclose(stats["mean"], mean, rtol=1e-5, atol=1e-6):
                    raise ValueError("Derived error differs from canonical SLCSMetrics")
                summaries.append(
                    {
                        "condition": condition,
                        "group": row["group"],
                        "entity": entity,
                        **stats,
                        "mean": mean,
                    }
                )
    return (
        {"selection": receipt, "domains": domains, "summaries": summaries},
        full,
        source_hashes,
    )


def position_errors(
    arrays: dict[str, np.ndarray], entity: str, selection: np.ndarray
) -> np.ndarray:
    """Same masked L2-in-meters definition as SLCSMetrics; float64 diagnostics."""
    mask = arrays[f"{entity}_mask"][selection]
    scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float64)
    pred = arrays[f"pred_{entity}_position"][selection][mask].astype(np.float64) * scale
    target = (
        arrays[f"target_{entity}_position"][selection][mask].astype(np.float64) * scale
    )
    return cast(np.ndarray, np.linalg.norm(pred - target, axis=-1))


def _save(fig: Any, path: Path, subtitle: str) -> None:
    fig.text(0.5, 0.035, CAPTION, ha="center", fontsize=12, color="#334155")
    fig.text(0.5, 0.012, subtitle, ha="center", fontsize=10, color="#475569")
    fig.savefig(
        path, dpi=180, facecolor="white", metadata={"Software": "SLCS PR report"}
    )
    plt.close(fig)


def _comparison_plot(runs: dict[str, dict[str, Any]], output: Path) -> None:
    domains = sorted(set(next(iter(runs.values()))["domains"].values()))
    groups = ["all", *domains]
    fig, axes = plt.subplots(
        2, len(groups), figsize=(6 * len(groups), 10), squeeze=False
    )
    fig.subplots_adjust(
        left=0.065, right=0.98, bottom=0.20, top=0.85, wspace=0.23, hspace=0.40
    )
    fig.suptitle(
        "Matched validation conditions", fontsize=23, fontweight="bold", y=0.97
    )
    handles = []
    for column, group in enumerate(groups):
        for row, entity in enumerate(("player", "ball")):
            ax = axes[row, column]
            for index, (label, run) in enumerate(runs.items()):
                selected = [
                    next(
                        item
                        for item in run["summaries"]
                        if item["group"] == group
                        and item["entity"] == entity
                        and item["condition"] == condition
                    )
                    for condition in REPORT_CONDITIONS
                ]
                x = (
                    np.arange(len(REPORT_CONDITIONS))
                    + (index - (len(runs) - 1) / 2) * 0.34
                )
                bars = ax.bar(
                    x,
                    [item["mean"] for item in selected],
                    width=0.31,
                    color=COLORS[index],
                    label=label,
                )
                ax.scatter(
                    x,
                    [item["p95"] for item in selected],
                    marker="D",
                    s=35,
                    color=COLORS[index],
                    edgecolors="white",
                    zorder=3,
                )
                if row == column == 0:
                    handles.append(bars)
            ax.set_title(
                f"{'Overall' if group == 'all' else group.title()} · {entity.title()}",
                fontsize=16,
            )
            ax.set_xticks(
                np.arange(len(REPORT_CONDITIONS)),
                ["Full", "No RGB", "Detector\ngap", "RGB only", "Gap +\nno RGB"],
                fontsize=11,
            )
            ax.set_ylabel("Position error (m)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.legend(
        handles,
        list(runs),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(runs),
        frameon=False,
    )
    _save(
        fig,
        output / "conditions.png",
        "Bars: mean · Diamonds: p95 (not uncertainty). "
        + AGGREGATION.split(". No")[0]
        + ".",
    )


def _distribution_plot(
    arrays_by_run: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.77, wspace=0.25)
    fig.suptitle(
        "Full-input validation · error distribution", fontsize=22, fontweight="bold"
    )
    for ax, entity in zip(axes, ("player", "ball"), strict=True):
        for index, (label, arrays) in enumerate(arrays_by_run.items()):
            values = np.sort(
                position_errors(
                    arrays, entity, np.ones(len(arrays["video_ids"]), dtype=bool)
                )
            )
            x = np.concatenate(([0.0], values))
            y = np.arange(len(values) + 1) / len(values)
            ax.step(
                x,
                y,
                where="post",
                color=COLORS[index],
                linewidth=2,
                label=f"{label} (n={len(values):,})",
            )
        ax.set(
            title=entity.title(),
            xlabel="Position error (m)",
            ylabel="Fraction of valid occurrences",
            xlim=(0, None),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=11, loc="lower right")
    _save(fig, output / "distribution.png", AGGREGATION)


def _curves_plot(runs: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.14, top=0.82, hspace=0.38, wspace=0.22
    )
    fig.suptitle(
        "Learning curves · validation-selected checkpoints",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )
    titles = (
        "Player position (m)",
        "Ball position (m)",
        "Scene position (m)",
        "Objective loss (dimensionless)",
    )
    for index, (label, run) in enumerate(runs.items()):
        epoch = run["selection"]["selected"]["epoch_zero_based"]
        for panel, (ax, tag, title) in enumerate(
            zip(axes.flat, TAGS[:4], titles, strict=True)
        ):
            points = np.array(run["curves"][tag])
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=COLORS[index],
                linewidth=2,
                label=f"{label} · val",
            )
            selected = points[points[:, 0] == epoch][0]
            ax.scatter(*selected, color=COLORS[index], s=100, marker="*", zorder=4)
            ax.axvline(epoch, color=COLORS[index], linestyle=":", alpha=0.65)
            if panel == 3:
                train = np.array(run["curves"]["train/loss_epoch"])
                ax.plot(
                    train[:, 0],
                    train[:, 1],
                    color=COLORS[index],
                    linestyle="--",
                    alpha=0.75,
                    label=f"{label} · train",
                )
            ax.set(title=title, xlabel="Epoch (zero-based)")
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            ax.grid(alpha=0.2)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    epochs = "; ".join(
        f"{label}: epoch {run['selection']['selected']['epoch_zero_based']}"
        for label, run in runs.items()
    )
    _save(
        fig,
        output / "learning_curves.png",
        f"Stars/dotted lines: selected checkpoint ({epochs}). Raw epoch values; no smoothing.",
    )


def generate_report(
    *,
    evaluations: dict[str, Path],
    output_root: Path,
    output: str,
    training: dict[str, Path] | None = None,
) -> Path:
    """Validate all requested artifacts before exclusively creating a report run."""
    if not 1 <= len(evaluations) <= 2 or any(
        not label.strip() for label in evaluations
    ):
        raise ValueError("Provide one or two uniquely named evaluations")
    if training is not None and set(training) != set(evaluations):
        raise ValueError("Training labels must exactly match evaluation labels")
    parts = output.split("/")
    if (
        not output_root.is_absolute()
        or len(parts) != 4
        or parts[:2] != ["slcs", "visualize"]
        or any(p in {"", ".", ".."} or ":" in p or "\\" in p for p in parts)
    ):
        raise ValueError(
            "Use an absolute output_root and slcs/visualize/<experiment>/<run-id>"
        )
    destination = (output_root / output).resolve()
    if not destination.is_relative_to(output_root.resolve()):
        raise ValueError("Output escapes output_root")
    if destination.exists():
        raise FileExistsError(f"Refusing existing report: {destination}")
    runs: dict[str, dict[str, Any]] = {}
    arrays_by_run: dict[str, dict[str, np.ndarray]] = {}
    source_hashes: dict[str, str] = {}
    for label, path in evaluations.items():
        run, arrays, hashes = _load_run(path.resolve(strict=True))
        if runs:
            reference = next(iter(arrays_by_run.values()))
            if run["domains"] != next(iter(runs.values()))["domains"]:
                raise ValueError("Domain mappings differ between runs")
            for key in MATCH_KEYS:
                if arrays[key].dtype != reference[key].dtype or not np.array_equal(
                    arrays[key], reference[key]
                ):
                    raise ValueError(f"Unmatched evaluation runs: {key}")
        if training is not None:
            train_path = training[label].resolve(strict=True)
            selected_path = Path(run["selection"]["selected"]["path"]).resolve()
            if not selected_path.is_relative_to(train_path):
                raise ValueError(
                    "Selected checkpoint does not belong to requested training run"
                )
            hashes.update(
                {
                    str(source.resolve()): _digest(source)
                    for source in sorted(
                        train_path.glob("logs/version_*/events.out.tfevents.*")
                    )
                }
            )
            epoch = run["selection"]["selected"]["epoch_zero_based"]
            run["curves"] = read_curves(train_path, epoch)
            score = next(
                value for at_epoch, value in run["curves"][TAGS[2]] if at_epoch == epoch
            )
            if not np.isclose(
                score,
                run["selection"]["selected"]["validation_score"],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("Selected validation score differs from TensorBoard")
        runs[label], arrays_by_run[label] = run, arrays
        source_hashes.update(hashes)
    if any(_digest(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError("Source artifact changed while loading")
    destination.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    ):
        _comparison_plot(runs, destination)
        _distribution_plot(arrays_by_run, destination)
        if training is not None:
            _curves_plot(runs, destination)
    if any(_digest(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError(
            "Source artifact changed during rendering; report is incomplete"
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "interpretation": CAPTION,
        "aggregation": AGGREGATION,
        "p95_definition": "numpy percentile 95, linear interpolation; descriptive tail statistic",
        "git_commit": git.stdout.strip(),
        "source_sha256": source_hashes,
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "renderer_sha256": _digest(Path(__file__)),
        "evaluations": {
            label: str(path.resolve()) for label, path in evaluations.items()
        },
        "training": None
        if training is None
        else {label: str(path.resolve()) for label, path in training.items()},
        "runs": runs,
        "artifacts": {
            path.name: _digest(path) for path in sorted(destination.glob("*.png"))
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return destination
