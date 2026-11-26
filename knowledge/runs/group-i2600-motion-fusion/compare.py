"""Re-score the fixed Meiji test predictions and summarize all three training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.tasks.ball_refiner_3d.post_training.dataset import PreparedDataset, sha256_file
from src.tasks.ball_refiner_3d.post_training.fusion import FUSION_MODES
from src.tasks.ball_refiner_3d.post_training.labels import MATCH_TOLERANCE
from src.tasks.ball_refiner_3d.post_training.runner import FUSION_CHECKPOINT
from src.tasks.ball_refiner_3d.post_training.scoring import report, score_clip

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.utils.io import json_value, write_json_atomic

MODES = ("ball-only", *FUSION_MODES)
SOURCES = ("annotation", "independent")
COLORS = ("#3a454f", "#00806b", "#a84412", "#3155af")
LABELS = ("Ball-only", "Shared", "Per Player", "Set Attention")
EXAMPLES = (
    ("video_000/clip_040", 230, "shot", "Volley"),
    ("video_000/clip_048", 173, "bounce", "Bounce"),
    ("video_000/clip_056", 681, "net", "Net contact"),
)


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def audit_run(
    run: Path,
    run_bundle: Path,
    job: dict[str, Any],
    pins: dict[str, Any],
    test: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = read(run_bundle / "queue-completion.json")
    capture = read(run_bundle / "run.json")
    assert status["job"] == job["job_id"] and status["state"] == "done"
    assert capture["run_id"] + ".job" == status["job"]
    assert capture["commit"] == pins["head"]
    assert capture["logical_gpu_slot"] == capture["resource"] == "all"
    run_info, metrics, matches = (
        read(run / name) for name in ("run.json", "metrics.json", "matches.json")
    )
    assert (run_bundle / "metrics.json").read_bytes() == (
        run / "metrics.json"
    ).read_bytes()
    assert run_info["state"] == "complete" and run_info["code_commit"] == pins["head"]
    assert run_info["device"] == "cuda"
    assert run_info["picking"] == {"threshold": 0.5, "min_separation": 3}
    assert run_info["tolerance_frames"] == MATCH_TOLERANCE == 2
    assert run_info["event_sigma_frames"] == 2.0
    for key in ("manifest_sha256", "prepared_sha256"):
        assert all(doc[key] == pins[key] for doc in (run_info, metrics, matches))
    assert run_info["checkpoint_sha256"] == pins["checkpoint_sha256"]
    settings = run_info["settings"]
    assert settings == {
        "mode": job["mode"],
        "seed": job["seed"],
        "epochs": 12,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
    }
    selected = None
    validation_range = None
    files = [run / name for name in ("run.json", "metrics.json", "matches.json")]
    if job["mode"] != "ball-only":
        history = read(run / "history.json")
        assert (run_bundle / "history.json").read_bytes() == (
            run / "history.json"
        ).read_bytes()
        assert history["manifest_sha256"] == pins["manifest_sha256"]
        epochs = history["epochs"]
        assert [row["updates"] for row in epochs] == list(range(47, 565, 47))
        f1_by_epoch = [row["val_macro_f1"] for row in epochs]
        validation_range = {"min": min(f1_by_epoch), "max": max(f1_by_epoch)}
        selected = max(
            epochs,
            key=lambda row: (
                row["val_macro_f1"],
                -row["val_bce"],
                -row["epoch"],
            ),
        )
        best = run / "best.pt"
        assert sha256_file(best) == metrics["fusion_checkpoint_sha256"]
        saved = torch.load(best, weights_only=True, map_location="cpu")
        assert (
            saved["epoch"]
            == selected["epoch"]
            == metrics["best_epoch"]
            == run_info["best_epoch"]
        )
        assert saved["schema"] == FUSION_CHECKPOINT
        assert all(saved["identity"][key] == run_info[key] for key in saved["identity"])
        params = list(saved["fusion"].values())
        assert all(torch.isfinite(p).all() for p in params)
        assert sum(p.numel() for p in params) == metrics["added_parameters"]
        files.extend((run / "history.json", best))
    else:
        assert metrics["best_epoch"] is None and metrics["added_parameters"] == 0

    assert set(matches["clips"]) == set(matches["predictions"]) == set(test)
    collected: dict[str, list] = {name: [] for name in SOURCES}
    for clip_id, clip in test.items():
        row = matches["predictions"][clip_id]
        path = run / row["path"]
        assert sha256_file(path) == row["sha256"]
        files.append(path)
        n = len(clip.labels.coverage)
        with np.load(path, allow_pickle=False) as data:
            assert data["fps"].item() == clip.fps and data["clip_id"].item() == clip_id
            assert tuple(data["event_classes"].tolist()) == EVENT_CLASSES
            assert all(
                data[key].item() == pins[key]
                for key in ("manifest_sha256", "prepared_sha256")
            )
            np.testing.assert_array_equal(data["frame"], np.arange(n))
            for key in ("ball_m", "ball_valid", "root_valid", "smpl_valid"):
                np.testing.assert_array_equal(data[key], getattr(clip.motion, key))
            p = data["event_probability"]
            logits, xyz = data["event_logits"], data["coordinates_m"]
            assert p.shape == logits.shape == xyz.shape == (n, 3)
            assert np.isfinite(logits).all() and np.isfinite(xyz).all()
            np.testing.assert_allclose(
                p, torch.from_numpy(logits).sigmoid().numpy(), atol=2e-7, rtol=0
            )
        labels = {"annotation": clip.labels}
        if clip.reference is not None:
            labels["independent"] = clip.reference
        for source, truth in labels.items():
            decisions = score_clip(p, truth, clip.fps, clip.motion)
            assert json_value(decisions) == matches["clips"][clip_id][source]
            collected[source].extend(decisions)
    for name, decisions in collected.items():
        assert report(decisions) == metrics[name]

    timing = metrics["timing"]
    assert set(timing["clips"]) == set(test)
    medians = []
    for row in timing["clips"].values():
        times = row["elapsed_ms"]
        assert len(times) == run_info["timing_repeats"] == 3
        assert all(math.isfinite(x) and x > 0 for x in times)
        medians.append(float(np.median(times)))
    assert math.isclose(timing["median_clip_ms"], np.median(medians))
    artifacts = {str(path.relative_to(run)): sha256_file(path) for path in files}
    return metrics, {
        "settings": settings,
        "environment": [run_info[k] for k in ("gpu", "torch", "cuda")],
        "selected_validation": selected,
        "val_macro_f1_range": validation_range,
        "artifacts_sha256": artifacts,
    }


def metric_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str, dict[str, Any]]]:
    rows = []
    for source in SOURCES:
        scopes = {name: metrics[source][name] for name in ("all", "in_point")}
        scopes.update(metrics[source]["conditions"])
        for scope, values in scopes.items():
            rows.append((source, scope, "macro", {"f1": values["macro_f1"]}))
            rows.extend(
                (source, scope, kind, row) for kind, row in values["by_class"].items()
            )
    return rows


def summarize(scores: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(scores),
        "min": min(scores),
        "max": max(scores),
        "sample_sd": stdev(scores) if len(scores) > 1 else None,
    }


def plot_examples(runs: Path, output: Path) -> None:
    figure, plots = plt.subplots(3, 1, figsize=(9.0, 7.6), layout="constrained")
    for axis, (clip_id, frame, kind, label) in zip(plots, EXAMPLES, strict=True):
        channel = EVENT_CLASSES.index(kind)
        path_name = clip_id.replace("/", "--") + ".npz"
        with np.load(runs / "ball-only-s42" / "predictions" / path_name) as raw:
            bounds = slice(max(0, frame - 14), min(len(raw["frame"]), frame + 15))
            time = raw["frame"][bounds]
        axis.axvspan(frame - 2, frame + 2, color="#dbe3ef", alpha=0.75, linewidth=0)
        axis.axvline(frame, color="#182333", linewidth=1)
        axis.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
        for mode, color, name in zip(MODES, COLORS, LABELS, strict=True):
            seeds = (42,) if mode == "ball-only" else (42, 43, 44)
            estimates = []
            for seed in seeds:
                run = runs / f"{mode}-s{seed}"
                with np.load(run / "predictions" / path_name) as inherited:
                    estimates.append(inherited["event_probability"][bounds, channel])
            trajectories = np.stack(estimates)
            if len(seeds) > 1:
                axis.fill_between(
                    time,
                    trajectories.min(0),
                    trajectories.max(0),
                    color=color,
                    alpha=0.12,
                )
            axis.plot(
                time, trajectories.mean(0), color=color, label=name, linewidth=1.6
            )
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlim(time[0], time[-1])
        axis.set_ylabel("Probability")
        axis.set_title(
            f"{label} · {clip_id.rsplit('/', 1)[-1]} · frame {frame}", loc="left"
        )
        axis.grid(axis="y", color="#e3e7eb", linewidth=0.6)
    plots[0].legend(ncol=4, loc="upper right", fontsize=8)
    plots[-1].set_xlabel("Original frame")
    figure.savefig(output / "event-examples.png", dpi=150)
    plt.close(figure)


def plot_learning(runs: Path, output: Path) -> None:
    figure, plots = plt.subplots(
        2, 3, figsize=(10.5, 5.8), layout="constrained", sharex=True, sharey="row"
    )
    seed_colors = ("#00796b", "#a84412", "#3155af")
    for column, mode in enumerate(FUSION_MODES):
        for seed, color in zip((42, 43, 44), seed_colors, strict=True):
            run = runs / f"{mode}-s{seed}"
            history = read(run / "history.json")["epochs"]
            best = read(run / "metrics.json")["best_epoch"]
            epochs = [r["epoch"] for r in history]
            for index, field in enumerate(("train_bce", "val_macro_f1")):
                values = [r[field] for r in history]
                axis = plots[index, column]
                axis.plot(
                    epochs, values, color=color, label=f"Seed {seed}", linewidth=1.35
                )
                if index == 1:
                    axis.scatter(best, values[best - 1], color=color, s=25, zorder=3)
        plots[0, column].set_title(LABELS[column + 1])
        plots[1, column].set_xlabel("Epoch")
    plots[0, 0].set_ylabel("Training masked BCE")
    plots[1, 0].set_ylabel("Validation macro-F1")
    plots[0, 0].legend(fontsize=8)
    for axis in plots.flat:
        axis.grid(color="#e3e7eb", linewidth=0.6)
    figure.savefig(output / "learning-curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = read(Path(__file__).with_name("pins.json"))
    identity = pins["identity"]
    prepared = PreparedDataset.load(
        args.run_root / f"prepared-{identity['manifest_sha256'][:8]}",
        identity["prepared_sha256"],
        identity["manifest_sha256"],
    )
    test = {clip.clip_id: clip for clip in prepared.split("test")}
    assert (
        len(test) == 11
        and sum(clip.reference is not None for clip in test.values()) == 6
    )
    runs = args.run_root / "runs"
    verified, all_metrics, grouped = {}, {}, defaultdict(list)
    table = []
    for key, job in pins["jobs"].items():
        run_bundle = Path(__file__).resolve().parents[1] / f"run-i2600-motion-{key}-r2"
        metrics, row = audit_run(runs / key, run_bundle, job, identity, test)
        verified[key] = row
        all_metrics[key] = {
            "macro_f1": {
                source: {
                    scope: metrics[source][scope]["macro_f1"]
                    for scope in ("all", "in_point")
                }
                for source in SOURCES
            },
            "test_f1_by_class": {
                kind: v["f1"]
                for kind, v in metrics["annotation"]["all"]["by_class"].items()
            },
            "median_clip_ms": metrics["timing"]["median_clip_ms"],
            "peak_cuda_mib": metrics["timing"]["max_cuda_allocated_bytes"] / 2**20,
            "added_parameters": metrics["added_parameters"],
            "best_epoch": metrics["best_epoch"],
            "val_macro_f1": row["selected_validation"]["val_macro_f1"]
            if row["selected_validation"]
            else None,
            "val_macro_f1_range": row["val_macro_f1_range"],
        }
        grouped[job["mode"]].append(all_metrics[key])
        for source, scope, kind, score in metric_rows(metrics):
            table.append(
                {
                    "mode": job["mode"],
                    "seed": job["seed"],
                    "source": source,
                    "scope": scope,
                    "class": kind,
                    **score,
                }
            )
    assert (
        len(verified) == 10
        and len({tuple(row["environment"]) for row in verified.values()}) == 1
    )
    aggregates = {}
    for mode in MODES:
        members = grouped[mode]
        aggregates[mode] = {
            "macro_f1": {
                source: {
                    scope: summarize([r["macro_f1"][source][scope] for r in members])
                    for scope in ("all", "in_point")
                }
                for source in SOURCES
            },
            "test_f1_by_class": {
                kind: summarize([r["test_f1_by_class"][kind] for r in members])
                for kind in EVENT_CLASSES
            },
            "median_clip_ms": summarize([r["median_clip_ms"] for r in members]),
            "peak_cuda_mib": max(r["peak_cuda_mib"] for r in members),
            "added_parameters": members[0]["added_parameters"],
        }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        args.output / "comparison.json",
        {
            "identity": identity,
            "prediction_files_verified": sum(
                sum(name.startswith("predictions/") for name in v["artifacts_sha256"])
                for v in verified.values()
            ),
            "per_run": all_metrics,
            "per_mode": aggregates,
        },
    )
    write_json_atomic(
        args.output / "run-evidence.json",
        {
            "identity": identity,
            "runs": verified,
        },
    )
    fields = (
        "mode",
        "seed",
        "source",
        "scope",
        "class",
        "events",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "time_mae_ms",
        "excluded_labels",
        "excluded_predictions",
    )
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in table:
            writer.writerow(
                {
                    field: "N/A" if value is None else value
                    for field, value in row.items()
                }
            )
    plot_learning(runs, args.output)
    plot_examples(runs, args.output)
    print(
        f"Verified {len(verified)} runs, {len(verified) * len(test)} prediction files and both label sources"
    )


if __name__ == "__main__":
    main()
