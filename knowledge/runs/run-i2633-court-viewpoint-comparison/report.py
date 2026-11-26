"""Verify the completed campaign and write paired real-image comparison tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import chain, product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from protocol import (
    CONDITIONS,
    H_ACCURACY_RATIO,
    REAL_REGISTRY_SHA256,
    REAL_SPLIT_COUNTS,
    STRATA,
    TRAIN_SEEDS,
    UPDATES,
    stratum_counts,
)
from train_and_evaluate import checkpoint_for, run_name

from src.synthetic_data_generation.dataset.court.scale.publication import read_json
from src.tasks.court_detection.scaling.real_split import canonical_sha256
from src.utils.data.image_record_store import file_sha256
from src.utils.io import write_json_atomic

matplotlib.use("Agg")

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2000
EFFECT_METRICS = ("kp_mean_px512", "h_formed_pct", "h_at_1pct_pct")
RUNS = tuple(product(CONDITIONS, TRAIN_SEEDS))


def verified_rows(
    run_dir: Path,
    split: str,
    info: dict[str, Any],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = run_dir / f"{split}.json"
    ids = [view["sample_id"] for view in views]
    if (
        info["path"] != path.name
        or info["sha256"] != file_sha256(path)
        or info["ordered_id_sha256"] != canonical_sha256(ids)
        or info["strata"] != stratum_counts(views)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(
            f"Evaluation inventory or receipt drift: {run_dir.name}/{split}."
        )
    document = read_json(path)
    if document["split"] != split or len(document["rows"]) != len(views):
        raise ValueError(f"Incomplete evaluation: {path}.")
    for row, view in zip(document["rows"], views, strict=True):
        if any(row[key] != value for key, value in view.items()):
            raise ValueError(
                f"Sample identity or stratum drift in {path}: {view['sample_id']}."
            )
        errors = row["kp_errors_px512"]
        if len(errors) != view["visible_kp"] or any(
            not math.isfinite(e) or e < 0 for e in errors
        ):
            raise ValueError(
                f"Invalid ordered-KP scoring in {path}: {view['sample_id']}."
            )
        h_error = row["h_mean_error_diagonal"]
        if (
            (row["h_status"] == "ok") != (h_error is not None)
            or (h_error is not None and (not math.isfinite(h_error) or h_error < 0))
            or row["h_at_1pct"]
            is not (h_error is not None and h_error <= H_ACCURACY_RATIO)
        ):
            raise ValueError(
                f"Invalid homography scoring in {path}: {view['sample_id']}."
            )
    return document["rows"]


def verified_run(
    campaign: Path, condition: str, seed: int, prep: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = campaign / "runs" / run_name(condition, seed)
    receipt = read_json(run_dir / "complete.json")
    fit = read_json(run_dir / "fit.json")
    if any(receipt[key] != value for key, value in fit.items()):
        raise ValueError(f"Fit and final evaluation receipts disagree for {run_dir}.")
    if (
        (receipt["condition"], receipt["seed"], receipt["updates"])
        != (condition, seed, UPDATES)
        or receipt["real_registry_sha256"] != REAL_REGISTRY_SHA256
        or receipt["real_strata_sha256"] != prep["real_strata_sha256"]
        or receipt["data_preparation_sha256"]
        != file_sha256(campaign / "preparation.json")
    ):
        raise ValueError(f"Completed run does not match its frozen inputs: {run_dir}.")
    checkpoint = checkpoint_for(run_dir, prep, campaign, condition, seed)
    if receipt["checkpoint"] != checkpoint.relative_to(run_dir).as_posix():
        raise ValueError(f"Unexpected checkpoint provenance for {run_dir}.")
    config_path = run_dir / "config.yaml"
    for path, key in (
        (checkpoint, "checkpoint_sha256"),
        (config_path, "train_config_sha256"),
        (run_dir / "data-condition.json", "train_data_condition_sha256"),
    ):
        if file_sha256(path) != receipt[key]:
            raise ValueError(f"Recorded file changed: {path}.")
    config = yaml.safe_load(config_path.read_text())
    # Drop only the agreed differences. Duplicate source configuration is fully
    # resolved into the saved YAML and must agree with the primary declaration.
    manifest = f"pool/{condition}.train.subset.json"
    if (
        config["data"]["source"].pop("manifest") != manifest
        or config["mixed"]["sources"]["synthetic_court"].pop("manifest") != manifest
        or config["run"].pop("seed") != seed
        or config["run"].pop("output_dir") != f"runs/{run_name(condition, seed)}"
    ):
        raise ValueError(f"Unexpected condition override for {run_dir}.")
    return receipt, config


def metrics(frames: list[dict[str, Any]]) -> dict[str, int | float | None]:
    n = len(frames)
    points = list(chain.from_iterable(frame["kp_errors_px512"] for frame in frames))
    formed = sum(frame["h_status"] == "ok" for frame in frames)
    accurate = sum(frame["h_at_1pct"] for frame in frames)
    return {
        "images": n,
        "videos": len({frame["video_id"] for frame in frames}),
        "visible_points": len(points),
        "h_formed_count": formed,
        "h_at_1pct_count": accurate,
        "kp_mean_px512": float(np.mean(points)) if n else None,
        "kp_median_px512": float(np.median(points)) if n else None,
        "kp_p90_px512": float(np.quantile(points, 0.9)) if n else None,
        "h_formed_pct": 100 * formed / n if n else None,
        "h_at_1pct_pct": 100 * accurate / n if n else None,
    }


def paired_video_interval(
    subsets: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, list[float] | None]:
    """Paired percentile intervals for the mean over the two matched seeds."""
    groups: dict[str, list[int]] = defaultdict(list)
    reference = subsets["narrow", TRAIN_SEEDS[0]]
    for i, row in enumerate(reference):
        groups[row["video_id"]].append(i)
    if len(groups) < 2:
        return dict.fromkeys(EFFECT_METRICS)
    # Each draw resamples entire video clusters. KP is point-pooled, whereas
    # H rates count images even when the fit fails.
    sums = []
    denominators = []
    for indices in groups.values():
        by_seed = []
        for seed in TRAIN_SEEDS:
            narrow = [subsets["narrow", seed][i] for i in indices]
            broad = [subsets["broad", seed][i] for i in indices]
            by_seed.append(
                [
                    math.fsum(
                        chain.from_iterable(row["kp_errors_px512"] for row in broad)
                    )
                    - math.fsum(
                        chain.from_iterable(row["kp_errors_px512"] for row in narrow)
                    ),
                    100
                    * (
                        sum(r["h_status"] == "ok" for r in broad)
                        - sum(r["h_status"] == "ok" for r in narrow)
                    ),
                    100
                    * (
                        sum(r["h_at_1pct"] for r in broad)
                        - sum(r["h_at_1pct"] for r in narrow)
                    ),
                ]
            )
        sums.append(np.mean(by_seed, axis=0))
        denominators.append(
            [
                sum(len(reference[i]["kp_errors_px512"]) for i in indices),
                len(indices),
                len(indices),
            ]
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    replicates = np.asarray(sums)[draws].sum(axis=1) / np.asarray(denominators)[
        draws
    ].sum(axis=1)
    return dict(
        zip(
            EFFECT_METRICS,
            np.quantile(replicates, [0.025, 0.975], axis=0).T.tolist(),
            strict=True,
        )
    )


def classify(effects: list[dict[str, Any]]) -> str:
    primary = [
        row["broad_minus_narrow_by_seed"]
        for row in effects
        if row["split"] == "test" and row["stratum"] in {"all", "challenging"}
    ]
    if len(primary) != 2:
        raise ValueError(
            "The prespecified two test populations are required for interpretation."
        )
    improved = [
        better
        for population in primary
        for record in population.values()
        for better in (record["kp_mean_px512"] < 0, record["h_at_1pct_pct"] > 0)
    ]
    if all(improved):
        return "supported"
    return "partially_supported" if any(improved) else "unsupported"


def comparison(campaign: Path) -> dict[str, Any]:
    prep = read_json(campaign / "preparation.json")
    registry = read_json(campaign / "registry.json")
    registry_payload = {
        key: value for key, value in registry.items() if key != "sha256"
    }
    if canonical_sha256(registry_payload) != REAL_REGISTRY_SHA256:
        raise ValueError("The registry population has changed.")
    for filename, info in prep["manifests"].items():
        if file_sha256(campaign / "inputs/pool" / filename) != info["sha256"]:
            raise ValueError(f"Frozen subset changed: {filename}.")
    strata_path = campaign / "strata.json"
    if file_sha256(strata_path) != prep["real_strata_sha256"]:
        raise ValueError("Frozen stratum annotations changed.")
    views = read_json(strata_path)
    runs = {run: verified_run(campaign, *run, prep) for run in RUNS}
    configurations = {canonical_sha256(config) for _, config in runs.values()}
    shared = (
        "source_head",
        "dino_commit",
        "dino_sha256",
        "gpu",
        "torch",
        "cuda",
        "opencv",
        "eval_precision",
        "eval_batch_limit",
        "eval_nms_kernel",
        "eval_decoder_threshold",
    )
    if (
        len(configurations) != 1
        or len({tuple(receipt[key] for key in shared) for receipt, _ in runs.values()})
        != 1
    ):
        raise ValueError(
            "Run configurations, implementation or evaluation settings differ."
        )
    rows = []
    effects = []
    for split, (images, videos) in REAL_SPLIT_COUNTS.items():
        counts = stratum_counts(views[split])
        if (
            counts != prep["real_strata_counts"][split]
            or counts["all"] != {"images": images, "videos": videos}
            or [view["sample_id"] for view in views[split]] != registry["splits"][split]
        ):
            raise ValueError(f"Unmatched fixed registry/strata for {split}.")
        measured = {
            run: verified_rows(
                campaign / "runs" / run_name(*run),
                split,
                receipt["evaluations"][split],
                views[split],
            )
            for run, (receipt, _) in runs.items()
        }
        for stratum in STRATA:
            index = [
                i for i, view in enumerate(views[split]) if stratum in view["strata"]
            ]
            subsets = {
                run: [records[i] for i in index] for run, records in measured.items()
            }
            scores = {run: metrics(subsets[run]) for run in RUNS}
            rows.extend(
                {
                    "split": split,
                    "stratum": stratum,
                    "condition": run[0],
                    "seed": run[1],
                    **score,
                }
                for run, score in scores.items()
            )
            by_seed = {
                str(seed): {
                    metric: scores["broad", seed][metric]
                    - scores["narrow", seed][metric]
                    if index
                    else None
                    for metric in EFFECT_METRICS
                }
                for seed in TRAIN_SEEDS
            }
            means = {
                condition: {
                    metric: float(
                        np.mean(
                            [scores[condition, seed][metric] for seed in TRAIN_SEEDS]
                        )
                    )
                    if index
                    else None
                    for metric in EFFECT_METRICS
                }
                for condition in CONDITIONS
            }
            effects.append(
                {
                    "split": split,
                    "stratum": stratum,
                    **counts[stratum],
                    "seed_means": means,
                    "broad_minus_narrow_by_seed": by_seed,
                    "broad_minus_narrow_mean": {
                        metric: means["broad"][metric] - means["narrow"][metric]
                        if index
                        else None
                        for metric in EFFECT_METRICS
                    },
                    "paired_video_95pct_interval": paired_video_interval(subsets),
                }
            )
    summary = {
        "schema": "court_viewpoint_comparison_v1",
        "registry_sha256": REAL_REGISTRY_SHA256,
        "common_training_config_sha256": next(iter(configurations)),
        "source_head": runs[RUNS[0]][0]["source_head"],
        "classification": classify(effects),
        "train_seeds": TRAIN_SEEDS,
        "updates": UPDATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "runs": {run_name(*run): receipt for run, (receipt, _) in runs.items()},
        "sum_run_seconds": math.fsum(
            receipt["seconds"] for receipt, _ in runs.values()
        ),
        "metrics": rows,
        "effects": effects,
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        table = csv.DictWriter(stream, list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def plot_test(effects: list[dict[str, Any]], path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    rows = [row for row in effects if row["split"] == "test" and row["images"]]
    titles = (
        "Mean KP error (px at 512)",
        "H formed (percentage points)",
        "H within 1% (percentage points)",
    )
    titles_by_stratum = {
        "all": "All",
        "challenging": "Challenging union",
        "low_angle": "Low angle",
        "oblique": "Oblique",
        "cropped": "Cropped",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), sharey=True)
    for axis, metric, title in zip(axes, EFFECT_METRICS, titles, strict=True):
        axis.axvline(0, color="0.5", linewidth=1, linestyle="--")
        for row_index, row in enumerate(rows):
            interval = row["paired_video_95pct_interval"][metric]
            if interval is not None:
                axis.plot(
                    interval, [row_index, row_index], linewidth=2, color="#333b48"
                )
            axis.scatter(
                row["broad_minus_narrow_mean"][metric],
                row_index,
                marker="D",
                s=28,
                color="#333b48",
                zorder=3,
            )
            axis.scatter(
                row["broad_minus_narrow_by_seed"]["42"][metric],
                row_index - 0.16,
                marker="o",
                s=24,
                color="#ad4b27",
            )
            axis.scatter(
                row["broad_minus_narrow_by_seed"]["43"][metric],
                row_index + 0.16,
                marker="s",
                s=24,
                color="#236e94",
            )
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
    labels = [
        f"{titles_by_stratum[row['stratum']]} (n={row['images']:,})" for row in rows
    ]
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Real-video test: broad minus narrow viewpoint training", y=0.96)
    fig.legend(
        handles=[
            Line2D(
                [], [], color="#ad4b27", marker="o", linestyle="none", label="Seed 42"
            ),
            Line2D(
                [], [], color="#236e94", marker="s", linestyle="none", label="Seed 43"
            ),
            Line2D(
                [],
                [],
                color="#333b48",
                marker="D",
                linewidth=2,
                label="Seed mean and paired video 95% interval",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncols=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.04,
        "Challenging union includes low-angle, oblique and cropped frames. Side-on: 0 frames.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.19, bottom=0.25, right=0.97, top=0.79, wspace=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = comparison(args.campaign.resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "comparison.json", result)
    write_csv(args.output / "metrics.csv", result["metrics"])
    plot_test(result["effects"], args.output / "test-effects.png")
    print(
        f"{result['classification']}: {len(result['runs'])} exact-update runs; {len(result['metrics'])} metric rows in {args.output}"
    )


if __name__ == "__main__":
    main()
