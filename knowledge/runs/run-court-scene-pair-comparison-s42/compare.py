"""Aggregate frozen Court evaluations on CPU; no model inference or training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

METRICS = (
    "kp_mean_distance_px",
    "kp_median_distance_px",
    "seg_miou",
    "line_dice",
    "semantic_line_miou",
)
SCENES = ("B00", "B01", "B02", "B03")
GROUPS = {"B00": "omni", "B01": "omni", "B02": "clay", "B03": "clay"}
MEMBERS = {
    "scene-1": ("B00",),
    "scene-2": ("B00", "B01"),
    "scene-3": ("B00", "B01", "B02"),
    "scene-4": SCENES,
    "P02": ("B00", "B02"),
    "P03": ("B00", "B03"),
    "P12": ("B01", "B02"),
    "P13": ("B01", "B03"),
    "P23": ("B02", "B03"),
}
CONDITIONS = tuple(MEMBERS)
PAIR_NAMES = ("P01", "P02", "P03", "P12", "P13", "P23")
SCENE_COUNTS = {"B00": 1056, "B01": 643, "B02": 462, "B03": 278}
ALIASES = {"scene-2": "P01"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def ordered_digest(ids: list[str]) -> str:
    # Same serialization as the frozen evaluation's ordered_sample_ids_sha256.
    return sha_bytes(json.dumps(ids, separators=(",", ":")).encode())


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_metrics(values: dict[str, Any]) -> None:
    require(set(METRICS) <= values.keys(), "Missing required metrics")
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values.values()),
        "Non-finite or non-numeric metric",
    )


def validate_report(
    condition: str, report: dict[str, Any], sources: dict[str, Any]
) -> None:
    pin, common = sources["conditions"][condition], sources["common"]
    require(
        report["schema"] == "court_data_scaling_evaluation_v2"
        and report["series"] == "scene-pairs",
        "Unexpected evaluation schema/series",
    )
    require(report["global_step"] == 33160, "Final update changed")
    require(
        report["checkpoint_sha256"] == pin["checkpoint_sha256"]
        and report["checkpoint"] == "/content/tennis-lab/" + pin["checkpoint"],
        "Checkpoint identity changed",
    )
    require(digest(report["config"]) == pin["config_digest"], "Config identity changed")
    require(
        digest(report["data_condition"]) == pin["data_condition_digest"],
        "Data identity changed",
    )
    require(
        report["config"]["run"]["seed"] == 42
        and report["config"]["data"]["batch_size"] == 8,
        "Training seed/batch changed",
    )
    data = report["data_condition"]
    require(
        data["accepted_images"] == 5500
        and data["training_seed"] == 42
        and data["synthetic_validation"]["images"] == 500,
        "Training/fit-validation budget changed",
    )
    require(
        set(data["scene_images"]) == set(MEMBERS[condition]),
        "Training scene membership changed",
    )
    count = len(MEMBERS[condition])
    expected_val = {
        scene: 500 // count + int(i < 500 % count)
        for i, scene in enumerate(MEMBERS[condition])
    }
    expected_train = {
        scene: 6000 // count - images for scene, images in expected_val.items()
    }
    require(
        data["scene_images"] == expected_train
        and data["synthetic_validation"]["scene_images"] == expected_val,
        "Per-scene training/fit-validation allocation changed",
    )
    require(
        report["synthetic_evaluation_manifest"]["sha256"]
        == common["evaluation_manifest_sha256"],
        "Evaluation manifest changed",
    )
    splits = report["splits"]
    require(
        set(splits) == {"tennis_court_detector-val", "synthetic_court-val"},
        "Both domains required",
    )
    for split_name, count, ids_key, pose in (
        ("tennis_court_detector-val", 2211, "real_ids_sha256", False),
        ("synthetic_court-val", 2439, "synthetic_ids_sha256", True),
    ):
        split = splits[split_name]
        ids = split["sample_ids"]
        require(
            split["count"] == count and len(ids) == count and len(set(ids)) == count,
            "Evaluation count/uniqueness changed",
        )
        require(
            split["ordered_sample_ids_sha256"]
            == common[ids_key]
            == ordered_digest(ids),
            "Evaluation ID order changed",
        )
        require(
            split["batch_size"] == 8 and split["pose_supervised"] is pose,
            "Evaluation batch/supervision changed",
        )
        finite_metrics(split["metrics"])
    synthetic = splits["synthetic_court-val"]
    require(
        synthetic["scene_order"] == list(SCENES)
        and set(synthetic["scenes"]) == set(SCENES),
        "Scene order/inventory changed",
    )
    for scene in SCENES:
        row = synthetic["scenes"][scene]
        ids = [i for i in synthetic["sample_ids"] if i.startswith(scene + ":")]
        require(
            row["count"] == SCENE_COUNTS[scene] == len(ids)
            and row["sample_ids"] == ids
            and row["ordered_sample_ids_sha256"] == ordered_digest(ids),
            "Scene count/order changed",
        )
        finite_metrics(row["metrics"])
    for metric in METRICS:
        # Reports were produced with Python 3.11. Explicit left-fold preserves
        # its sum exactly; Python 3.12 changed float sum's rounding algorithm.
        mean = (
            reduce(
                add, (synthetic["scenes"][s]["metrics"][metric] for s in SCENES), 0.0
            )
            / 4
        )
        require(
            synthetic["scene_macro_mean"][metric] == mean,
            "Four-scene macro mean changed",
        )


def load_reports(directory: Path, sources: dict[str, Any]) -> dict[str, Any]:
    require(
        set(sources["conditions"]) == set(CONDITIONS),
        "Source condition inventory changed",
    )
    require(
        {p.stem for p in directory.glob("*.json")} == set(CONDITIONS),
        "Evaluation inventory must be exactly nine conditions",
    )
    reports = {}
    for condition in CONDITIONS:
        raw = (directory / f"{condition}.json").read_bytes()
        require(
            sha_bytes(raw) == sources["conditions"][condition]["evaluation_sha256"],
            f"{condition}: raw evaluation hash changed",
        )
        report = json.loads(raw)
        validate_report(condition, report, sources)
        reports[condition] = report
    return reports


def contrasts(
    pairs: dict[str, dict[str, float]], metrics: tuple[str, ...] = METRICS
) -> list[dict[str, Any]]:
    require(set(pairs) == set(PAIR_NAMES), "Exactly six pairs required; P01 is scene-2")
    result = []
    for metric in metrics:
        same = math.fsum(pairs[p][metric] for p in ("P01", "P23")) / 2
        cross = math.fsum(pairs[p][metric] for p in ("P02", "P03", "P12", "P13")) / 4
        additivity = (pairs["P02"][metric] + pairs["P13"][metric]) - (
            pairs["P03"][metric] + pairs["P12"][metric]
        )
        result.append(
            {
                "metric": metric,
                "same_group_mean": same,
                "cross_group_mean": cross,
                "delta": same - cross,
                "additivity": additivity,
                "same_group_worse_direction": "positive"
                if metric.startswith("kp_")
                else "negative",
            }
        )
    return result


def category(condition: str, scene: str) -> str:
    require(condition in MEMBERS and scene in GROUPS, "Unknown condition/scene")
    seen = MEMBERS[condition]
    if scene in seen:
        return "seen"
    if GROUPS[scene] in {GROUPS[s] for s in seen}:
        return "unseen-with-same-group"
    return "unseen-without-same-group"


def summarize(reports: dict[str, Any]) -> dict[str, Any]:
    require(set(reports) == set(CONDITIONS), "Exactly nine conditions required")
    real_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    matrices: dict[str, list[dict[str, Any]]] = {m: [] for m in METRICS}
    for condition in CONDITIONS:
        report = reports[condition]
        label = ALIASES.get(condition, condition)
        real_rows.append(
            {
                "condition": label,
                "source_condition": condition,
                "training_scenes": "+".join(MEMBERS[condition]),
                **{
                    m: report["splits"]["tennis_court_detector-val"]["metrics"][m]
                    for m in METRICS
                },
            }
        )
        scenes = report["splits"]["synthetic_court-val"]["scenes"]
        for scene in SCENES:
            synthetic_rows.append(
                {
                    "condition": label,
                    "scene": scene,
                    "count": SCENE_COUNTS[scene],
                    "category": category(condition, scene),
                    **{m: scenes[scene]["metrics"][m] for m in METRICS},
                }
            )
        for metric in METRICS:
            values = {s: scenes[s]["metrics"][metric] for s in SCENES}
            matrices[metric].append(
                {
                    "condition": label,
                    **values,
                    "four_scene_mean": math.fsum(values.values()) / 4,
                }
            )
    by_name = {r["condition"]: r for r in real_rows}
    pairs = [by_name[p] for p in PAIR_NAMES]
    additions = [
        {**by_name[p], "added_2750": s}
        for p, s in zip(("scene-1", "P01", "P02", "P03"), SCENES, strict=True)
    ]
    return {
        "schema": "court_scene_pair_comparison_v1",
        "seed": 42,
        "updates": 33160,
        "train_images": 5500,
        "real_val_images": 2211,
        "synthetic_eval_images": 2439,
        "primary_metric": "kp_median_distance_px",
        "groups": GROUPS,
        "real_all": real_rows,
        "real_pairs": pairs,
        "contrasts": contrasts({r["condition"]: r for r in pairs}),
        "b00_addition": additions,
        "synthetic_long": synthetic_rows,
        "synthetic_matrix": matrices,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(result: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rows = result["synthetic_matrix"]["kp_median_distance_px"]
    codes = {
        "seen": "S",
        "unseen-with-same-group": "G",
        "unseen-without-same-group": "N",
    }
    cells = {
        (r["condition"], r["scene"]): codes[r["category"]]
        for r in result["synthetic_long"]
    }
    values = [[row[s] for s in SCENES] for row in rows]
    fig, ax = plt.subplots(figsize=(9.4, 7.5), layout="constrained")
    heat = ax.imshow(
        values,
        cmap="YlOrRd",
        norm=LogNorm(
            vmin=min(min(v) for v in values), vmax=max(max(v) for v in values)
        ),
        aspect="auto",
    )
    ax.set_xticks(
        range(4),
        [
            "B00 (omni)\nn=1,056",
            "B01 (omni)\nn=643",
            "B02 (clay)\nn=462",
            "B03 (clay)\nn=278",
        ],
    )
    ax.set_yticks(
        range(9),
        [f"{row['condition']}   mean {row['four_scene_mean']:.2f}" for row in rows],
    )
    for i, row in enumerate(rows):
        for j, scene in enumerate(SCENES):
            value = row[scene]
            ax.text(
                j,
                i,
                f"{value:.2f}  {cells[row['condition'], scene]}",
                ha="center",
                va="center",
                color="white" if heat.norm(value) > 0.65 else "black",
                fontsize=10,
            )
    ax.set_title(
        "Common synthetic evaluation: KP median (px, lower is better)\n9 checkpoints / seed 42 / final 33,160 updates",
        fontsize=12,
        pad=14,
    )
    ax.set_xlabel(
        "S: trained venue   G: untrained venue, same group represented\nN: untrained venue, group absent | row mean: unweighted 4-venue mean\nAll evaluation trajectories are held out within these four venues.",
        labelpad=12,
    )
    fig.colorbar(heat, ax=ax, label="KP median px (log color scale)", shrink=0.78)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument(
        "--sources", type=Path, default=Path(__file__).with_name("sources.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = load_reports(args.evaluations, json.loads(args.sources.read_text()))
    result = summarize(reports)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", result)
    for name in (
        "real_all",
        "real_pairs",
        "contrasts",
        "b00_addition",
        "synthetic_long",
    ):
        write_csv(args.output / f"{name}.csv", result[name])
    for metric, rows in result["synthetic_matrix"].items():
        write_csv(args.output / f"synthetic-{metric}.csv", rows)
    plot_matrix(result, args.output / "synthetic-kp-median.png")
    print(
        json.dumps(
            {"conditions": len(reports), "contrasts": result["contrasts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
