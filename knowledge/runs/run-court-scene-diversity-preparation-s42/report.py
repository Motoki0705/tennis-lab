"""Compare the four fixed-budget runs, keeping synthetic and real validation separate."""

from __future__ import annotations

# Standalone reproduction scripts bootstrap this checkout before importing src.
# ruff: noqa: E402
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.tasks.court_detection.ablation.evaluation import DENSE_METRICS
from src.tasks.court_detection.scaling.comparison import comparison_contract


def summarize(reports):
    if len(reports) != 4:
        raise ValueError("Exactly four completed conditions are required.")
    reference = None
    rows = []
    seen = set()
    for report in reports:
        if report.get("schema") != "court_data_scaling_evaluation_v1":
            raise ValueError("Expected fixed-update evaluation receipts.")
        condition = report["data_condition"]
        scenes = len(condition["scene_images"])
        if scenes not in (1, 2, 3, 4) or scenes in seen:
            raise ValueError("Missing or repeated scene-count condition.")
        seen.add(scenes)
        expected_val = {
            f"B{i:02d}": 500 // scenes + int(i < 500 % scenes) for i in range(scenes)
        }
        expected_train = {
            scene: 6000 // scenes - n for scene, n in expected_val.items()
        }
        if (
            condition["scene_images"] != expected_train
            or condition["synthetic_validation"]["scene_images"] != expected_val
            or condition["training_seed"] != 42
        ):
            raise ValueError(
                "Per-scene counts or training seed differ from the agreed conditions."
            )
        if (
            report["global_step"] != 33160
            or condition["accepted_images"] != 5500
            or condition["synthetic_validation"]["images"] != 500
        ):
            raise ValueError("Training/validation size or update budget differs.")
        contract = comparison_contract(report)
        synthetic = contract["splits"].pop("synthetic_court-val")
        if synthetic["count"] != 500 or not synthetic["pose_supervised"]:
            raise ValueError(
                "Synthetic validation must contain 500 pose-supervised samples."
            )
        synthetic.pop("ordered_sample_ids_sha256")
        contract["synthetic_validation_contract"] = synthetic
        if (
            set(contract["splits"]) != {"tennis_court_detector-val"}
            or contract["splits"]["tennis_court_detector-val"]["count"] != 2211
        ):
            raise ValueError("Expected the original 2211-image real validation split.")
        if reference is None:
            reference = contract
        elif contract != reference:
            raise ValueError(
                "Model/training/preprocessing/real validation contracts differ."
            )
        row = {
            "scenes": scenes,
            "train_images": 5500,
            "synthetic_val_images": 500,
            "real_val_images": 2211,
            "updates": 33160,
        }
        for metric in DENSE_METRICS:
            syn = report["splits"]["synthetic_court-val"]["metrics"][metric]
            real = report["splits"]["tennis_court_detector-val"]["metrics"][metric]
            if any(
                type(v) not in (float, int) or not math.isfinite(v) for v in (syn, real)
            ):
                raise ValueError("Validation metrics must be finite.")
            row[f"synthetic/{metric}"] = syn
            row[f"real/{metric}"] = real
            row[f"real_minus_synthetic/{metric}"] = real - syn
        rows.append(row)
    return sorted(rows, key=lambda row: row["scenes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize([json.loads(path.read_text()) for path in args.evaluations])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(DENSE_METRICS), figsize=(22, 4), squeeze=False)
    for metric, axis in zip(DENSE_METRICS, axes[0], strict=True):
        for domain in ("synthetic", "real"):
            axis.plot(
                [row["scenes"] for row in rows],
                [row[f"{domain}/{metric}"] for row in rows],
                marker="o",
                label=domain,
            )
        axis.set(title=metric, xlabel="Training scenes", xticks=[1, 2, 3, 4])
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png")
    plt.close(figure)
    print(args.output / "comparison.csv")


if __name__ == "__main__":
    main()
