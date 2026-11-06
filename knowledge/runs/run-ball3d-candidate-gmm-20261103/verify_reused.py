"""Compare complete typed prediction records and all non-runtime scores."""

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from src.tennis_scene.evaluation.ball3d.runner import load_run


def same(left, right):
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype:
            raise ValueError("Prediction dtype changed")
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError("Prediction diagnostic fields changed")
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        for a, b in zip(left, right, strict=True):
            same(a, b)
    elif left != right:
        raise ValueError(f"Prediction scalar changed: {left} vs {right}")


def without_runtime(value):
    if isinstance(value, dict):
        return {key: without_runtime(v) for key, v in value.items() if not key.startswith("runtime_seconds")}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    checked = []
    for path in sorted((args.before/"results").glob("*/*/result.json")):
        relative = path.parent.relative_to(args.before)
        same(load_run(path.parent), load_run(args.after/relative))
        checked.append(str(relative))
    if len(checked) != 20:
        raise ValueError("Expected all 20 prediction runs")
    for name in ("scores.json", "paired-scores.json", "uncertainty.json", "frozen-calibration.json"):
        same(without_runtime(json.loads((args.before/name).read_text())),
             without_runtime(json.loads((args.after/name).read_text())))
    result = {"prediction_records_exact": checked, "non_runtime_scores_exact": True,
        "calibration_exact": True, "uncertainty_exact": True,
        "presentation_only_changes": "panel clipping, distinct ellipsoid colours, representative frames selected by availability"}
    (Path(__file__).parent/"reused-predictions-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print("All 20 complete typed prediction records and non-runtime scores are unchanged")


if __name__ == "__main__":
    main()
