"""Compare distinct checkpoints only after exact teacher/sample correspondence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.slcs.evaluation.comparison import (
    CONDITIONS,
    MATCH_KEYS,
    METRICS,
    save_comparison,
)


def main() -> None:
    output = Path("/home/kamimura/projects/tennis-lab/outputs")
    summaries: dict[str, Any] = {}
    rows = []
    for split in ("val", "test"):
        summaries[split] = {}
        for model in ("baseline", "augmented"):
            directory = output / f"slcs/analyze/real_rgb_pilot_{model}/s42-002/{split}"
            summaries[split][model] = json.loads((directory / "comparison.json").read_text())
        assert summaries[split]["baseline"]["checkpoint_sha256"] != summaries[split]["augmented"]["checkpoint_sha256"]
        for condition in CONDITIONS:
            files = [output / f"slcs/evaluate/real_rgb_pilot_{model}_{split}_{condition}/s42-002/eval_arrays.npz" for model in ("baseline", "augmented")]
            with np.load(files[0], allow_pickle=False) as baseline, np.load(files[1], allow_pickle=False) as augmented:
                for key in MATCH_KEYS:
                    assert baseline[key].dtype == augmented[key].dtype and np.array_equal(baseline[key], augmented[key]), (split, condition, key)
        baseline_rows = summaries[split]["baseline"]["rows"]
        augmented_rows = summaries[split]["augmented"]["rows"]
        assert len(baseline_rows) == len(augmented_rows)
        for baseline_row, augmented_row in zip(baseline_rows, augmented_rows, strict=True):
            row = {"split": split}
            for key in ("group_type", "group", "condition"):
                assert baseline_row[key] == augmented_row[key]
                row[key] = baseline_row[key]
            for metric in METRICS:
                row[f"baseline_{metric}"] = baseline_row[metric]
                row[f"augmented_{metric}"] = augmented_row[metric]
                row[f"augmented_minus_baseline_{metric}"] = augmented_row[metric] - baseline_row[metric]
            rows.append(row)
    save_comparison({"interpretation": "Paired pseudo-teacher agreement for different validation-selected checkpoints. Negative augmented-minus-baseline favors augmentation; not independent 3D accuracy or a replicated causal estimate. No Meiji test recording in pilot.", "correspondence": "All MATCH_KEYS, including IDs/order/targets/masks/weights, exactly match for both splits and four conditions", "rows": rows}, output / "slcs/analyze/real_rgb_pilot_comparison/s42-002/paired_models")


if __name__ == "__main__":
    main()
