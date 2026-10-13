"""Choose the additional-seed arm using seed42 validation receipts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .issue1044 import ARMS


def select_additional_arm(root: Path) -> dict[str, Any]:
    receipts = []
    for arm in ARMS:
        receipt = json.loads((root / f"{arm}-s42-u4000/selection.json").read_text())
        if (
            receipt["arm"] != arm
            or receipt["seed"] != 42
            or receipt["optimizer_updates"] != 4000
            or receipt["smoke"]
            or receipt["selection_used_test"]
            or receipt["protocol"] != "issue1044-v1"
        ):
            raise ValueError(f"Incomplete or inconsistent seed42 receipt: {arm}")
        receipts.append(receipt)
    candidates = [row for row in receipts if row["arm"] in ARMS[2:]]
    chosen = min(candidates, key=lambda row: row["validation_trajectory_rmse_m"])
    selection = {
        "schema": "blcs.issue1044.additional_seeds.v1",
        "purpose": "additional seeds only; no adoption decision",
        "criterion": "minimum checkpoint validation trajectory_rmse_m",
        "used_test": False,
        "selected_arm": chosen["arm"],
        "seed42_receipts": receipts,
    }
    path = root / "additional-seed-selection.json"
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError("Previously frozen additional-seed selection changed")
    else:
        path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection
