"""Compare identical Meiji 3cam reference units using the saved #964 and #2445 scores."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from src.tasks.person_tracking.evaluation import score_units
from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

ID_FIELDS = (
    "idf1",
    "idtp",
    "idfp",
    "idfn",
    "player_units",
    "player_units_kept",
    "nonplayer_units",
    "nonplayer_units_kept",
    "id_switches",
    "fragments",
)


def selection_metrics(ids: dict) -> dict:
    result = {name: ids[name] for name in ID_FIELDS}
    kept, nonplayer = result["player_units_kept"], result["nonplayer_units_kept"]
    result["precision"] = kept / (kept + nonplayer) if kept + nonplayer else None
    result["recall"] = kept / result["player_units"] if result["player_units"] else None
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pair(old: dict, new: dict) -> None:
    for field in ("player_units", "nonplayer_units"):
        if old[field] != new[field]:
            raise ValueError(f"Mismatched reference denominator: {field}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True, help="Archived outputs root")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evidence = Path(__file__).resolve().parents[1]
    dev_new = read_json(evidence / "run-i2445-player-slot-integration/meiji3.json")
    holdout_new = read_json(evidence / "run-i2445-reserved-tracking/meiji3.json")
    old = args.archive / "person_tracking/evaluate"
    r10 = old / "tracker_matrix/i964-hybrids-r10-20260930"
    dev_csv = r10 / "comparison.csv"
    identity = read_json(r10 / "identity.json")
    with dev_csv.open(newline="", encoding="utf-8") as handle:
        (row,) = [
            r
            for r in csv.DictReader(handle)
            if (r["variant"], r["camera"], r["near_far"])
            == ("strongsort_pp_pose__clipreid_vitb16_market1501", "all", "all")
        ]
    if (
        set(dev_new["overall"]["clips"]) != set(dev_new["clips"])
        or len(dev_new["clips"]) != 4
    ):
        raise ValueError("Wrong development cohort")
    for key, record in dev_new["clips"].items():
        if record["inputs"]["r7_plan_sha256"] != identity["plan"]["sha256"] or record[
            "inputs"
        ]["labels_sha256"] != dual_sha256(label_path(args.dataset, key)):
            raise ValueError(f"Development plan/label pair changed: {key}")
    dev_legacy = selection_metrics(
        {
            name: float(row["group_" + name])
            if name == "idf1"
            else int(row["group_" + name])
            for name in ID_FIELDS
        }
    )
    dev_projection = selection_metrics(dev_new["overall"]["projected"]["selected_id"])
    validate_pair(dev_legacy, dev_projection)
    reserved = old / "i964-unseen-r16-20261001/scoring"
    original = read_json(reserved / "summary.json")
    attempt = read_json(reserved / "attempt.json")
    if (
        set(attempt["labels"]) != set(holdout_new["clips"])
        or len(holdout_new["clips"]) != 3
    ):
        raise ValueError("Wrong reserved cohort")
    for key, record in holdout_new["clips"].items():
        if record["inputs"]["labels_sha256"] != attempt["labels"][key][
            "sha256"
        ] or record["inputs"]["labels_sha256"] != dual_sha256(
            label_path(args.dataset, key)
        ):
            raise ValueError(f"Reserved labels changed: {key}")
    with gzip.open(reserved / "group-units.jsonl.gz", "rt", encoding="utf-8") as handle:
        units = [json.loads(line) for line in handle]
    if set(u["clip"] for u in units) != set(holdout_new["clips"]):
        raise ValueError("Reserved legacy units do not cover the declared clips")
    recomputed = selection_metrics(score_units(units))
    if recomputed != selection_metrics(original["tracking"]["group"]):
        raise ValueError("Reserved legacy group scoring did not reproduce")
    result = {
        "development": {
            "clips": dev_new["overall"]["clips"],
            "legacy": dev_legacy,
            "multiview_projected": dev_projection,
            "multiview_supporting_rows": selection_metrics(
                dev_new["overall"]["supporting_rows"]["selected_id"]
            ),
        },
        "provenance": {
            "legacy_development_score_sha256": dual_sha256(dev_csv),
            "legacy_reserved_attempt_sha256": dual_sha256(reserved / "attempt.json"),
            "legacy_reserved_units_sha256": dual_sha256(
                reserved / "group-units.jsonl.gz"
            ),
            "legacy_reserved_clip_status": {
                key: read_json(reserved / key / "result.json")["status"]
                for key in holdout_new["clips"]
            },
        },
    }
    for group in ("all_three", "without_repaired_clip"):
        block = holdout_new[group]
        reference = selection_metrics(
            score_units([unit for unit in units if unit["clip"] in block["clips"]])
        )
        projected = selection_metrics(block["projected"]["selected_id"])
        validate_pair(reference, projected)
        result[group] = {
            "clips": block["clips"],
            "legacy": reference,
            "multiview_projected": projected,
            "multiview_supporting_rows": selection_metrics(
                block["supporting_rows"]["selected_id"]
            ),
        }
    write_json_atomic(args.output, result)
    for split in ("development", "all_three", "without_repaired_clip"):
        print(
            split,
            {
                k: {
                    f: v[f]
                    for f in ("idf1", "id_switches", "fragments", "precision", "recall")
                }
                for k, v in result[split].items()
                if k != "clips"
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
