"""Extract small, pinned provenance from the closed campaign; raw reports stay on Drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare import (
    CONDITIONS,
    MEMBERS,
    SCENES,
    digest,
    ordered_digest,
    require,
    sha_bytes,
    write_json,
)

CAMPAIGN = "outputs/court_detection/campaign/scene-pair-5500-s384-colab"
SCIENTIFIC_REF = "17ad301bf8d80a5d094f28bb48f3431c7d9b5d89"


def extract(state: Path, evaluations: Path, legacy: Path) -> dict[str, Any]:
    inputs = json.loads((state / "inputs.json").read_text())
    identity = digest(inputs)
    require(
        identity == "4d7b39d613471fd9f94a64dcb36d8fbba3a939fa0517773125af394cc3c1ca33",
        "Canonical inputs identity changed",
    )
    progress = json.loads((state / "progress.json").read_text())
    require(
        progress["stage"] == "complete" and progress["status"] == "completed",
        "Campaign is not complete",
    )
    expected_stages = {
        "inputs",
        "probe-P02",
        *(f"evaluate-{condition}" for condition in CONDITIONS),
        *(
            f"train-{condition}"
            for condition in CONDITIONS
            if condition.startswith("P")
        ),
    }
    require(
        {p.stem for p in (state / "done").glob("*.json")} == expected_stages,
        "Expected the exact sixteen closed stages",
    )
    result: dict[str, Any] = {
        "schema": "court_scene_pair_comparison_sources_v1",
        "scientific_ref": SCIENTIFIC_REF,
        "group_source": "https://github.com/Motoki0705/tennis-lab/issues/1581#issuecomment-6533537917",
        "inputs_identity": identity,
        "drive_campaign": CAMPAIGN,
        "campaign_progress": progress,
        "common": {
            "evaluation_manifest_sha256": inputs["evaluation_manifest_sha256"],
            "real_ids_sha256": ordered_digest(inputs["real_ids"]),
            "synthetic_ids_sha256": ordered_digest(inputs["synthetic_ids"]),
            "source_objects": inputs["source_objects"],
            "pool_sha256": inputs["pool_sha256"],
            "pretrained_sha256": inputs["pretrained_sha256"],
        },
        "conditions": {},
        "restarts": [],
        "verification_scope": "Hashes and report identities extracted from the actual verified Drive campaign. The original Colab campaign checked full checkpoint payloads before publishing done markers. This CPU aggregation does not reload checkpoint payloads.",
    }
    for condition in CONDITIONS:
        marker_path = state / "done" / f"evaluate-{condition}.json"
        marker = json.loads(marker_path.read_text())
        checked = marker["result"]
        raw = (evaluations / f"{condition}.json").read_bytes()
        report = json.loads(raw)
        require(
            marker["schema"] == "court_scene_pair_stage_v1"
            and marker["inputs_identity"] == identity
            and marker["stage"] == {"kind": "evaluate", "condition": condition},
            "Evaluation marker identity changed",
        )
        require(
            sha_bytes(raw) == checked["evaluation_sha256"],
            "Report differs from original done marker",
        )
        require(
            report["checkpoint_sha256"] == checked["checkpoint_sha256"]
            and report["data_condition"] == checked["data_condition"],
            "Report/marker checkpoint or data changed",
        )
        require(
            report["global_step"] == checked["global_step"] == 33160,
            "Checkpoint update changed",
        )
        require(
            set(report["data_condition"]["scene_images"]) == set(MEMBERS[condition]),
            "Condition membership differs",
        )
        pin = {
            k: checked[k]
            for k in (
                "checkpoint",
                "checkpoint_sha256",
                "evaluation",
                "evaluation_sha256",
                "global_step",
            )
        }
        pin.update(
            {
                "config_digest": digest(report["config"]),
                "data_condition_digest": digest(report["data_condition"]),
                "train_scene_images": report["data_condition"]["scene_images"],
                "fit_val_scene_images": report["data_condition"][
                    "synthetic_validation"
                ]["scene_images"],
                "data_manifest_sha256": report["data_manifest_sha256"],
                "original_marker": f"{CAMPAIGN}/done/{marker_path.name}",
                "original_marker_sha256": sha_bytes(marker_path.read_bytes()),
                "evaluation_artifact_count": len(checked["evaluation_files"]),
                "training_source": report["config"]["data"]["source"],
            }
        )
        if condition.startswith("scene-"):
            old_path = (
                legacy
                / f"run-court-scene-diversity-colab-{condition}-s42"
                / "evaluation.json"
            )
            old_report = json.loads(old_path.read_text())
            require(
                old_report["checkpoint_sha256"] == report["checkpoint_sha256"],
                "Legacy checkpoint changed",
            )
            old = old_report["splits"]["tennis_court_detector-val"]
            new = report["splits"]["tennis_court_detector-val"]
            require(
                old["sample_ids"] == new["sample_ids"]
                and old["metrics"] == new["metrics"],
                "Legacy real evaluation changed",
            )
            pin["legacy_real_five_metrics_and_ids_exact_equal"] = True
            pin["legacy_evaluation_sha256"] = sha_bytes(old_path.read_bytes())
        else:
            train = json.loads((state / "done" / f"train-{condition}.json").read_text())
            require(
                train["inputs_identity"] == identity
                and all(
                    train["result"][k] == checked[k]
                    for k in (
                        "checkpoint",
                        "checkpoint_sha256",
                        "global_step",
                        "data_condition",
                    )
                ),
                "Training/evaluation marker changed",
            )
        result["conditions"][condition] = pin
    for path in sorted((state / "restarts").glob("*.json")):
        restart = json.loads(path.read_text())
        checkpoint = restart["resume_checkpoint"]
        result["restarts"].append(
            {
                "record": f"{CAMPAIGN}/restarts/{path.name}",
                "sha256": sha_bytes(path.read_bytes()),
                "condition": restart["condition"],
                "probe": restart["probe"],
                "resume_update": restart["resume_update"],
                "discarded_updates": restart["discarded_updates"],
                "limit": restart["limit"],
                "resume_checkpoint": {
                    k: checkpoint[k]
                    for k in ("checkpoint", "checkpoint_sha256", "global_step")
                }
                if checkpoint
                else None,
            }
        )
    require(len(result["restarts"]) == 9, "Restart inventory changed")
    require(
        {i.split(":", 1)[0] for i in inputs["synthetic_ids"]} == set(SCENES),
        "Synthetic scene membership changed",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument(
        "--legacy-runs", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, extract(args.campaign, args.evaluations, args.legacy_runs))


if __name__ == "__main__":
    main()
