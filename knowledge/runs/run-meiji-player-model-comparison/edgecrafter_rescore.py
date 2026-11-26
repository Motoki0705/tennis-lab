"""Rescore saved EdgeCrafter predictions with the checked-out common scorer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prediction_rows import PredictionHeader, read_predictions

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
)
from src.tennis_scene.evaluation.player_pose.scoring import score_players
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

SCORER_PATH = Path("src/tennis_scene/evaluation/player_pose/scoring.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("metrics", type=Path, nargs="+")
    args = parser.parse_args()
    manifest = load_manifest(args.cohort, expected_sha256=args.manifest_sha256)
    benchmark_id = manifest["benchmark_id"]
    references = load_reference_frames(
        args.cohort, expected_sha256=args.manifest_sha256
    )
    scorer_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(SCORER_PATH)],
        text=True,
    ).strip()
    checkout_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "scorer_commit": scorer_commit,
        "scorer_source_sha256": dual_sha256(SCORER_PATH),
        "scoring_checkout_commit": checkout_commit,
    }
    records = []
    for previous_path in args.metrics:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if (previous["benchmark_id"], previous["manifest_sha256"]) != (
            benchmark_id,
            args.manifest_sha256,
        ):
            raise ValueError(f"Saved score is for a different cohort: {previous_path}")
        model_id, backend = previous["model_id"], previous["backend"]
        raw = previous_path.with_name("predictions.jsonl")
        raw_sha256 = dual_sha256(raw)
        if raw_sha256 != previous["predictions_sha256"]:
            raise ValueError(f"Saved predictions differ from original receipt: {raw}")
        header = PredictionHeader(
            benchmark_id, args.manifest_sha256, model_id, backend, previous["has_pose"]
        )
        predictions = read_predictions(raw, header, expected_frames=references)
        score = score_players(references, predictions)
        old_score = previous["score"]
        affected = sorted(
            key
            for key, value in score.items()
            if key != "policy" and value != old_score[key]
        )
        old_sha256 = dual_sha256(previous_path)
        updated_path = previous_path.with_name(
            f"metrics.scorer-{scorer_commit[:9]}.json"
        )
        write_json_atomic(
            updated_path,
            {
                **previous,
                **provenance,
                "score": score,
                "previous_metrics_sha256": old_sha256,
            },
        )
        parity_path = previous_path.with_name("parity-vs-pt32.json")
        parity = None
        if parity_path.is_file():
            proof = json.loads(parity_path.read_text(encoding="utf-8"))
            if proof["candidate_predictions_sha256"] != raw_sha256:
                raise ValueError(
                    f"Parity receipt uses different candidate predictions: {parity_path}"
                )
            parity = {
                "path": str(parity_path),
                "sha256": dual_sha256(parity_path),
                "status": proof["status"],
                "matched_fraction_both_directions": proof["metrics"][
                    "matched_fraction_both_directions"
                ],
                "matched_iou_p05": proof["metrics"]["matched_iou_p05"],
                "abs_score_error_p95": proof["metrics"]["abs_score_error"]["p95"],
            }
        detection = score["overall"]["detection"]
        records.append(
            {
                "model_id": model_id,
                "backend": backend,
                "images": score["images"],
                "eligible_boxes": detection["references"],
                "recall_by_iou": detection["recall_by_iou"],
                "previous_recall_by_iou": old_score["overall"]["detection"][
                    "recall_by_iou"
                ],
                "changed_aggregates": affected,
                "predictions_path": str(raw),
                "predictions_sha256": raw_sha256,
                "accuracy_path": str(updated_path),
                "accuracy_sha256": dual_sha256(updated_path),
                "previous_metrics_sha256": old_sha256,
                "parity": parity,
                "weight_sha256": previous["weight_sha256"],
                "engine_sha256": previous["engine_sha256"],
            }
        )
        print(
            f"{model_id} {backend} scored {score['images']} images. Affected aggregates {affected}",
            flush=True,
        )
    write_json_atomic(
        args.summary,
        {
            **provenance,
            "benchmark_id": benchmark_id,
            "manifest_sha256": args.manifest_sha256,
            "conditions": records,
        },
    )


if __name__ == "__main__":
    main()
