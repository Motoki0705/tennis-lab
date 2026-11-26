"""Check the reserved holdout inputs without reading or scoring the labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe import read_json, read_output

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import require_alignment
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic

_EVIDENCE = Path("knowledge/runs/run-i964-unseen-r16-20261001")


def inspect(dataset: Path, archive: Path) -> dict:
    plan_path = archive / "plan.json"
    plan = read_json(plan_path)
    pinned = read_json(_EVIDENCE / "preparation-complete-r17.json")["plan"]["sha256"]
    recorded = read_json(_EVIDENCE / "prepared-plan-r17.json")
    if dual_sha256(plan_path) != pinned or plan["records"] != recorded["records"]:
        raise ValueError("The unseen archive plan differs from the reservation")

    results = {}
    for record in plan["records"]:
        clip = record["clip"]
        directory = archive / clip / "store"
        store = read_json(directory / "scene.json")
        source = restore_type(record["source"], ClipSource)
        if document_digest(source) != store["source_sha256"]:
            raise ValueError(f"Unexpected clip metadata: {clip}")
        rows_by_camera = {}
        court_path = directory.parent / "court-execute.json"
        courts = read_json(court_path)
        label = label_path(dataset, clip)
        if not label.is_file():
            raise FileNotFoundError(label)
        for video in source.videos:
            if (
                dual_sha256(label.parents[2] / "media" / video.path.name)
                != video.sha256
            ):
                raise ValueError(f"Unexpected media: {clip}/{video.camera_id}")
            node = f"person_detection/{video.camera_id}"
            det, deps = read_output(
                directory, store, node, "person_detections", 2, PersonDetectionOutput
            )
            descriptor = read_json(directory / store["artifacts"][node]["path"])
            settings = descriptor["identity"]["settings"]
            if (
                deps
                or settings["detector"] != "dino"
                or settings["scope"] != "full_frame"
                or settings["merge_duplicates"]
                or settings["runtime"]["confidence"] != 0.3
            ):
                raise ValueError(
                    f"Unexpected raw detection contract: {clip}/{video.camera_id}"
                )
            rows = require_alignment(video, det)
            if not np.array_equal(rows, np.arange(len(rows))):
                raise ValueError(f"Missing source rows: {clip}/{video.camera_id}")
            rows_by_camera[video.camera_id] = len(rows)
            court = courts[video.camera_id]
            if court["status"] == "ok":
                document = court["calibration"]["views"][0]["camera"]
                camera = PinholeCamera(
                    document["camera_id"],
                    np.asarray(document["intrinsic"], np.float64),
                    np.asarray(document["rotation"], np.float64),
                    np.asarray(document["translation"], np.float64),
                )
                if camera.camera_id != video.camera_id:
                    raise ValueError(
                        f"Unexpected court calibration: {clip}/{video.camera_id}"
                    )
        results[clip] = {
            "frames": source.num_frames,
            "raw_dino_rows": rows_by_camera,
            "source_sha256": store["source_sha256"],
            "court_receipt_sha256": dual_sha256(court_path),
            "calibrated_cameras": [
                cam for cam in source.camera_ids if courts[cam]["status"] == "ok"
            ],
            "reviewed_ball_turns": record["view_half_turns"],
            "saved_pre_tracking_pose_ports": [
                cam
                for cam in source.camera_ids
                if f"person_pose_features/{cam}" in store["artifacts"]
            ],
            "partial_reference_file_present": True,
        }
    return {"r18_plan_sha256": pinned, "clips": results, "label_scoring_attempts": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, inspect(args.dataset, args.archive))


if __name__ == "__main__":
    main()
