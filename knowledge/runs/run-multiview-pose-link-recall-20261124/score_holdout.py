"""Score the three reserved Meiji 3cam clips once after queued ViTPose succeeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assignment_probe import player_only
from holdout_pose import _solver_config
from probe import read_json, read_output
from recover_holdout_camera import CLIP as REPAIRED_CLIP
from recover_holdout_camera import _camera
from replay import micro_pairs, predictions_from_rows

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tennis_scene.multiview_pose.inputs import PoseInputs
from src.tennis_scene.multiview_pose.reconstruct import reconstruct
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic

_RUN = Path(__file__).resolve().parent
_RESERVATION = _RUN.parent / "run-i964-unseen-r16-20261001"


def _cameras(
    source: ClipSource, entry: dict[str, Any], archive: Path, repair: dict[str, Any]
) -> tuple[PinholeCamera, ...]:
    receipt = archive / source.clip_id / "court-execute.json"
    if dual_sha256(receipt) != entry["court_receipt_sha256"]:
        raise ValueError(f"The reserved Court calibration changed: {source.clip_id}")
    if source.clip_id == REPAIRED_CLIP:
        side = repair["ball_decision"]
        if (
            repair["clip"] != source.clip_id
            or repair["source_sha256"] != entry["source_sha256"]
            or repair["prior_calibration_receipt_sha256"]
            != entry["court_receipt_sha256"]
            or repair["court_fit"]["rmse_px"] > repair["court_fit"]["limit_px"]
            or side["status"] != "decided"
            or tuple(side["camera_ids"]) != source.camera_ids
            or side["reference_camera"] != "cam0"
            or side["frames"] != 1249
            or side["view_half_turns"] != [False, False, True]
        ):
            raise ValueError("Expected the approved Court and ball recovery")
        documents = repair["cameras"]
        turns = side["view_half_turns"]
    else:
        courts = read_json(receipt)
        if any(courts[cam]["status"] != "ok" for cam in source.camera_ids):
            raise ValueError(f"A reserved calibration is missing: {source.clip_id}")
        documents = [
            courts[cam]["calibration"]["views"][0]["camera"]
            for cam in source.camera_ids
        ]
        turns = entry["reviewed_ball_turns"]
    if (
        tuple(doc["camera_id"] for doc in documents) != source.camera_ids
        or turns is None
    ):
        raise ValueError(f"Calibration and ball-decision axes differ: {source.clip_id}")
    return tuple(
        _camera(doc).half_turned(turn)
        for doc, turn in zip(documents, turns, strict=True)
    )


def _observations(
    stores: Path,
    clip: str,
    entry: dict[str, Any],
    complete: dict[str, Any],
    plan_sha: str,
    archive: Path,
    repair: dict[str, Any],
) -> PoseInputs:
    directory = (stores / clip).resolve()
    index = read_json(directory / "scene.json")
    source = restore_type(index["source"], ClipSource)
    source_sha = entry["source_sha256"]
    nodes = {
        f"{port}/{camera}"
        for port in ("person_detection", "person_pose_features")
        for camera in source.camera_ids
    }
    if (
        source.clip_id != clip
        or index["source"] != entry["source"]
        or index["source_sha256"] != source_sha
        or document_digest(source) != source_sha
        or set(index["artifacts"]) != nodes
        or set(complete) != set(source.camera_ids)
    ):
        raise ValueError(f"Completed observations differ from the reservation: {clip}")
    detections, poses = [], []
    for camera in source.camera_ids:
        det_node, pose_node = (
            f"person_detection/{camera}",
            f"person_pose_features/{camera}",
        )
        det_ref, pose_ref = (index["artifacts"][node] for node in (det_node, pose_node))
        identity = {
            "holdout_plan_sha256": plan_sha,
            "raw_detection": entry["detections"][camera],
        }
        for node, ref in ((det_node, det_ref), (pose_node, pose_ref)):
            desc = read_component_descriptor(
                directory, ref, node=node, source_sha256=source_sha
            )
            if desc["identity"] != identity:
                raise ValueError(
                    f"Published port belongs to different inputs: {clip}/{node}"
                )
        det, det_deps = read_output(
            directory, index, det_node, "person_detections", 2, PersonDetectionOutput
        )
        pose, pose_deps = read_output(
            directory,
            index,
            pose_node,
            "person_pose_features",
            2,
            PersonPoseFeaturesOutput,
        )
        if (
            det_deps
            or pose_deps != {"detections": det_ref}
            or complete[camera] != {"rows": len(det.boxes_xyxy), "pose": pose_ref}
            or len(det.boxes_xyxy) != entry["raw_dino_rows"][camera]
        ):
            raise ValueError(
                f"Original DINO and completed pose rows differ: {clip}/{camera}"
            )
        detections.append(det)
        poses.append(pose)
    return PoseInputs(
        source,
        _cameras(source, entry, archive, repair),
        tuple(detections),
        tuple(poses),
    )


def _label_pins(dataset: Path, clips: set[str]) -> dict[str, tuple[Path, str]]:
    receipt = read_json(_RESERVATION / "label-receipt-r18.json")
    pinned = {
        "/".join(Path(meta["snapshot"]).parts[-3:-1]): meta["sha256"]
        for meta in receipt["files"].values()
        if Path(meta["snapshot"]).name == "labels.json"
    }
    if set(pinned) != clips:
        raise ValueError("The partial references cover a different reserved clip set")
    paths = {clip: (label_path(dataset, clip), sha) for clip, sha in pinned.items()}
    if any(dual_sha256(path) != sha for path, sha in paths.values()):
        raise ValueError("A reserved partial reference changed")
    return paths


def score(plan_path: Path, poses: Path, output: Path) -> None:
    attempt = poses / "scoring-attempt.json"
    if attempt.exists() or output.exists():
        raise FileExistsError("The single reserved scoring attempt already started")
    plan_sha = dual_sha256(plan_path)
    preparation = read_json(_RUN / "holdout-pose-preparation.json")
    plan = read_json(plan_path)
    if (
        plan_sha != preparation["plan_sha256"]
        or plan["solver_commit"] != preparation["solver_commit"]
        or plan["solver_config"] != _solver_config()
    ):
        raise ValueError("Expected the pinned plan and unchanged solver")
    completion = poses / "complete.json"
    complete = read_json(completion)
    entries = plan["clips"]
    if (
        complete["plan_sha256"] != plan_sha
        or complete["pose_model"] != plan["pose_model"]
        or set(complete["clips"]) != set(entries)
    ):
        raise ValueError("Expected completed ViTPose for all three reserved clips")
    repair_path = _RUN / "holdout-camera-recovery.json"
    repair = read_json(repair_path)
    labels_by_clip = _label_pins(Path(plan["dataset"]), set(entries))
    predictions: dict[str, dict[str, CameraPrediction]] = {}
    for clip, entry in entries.items():
        inputs = _observations(
            poses / "stores",
            clip,
            entry,
            complete["clips"][clip],
            plan_sha,
            Path(plan["archive"]),
            repair,
        )
        predictions[clip] = predictions_from_rows(
            inputs.detections, reconstruct(inputs)
        )
    pins = {
        "plan_sha256": plan_sha,
        "pose_completion_sha256": dual_sha256(completion),
        "camera_recovery_sha256": dual_sha256(repair_path),
        "solver_commit": plan["solver_commit"],
        "config": plan["solver_config"],
        "labels_sha256": {clip: sha for clip, (_, sha) in labels_by_clip.items()},
    }
    with attempt.open("x") as stream:
        json.dump(pins, stream, indent=2)
    results = {}
    for clip, (label, _) in labels_by_clip.items():
        labels = ClipLabels.load(label)
        if labels.clip_id != clip or labels.num_frames != entries[clip]["frames"]:
            raise ValueError(f"Wrong partial reference scope: {clip}")
        metrics = evaluate(labels, predictions[clip])
        results[clip] = {
            "pairs": metrics["pairs"],
            "player_pairs": evaluate(player_only(labels), predictions[clip])["pairs"],
            "label_coverage": metrics["coverage"],
        }

    def aggregate(clip_ids: list[str]) -> dict[str, Any]:
        return {
            "clips": clip_ids,
            **{
                key: micro_pairs(results[clip][key] for clip in clip_ids)
                for key in ("pairs", "player_pairs")
            },
        }

    write_json_atomic(
        output,
        {
            "pins": pins,
            "clips": results,
            "all_three": aggregate(sorted(results)),
            "without_repaired_clip": aggregate(
                sorted(results.keys() - {REPAIRED_CLIP})
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.plan, args.poses, args.output)


if __name__ == "__main__":
    main()
