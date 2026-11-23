"""CPU acceptance replay of framewise multiview pose on the pinned Meiji inputs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from assignment_probe import (
    DEV_CLIPS,
    OBSERVE,
    R7_EVIDENCE,
    SIDE_REPORT,
    player_only,
    read_all_pose_rows,
)
from numpy.typing import NDArray
from probe import read_output

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    evaluate,
    match_to_labels,
)
from src.tennis_scene.multiview_pose.geometry import view_rays
from src.tennis_scene.multiview_pose.inputs import (
    PinnedCourtSide,
    PoseInputs,
    read_saved_inputs,
)
from src.tennis_scene.multiview_pose.reconstruct import DEFAULT_CONFIG, reconstruct
from src.tennis_scene.multiview_pose.schema import FramePoseCandidates, RowRejection
from src.tennis_scene.multiview_pose.storage import publish_candidates, read_candidates
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_human_eval_inputs(
    dataset: Path, archives: Path, clip: str
) -> tuple[PoseInputs, ClipLabels, dict[str, Any]]:
    """Replay old all-row FP32 evidence and the reviewed-ball calibration for this evaluation only."""
    store = (archives / OBSERVE / clip).resolve()
    index = _read(store / "scene.json")
    source = restore_type(index["source"], ClipSource)
    calibration, _ = read_output(
        store,
        index,
        "court_calibration",
        "local_court_calibration",
        1,
        CourtCalibrationOutput,
    )
    side_path = archives / SIDE_REPORT
    side = next(item for item in _read(side_path)["clips"] if item["clip_id"] == clip)
    if (
        source.clip_id != clip
        or document_digest(source) != index["source_sha256"]
        or tuple(side["camera_ids"]) != source.camera_ids
        or not side["annotation"]["decided"]
        or calibration.calibration.camera_ids != source.camera_ids
        or calibration.calibration.excluded
        or calibration.reference_camera != side["reference_camera"]
    ):
        raise ValueError(f"Mismatched Meiji 3cam observation/side archive: {clip}")
    turns = side["annotation"]["view_half_turns"]
    cameras = tuple(
        v.camera.half_turned(turn)
        for v, turn in zip(calibration.calibration.views, turns, strict=True)
    )
    plan = _read(R7_EVIDENCE / "plan.json")
    features = _read(R7_EVIDENCE / "features.json")
    plan_sha = dual_sha256(R7_EVIDENCE / "plan.json")
    if (
        plan["source_name"] != "coco_fullframe_0.30"
        or plan_sha != features["plan_sha256"]
    ):
        raise ValueError("Meiji 3cam replay requires the pinned full-frame r7 evidence")
    label = label_path(dataset, clip)
    detections, poses = [], []
    for video in source.videos:
        if dual_sha256(label.parents[2] / "media" / video.path.name) != video.sha256:
            raise ValueError(
                f"Meiji 3cam media SHA differs from the observations: {clip} {video.camera_id}"
            )
        detection, joints = read_all_pose_rows(
            archives, clip, video, plan, features, plan_sha
        )
        rows = np.arange(len(joints), dtype=np.int64)
        detections.append(
            PersonDetectionOutput(
                video.camera_id,
                detection.offsets,
                detection.boxes,
                detection.scores,
                rows,
            )
        )
        poses.append(
            PersonPoseFeaturesOutput(
                video.camera_id,
                detection.offsets,
                rows,
                joints,
                document_digest(plan["models"]["pose"]),
                document_digest(plan["feature_config"]),
            )
        )
    labels = ClipLabels.load(label)
    if labels.clip_id != clip or labels.num_frames != source.num_frames:
        raise ValueError(f"Wrong Meiji 3cam label scope: {clip}")
    return (
        PoseInputs(source, cameras, tuple(detections), tuple(poses)),
        labels,
        {
            "labels_sha256": dual_sha256(label),
            "source_sha256": index["source_sha256"],
            "calibration_artifact_id": index["artifacts"]["court_calibration"][
                "artifact_id"
            ],
            "side_report_sha256": dual_sha256(side_path),
            "r7_plan_sha256": plan_sha,
            "r7_features_sha256": dual_sha256(R7_EVIDENCE / "features.json"),
            "turns": turns,
        },
    )


def _predictions(
    inputs: PoseInputs, result: FramePoseCandidates
) -> dict[str, CameraPrediction]:
    frames = np.repeat(
        np.arange(inputs.source.num_frames), np.diff(result.frame_offsets)
    )
    predictions = {}
    for index, detection in enumerate(inputs.detections):
        axis = detection.frame_offsets
        counts = np.diff(axis)
        observed = np.arange(counts.max())[:, None] < counts
        boxes = np.zeros((*observed.shape, 4), np.float64)
        boxes.transpose(1, 0, 2)[observed.T] = detection.boxes_xyxy
        ids = np.full(observed.shape, -1, np.int64)
        present = result.source_rows[:, index] >= 0
        f = frames[present]
        ids[result.source_rows[present, index] - axis[f], f] = np.flatnonzero(present)
        predictions[detection.camera_id] = CameraPrediction(
            np.arange(len(boxes), dtype=np.int64), boxes, observed, ids
        )
    return predictions


def _label_rows(
    inputs: PoseInputs, labels: ClipLabels, matched: dict[str, NDArray[np.int64]]
) -> list[list[dict[int, int]]]:
    all_rows = []
    for detection in inputs.detections:
        per_frame = []
        for frame, (start, end) in enumerate(
            zip(detection.frame_offsets[:-1], detection.frame_offsets[1:], strict=True)
        ):
            rows: dict[int, int] = {}
            for local, person in enumerate(
                matched[detection.camera_id][: end - start, frame]
            ):
                row = int(start + local)
                if (
                    person >= 0
                    and labels.people[person].role == "player"
                    and (
                        person not in rows
                        or detection.confidence[row]
                        > detection.confidence[rows[int(person)]]
                    )
                ):
                    rows[int(person)] = row
            per_frame.append(rows)
        all_rows.append(per_frame)
    return all_rows


def _holdout(
    inputs: PoseInputs,
    matched: dict[str, NDArray[np.int64]],
    labelled: list[list[dict[int, int]]],
    camera_idx: int,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    retained = [i for i in range(len(inputs.cameras)) if i != camera_idx]
    subset = PoseInputs(
        replace(inputs.source, videos=tuple(inputs.source.videos[i] for i in retained)),
        tuple(inputs.cameras[i] for i in retained),
        tuple(inputs.detections[i] for i in retained),
        tuple(inputs.poses[i] for i in retained),
    )
    held = inputs.cameras[camera_idx]
    pose = view_rays(
        inputs.poses[camera_idx].keypoints, held, inputs.source.size, DEFAULT_CONFIG
    )
    result = reconstruct(subset)
    eligible_people = scored_people = eligible_joints = 0
    errors = []
    for frame, (start, end) in enumerate(
        zip(result.frame_offsets[:-1], result.frame_offsets[1:], strict=True)
    ):
        available = set.intersection(*(set(per_frame[frame]) for per_frame in labelled))
        eligible_people += len(available)
        eligible_joints += sum(
            int(pose.observed[labelled[camera_idx][frame][person]].sum())
            for person in available
        )
        winners: dict[int, int] = {}
        for candidate in range(start, end):
            people = [
                int(
                    matched[inputs.source.camera_ids[i]][
                        row - inputs.detections[i].frame_offsets[frame], frame
                    ]
                )
                for i, row in zip(retained, result.source_rows[candidate], strict=True)
            ]
            person = people[0]
            if (
                person in available
                and len(set(people)) == 1
                and (
                    person not in winners
                    or result.scores[candidate] > result.scores[winners[person]]
                )
            ):
                winners[person] = candidate
        for person, candidate in winners.items():
            row = labelled[camera_idx][frame][person]
            projected, front = held.project(result.positions_m[candidate])
            valid = pose.observed[row] & result.joint_valid[candidate] & front
            if valid.any():
                errors.append(
                    np.linalg.norm(projected[valid] - pose.uv[row, valid], axis=1)
                )
                scored_people += 1
    residuals = np.concatenate(errors) if errors else np.empty(0, np.float64)
    return {
        "eligible_person_frames": eligible_people,
        "scored_person_frames": scored_people,
        "eligible_joints": eligible_joints,
        "scored_joints": len(residuals),
        **_error_summary(residuals),
    }, residuals


def _error_summary(residuals: NDArray[np.float64]) -> dict[str, float | None]:
    return {
        f"p{q}_px": float(np.percentile(residuals, q)) if len(residuals) else None
        for q in (50, 90, 95)
    }


def _coverage(result: FramePoseCandidates) -> dict[str, Any]:
    present = (result.source_rows >= 0).sum(axis=1)
    reasons = {
        reason.name: sum(
            int((values == reason).sum()) for values in result.row_rejection_codes
        )
        for reason in RowRejection
    }
    return {
        "frames": len(result.frame_offsets) - 1,
        "populated_frames": int((np.diff(result.frame_offsets) > 0).sum()),
        "candidates": len(result.positions_m),
        "by_supported_views": {
            str(v): int((present == v).sum())
            for v in range(2, len(result.camera_ids) + 1)
        },
        "valid_joints": int(result.joint_valid.sum()),
        "ambiguous_candidates": int(result.ambiguous.sum()),
        "dino_rows": reasons,
    }


def evaluate_3cam(dataset: Path, archives: Path) -> dict[str, Any]:
    started = perf_counter()
    clips = {}
    all_errors = []
    for clip in DEV_CLIPS:
        inputs, labels, pins = _read_human_eval_inputs(dataset, archives, clip)
        start = perf_counter()
        result = reconstruct(inputs)
        reconstruction_seconds = perf_counter() - start
        preds = _predictions(inputs, result)
        legacy = evaluate(labels, preds)
        matched = match_to_labels(labels, preds, 0.5)
        labelled = _label_rows(inputs, labels, matched)
        holdouts = {}
        for i, camera in enumerate(inputs.cameras):
            holdouts[camera.camera_id], errors = _holdout(inputs, matched, labelled, i)
            all_errors.append(errors)
        clips[clip] = {
            "pins": pins,
            "pairs": legacy["pairs"],
            "player_pairs": evaluate(player_only(labels), preds)["pairs"],
            "label_coverage": legacy["coverage"],
            "reconstruction_seconds": reconstruction_seconds,
            "coverage": _coverage(result),
            "holdout": holdouts,
        }
        print(f"{clip} scored", flush=True)
    overall = {}
    for key in ("pairs", "player_pairs"):
        tp, fp, fn = (
            sum(clip[key][count] for clip in clips.values())
            for count in ("tp", "fp", "fn")
        )
        overall[key] = {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn)}
    holdout = {
        key: sum(s[key] for clip in clips.values() for s in clip["holdout"].values())
        for key in (
            "eligible_person_frames",
            "scored_person_frames",
            "eligible_joints",
            "scored_joints",
        )
    }
    return {
        "clips": clips,
        **overall,
        "holdout": {**holdout, **_error_summary(np.concatenate(all_errors))},
        "wall_seconds": perf_counter() - started,
        "config": json_value(DEFAULT_CONFIG),
    }


_side: PinnedCourtSide | None = None


def _open_side(
    side_store: Path, digest: str, court_store: Path, drift_store: Path
) -> None:
    global _side
    _side = PinnedCourtSide.load(side_store, digest, court_store, drift_store)


def _run_clip(dataset: Path, output: Path, clip: str) -> dict[str, Any]:
    assert _side is not None
    video, name = clip.split("/")
    store = dataset / "videos" / video / "clips" / name / "annotations" / "tennis_scene"
    started = perf_counter()
    inputs = read_saved_inputs(store, _side)
    loaded = perf_counter()
    result = reconstruct(inputs.observations)
    reconstructed = perf_counter()
    ref = publish_candidates(output / clip, store, inputs, result, DEFAULT_CONFIG)
    published = read_candidates(output / clip)
    return {
        "source_sha256": document_digest(inputs.observations.source),
        "inputs_digest": document_digest(inputs.references),
        "side": inputs.court_side,
        "artifact_id": ref.artifact_id,
        "descriptor_sha256": ref.sha256,
        "read_seconds": loaded - started,
        "reconstruction_seconds": reconstructed - loaded,
        "publish_readback_seconds": perf_counter() - reconstructed,
        **_coverage(published),
    }


def evaluate_4cam(
    dataset: Path, output: Path, shared: Path, workers: int
) -> dict[str, Any]:
    started = perf_counter()
    bundle = Path("knowledge/runs/run-court-side-video-shared-20261118")
    digest = _read(bundle / "summary.json")["side_digest"]
    courts = shared / "court-video-shared/tennis_scene/evaluate"
    args = (
        bundle / "side-store",
        digest,
        courts / "court-video-shared-acceptance-20261117/store",
        courts / "court-video-drift-lines-20261117/drift-store",
    )
    _open_side(*args)
    assert _side is not None
    clips = _side.clip_ids
    results = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_open_side,
        initargs=args,
    ) as executor:
        pending = {
            clip: executor.submit(_run_clip, dataset, output, clip) for clip in clips
        }
        for clip, future in pending.items():
            results[clip] = future.result()
            write_json_atomic(output / "progress.json", results)
            print(
                f"{clip} reconstructed in {results[clip]['reconstruction_seconds']:.2f} s",
                flush=True,
            )
    return {
        "clips": results,
        "workers": workers,
        "wall_seconds": perf_counter() - started,
        "config": json_value(DEFAULT_CONFIG),
        "side_digest": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("meiji3", "meiji4"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "meiji3":
        result = evaluate_3cam(args.dataset, args.inputs_root)
    else:
        result = evaluate_4cam(
            args.dataset, args.output, args.inputs_root, args.workers
        )
    write_json_atomic(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
