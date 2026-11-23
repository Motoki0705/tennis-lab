"""Score the published #2442 samples against the saved 4cam candidates on CPU."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from assignment_probe import player_only
from probe import read_json, read_output
from replay import micro_pairs, predictions_from_rows

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tennis_scene.multiview_pose.schema import NODE
from src.tennis_scene.multiview_pose.storage import read_candidates
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic

_RUN = Path(__file__).resolve().parent
_PUBLICATION_SHA = "3a27b03bda2a6e30879eec622db1570b8fc364b0c31adf2c22f4df485fedc33c"
_DATASET_SHA = "968da7df0a235c4213af549ea661fa1b66c8e8c57c94b2c66d07775b23f50099"
_COUNT_FIELDS = ("frames", "scored_frames", "label_boxes", "matched_label_boxes")
_PAIR_FIELDS = ("pairs", "player_pairs")


def _score_frames(
    labels: ClipLabels, predictions: dict[str, CameraPrediction], frames: list[int]
) -> dict[str, Any]:
    """Evaluate the original frames in this sample and return frame-level counts."""
    selected = np.asarray(frames, dtype=np.int64)
    views = {}
    for camera, view in labels.cameras.items():
        keep = np.isin(view.frames, selected)
        views[camera] = replace(
            view,
            frames=np.searchsorted(selected, view.frames[keep]),
            person_index=view.person_index[keep],
            boxes_xyxy=view.boxes_xyxy[keep],
        )
    subset = replace(labels, num_frames=len(frames), cameras=views, provenance={})
    preds = {
        camera: replace(
            value,
            boxes_xyxy=value.boxes_xyxy[:, selected],
            observed=value.observed[:, selected],
            player_ids=value.player_ids[:, selected],
        )
        for camera, value in predictions.items()
    }
    score = evaluate(subset, preds)
    coverage = score["coverage"]
    return {
        "frames": len(frames),
        "scored_frames": score["group_accuracy"]["frames_scored"],
        "label_boxes": sum(view["label_boxes"] for view in coverage.values()),
        "matched_label_boxes": sum(
            view["matched_label_boxes"] for view in coverage.values()
        ),
        "pairs": score["pairs"],
        "player_pairs": evaluate(player_only(subset), preds)["pairs"],
    }


def _counts(scores: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        field: sum(score[field] for score in scores) for field in _COUNT_FIELDS
    }
    for field in _PAIR_FIELDS:
        totals[field] = micro_pairs(score[field] for score in scores)
    return totals


def _predictions(
    clip: str, raw_store: Path, pose_store: Path, expected: dict[str, Any]
) -> tuple[dict[str, CameraPrediction], dict[str, str]]:
    raw_store, pose_store = raw_store.resolve(), pose_store.resolve()
    index = read_json(pose_store / "scene.json")
    raw_index = read_json(raw_store / "scene.json")
    source_sha = expected["source_sha256"]
    ref = index["artifacts"][NODE]
    if (
        index["source"]["clip_id"] != clip
        or index["source_sha256"] != source_sha
        or raw_index["source_sha256"] != source_sha
        or document_digest(raw_index["source"]) != source_sha
        or ref["artifact_id"] != expected["artifact_id"]
        or ref["sha256"] != expected["descriptor_sha256"]
    ):
        raise ValueError(
            f"Expected the measured output and its original source: {clip}"
        )
    desc = read_component_descriptor(
        pose_store, ref, node=NODE, source_sha256=source_sha
    )
    observations = desc["provenance"]["observations"]
    if document_digest(observations) != expected["inputs_digest"]:
        raise ValueError(f"Candidate input provenance changed: {clip}")

    result = read_candidates(pose_store)
    detections = []
    for camera in result.camera_ids:
        name = f"person_detection/{camera}"
        detection, dependencies = read_output(
            raw_store, raw_index, name, "person_detections", 2, PersonDetectionOutput
        )
        if (
            raw_index["artifacts"][name] != observations[name]
            or dependencies
            or detection.duplicate_merges
            or not np.array_equal(
                detection.source_rows, np.arange(len(detection.boxes_xyxy))
            )
        ):
            raise ValueError(
                f"Expected the exact all-row DINO observations: {clip}/{camera}"
            )
        detections.append(detection)
    media = {video["camera_id"]: video["sha256"] for video in index["source"]["videos"]}
    return predictions_from_rows(tuple(detections), result), media


def _score_clip(
    dataset: Path,
    entry: dict[str, Any],
    published: dict[str, Any],
    picked: dict[str, Any],
    predictions: dict[str, CameraPrediction],
    media: dict[str, str],
    snapshot_sha: str,
) -> dict[str, Any]:
    clip = entry["clip_id"]
    path = label_path(dataset, clip)
    label_sha = dual_sha256(path)
    labels = ClipLabels.load(path)
    annotation = labels.provenance["pseudo_annotation"]
    strata = picked["sampled"]
    membership: dict[str, list[str]] = {}
    for tag, frames in strata.items():
        for frame in frames:
            membership.setdefault(str(frame), []).append(tag)
    frames = sorted(map(int, membership))
    if (
        published["status"] != "published"
        or label_sha != published["labels_sha256"]
        or labels.clip_id != clip
        or labels.num_frames != entry["num_frames"]
        or labels.num_frames != picked["num_frames"]
        or any(
            view.observed.shape[1] != labels.num_frames for view in predictions.values()
        )
        or set(labels.cameras) != set(media)
        or int((labels.roles == "player").sum()) != 4
        or annotation["media_sha256"] != media
        or annotation["input_snapshot_sha256"] != snapshot_sha
        or annotation["show_prior_association"] is not False
        or annotation["annotated_frames"] != frames
        or annotation["annotated_frames"] != picked["sampled_frames"]
        or {key: sorted(tags) for key, tags in annotation["frame_strata"].items()}
        != {key: sorted(tags) for key, tags in membership.items()}
        or len(frames) != published["frame_count"]
        or any(
            not np.isin(view.frames, frames).all() for view in labels.cameras.values()
        )
    ):
        raise ValueError(
            f"Pseudo-label publication and sampled frames disagree: {clip}"
        )
    overall = _score_frames(labels, predictions, frames)
    if overall["label_boxes"] != published["boxes"]:
        raise ValueError(f"Published label box count changed: {clip}")
    return {
        "labels_sha256": label_sha,
        "overall": overall,
        "by_stratum": {
            tag: _score_frames(labels, predictions, sorted(sampled))
            for tag, sampled in strata.items()
            if sampled
        },
    }


def score(root: Path, raw_dataset: Path, poses: Path) -> dict[str, Any]:
    dataset = root / "full/dataset"
    publication_path = root / "full/results.json"
    selection_path = root / "selection.json"
    if (
        dual_sha256(publication_path) != _PUBLICATION_SHA
        or dual_sha256(dataset / "dataset.json") != _DATASET_SHA
    ):
        raise ValueError("Expected the fixed #2442 full publication")
    publication = read_json(publication_path)
    if dual_sha256(selection_path) != publication["selection_sha256"]:
        raise ValueError("The sampling plan differs from the published evaluation")
    selection = read_json(selection_path)
    manifest = read_json(dataset / "dataset.json")
    measured_path = _RUN / "meiji4.json"
    measured = read_json(measured_path)
    clip_ids = {entry["clip_id"] for entry in manifest["clips"]}
    if (
        not clip_ids
        == set(publication["clips"])
        == set(selection["clips"])
        == set(measured["clips"])
    ):
        raise ValueError(
            "Expected the candidate, pseudo-label and selection clip sets to match"
        )

    clips = {}
    for entry in manifest["clips"]:
        clip = entry["clip_id"]
        expected = measured["clips"][clip]
        raw = raw_dataset / entry["path"] / "annotations/tennis_scene"
        predictions, media = _predictions(clip, raw, poses / clip, expected)
        clips[clip] = _score_clip(
            dataset,
            entry,
            publication["clips"][clip],
            selection["clips"][clip],
            predictions,
            media,
            publication["input_snapshot_sha256"],
        )
    overall = _counts([item["overall"] for item in clips.values()])
    by_stratum = {
        tag: _counts(
            [
                item["by_stratum"][tag]
                for item in clips.values()
                if tag in item["by_stratum"]
            ]
        )
        for tag in publication["by_stratum"]
    }
    if overall["frames"] != publication["frame_counts"]["published"] or {
        tag: value["frames"] for tag, value in by_stratum.items()
    } != {tag: facts["published"] for tag, facts in publication["by_stratum"].items()}:
        raise ValueError("Scored frame counts differ from the full publication")
    return {
        "dataset_id": manifest["dataset_id"],
        "label_source": {
            "issue": 2442,
            "model": publication["model"],
            "effort": publication["effort"],
            "publication_sha256": _PUBLICATION_SHA,
            "dataset_sha256": _DATASET_SHA,
            "selection_sha256": publication["selection_sha256"],
            "input_snapshot_sha256": publication["input_snapshot_sha256"],
            "human_error_rate": None,
        },
        "candidate_metrics_sha256": dual_sha256(measured_path),
        "min_iou": 0.5,
        "overall": overall,
        "by_stratum": by_stratum,
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.publication, args.raw_dataset, args.candidates)
    write_json_atomic(args.output, result)
    print(result["overall"], flush=True)


if __name__ == "__main__":
    main()
