"""Score the saved full-frame DINO and ViTPose-H TensorRT FP16 predictions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from prediction_rows import PredictionHeader, write_predictions

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    ReferenceFrame,
    score_players,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic


def camera_predictions(
    video: SourceVideo,
    detections: PersonDetectionOutput,
    poses: PersonPoseFeaturesOutput,
    frames: Sequence[int],
) -> dict[int, Predictions]:
    require_alignment(video, detections, poses)
    result = {}
    for frame in frames:
        start, stop = detections.frame_offsets[frame : frame + 2]
        joints = poses.keypoints[start:stop]
        result[frame] = Predictions(
            detections.boxes_xyxy[start:stop],
            detections.confidence[start:stop],
            np.where(joints[..., 2:] > 0, joints, 0),
        )
    return result


def _model_record(
    descriptor: dict[str, Any], slot: str, architecture: str
) -> dict[str, Any]:
    component = descriptor["identity"]
    config = component["settings"]
    model = config["models"][slot]
    if (
        model["architecture"] != architecture
        or model["dtype"] != "float16"
        or model["settings"]["backend"] != "tensorrt"
        or model["settings"]["precision"] != "fp16"
    ):
        raise ValueError(f"Unexpected {slot} precision, architecture or backend")
    return {
        "architecture": architecture,
        "implementation": component["implementation"],
        "device": model["device"],
        "dtype": model["dtype"],
        "checkpoint_sha256": model["assets"]["checkpoint"]["sha256"],
        "model_settings": model["settings"],
        "merge_duplicates": config.get("merge_duplicates"),
        "bbox_enlarge": config.get("bbox_enlarge"),
    }


def load_archived_predictions(
    dataset: Path, manifest: dict[str, Any]
) -> tuple[dict[FrameKey, Predictions], dict[str, Any]]:
    result: dict[FrameKey, Predictions] = {}
    sources = {}
    models: dict[str, Any] = {}
    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        index = (
            dataset
            / "videos"
            / clip_id.replace("/", "/clips/", 1)
            / "annotations"
            / "tennis_scene"
            / "scene.json"
        )
        store = StoreReader(index)
        if (
            store.source["clip_id"] != clip_id
            or store.camera_ids != tuple(clip["camera_ids"])
            or store.size != (clip["width"], clip["height"])
            or store.frame_count != clip["source_frames"]
        ):
            raise ValueError(f"Source store differs from the fixed clip: {clip_id}")
        cameras = {}
        for saved, pinned in zip(
            store.source["videos"], clip["source_videos"], strict=True
        ):
            camera = pinned["camera_id"]
            if any(
                saved[key] != pinned[key]
                for key in ("camera_id", "sha256", "fps", "num_frames")
            ):
                raise ValueError(f"Source media differs for {clip_id}/{camera}")
            video = SourceVideo(**{**saved, "path": Path(saved["path"])})
            det_node, pose_node = (
                f"person_detection/{camera}",
                f"person_pose_features/{camera}",
            )
            det, det_descriptor = store.payload(det_node)
            pose, pose_descriptor = store.payload(pose_node)
            if (
                (det_descriptor["output_schema"], det_descriptor["output_version"])
                != ("person_detections", 2)
                or (pose_descriptor["output_schema"], pose_descriptor["output_version"])
                != ("person_pose_features", 2)
                or pose_descriptor["dependencies"]
                != {"detections": store.references[det_node]}
            ):
                raise ValueError(
                    f"Unsupported detection/pose lineage for {clip_id}/{camera}"
                )
            profile = {
                "dino": _model_record(det_descriptor, "detector", "dino_4scale_swin_l"),
                "vitpose": _model_record(
                    pose_descriptor, "pose", "vitpose_huge_coco17"
                ),
            }
            if models and profile != models:
                raise ValueError(f"Model configuration changes at {clip_id}/{camera}")
            models = profile
            rows = camera_predictions(
                video,
                PersonDetectionOutput(**det),
                PersonPoseFeaturesOutput(**pose),
                clip["evaluation_frames"],
            )
            result.update(
                (FrameKey(clip_id, camera, frame), row) for frame, row in rows.items()
            )
            cameras[camera] = {
                "video_sha256": saved["sha256"],
                "detections": store.references[det_node],
                "poses": store.references[pose_node],
            }
        sources[clip_id] = {
            "path": index.relative_to(dataset).as_posix(),
            "sha256": dual_sha256(index),
            "cameras": cameras,
        }
    return result, {"models": models, "archives": sources}


def write_baseline_outputs(
    output: Path,
    benchmark_id: str,
    manifest_sha256: str,
    backend: str,
    references: dict[FrameKey, ReferenceFrame],
    paired: dict[FrameKey, Predictions],
    provenance: dict[str, Any],
    *,
    scorer_commit: str | None = None,
) -> None:
    detections = {
        key: Predictions(row.boxes_xyxy, row.scores) for key, row in paired.items()
    }
    for model_id, has_pose, predictions in (
        ("dino-swin-l", False, detections),
        ("dino-swin-l-vitpose-h", True, paired),
    ):
        header = PredictionHeader(
            benchmark_id, manifest_sha256, model_id, backend, has_pose
        )
        directory = output / model_id / backend
        rows_path = directory / "predictions.jsonl"
        write_predictions(rows_path, header, predictions, expected_frames=references)
        score = score_players(references, predictions)
        receipt = {
            "condition": header.as_json(),
            "predictions_sha256": dual_sha256(rows_path),
            "provenance": provenance,
            "score": score,
        }
        if scorer_commit is not None:
            receipt["scorer_commit"] = scorer_commit
        write_json_atomic(directory / "accuracy.json", receipt)
        print(
            model_id,
            score["images"],
            score["overall"]["detection"]["recall_by_iou"],
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.benchmark, expected_sha256=args.manifest_sha256)
    references = load_reference_frames(
        args.benchmark, expected_sha256=args.manifest_sha256
    )
    paired, provenance = load_archived_predictions(args.dataset, manifest)
    write_baseline_outputs(
        args.output,
        manifest["benchmark_id"],
        args.manifest_sha256,
        "trt16",
        references,
        paired,
        provenance,
    )


if __name__ == "__main__":
    main()
