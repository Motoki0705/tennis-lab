"""Compare the staged RTMPose adapter to official CPU FP32 inference."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate import AccuracyStages
from pose_contract import PoseInput
from rtmpose import COCO_JOINTS, MODELS, load_estimator, load_recipe, make_adapter

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey
from src.utils.inference_benchmark import Backend
from src.utils.video import OpenCVVideoFrameReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    key = FrameKey("video_000/clip_003", "cam0", 120)
    manifest = load_manifest(args.bundle, expected_sha256=args.bundle_sha256)
    references = load_reference_frames(args.bundle, expected_sha256=args.bundle_sha256)
    clip = next(clip for clip in manifest["clips"] if clip["clip_id"] == key.clip_id)
    record = next(v for v in clip["source_videos"] if v["camera_id"] == key.camera_id)
    video = pinned_file(args.media_root, record)
    cv2.setNumThreads(1)
    reader = OpenCVVideoFrameReader(video, max_frames=key.source_frame + 1)
    frame = next(packet.frame for packet in reader if packet.index == key.source_frame)
    boxes = references[key].boxes_xyxy[:2].astype(np.float32)
    if len(boxes) != 2:
        raise ValueError(f"Expected two fixed boxes at {key}")
    original_frame, original_boxes = frame.copy(), boxes.copy()
    rows = tuple(PoseInput(frame, box) for box in boxes)
    infer_official = import_module("mmpose.apis").inference_topdown
    for model_id, spec in MODELS.items():
        checkpoint = args.weights / spec.checkpoint_name
        estimator = load_estimator(load_recipe(model_id, checkpoint, args.source_root))
        if (
            sum(parameter.numel() for parameter in estimator.parameters())
            != spec.parameters
        ):
            raise ValueError(f"Unexpected parameter count for {model_id}")
        official = [
            sample.pred_instances for sample in infer_official(estimator, frame, boxes)
        ]
        xy = np.concatenate(
            [sample.keypoints[:, :COCO_JOINTS] for sample in official]
        ).astype(np.float32)
        scores = np.concatenate(
            [sample.keypoint_scores[:, :COCO_JOINTS] for sample in official]
        )
        # MMPose projects its invalid -0.5 coordinates into native pixels.
        xy[scores <= 0] = np.nan
        infer = make_adapter(
            model_id=model_id,
            backend=Backend.PT32,
            checkpoint=checkpoint,
            device="cpu",
            source_root=args.source_root,
        ).load()
        stages = AccuracyStages(Backend.PT32)
        tail_stages = AccuracyStages(Backend.PT32)
        with torch.inference_mode():
            staged = infer(rows, stages).keypoints_xyc
            tail = infer(rows[-1:], tail_stages).keypoints_xyc
        np.testing.assert_allclose(staged[..., :2], xy, rtol=0, atol=1e-4)
        np.testing.assert_allclose(staged[..., 2], scores, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            tail[..., :2], staged[-1:, ..., :2], rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(tail[..., 2], staged[-1:, ..., 2], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(frame, original_frame)
        np.testing.assert_array_equal(boxes, original_boxes)
        if stages.forward_count != 2 or tail_stages.forward_count != 2:
            raise ValueError("The comparison requires two RTMPose forwards per batch")
        print(
            f"{model_id} {key}, 2 boxes, shape {staged.shape}, "
            f"max xy difference {np.nanmax(np.abs(staged[..., :2] - xy)):.7g} px, "
            f"max score difference {np.max(np.abs(staged[..., 2] - scores)):.7g}, "
            f"max tail score difference {np.max(np.abs(tail[..., 2] - staged[-1:, ..., 2])):.7g}, "
            "2 forwards at both batch lengths",
            flush=True,
        )


if __name__ == "__main__":
    main()
