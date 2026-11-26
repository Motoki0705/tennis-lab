"""Qualify the production people models against the pinned comparison adapters."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.submodules.models import (
    PersonDetectionRequest,
    RFDetrPersonDetector,
    RTMPoseBoxSequenceRequest,
    RTMPoseXPose2D,
)
from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey
from src.tennis_scene.pipeline.source_identity import source_commit
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner
from src.utils.video.reader import RandomAccessVideoReader

BUNDLE_SHA256 = "3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580"
CLIP = "video_000/clip_003"
FRAME = 120
POSE_MODEL = "rtmpose-x"


def _rows(args: argparse.Namespace, cameras: tuple[str, ...]) -> tuple[list[Any], np.ndarray]:
    manifest = load_manifest(args.bundle, expected_sha256=BUNDLE_SHA256)
    refs = load_reference_frames(args.bundle, expected_sha256=BUNDLE_SHA256)
    clip = next(item for item in manifest["clips"] if item["clip_id"] == CLIP)
    images, boxes = [], []
    for camera in cameras:
        video = pinned_file(args.media_root, next(row for row in clip["source_videos"] if row["camera_id"] == camera))
        with RandomAccessVideoReader(video) as reader:
            images.append(reader.read(FRAME))
        chosen = refs[FrameKey(CLIP, camera, FRAME)].boxes_xyxy[:2]
        if len(chosen) != 2:
            raise ValueError(f"Expected two fixed boxes at {CLIP}/{camera}/{FRAME}")
        boxes.extend(chosen)
    return images, np.asarray(boxes, dtype=np.float32)


def _request(images: list[Any], boxes: np.ndarray) -> RTMPoseBoxSequenceRequest:
    return RTMPoseBoxSequenceRequest(enumerate(images), torch.from_numpy(boxes),
                                    torch.arange(len(images), dtype=torch.int64).repeat_interleave(2))


def _pose_delta(predicted: np.ndarray, reference: np.ndarray, *,
                xy_tolerance: float, score_tolerance: float) -> dict[str, Any]:
    if predicted.shape != reference.shape:
        raise ValueError("Pose output shape or row count differs")
    valid = reference[..., 2] > 0
    np.testing.assert_array_equal(predicted[..., 2] > 0, valid)
    np.testing.assert_array_equal(predicted[..., :2][~valid], np.full((int((~valid).sum()), 2), -1.))
    delta = np.linalg.norm(predicted[..., :2][valid] - reference[..., :2][valid], axis=-1)
    score_delta = np.abs(predicted[..., 2] - reference[..., 2])
    maximum_xy = float(delta.max(initial=0.))
    maximum_score = float(score_delta.max(initial=0.))
    if maximum_xy > xy_tolerance or maximum_score > score_tolerance:
        raise ValueError(f"Pose parity exceeded tolerances: {maximum_xy} px, {maximum_score} score")
    return {"rows": len(predicted), "valid_joints": int(valid.sum()), "max_xy_l2_px": maximum_xy,
            "max_abs_score": maximum_score, "xy_tolerance_px": xy_tolerance, "score_tolerance": score_tolerance}


def _cpu(args: argparse.Namespace, reference: Any, recipe: Any) -> dict[str, Any]:
    frames, boxes = _rows(args, ("cam0",))
    estimator = reference.load_estimator(recipe)
    official = import_module("mmpose.apis").inference_topdown(estimator, frames[0], boxes)
    samples = [prediction.pred_instances for prediction in official]
    expected = np.concatenate([
        np.concatenate((sample.keypoints[:, :17], sample.keypoint_scores[:, :17, None]), axis=-1)
        for sample in samples
    ])
    model = RTMPoseXPose2D(args.rtmpose_checkpoint, device="cpu", batch_size=2, show_progress=False)
    try:
        result = model.predict(_request(frames, boxes)).keypoints.numpy()
    finally:
        model.unload()
    return {"RTMPose-X PT32": _pose_delta(result, expected, xy_tolerance=.0001, score_tolerance=.00001),
            "cameras": ["cam0"]}


def _rf_delta(args: argparse.Namespace, frames: list[Any], backend: Backend,
              engine: EngineBundle | None, device: torch.device) -> dict[str, Any]:
    comparison = import_module("knowledge.runs.run-meiji-player-model-comparison.rfdetr")
    checkpoint = comparison.Model.open(args.rfdetr_checkpoint)
    official = comparison.Detector(checkpoint, backend, device, engine)
    model = RFDetrPersonDetector(args.rfdetr_checkpoint, device=device, precision="fp16",
                                 confidence=.25, engines=() if engine is None else (engine,))
    requests = [PersonDetectionRequest(frame) for frame in frames]
    try:
        ours = model.predict_batch(requests, batch_size=1)
        expected = [official.predict([frame])[0] for frame in frames]
    finally:
        model.unload()
    for ours_row, expected_row in zip(ours, expected, strict=True):
        np.testing.assert_allclose(ours_row.boxes_xyxy, expected_row.boxes_xyxy, atol=.001, rtol=0)
        np.testing.assert_allclose(ours_row.scores, expected_row.scores, atol=.000001, rtol=0)
    return {
        "frames": len(frames), "boxes": sum(len(row.scores) for row in ours), "cutoff": .25,
        "max_box_delta_px": max(float(np.abs(o.boxes_xyxy - e.boxes_xyxy).max(initial=0.))
                                for o, e in zip(ours, expected, strict=True)),
        "max_abs_score": max(float(np.abs(o.scores - e.scores).max(initial=0.))
                             for o, e in zip(ours, expected, strict=True)),
        "plan_sha256": None if engine is None else engine.engine.plan_sha256,
    }


def _rtm_delta(args: argparse.Namespace, reference: Any, recipe: Any, device: torch.device,
               backend: Backend, engine: EngineBundle | None, frames: list[Any], boxes: np.ndarray) -> dict[str, Any]:
    if engine is None:
        graph = reference.RTMPoseGraph(reference.load_estimator(recipe)).eval().to(device=device, dtype=torch.float16)
        forward = graph.forward
    else:
        runner = TensorRTRunner(engine, device)

        def forward(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            axes = runner({"pixels": pixels})
            return axes["simcc_x"], axes["simcc_y"]

    infer = reference._make_infer(recipe, backend, forward, device)
    stages = import_module("evaluate").AccuracyStages(backend)
    input_row = import_module("pose_contract").PoseInput
    rows = tuple(input_row(image, box) for i, image in enumerate(frames) for box in boxes[2*i:2*i+2])
    model = RTMPoseXPose2D(args.rtmpose_checkpoint, device=device, precision="fp16", engine=engine,
                            batch_size=len(rows), show_progress=False)
    try:
        with torch.inference_mode():
            expected = infer(rows, stages).keypoints_xyc
            result = model.predict(_request(frames, boxes)).keypoints.numpy()
    finally:
        model.unload()
    if stages.forward_count != 2:
        raise ValueError("The comparison requires two RTMPose forwards")
    return {**_pose_delta(result, expected, xy_tolerance=.001, score_tolerance=.0001),
            "plan_sha256": None if engine is None else engine.engine.plan_sha256}


def _gpu(args: argparse.Namespace, reference: Any, recipe: Any) -> dict[str, Any]:
    cameras = ("cam0", "cam1", "cam2", "cam3")
    frames, boxes = _rows(args, cameras)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    rf = EngineBundle.read(args.rfdetr_engine)
    rtm = EngineBundle.read(args.rtmpose_engine)
    return {
        "RF-DETR-L PT16": _rf_delta(args, frames, Backend.PT16, None, device),
        "RF-DETR-L TRT16": _rf_delta(args, frames, Backend.TRT16, rf, device),
        "RTMPose-X PT16": _rtm_delta(args, reference, recipe, device, Backend.PT16, None, frames, boxes),
        "RTMPose-X TRT16": _rtm_delta(args, reference, recipe, device, Backend.TRT16, rtm, frames, boxes),
        "cameras": list(cameras), "device": torch.cuda.get_device_name(device),
        "rfdetr_sha256": dual_sha256(args.rfdetr_checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    for option in ("bundle", "media-root", "mmpose-source", "rtmpose-checkpoint", "report"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("rfdetr-checkpoint", "rfdetr-engine", "rtmpose-engine"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.mode == "gpu" and None in (args.rfdetr_checkpoint, args.rfdetr_engine, args.rtmpose_engine):
        parser.error("GPU parity requires --rfdetr-checkpoint, --rfdetr-engine and --rtmpose-engine")
    cv2.setNumThreads(1)
    # The pinned comparison directory is supplied on PYTHONPATH.
    reference = import_module("rtmpose")
    recipe = reference.load_recipe(POSE_MODEL, args.rtmpose_checkpoint, args.mmpose_source)
    results = _cpu(args, reference, recipe) if args.mode == "cpu" else _gpu(args, reference, recipe)
    report = {
        "head": source_commit(), "bundle_sha256": BUNDLE_SHA256, "clip": CLIP, "source_frame": FRAME,
        "rtmpose_sha256": dual_sha256(args.rtmpose_checkpoint), "mmpose_commit": reference.SOURCE_COMMIT,
        "torch": torch.__version__, "results": results,
    }
    write_json_atomic(args.report, report)
    print(results, flush=True)


if __name__ == "__main__":
    main()
