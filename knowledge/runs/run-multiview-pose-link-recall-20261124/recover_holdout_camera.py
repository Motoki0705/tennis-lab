"""Check an independent camera and ball-side repair for the reserved Meiji clip."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from probe import read_json

from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    CourtSideUndecided,
    decide_court_side,
)
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports.ball_annotations import convert_ball_annotation
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.checksum import dual_sha256
from src.utils.geometry.planar_camera import fit_planar_camera
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.io import write_json_atomic
from src.utils.schema.court import CourtConfig, court_keypoints_3d

CLIP = "video_001/clip_003"
_ATTESTATION = Path(__file__).with_name("holdout-inputs.json")
_COURT_REPORT = Path("knowledge/runs/run-slcs-meiji-court-visual-qc-v1/results.json")
_RAW_ARCHIVE = "person_tracking/evaluate/i964-unseen-r16-20261001"
_SIDE_CONFIG = "court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.pipeline_config.yaml"


def _camera(document: dict[str, Any]) -> PinholeCamera:
    return PinholeCamera(
        document["camera_id"],
        np.asarray(document["intrinsic"], np.float64),
        np.asarray(document["rotation"], np.float64),
        np.asarray(document["translation"], np.float64),
    )


def recover(dataset: Path, archives: Path) -> dict[str, Any]:
    attestation = read_json(_ATTESTATION)
    recorded = attestation["clips"][CLIP]
    archive = archives / _RAW_ARCHIVE
    plan_path = archive / "plan.json"
    court_receipt = archive / CLIP / "court-execute.json"
    if (
        dual_sha256(plan_path) != attestation["r18_plan_sha256"]
        or dual_sha256(court_receipt) != recorded["court_receipt_sha256"]
    ):
        raise ValueError("The reserved input plan or its calibration receipt changed")
    records = [item for item in read_json(plan_path)["records"] if item["clip"] == CLIP]
    (record,) = records
    source = restore_type(record["source"], ClipSource)
    if document_digest(source) != recorded["source_sha256"]:
        raise ValueError("The reserved clip source changed")

    qc = read_json(_COURT_REPORT)
    if not qc["numerical_contract_passed"] or not qc["inputs_unchanged"]:
        raise ValueError(
            "The historical Court projection receipt did not pass its checks"
        )
    video_id, clip_id = CLIP.split("/")
    clip = dataset / "videos" / video_id / "clips" / clip_id
    for video in source.videos:
        suffix = f"/{video_id}/clips/{clip_id}/media/{video.camera_id}.mp4"
        (recorded_sha,) = (
            digest
            for path, digest in qc["input_hashes_before"].items()
            if path.endswith(suffix)
        )
        if (
            recorded_sha != video.sha256
            or dual_sha256(clip / "media" / f"{video.camera_id}.mp4") != video.sha256
        ):
            raise ValueError(
                f"The Court report uses a different video for {video.camera_id}"
            )

    (cam0_points,) = (
        view
        for view in qc["views"]
        if view["clip"] == CLIP and view["camera"] == "cam0"
    )
    points = cam0_points["points"]
    if [point["point"] for point in points] != list(range(14)):
        raise ValueError("Expected the original ordered CourtKP14 points")
    court_xyz = (
        court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14].astype(np.float64)
    )
    uv = np.asarray([point["fit_uv_px"] for point in points], np.float64)
    if len(cam0_points["inlier_channels"]) < 6 or len(cam0_points["frame_indices"]) < 2:
        raise ValueError("The independent Court projection is missing its support")
    fit = fit_planar_camera(
        court_xyz, uv, np.ones(14), source.size, require_in_image=False
    )

    side_config = archives / _SIDE_CONFIG
    config = yaml.safe_load(side_config.read_text())
    camera = PinholeCamera("cam0", fit.K, fit.R, fit.t)
    limit = config["camera_geometry"]["calibration_error_ratio"] * math.hypot(
        *source.size
    )
    if (
        camera.center[1] > 0
        or fit.rmse_px > limit
        or not camera.project(court_xyz)[1].all()
    ):
        raise ValueError(
            f"cam0 did not pass the standard local Court fit gate ({fit.rmse_px:.3f} px)"
        )
    prior = read_json(court_receipt)
    if (
        tuple(source.camera_ids) != ("cam0", "cam1", "cam2")
        or prior["cam0"]["status"] != "stopped"
    ):
        raise ValueError("This repair is only for the missing cam0 calibration")
    cameras = (camera,) + tuple(
        _camera(prior[name]["calibration"]["views"][0]["camera"])
        for name in ("cam1", "cam2")
    )
    if tuple(model.camera_id for model in cameras) != source.camera_ids:
        raise ValueError("Calibration camera axes differ from the reservation")

    ball_paths = [
        clip / "outsource" / f"{video.camera_id}_annotations.json"
        for video in source.videos
    ]
    balls = [
        convert_ball_annotation(path, video)[0]
        for path, video in zip(ball_paths, source.videos, strict=True)
    ]
    side = CourtSideConfig(**config["court_side"])
    side = replace(
        side,
        reprojection_px=side.reprojection_px * source.pixel_threshold_scale,
        min_motion_px=side.min_motion_px * source.pixel_threshold_scale,
    )
    if (
        config["max_frames"] is not None
        or config["camera_geometry"]["reference_camera"] is not None
    ):
        raise ValueError(
            "Expected the full-timeline ball decision and default reference camera"
        )
    reference = min(source.camera_ids)
    try:
        decision = decide_court_side(
            cameras,
            reference,
            np.stack([ball.uv_px for ball in balls]),
            np.stack([ball.observed for ball in balls]),
            side,
        )
        result = {"status": "decided", **json_value(decision)}
    except CourtSideUndecided as error:
        result = {
            "status": "undecided",
            "reason": error.reason,
            "frames": error.frames,
            "hypotheses": json_value(error.hypotheses),
        }
    return {
        "clip": CLIP,
        "source_sha256": recorded["source_sha256"],
        "court_projection_report_sha256": dual_sha256(_COURT_REPORT),
        "prior_calibration_receipt_sha256": recorded["court_receipt_sha256"],
        "court_fit": {
            "rmse_px": fit.rmse_px,
            "limit_px": limit,
            "historical_supported_points": len(cam0_points["supported_channels"]),
            "historical_inlier_points": len(cam0_points["inlier_channels"]),
            "historical_p95_to_detected_px": cam0_points["fit_error_px_p95"],
        },
        "cameras": [json_value(model) for model in cameras],
        "side_config": json_value(side),
        "side_config_source_sha256": dual_sha256(side_config),
        "observed_ball_frames": [int(ball.observed.sum()) for ball in balls],
        "ball_annotation_sha256": [dual_sha256(path) for path in ball_paths],
        "ball_decision": result,
        "person_label_scoring_attempts": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    write_json_atomic(args.output, recover(args.dataset, args.archives))


if __name__ == "__main__":
    main()
