"""Meiji JPEG context through #964's frozen detection/feature/tracking/selection."""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    ViTPosePose2D,
)
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_refiner.data.context_arrays import ContextArrays, GeneratedContext
from src.tasks.ball_refiner.data.context_inference import StoredJPEGContextProducer
from src.tasks.court_detection.inference.regions import CourtRegionUnavailable
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.person_tracking.sequence import TrackingSequence, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.association.associate import CameraTracks
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.components.camera_geometry import calibrate_local_courts
from src.tennis_scene.pipeline.components.court_kp import (
    CourtImagePrediction,
    CourtKPModule,
    CourtKPResult,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.keypoints import normalize_grid_keypoints
from src.utils.paths import PROJECT_ROOT


def require_meiji(clip: ClipRecord) -> None:
    video = {"train": "video_002", "val": "video_000"}.get(clip.split)
    if clip.source != "meiji" or video is None or f"/{video}/" not in clip.clip_id:
        raise ValueError(f"Only Meiji train/video_002 and val/video_000 are permitted: {clip.clip_id}")


def load_meiji_producer(scene_config: Path, freeze: Path) -> MeijiContextProducer:
    config = OmegaConf.load(scene_config)
    if not isinstance(config, DictConfig):
        raise ValueError("A complete scene configuration is required")
    return MeijiContextProducer(PipelineRuntimeConfig.from_config(config, bind_inputs=False), freeze)


class MeijiContextProducer(StoredJPEGContextProducer):
    """Models run in separate detector, pose/CLIP, and court memory stages.

    No cross-camera association runs. Candidate A's configuration is checked
    because its footpoint/sampling contract is used by the shared selection.
    A failure receipt covers every frame; it never becomes a readable cache.
    """

    def __init__(self, scene: PipelineRuntimeConfig, freeze: Path) -> None:
        super().__init__(people=scene.people, court=scene.court_kp, max_tracks=scene.max_tracks_per_camera)
        self.scene, self.freeze = scene, freeze
        self.progress: dict[str, Any] = {}
        frozen = json.loads(freeze.read_text())["person"]
        actual = {
            "detector_runtime": asdict(self.people.runtime.dino_detector),
            "pose_runtime": json_value(asdict(self.people.runtime.vitpose)),
            "tracking": json_value(scene.tracking.identity()),
            "association": json_value(asdict(scene.player_association)),
            "selection": asdict(LinkingConfig(max_candidates=scene.max_tracks_per_camera)),
            "merge_duplicate_person_boxes": scene.merge_duplicate_person_boxes,
        }
        if any(actual[k] != frozen[k] for k in actual):
            raise ValueError("Meiji producer differs from the frozen #964 person defaults")

    def identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = super().identity()
        frozen = json.loads(self.freeze.read_text())
        assets = {"dino": "detector", "vitpose": "vitpose", "court": "court"}
        for key, path in (("aflink", self.scene.aflink_checkpoint), ("tracking_encoder", self.scene.tracking_encoder_weights)):
            identity["assets"][key] = {"path": str(path), "sha256": dual_sha256(path)}
            assets[key] = key
        for key, original in assets.items():
            if identity["assets"][key]["sha256"] != frozen["assets"][original]["sha256"]:
                raise ValueError(f"Frozen person/court weight changed: {key}")
        for name, digest in frozen["source"].items():
            if name.startswith(("src/tasks/person_tracking/", "src/tasks/player_association/", "src/submodules/models/")) \
                    and dual_sha256(PROJECT_ROOT / name) != digest:
                raise ValueError(f"Frozen shared person implementation changed: {name}")
        identity.update(producer="meiji_i964_shared_sequence.v1", tracking=self.scene.tracking.identity(),
                        selection=asdict(LinkingConfig(max_candidates=self.max_tracks)),
                        association=json_value(asdict(self.scene.player_association)),
                        camera_geometry=asdict(self.scene.camera_geometry),
                        frozen_person={"sha256": dual_sha256(self.freeze), "person": frozen["person"]})
        return identity

    def _tracking(self, store: BallFrameStore, clip: ClipRecord) -> tuple[TrackingSequence, np.ndarray]:
        rows = store.clip_rows(clip)
        runtime = self.people.runtime
        detector = DinoPersonDetector(self.people.dino_checkpoint, self.people.dino_repository,
            device=runtime.device, confidence=runtime.dino_detector.confidence,
            short_side=runtime.dino_detector.short_side, max_long_side=runtime.dino_detector.max_long_side)
        detections = []
        try:
            for index, row in enumerate(rows):
                self.progress.update(stage="detection", frame=index)
                detections.append(detector.predict(PersonDetectionRequest(store.read_bgr(int(row)))))
        finally:
            detector.unload()
            release_inference_memory(runtime.device)
        counts = np.asarray([len(d.scores) for d in detections], np.int32)
        pose = ViTPosePose2D(self.people.vitpose_checkpoint, device=runtime.device,
            flip_test=runtime.vitpose.flip_test, batch_size=runtime.vitpose.batch_size,
            head_config=runtime.vitpose.head, precision="float32")
        encoder = None
        try:
            encoder = build_encoder(self.scene.tracking.encoder, checkpoint_root=self.scene.roots.checkpoint_root,
                                    external_root=self.scene.roots.external_asset_root, device=runtime.device)
            extractor = FeatureExtractor(pose, UnpromptedEncoder(encoder, 1280), self.scene.tracking.features)

            def frames() -> Iterator[DetectionFeatures]:
                offset = 0
                for index, (row, detection) in enumerate(zip(rows, detections, strict=True)):
                    self.progress.update(stage="pose_clip_tracking", frame=index)
                    count = len(detection.scores)
                    yield extractor.extract(index, store.read_bgr(int(row)), np.arange(offset, offset + count, dtype=np.int64),
                                            detection.boxes_xyxy, detection.scores)
                    offset += count

            result = track_sequence(frames(), fps=float(Fraction(clip.fps)), config=self.scene.tracking,
                                    aflink=AFLink(self.scene.aflink_checkpoint))
        finally:
            pose.unload()
            del encoder
            release_inference_memory(runtime.device)
        return result, counts

    def _court(self, image: np.ndarray) -> CourtImagePrediction:
        court = CourtKPModule(self.court)
        try:
            try:
                return court.predict_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except CourtRegionUnavailable as error:
                return CourtImagePrediction(np.zeros((14, 2), np.float32), np.zeros(14, bool),
                    {"status": "no_supported_region", "candidates": list(error.candidates)}, None)
        finally:
            court.unload()
            release_inference_memory(self.people.runtime.device)

    def _select(self, result: TrackingSequence, clip: ClipRecord, court: CourtImagePrediction) -> tuple[Any, Any, dict[str, Any]]:
        assert clip.camera_id is not None
        observation = CourtKPResult(normalize_grid_keypoints(court.points_px, clip.width, clip.height)[None, None],
            court.valid[None, None], np.array([0], np.int32),
            {"output_keypoint_contract": "camera_view_v2", "cameras": [{"frames": [{"frame_index": 0, **court.diagnostics}]}]})
        calibration = calibrate_local_courts(observation, (clip.camera_id,), size=(clip.width, clip.height), config=self.scene.camera_geometry)
        if not calibration.views:
            return (np.zeros((0, clip.frame_count, 4), np.float32),
                    result.evidence.regroup(np.full((0, clip.frame_count), -1, np.int64)),
                    {"status": "camera_not_calibrated", "excluded": calibration.excluded})
        scale = np.hypot(clip.width, clip.height) / np.hypot(1920, 1080)
        sampling = CropSamplingConfig()
        sampling = replace(sampling, min_height_px=sampling.min_height_px * scale, border_px=sampling.border_px * scale)
        appearances = evidence_appearance(result.boxes, result.observed, result.evidence, (clip.width, clip.height), sampling)
        tracks = CameraTracks(calibration.views[0].camera, (clip.width, clip.height), result.track_ids,
                              result.boxes, result.observed, appearances)
        footpoints = replace(self.scene.player_association.footpoints,
                            bottom_border_px=self.scene.player_association.footpoints.bottom_border_px * scale)
        _, diagnostic = select_linked_candidates(tracks, float(Fraction(clip.fps)), LinkingConfig(max_candidates=self.max_tracks), footpoints)
        grouped, origins = linked_timeline(tracks, diagnostic)
        return grouped.boxes_xyxy, result.evidence.regroup(origins), {"status": "complete", "linking": diagnostic,
            "origin_rows": origins.tolist(), "raw_track_ids": result.track_ids.tolist(),
            "aflink_source_track_ids": result.source_track_ids}

    def predict(self, store: BallFrameStore, clip: ClipRecord) -> GeneratedContext:
        require_meiji(clip)
        if store.clip_by_id(clip.clip_id) != clip or clip.camera_id is None:
            raise ValueError("Meiji context requires matching store and an explicit camera")
        start = time.monotonic()
        self.progress = {"clip_id": clip.clip_id, "stage": "starting", "frame": 0, "frames": clip.frame_count}
        result, counts = self._tracking(store, clip)
        self.progress.update(stage="court", frame=0)
        rows = store.clip_rows(clip)
        court = self._court(store.read_bgr(int(rows[0])))
        self.progress.update(stage="selection", frame=None)
        xyxy, evidence, diagnostic = self._select(result, clip, court)
        observed = evidence.detection_rows.T >= 0
        boxes = np.concatenate(((xyxy[..., :2] + xyxy[..., 2:]) * .5,
            (xyxy[..., 2:] - xyxy[..., :2]).max(-1, keepdims=True) * self.scene.tracking.features.bbox_enlarge), -1).transpose(1, 0, 2)
        boxes[~observed] = 0
        arrays = ContextArrays(store.frames["frame_index"][rows].copy(), store.frames["pts"][rows].copy(), counts,
            np.arange(observed.shape[1], dtype=np.int64), boxes.astype(np.float32), observed,
            evidence.poses.transpose(1, 0, 2, 3).copy(), court.points_px, court.valid)
        # Every source frame has an explicit pose state, independent of ball labels.
        states = np.where(observed.any(1), "observed", np.where(counts == 0, "no_detection", "no_selected_track")).tolist()
        if diagnostic["status"] == "camera_not_calibrated":
            states = ["court_calibration_failed"] * clip.frame_count
        self.progress.update(stage="complete", frame=None)
        return GeneratedContext(arrays, {"status": "complete", "person_region_policy": "full_frame",
            "person_detection_frames": clip.frame_count, "tracking_frames": clip.frame_count,
            "pose_crops": int(observed.sum()), "inferred_detection_pose_crops": int(counts.sum()),
            "court_frame_indices": [0], "court_diagnostics": court.diagnostics,
            "selection": diagnostic, "pose_frame_status": states, "seconds": time.monotonic() - start,
            "raw_negative_pose_peaks": int((arrays.keypoints[..., 2] < 0).sum()),
            "synthetic_gsi_used_as_observed": False})
