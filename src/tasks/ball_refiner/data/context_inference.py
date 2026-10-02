"""Full-frame DINO -> BoT-SORT -> ViTPose and frame-zero court on stored JPEGs."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.submodules.models import (
    BotSortAssociator,
    DinoPersonDetector,
    PersonDetectionRequest,
    Pose2DFrameSequenceRequest,
    TrackRequest,
    ViTPosePose2D,
    select_and_complete_tracks,
    validate_dino_extension,
)
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_refiner.data.context_arrays import ContextArrays, GeneratedContext
from src.tasks.court_detection.inference.regions import CourtRegionUnavailable
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.court_kp import CourtKPConfig, CourtKPModule
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def load_context_producer(scene_config: Path, *, max_tracks: int) -> StoredJPEGContextProducer:
    """Use the scene's strict config adapters without binding video inputs."""
    config = OmegaConf.load(scene_config)
    if not isinstance(config, DictConfig):
        raise TypeError("scene_config must contain the complete composed scene configuration")
    scene = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    return StoredJPEGContextProducer(people=scene.people, court=scene.court_kp, max_tracks=max_tracks)


class StoredJPEGContextProducer:
    """Execute each stage explicitly; keep only observed (not filled) pose crops.

    Full-frame detection is a fixed policy, not a fallback for missing court
    ROI. The refiner has no track-ID embedding, so this producer keeps raw
    BoT-SORT IDs without scene-specific tracklet linking or player selection.
    The existing tracker box completion/smoothing is used only to crop the
    actually observed frames. All stages share unmodified stored images.
    """

    def __init__(self, *, people: PeopleModelConfig, court: CourtKPConfig, max_tracks: int) -> None:
        if people.detector != "dino" or people.runtime.device != court.device:
            raise ValueError("Context generation requires explicit DINO and a common model device")
        if type(max_tracks) is not int or max_tracks < 1:
            raise ValueError("max_tracks must be a positive integer; truncation is forbidden")
        self.people, self.court, self.max_tracks = people, court, max_tracks

    def identity(self) -> dict[str, Any]:
        extension_path = validate_dino_extension()
        assets = {"dino": self.people.dino_checkpoint, "vitpose": self.people.vitpose_checkpoint,
                  "court": self.court.checkpoint, "dino_extension": extension_path}
        source_root = PROJECT_ROOT / "src"
        repository = self.people.dino_repository
        if not repository.is_absolute() or not (repository / "models").is_dir():
            raise ValueError("DINO repository must be an existing absolute source directory")
        return {
            "assets": {name: {"path": str(path), "sha256": dual_sha256(path)} for name, path in assets.items()},
            "source_sha256": {str(path.relative_to(source_root)): dual_sha256(path)
                              for path in sorted(source_root.rglob("*.py"))},
            "dino_repository": {"path": str(repository), "sha256": {
                str(path.relative_to(repository)): dual_sha256(path) for path in sorted(repository.rglob("*.py"))}},
            "person_region_policy": "full_frame", "max_tracks": self.max_tracks,
            "tracking": "botsort_raw_ids; complete_and_smooth_boxes; observed_crops_only",
            "people_runtime": json_value(asdict(self.people.runtime)),
            "pose_precision": "float32",
            "dino_extension_preflight": "cpu_dispatch_forward_backward_v1",
            "court": {"subpixel_refine": self.court.subpixel_refine,
                      "postprocess": asdict(self.court.postprocess), "region_search": asdict(self.court.region_search)},
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "ultralytics")},
        }

    def predict(self, store: BallFrameStore, clip: ClipRecord) -> GeneratedContext:
        if store.clip_by_id(clip.clip_id) != clip:
            raise ValueError("Context clip does not belong to the store")
        n = clip.frame_count
        rows = store.clip_rows(clip)
        runtime = self.people.runtime
        detector = DinoPersonDetector(
            self.people.dino_checkpoint, self.people.dino_repository, device=runtime.device,
            confidence=runtime.dino_detector.confidence, short_side=runtime.dino_detector.short_side,
            max_long_side=runtime.dino_detector.max_long_side,
        )
        associator = BotSortAssociator()
        history: list[list[dict[str, Any]]] = []
        counts: list[int] = []
        start = time.perf_counter()
        try:
            for row in rows:
                image = store.read_bgr(int(row))
                detection = detector.predict(PersonDetectionRequest(image))
                counts.append(len(detection.scores))
                history.append(associator.update(detection, image))
        finally:
            detector.unload()
        detection_seconds = time.perf_counter() - start
        # In all-track, noninteractive mode video_path is a diagnostic label;
        # select_and_complete_tracks performs no video read or UI operation.
        tracked = select_and_complete_tracks(history, TrackRequest(clip.clip_id, None, False), n)
        ids = tracked.track_ids
        if len(ids) > self.max_tracks:
            raise ValueError(f"{clip.clip_id}: {len(ids)} tracks exceed explicit cap {self.max_tracks}")
        p = len(ids)
        boxes: np.ndarray = np.zeros((n, p, 3), np.float32)
        observed: np.ndarray = np.zeros((n, p), bool)
        for person, track_id in enumerate(ids):
            mask = tracked.observed_mask(track_id).numpy()
            observed[:, person] = mask
            boxes[mask, person] = tracked.bbx_xys(track_id, base_enlarge=runtime.tracking.bbox_enlarge).numpy()[mask]
        keypoints: np.ndarray = np.zeros((n, p, 17, 3), np.float32)
        frames, people = np.nonzero(observed)
        pose = ViTPosePose2D(
            self.people.vitpose_checkpoint, device=runtime.device, flip_test=runtime.vitpose.flip_test,
            batch_size=runtime.vitpose.batch_size, head_config=runtime.vitpose.head, precision="float32",
        )
        start = time.perf_counter()
        completed_crops = 0
        try:
            if len(frames):
                request = Pose2DFrameSequenceRequest(
                    ((index, store.read_bgr(int(row))) for index, row in enumerate(rows)),
                    torch.from_numpy(boxes[frames, people]), torch.from_numpy(frames.astype(np.int64)),
                )
                result = pose.predict(request).keypoints
                if result.shape != (len(frames), 17, 3) or result.device.type != "cpu" or result.dtype != torch.float32:
                    raise ValueError("ViTPose must preserve every requested observed crop as CPU float32 COCO17")
                keypoints[frames, people] = result.numpy()
                completed_crops = len(result)
        finally:
            pose.unload()
        pose_seconds = time.perf_counter() - start
        court = CourtKPModule(self.court)
        start = time.perf_counter()
        try:
            rgb = cv2.cvtColor(store.read_bgr(int(rows[0])), cv2.COLOR_BGR2RGB)
            try:
                prediction = court.predict_image(rgb)
                court_points, court_valid = prediction.points_px, prediction.valid
                diagnostics: dict[str, Any] = {"geometry": prediction.diagnostics, "region_selection": prediction.region_selection}
            except CourtRegionUnavailable as exc:
                # Search really executed; this is a measured absent context.
                # Decoder/model/runtime exceptions are deliberately not caught.
                court_points = np.zeros((14, 2), np.float32)
                court_valid = np.zeros(14, bool)
                diagnostics = {"status": "no_supported_region", "candidates": list(exc.candidates)}
        finally:
            court.unload()
        arrays = ContextArrays(
            frame_index=store.frames["frame_index"][rows].copy(), pts=store.frames["pts"][rows].copy(),
            detection_count=np.asarray(counts, np.int32), track_ids=np.asarray(ids, np.int64),
            boxes_xys=boxes, track_observed=observed, keypoints=keypoints,
            court_points=court_points, court_valid=court_valid,
        )
        return GeneratedContext(arrays, {
            "status": "complete", "person_region_policy": "full_frame",
            "person_detection_frames": len(counts), "tracking_frames": len(history), "pose_crops": completed_crops,
            "court_frame_indices": [0], "court_diagnostics": diagnostics,
            "seconds": {"detection_tracking": detection_seconds, "pose": pose_seconds,
                        "court": time.perf_counter() - start},
        })
