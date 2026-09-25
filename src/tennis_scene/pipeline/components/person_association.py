"""Independently loaded PLCS player Re-ID and court-side pipeline components."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.tasks.plcs.inference.person_predictor import (
    CourtSidePredictor,
    PlayerReIDPredictor,
)
from src.tasks.plcs.model_io.person_association import (
    CourtSideResult,
    PersonInferencePolicy,
    PersonObservationRequest,
    PersonReIDResult,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.utilts.timeline import association_frame_indices


@dataclass(frozen=True)
class PersonAssociationInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    people: ObjectObservations
    has_ball: bool = False


@dataclass(frozen=True)
class PlayerReIDOutput:
    camera_ids: tuple[str, ...]
    result: PersonReIDResult | None


@dataclass(frozen=True)
class CourtSideOutput:
    camera_ids: tuple[str, ...]
    reference_camera: str
    view_half_turns: tuple[bool, ...] | None
    side_logits: np.ndarray | None

    def __post_init__(self) -> None:
        if self.reference_camera not in self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("Invalid side artifact camera/reference IDs")
        if self.view_half_turns is not None:
            if len(self.view_half_turns) != len(self.camera_ids) or any(type(v) is not bool for v in self.view_half_turns):
                raise ValueError("Side artifact must declare one boolean per camera")
            if self.view_half_turns[self.camera_ids.index(self.reference_camera)]:
                raise ValueError("Reference camera cannot be half-turned")
        if self.side_logits is not None and (self.side_logits.shape != (len(self.camera_ids),) or not np.isfinite(self.side_logits).all()):
            raise ValueError("Invalid side model logits")


def person_model_request(inputs: PersonAssociationInput, policy: PersonInferencePolicy, visibility: float) -> PersonObservationRequest:
    raw, court = inputs.people, inputs.calibration.court
    active = [v.source_index for v in inputs.calibration.calibration.views]
    sample = association_frame_indices(inputs.source.num_frames, inputs.source.fps, max_frames=policy.max_frames)
    uv, visible = raw.normalized(visibility)
    size = np.asarray(raw.size, np.float32)
    court_uv = court.keypoints[active][:, sample] * (np.maximum(size - 1, 1) / size)
    return PersonObservationRequest(torch.from_numpy(uv[:, sample]), torch.from_numpy(visible[:, sample]),
        torch.from_numpy(np.array(raw.local_track_ids, copy=True)), torch.from_numpy(court_uv.astype(np.float32)),
        torch.from_numpy(court.visibility[active][:, sample].astype(bool)), raw.camera_ids,
        inputs.calibration.reference_camera, torch.from_numpy(sample))


class PlayerReIDModule:
    def __init__(self, checkpoint: Path, *, device: str = "cpu", camera_ids: tuple[str, ...] = (),
                 policy: PersonInferencePolicy | None = None, visibility: float = .15, enabled: bool = True) -> None:
        self.checkpoint, self.device = checkpoint, device
        self.policy, self.visibility, self.enabled = policy, visibility, enabled
        self.io = ComponentIO("person_reid", PersonAssociationInput, PlayerReIDOutput,
            {"calibration": InputPort("local_court_calibration"), **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}}, "person_identities")

    def process(self, inputs: PersonAssociationInput) -> PlayerReIDOutput:
        if self.policy is None:
            raise ValueError("Re-ID component requires a declared inference policy")
        result = None
        if self.enabled:
            result = self.process_observations(person_model_request(inputs, self.policy, self.visibility), policy=self.policy)
        return PlayerReIDOutput(inputs.people.camera_ids, result)

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> PersonReIDResult | None:
        if not bool(request.human_vis.any()):
            return None
        predictor = PlayerReIDPredictor.load(self.checkpoint, device=self.device)
        try:
            return predictor.predict_observations(request, policy=policy)
        finally:
            del predictor
            gc.collect()
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()


class CourtSideModule:
    def __init__(self, checkpoint: Path, *, device: str = "cpu", camera_ids: tuple[str, ...] = (),
                 policy: PersonInferencePolicy | None = None, visibility: float = .15) -> None:
        self.checkpoint, self.device = checkpoint, device
        self.policy, self.visibility = policy, visibility
        self.io = ComponentIO("court_side", PersonAssociationInput, CourtSideOutput,
            {"calibration": InputPort("local_court_calibration"),
             **{f"pose_{c}": InputPort("person_poses") for c in camera_ids},
             **{f"ball_{c}": InputPort("ball_detections") for c in camera_ids}}, "court_side")

    def process(self, inputs: PersonAssociationInput) -> CourtSideOutput:
        if self.policy is None:
            raise ValueError("Side component requires a declared inference policy")
        result = None
        request = person_model_request(inputs, self.policy, self.visibility)
        if bool(request.human_vis.any()) or inputs.has_ball:
            result = self.process_observations(request, policy=self.policy)
        return CourtSideOutput(inputs.people.camera_ids, inputs.calibration.reference_camera,
            None if result is None else tuple(bool(v) for v in result.view_half_turns.tolist()),
            None if result is None else result.side_logits.numpy())

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> CourtSideResult:
        predictor = CourtSidePredictor.load(self.checkpoint, device=self.device)
        try:
            return predictor.predict_observations(request, policy=policy)
        finally:
            del predictor
            gc.collect()
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
