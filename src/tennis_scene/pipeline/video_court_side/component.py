"""Consume the shared decision with the current clip's calibration and ball evidence."""
from __future__ import annotations

from src.tennis_scene.pipeline.components.court_side import (
    CourtSideInput,
    CourtSideOutput,
    court_side_io,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact

from .artifact import VideoCourtSideArtifact, require_accepted_courts
from .prepare import calibration_digest, observations_digest


class SharedCourtSideModule:
    def __init__(self, camera_ids: tuple[str, ...], artifact: VideoCourtSideArtifact, court: VideoCourtArtifact, policy: str,
                 *, court_policy: str = "warn") -> None:
        self.io = court_side_io(camera_ids)
        self.artifact, self.court, self.policy = artifact, court, policy
        self.court_policy = court_policy

    def require_clip(self, clip: str) -> None:
        self.artifact.verify()
        self.artifact.check_court(self.court, self.court_policy)
        if clip not in self.artifact.decision.evidence_clip_ids:
            require_accepted_courts(self.court, (clip,), self.court_policy)
        self.artifact.require_clip(clip, self.policy)

    def process(self, inputs: CourtSideInput) -> CourtSideOutput:
        clip = inputs.source.clip_id
        self.require_clip(clip)
        self.court.check_source(inputs.source)
        decision = self.artifact.decision
        if (calibration_digest(inputs, self.court) != decision.calibration_digest
                or observations_digest(*inputs.sample(decision.max_frames)) != self.artifact.clips[clip].observations_digest):
            raise ValueError(f"Court side calibration or ball observations differ from the shared decision: {clip}")
        return self.artifact.output()
