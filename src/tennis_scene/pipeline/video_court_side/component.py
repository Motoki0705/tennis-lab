"""Consume the shared decision with the current clip's calibration and ball evidence."""
from __future__ import annotations

from src.tennis_scene.pipeline.components.court_side import (
    CourtSideInput,
    CourtSideOutput,
    court_side_io,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact

from .artifact import VideoCourtSideArtifact
from .prepare import calibration_digest, observations_digest


class SharedCourtSideModule:
    def __init__(self, camera_ids: tuple[str, ...], artifact: VideoCourtSideArtifact, court: VideoCourtArtifact, policy: str) -> None:
        self.io = court_side_io(camera_ids)
        self.artifact, self.court, self.policy = artifact, court, policy

    def process(self, inputs: CourtSideInput) -> CourtSideOutput:
        self.artifact.verify()
        self.court.check_source(inputs.source)
        clip = inputs.source.clip_id
        decision = self.artifact.decision
        if (calibration_digest(inputs, self.court) != decision.calibration_digest
                or observations_digest(*inputs.sample(decision.max_frames)) != self.artifact.clips[clip].observations_digest):
            raise ValueError(f"Court side calibration or ball observations differ from the shared decision: {clip}")
        report = self.artifact.clip_report(clip, self.policy)
        if report["status"] == "failed":
            raise ReconstructionUnavailable(report["reason"], f"{clip}: {report['reason']}", diagnostics=report)
        return self.artifact.output()
