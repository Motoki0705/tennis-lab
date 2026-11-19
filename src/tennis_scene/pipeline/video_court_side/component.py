"""Consume the shared decision with the current clip's calibration and evidence."""
from __future__ import annotations

from src.tasks.court_side.hypothesis import PlayerCourtSideConfig
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideInput,
    CourtSideOutput,
    PlayerCourtSideInput,
    court_side_io,
    player_court_side_io,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact

from .artifact import VideoCourtSideArtifact, require_accepted_courts
from .prepare import calibration_digest, observations_digest, player_observations_digest


class SharedCourtSideModule:
    def __init__(self, camera_ids: tuple[str, ...], artifact: VideoCourtSideArtifact, court: VideoCourtArtifact, policy: str,
                 *, court_policy: str = "warn") -> None:
        decision = artifact.decision
        self.player_config = PlayerCourtSideConfig(**decision.thresholds) if decision.evidence == "player" else None
        self.io = player_court_side_io(camera_ids) if self.player_config is not None else court_side_io(camera_ids)
        self.artifact, self.court, self.policy = artifact, court, policy
        self.court_policy = court_policy

    def require_clip(self, clip: str) -> None:
        self.artifact.verify()
        self.artifact.check_court(self.court, self.court_policy)
        if clip not in self.artifact.decision.evidence_clip_ids:
            require_accepted_courts(self.court, (clip,), self.court_policy)
        self.artifact.require_clip(clip, self.policy)

    def process(self, inputs: CourtSideInput | PlayerCourtSideInput) -> CourtSideOutput:
        clip = inputs.source.clip_id
        self.require_clip(clip)
        self.court.check_source(inputs.source)
        decision = self.artifact.decision
        mismatch = f"Court side calibration or {decision.evidence} observations differ from the shared decision: {clip}"
        if calibration_digest(inputs, self.court) != decision.calibration_digest:
            raise ValueError(mismatch)
        if isinstance(inputs, PlayerCourtSideInput) and self.player_config is not None:
            indices, views = inputs.sample(decision.max_frames, self.player_config)
            digest = player_observations_digest(indices, views)
        elif isinstance(inputs, CourtSideInput) and self.player_config is None:
            digest = observations_digest(*inputs.sample(decision.max_frames))
        else:
            raise ValueError("Current court side evidence differs from the shared decision")
        if digest != self.artifact.clips[clip].observations_digest:
            raise ValueError(mismatch)
        return self.artifact.output()
