"""Video-wide decision and standalone clip measurements, independent of execution policy."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tasks.court_side.hypothesis import (
    CourtSideUndecided,
    EvidenceKind,
    HypothesisScore,
    SideConfig,
    SideEvidence,
    evidence_kind,
    judge_side_evidence,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.tennis_scene.pipeline.video_artifact import load_document
from src.tennis_scene.pipeline.video_court.diagnostics import (
    clip_report,
    drift_identity,
)
from src.tennis_scene.pipeline.video_court.policy import (
    require_accepted_drift,
    validate_unobservable_policy,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact

SCHEMA = "video_court_side_v1"
DIAGNOSTICS_SCHEMA = "video_court_side_diagnostics_v1"


def require_accepted_courts(court: VideoCourtArtifact, clips: Iterable[str], policy: str) -> None:
    drift_digests = drift_identity(court).get("camera_digests", {})
    for clip in clips:
        for camera, document in court.cameras.items():
            require_accepted_drift(document, clip, clip_report(court, camera, clip), policy,
                                   artifact_digest=court.manifest["cameras"][camera], drift_digest=drift_digests.get(camera))


@dataclass(frozen=True)
class SideAssessment:
    reason: str | None
    hypotheses: tuple[HypothesisScore, ...]
    frames: int
    pair_frames: dict[str, int]
    margin: float | None

    @classmethod
    def judge(cls, evidence: SideEvidence, config: SideConfig) -> SideAssessment:
        reason = None
        try:
            judgement = judge_side_evidence(evidence, config)
            margin: float | None = judgement.margin
        except CourtSideUndecided as undecided:
            reason = undecided.reason
            margin = undecided.margin
        return cls(reason, evidence.hypotheses, evidence.frames, evidence.pair_record(), margin)


@dataclass(frozen=True)
class ClipMeasurement:
    sampled_frames: int
    scored_frame_indices: tuple[int, ...]  # source-frame indices within this clip only
    observations_digest: str
    standalone: SideAssessment


@dataclass(frozen=True)
class VideoSideDecision:
    schema: str
    dataset_id: str
    video_id: str
    court_digest: str
    court_drift_digest: str | None
    thresholds: dict[str, Any]
    max_frames: int
    implementation: str
    input_revisions: dict[str, str]
    evidence_clip_ids: tuple[str, ...]
    calibration_digest: str
    camera_ids: tuple[str, ...]
    reference_camera: str
    shared: SideAssessment
    diagnostics_digest: str
    evidence: EvidenceKind = "ball"

    def document(self) -> dict[str, Any]:
        value: dict[str, Any] = json_value(self)
        # Legacy v1 documents imply ball evidence and keep their original digest.
        if self.evidence == "ball":
            del value["evidence"]
        return value


@dataclass(frozen=True)
class VideoCourtSideArtifact:
    directory: Path
    digest: str
    decision: VideoSideDecision
    clips: dict[str, ClipMeasurement]

    @classmethod
    def load(cls, directory: Path, digest: str) -> VideoCourtSideArtifact:
        directory = directory.resolve()
        raw = load_document(directory, digest)
        decision: VideoSideDecision = restore_type({"evidence": "ball", **raw}, VideoSideDecision)
        if decision.schema != SCHEMA or raw != decision.document():
            raise ValueError("Unknown shared court side artifact schema")
        diagnostics = load_document(directory, decision.diagnostics_digest)
        if set(diagnostics) != {"schema", "clips"} or diagnostics["schema"] != DIAGNOSTICS_SCHEMA:
            raise ValueError("Invalid standalone court side diagnostics")
        clips: dict[str, ClipMeasurement] = restore_type(diagnostics["clips"], dict[str, ClipMeasurement])
        members = set(clips)
        selected = decision.evidence_clip_ids
        if (not selected or not set(selected) <= members or len(set(selected)) != len(selected)
                or members != set(decision.input_revisions)
                or any(len(c.scored_frame_indices) != c.standalone.frames for c in clips.values())
                or decision.shared.frames != sum(clips[c].standalone.frames for c in selected)):
            raise ValueError("Shared court side membership or scored-frame count disagrees with its diagnostics")
        return cls(directory, digest, decision, clips)

    def verify(self) -> None:
        if load_document(self.directory, self.digest) != self.decision.document():
            raise ValueError("Shared court side decision changed")
        diagnostics = load_document(self.directory, self.decision.diagnostics_digest)
        if diagnostics != {"schema": DIAGNOSTICS_SCHEMA, "clips": json_value(self.clips)}:
            raise ValueError("Standalone court side diagnostics changed")

    def check_court(self, court: VideoCourtArtifact, court_policy: str) -> None:
        court.verify()
        d = self.decision
        if (d.dataset_id, d.video_id, d.court_digest) != (court.manifest["dataset_id"], court.manifest["video_id"], court.digest):
            raise ValueError("Shared court side artifact belongs to another video or court")
        if d.court_drift_digest != drift_identity(court).get("artifact_digest"):
            raise ValueError("Shared court side was measured with different court drift evidence")
        require_accepted_courts(court, d.evidence_clip_ids, court_policy)

    def check_settings(self, court: VideoCourtArtifact, thresholds: SideConfig, max_frames: int, *, court_policy: str = "warn") -> None:
        self.check_court(court, court_policy)
        if self.decision.evidence != evidence_kind(thresholds):
            raise ValueError("Shared court side evidence disagrees with execution")
        if self.decision.thresholds != json_value(thresholds) or self.decision.max_frames != max_frames:
            raise ValueError("Shared court side thresholds or sampling disagree with execution")

    def clip_verdict(self, clip_id: str, policy: str) -> dict[str, Any]:
        validate_unobservable_policy(policy)
        d = self.decision
        local = self.clips[clip_id].standalone
        status, reason = "ok", None
        if d.shared.reason is not None:
            status, reason = "failed", f"court_side_video_shared_{d.shared.reason}"
        elif local.reason is not None:
            status = "warning" if policy == "warn" else "failed"
            reason = f"court_side_{local.reason}"
        elif local.hypotheses[0].view_half_turns != d.shared.hypotheses[0].view_half_turns:
            status, reason = "failed", "court_side_video_shared_conflict"
        return {"status": status, "reason": reason, **({"evidence": d.evidence} if d.evidence == "player" else {})}

    def clip_report(self, clip_id: str, policy: str) -> dict[str, Any]:
        d = self.decision
        measurement = self.clips[clip_id]
        return {"directory": str(self.directory), "artifact_digest": self.digest, "diagnostics_digest": d.diagnostics_digest,
                "camera_ids": list(d.camera_ids), "reference_camera": d.reference_camera, "clip_id": clip_id,
                "sampled_frames": measurement.sampled_frames, **self.clip_verdict(clip_id, policy), "unobservable_policy": policy,
                "standalone": json_value(measurement.standalone), "shared": json_value(d.shared)}

    def require_clip(self, clip_id: str, policy: str) -> None:
        if self.clip_verdict(clip_id, policy)["status"] == "failed":
            report = self.clip_report(clip_id, policy)
            raise ReconstructionUnavailable(report["reason"], f"{clip_id}: {report['reason']}", diagnostics=report)

    def node_settings(self, clip_id: str, side_policy: str, court_policy: str) -> dict[str, Any]:
        return {"artifact_digest": self.digest, "unobservable_policy": side_policy, "court_unobservable_policy": court_policy,
                **self.clip_verdict(clip_id, side_policy)}

    def publication_verdicts(self, court: VideoCourtArtifact, side_policy: str, court_policy: str) -> dict[str, dict[str, Any]]:
        """Per-clip admission for markers, including clips outside the evidence selection."""
        clips = {clip: self.clip_verdict(clip, side_policy) for clip in self.clips}
        for clip in self.clips.keys() - set(self.decision.evidence_clip_ids):
            try:
                require_accepted_courts(court, (clip,), court_policy)
            except ReconstructionUnavailable as error:
                clips[clip].update(status="failed", reason=error.reason, detail=str(error))
        return clips

    def output(self) -> CourtSideOutput:
        d = self.decision
        if d.shared.reason is not None or d.shared.margin is None:
            raise ValueError("Cannot use an undecided video court side")
        return CourtSideOutput(d.camera_ids, d.reference_camera, d.shared.hypotheses[0].view_half_turns,
                               d.shared.hypotheses, d.shared.margin, d.shared.frames)
