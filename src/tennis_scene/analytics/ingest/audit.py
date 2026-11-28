"""Compile confirmed point context and account for availability over all clips."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, TextIO

from src.tennis_scene.analytics.export import write_csv, write_json
from src.tennis_scene.analytics.ingest.score_adapter import (
    TIME_TOLERANCE_SEC,
    ScoredClip,
    ScoredVideo,
)
from src.tennis_scene.analytics.ingest.supplement import (
    ManualPoint,
    validate_supplement,
)
from src.tennis_scene.analytics.ingest.terminal_cause import (
    TerminalEvidence,
    derive_terminal_cause,
)
from src.tennis_scene.analytics.ingest.v4_reader import SPLITS
from src.tennis_scene.analytics.metrics import (
    DEFINITION_VERSION,
    ErrorAssessment,
    TerminalCause,
    TerminalCauseSource,
    counted_error_assessment,
)
from src.tennis_scene.ball_event_labels import (
    EVENT_CLASSES_V4,
    EVENT_STATUS,
    EVENT_SUBTYPES_V4,
)
from src.tennis_scene.clip_studio.project import count_to_dict


@dataclass(frozen=True)
class PreparedClip:
    scored: ScoredClip
    terminal: TerminalEvidence
    terminal_clip_id: str
    manual_point: ManualPoint | None
    manual_attempts: ManualPoint | None

    @property
    def terminal_cause(self) -> TerminalCause:
        if (
            self.manual_point is not None
            and self.manual_point.terminal_cause is not None
        ):
            return self.manual_point.terminal_cause
        return self.terminal.cause

    @property
    def terminal_cause_source(self) -> TerminalCauseSource:
        if (
            self.manual_point is not None
            and self.manual_point.terminal_cause is not None
        ):
            return "manual_confirmed"
        return "derived_v4_score" if self.terminal.cause != "unknown" else "unknown"

    @property
    def error_assessment(self) -> ErrorAssessment:
        if (
            self.manual_point is not None
            and self.manual_point.error_assessment is not None
        ):
            return self.manual_point.error_assessment
        return "undetermined"

    @property
    def error_assessment_source(self) -> str | None:
        if (
            self.manual_point is not None
            and self.manual_point.error_assessment is not None
        ):
            return "manual_confirmed"
        return None


@dataclass(frozen=True)
class PreparedIngest:
    video: ScoredVideo
    manual_supplement_sha256: str | None
    clips: tuple[PreparedClip, ...]


def prepare_ingest(
    video: ScoredVideo,
    supplement_sha256: str | None,
    supplements: dict[str, ManualPoint],
) -> PreparedIngest:
    by_id = {clip.v4.clip_id: clip for clip in video.clips}
    by_point: dict[str, ManualPoint] = {}
    for clip_id, supplement in supplements.items():
        scored = by_id.get(clip_id)
        if scored is None:
            raise ValueError(
                f"Manual supplement clip is outside the V4 release: {clip_id}"
            )
        validate_supplement(supplement, scored)
        key = scored.match_point_key
        assert key is not None
        if key in by_point:
            raise ValueError(
                f"Multiple manual supplements refer to the same match point: {key}"
            )
        by_point[key] = supplement

    # Keep evidence consistent across alternate exports of one point.
    evidence_by_point: dict[str, list[tuple[TerminalEvidence, str]]] = defaultdict(list)
    for scored in video.clips:
        if scored.match_point_key is not None:
            evidence_by_point[scored.match_point_key].append(
                (derive_terminal_cause(scored), scored.v4.clip_id)
            )
    clips = []
    for scored in video.clips:
        key = scored.match_point_key
        clip_id = scored.v4.clip_id
        if key is None:
            terminal, evidence_clip_id = derive_terminal_cause(scored), clip_id
            manual = None
        else:
            evidence = evidence_by_point[key]
            resolved = [
                (item, source) for item, source in evidence if item.cause != "unknown"
            ]
            if len({item.cause for item, _ in resolved}) > 1:
                terminal, evidence_clip_id = (
                    TerminalEvidence(
                        "unknown", missing_reason="conflicting_v4_terminal_labels"
                    ),
                    clip_id,
                )
            else:
                terminal, evidence_clip_id = (resolved or evidence)[0]
            manual = by_point.get(key)
        clips.append(
            PreparedClip(
                scored, terminal, evidence_clip_id, manual, supplements.get(clip_id)
            )
        )
    return PreparedIngest(video, supplement_sha256, tuple(clips))


def _event_details(clip: PreparedClip) -> dict[str, Any]:
    events = clip.scored.v4.events
    adopted = [event for event in events if event.adopted_in_v4]
    excluded = [
        {"source_event_id": event.source_event_id, "reason": event.excluded_reason}
        for event in events
        if not event.adopted_in_v4
    ]
    outside = {}
    for event in adopted:
        reason = event.outside_point_reason
        if clip.scored.score_status == "not_point":
            reason = "projects_not_point"
        if reason is not None:
            outside[event.source_event_id] = reason
    return {
        "adopted_events": len(adopted),
        "excluded_events": len(excluded),
        "adopted_source_event_ids": [event.source_event_id for event in adopted],
        "excluded_source_events": excluded,
        "adopted_outside_point": outside,
        "by_subtype": dict(Counter(event.subtype for event in adopted)),
        "unlabelable_frames": int((~clip.scored.v4.coverage).sum()),
    }


def _clip_report(clip: PreparedClip) -> dict[str, Any]:
    scored = clip.scored
    score = scored.confirmed
    manual = clip.manual_attempts
    score_missing = (
        "missing_score" if scored.score_status is None else scored.score_status
    )
    missing_reasons = (
        dict.fromkeys(
            ("score_before", "score_after", "server_team", "winner_team"), score_missing
        )
        if score is None
        else {}
    )
    if clip.terminal_cause == "unknown":
        missing_reasons["terminal_cause"] = (
            score_missing if score is None else clip.terminal.missing_reason
        )
    if (
        counted_error_assessment(clip.error_assessment, clip.error_assessment_source)
        is None
    ):
        missing_reasons["error_assessment"] = (
            score_missing
            if score is None
            else (
                "manually_undetermined"
                if clip.error_assessment_source
                else "no_manual_error_assessment"
            )
        )
    if manual is None or not manual.serve_attempts:
        missing_reasons["manual_serve_attempts"] = (
            score_missing if score is None else "no_manual_serve_attempts"
        )
    return {
        "clip_id": scored.v4.clip_id,
        "split": scored.v4.split,
        "match_point_key": scored.match_point_key,
        "score_status": scored.score_status,
        "score_before": count_to_dict(score.before)
        if score is not None and score.before is not None
        else None,
        "score_after": count_to_dict(scored.score_after)
        if scored.score_after is not None
        else None,
        "server_team": score.server if score is not None else None,
        "winner_team": score.winner if score is not None else None,
        "terminal_cause": clip.terminal_cause,
        "terminal_cause_source": clip.terminal_cause_source,
        "error_assessment": clip.error_assessment,
        "error_assessment_source": clip.error_assessment_source,
        "manual_serve_attempts": len(manual.serve_attempts)
        if manual is not None
        else 0,
        "missing_reasons": missing_reasons,
        "global_start_sec": scored.v4.global_start_sec,
        "global_end_sec": scored.v4.global_end_sec,
        "start_error_sec": scored.start_error_sec,
        "end_error_sec": scored.end_error_sec,
        **_event_details(clip),
    }


def audit_report(batch: PreparedIngest) -> dict[str, Any]:
    video = batch.video
    release = video.release
    clips = [_clip_report(item) for item in batch.clips]
    columns = {}
    for field in (
        "score_before",
        "score_after",
        "server_team",
        "winner_team",
        "terminal_cause",
        "error_assessment",
        "manual_serve_attempts",
        "adopted_events",
        "excluded_events",
        *(f"v4_subtype.{subtype}" for subtype in EVENT_SUBTYPES_V4),
    ):
        missing = Counter(
            row["missing_reasons"][field]
            for row in clips
            if field in row["missing_reasons"]
        )
        available = len(clips) - sum(missing.values())
        columns[field] = {
            "available": available,
            "total": len(clips),
            "rate": available / len(clips),
            "missing_by_reason": dict(missing),
        }
    adopted = [
        event for clip in video.clips for event in clip.v4.events if event.adopted_in_v4
    ]
    labels = {}
    for axis, keys in (
        ("kind", EVENT_CLASSES_V4),
        ("subtype", EVENT_SUBTYPES_V4),
        ("status", EVENT_STATUS),
    ):
        counts = Counter(getattr(event, axis) for event in adopted)
        labels[f"by_{axis}"] = {key: counts[key] for key in keys}
    excluded_reasons = Counter(
        item["reason"] for row in clips for item in row["excluded_source_events"]
    )
    split_counts = Counter(item.scored.v4.split for item in batch.clips)
    unresolved = [
        row["clip_id"]
        for row in clips
        if row["score_status"] not in ("confirmed", "not_point")
    ]
    return {
        "benchmark_id": release.benchmark_id,
        "dataset_id": release.dataset_id,
        "video_id": release.video_id,
        "definition_version": DEFINITION_VERSION,
        "input_hashes": {
            "v4_manifest_sha256": release.manifest_sha256,
            "v4_sha256sums_sha256": release.sha256sums_sha256,
            "projects_json_sha256": video.projects_sha256,
            "manual_supplement_sha256": batch.manual_supplement_sha256,
        },
        "checksum_verification": {
            "scope": "ingest_files",
            "files_verified": release.verified_files,
            "files_listed": release.listed_files,
        },
        "clip_count": len(clips),
        "match_point_count": len(
            {
                row["match_point_key"]
                for row in clips
                if row["match_point_key"] is not None
            }
        ),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "time_alignment": {
            "matched_clips": len(clips),
            "max_error_sec": max(
                max(row["start_error_sec"], row["end_error_sec"]) for row in clips
            ),
            "tolerance_sec": TIME_TOLERANCE_SEC,
        },
        "score_chain": {
            "status": "continuous"
            if video.chain_segments == 1 and not unresolved
            else "partial",
            "verified_links": video.chain_links,
            "confirmed_segments": video.chain_segments,
            "match_complete": video.match_complete,
            "unresolved_clip_ids": unresolved,
            "by_score_status": dict(
                Counter(row["score_status"] or "missing_score" for row in clips)
            ),
        },
        "terminal_cause_counts": dict(Counter(row["terminal_cause"] for row in clips)),
        "terminal_cause_sources": dict(
            Counter(row["terminal_cause_source"] for row in clips)
        ),
        "event_counts": {
            "adopted": len(adopted),
            "excluded": sum(excluded_reasons.values()),
            "excluded_by_reason": dict(excluded_reasons),
            "adopted_outside_point_by_reason": dict(
                Counter(
                    reason
                    for row in clips
                    for reason in row["adopted_outside_point"].values()
                )
            ),
            "unlabelable_frames": sum(row["unlabelable_frames"] for row in clips),
            **labels,
        },
        "columns": columns,
        "clips": clips,
    }


def write_audit(stream: TextIO, report: dict[str, Any], format: str) -> None:
    if format == "json":
        write_json(stream, report)
        return
    if format != "csv":
        raise ValueError(f"Unsupported audit format: {format}")
    rows = []
    for key, value in report.items():
        if key not in ("clips", "columns"):
            rows.append({"section": "summary", "key": key, "data": value})
    rows.extend(
        {"section": "column", "key": key, "data": value}
        for key, value in report["columns"].items()
    )
    rows.extend(
        {"section": "clip", "key": row["clip_id"], "data": row}
        for row in report["clips"]
    )
    write_csv(stream, ("section", "key", "data"), rows)
