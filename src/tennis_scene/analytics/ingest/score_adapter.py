"""Join the frozen timeline to the official Clip Studio point ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any

from src.tennis_scene.analytics.ingest.input_files import json_object, sha256_value
from src.tennis_scene.analytics.ingest.v4_reader import (
    V4Clip,
    V4Release,
)
from src.tennis_scene.clip_studio.project import (
    Clip,
    ClipScore,
    ClipStudioProjects,
    MatchSettings,
    ScoreStatus,
)
from src.tennis_scene.clip_studio.score_chain import score_views
from src.utils.configuration import PathResolver, PathRole
from src.utils.tennis_scoring import ScoreCount, advance_point, match_over

TIME_TOLERANCE_SEC = 1e-9
POINT_KEY = "point"


@dataclass(frozen=True)
class PointIdentityHold:
    reason: str
    candidate_clip_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "candidate_clip_ids": list(self.candidate_clip_ids),
        }


@dataclass(frozen=True)
class ScoredClip:
    v4: V4Clip
    recorded_score: ClipScore | None
    score_after: ScoreCount | None
    match_point_key: str | None
    start_error_sec: float
    end_error_sec: float
    point_identity_hold: PointIdentityHold | None = None

    @property
    def score_status(self) -> ScoreStatus | None:
        return self.recorded_score.status if self.recorded_score is not None else None

    @property
    def confirmed(self) -> ClipScore | None:
        if self.score_status == "confirmed":
            return self.recorded_score
        return None


@dataclass(frozen=True)
class ScoredVideo:
    release: V4Release
    projects_sha256: str
    match: MatchSettings
    sources: tuple[dict[str, str | float], ...]
    clips: tuple[ScoredClip, ...]
    chain_links: int
    chain_segments: int

    @property
    def starts_at_match_open(self) -> bool:
        for clip in self.clips:
            if clip.match_point_key is not None:
                score = clip.confirmed
                assert score is not None and score.before is not None
                opening = ScoreCount(
                    sets=(0, 0) if score.before.sets is not None else None
                )
                return bool(score.before == opening)
        return False

    @property
    def chain_continuous(self) -> bool:
        return (
            self.starts_at_match_open
            and self.chain_links > 0
            and self.chain_segments == 1
            and all(
                clip.score_status in ("confirmed", "not_point")
                and clip.point_identity_hold is None
                for clip in self.clips
            )
        )

    @property
    def reached_match_end(self) -> bool | None:
        counts = [
            clip.score_after for clip in self.clips if clip.score_after is not None
        ]
        return (
            any(match_over(self.match.rules, count) for count in counts)
            if counts
            else None
        )

    @property
    def match_complete(self) -> bool:
        return self.chain_continuous and self.reached_match_end is True


def _project_sources(clip: V4Clip, sources: tuple[dict[str, str | float], ...]) -> None:
    by_camera = {source["camera_id"]: source for source in sources}
    if by_camera.keys() != set(clip.clip.camera_ids):
        raise ValueError(f"{clip.clip_id}: camera IDs disagree with projects.json")
    for camera in clip.clip.cameras:
        saved = by_camera[camera["camera_id"]]
        relative = Path(str(saved["path"])).parts
        raw_path, offset = camera.get("source_path"), camera.get("offset_sec")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or Path(raw_path).parts[-len(relative) :] != relative
            or (type(offset) is not int and type(offset) is not float)
            or not isclose(
                offset,
                float(saved["offset_sec"]),
                rel_tol=0,
                abs_tol=TIME_TOLERANCE_SEC,
            )
        ):
            raise ValueError(
                f"{clip.clip_id}: source path or offset disagrees for {camera['camera_id']}"
            )


def _time_errors(clip: V4Clip, project_clip: Clip) -> tuple[float, float]:
    start = abs(clip.global_start_sec - project_clip.start_sec)
    end = abs(clip.global_end_sec - project_clip.end_sec)
    if max(start, end) > TIME_TOLERANCE_SEC:
        raise ValueError(
            f"{clip.clip_id}: source time mismatch with projects.json "
            f"(start error {start:.12g} s, end error {end:.12g} s)"
        )
    return start, end


def _overlapping_events(
    clip: V4Clip, start: float, end: float
) -> list[tuple[float, tuple[str, str | None, int | None]]]:
    return [
        (time, (event.kind, event.subtype, event.court_side))
        for event in clip.events
        if event.in_point
        and event.status == "certain"
        and event.kind != "unknown"
        and start <= (time := clip.global_start_sec + event.frame / clip.clip.fps) < end
    ]


def _same_source_point(left: V4Clip, right: V4Clip, start: float, end: float) -> bool:
    if (left.global_start_sec, left.global_end_sec) == (
        right.global_start_sec,
        right.global_end_sec,
    ):
        return True
    # Both samples can be rounded by half a frame during clip export.
    tolerance = 0.5 / left.clip.fps + 0.5 / right.clip.fps
    left_events = _overlapping_events(left, start, end)
    right_events = _overlapping_events(right, start, end)
    return any(
        left_label == right_label and abs(left_time - right_time) <= tolerance
        for left_time, left_label in left_events
        for right_time, right_label in right_events
    )


def _canonical_clips(
    timeline: list[Clip], v4_clips: dict[str, V4Clip]
) -> tuple[list[Clip], dict[str, str], dict[str, PointIdentityHold]]:
    representative: dict[str, str] = {}
    held: dict[str, PointIdentityHold] = {}
    confirmed: list[Clip] = []
    by_name = {clip.name: clip for clip in timeline}
    unique: list[Clip] = []
    for clip in timeline:
        score = clip.score
        if score is None or score.status != "confirmed":
            unique.append(clip)
            continue
        v4 = v4_clips[clip.name]
        matches: set[str] = set()
        same_before: set[str] = set()
        for previous in confirmed:
            source = v4_clips[previous.name]
            start, end = (
                max(source.global_start_sec, v4.global_start_sec),
                min(source.global_end_sec, v4.global_end_sec),
            )
            if start >= end:
                continue
            canonical = representative[previous.name]
            if _same_source_point(source, v4, start, end):
                matches.add(canonical)
            elif previous.score is not None and previous.score.before == score.before:
                same_before.add(canonical)

        if len(matches) == 1:
            original = by_name[next(iter(matches))]
            assert original.score is not None
            if (score.before, score.server, score.winner) != (
                original.score.before,
                original.score.server,
                original.score.winner,
            ):
                raise ValueError(
                    f"{v4.clip_id}: duplicate point has conflicting confirmed scores"
                )
            representative[clip.name] = original.name
        elif matches or same_before:
            held[clip.name] = PointIdentityHold(
                "multiple_source_point_matches"
                if matches
                else "overlapping_same_score_without_shared_event",
                tuple(
                    sorted(v4_clips[name].clip_id for name in (matches or same_before))
                ),
            )
            continue
        else:
            unique.append(clip)
            representative[clip.name] = clip.name
        confirmed.append(clip)
    return unique, representative, held


def _validate_chain(
    project_clips: list[Clip], match: MatchSettings, views: dict[str, dict[str, Any]]
) -> tuple[dict[str, ScoreCount], int, int]:
    after: dict[str, ScoreCount] = {}
    match_ended = False
    links = segments = 0
    for clip in project_clips:
        score = clip.score
        if score is None or score.status != "confirmed":
            continue
        if match_ended:
            raise ValueError(f"{clip.name}: confirmed point after match completion")
        view = views[clip.name]
        if (
            view["chain_broken"]
            or view["discontinuous"]
            or view["expected_server"] is not None
        ):
            raise ValueError(
                f"{clip.name}: score chain or server order disagrees with the previous point"
            )
        assert score.before is not None and score.winner is not None
        current = advance_point(match.rules, score.before, score.winner)
        after[clip.name] = current
        match_ended = match_over(match.rules, current)
        if view["chain_available"]:
            links += 1
        else:
            segments += 1
    return after, links, segments


def read_scores(
    release: V4Release,
    projects_path: Path,
    resolver: PathResolver,
    *,
    projects_sha256: str | None = None,
) -> ScoredVideo:
    path = resolver.validate(PathRole.DATA, projects_path)
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if projects_sha256 is not None and digest != sha256_value(
        projects_sha256, "projects_sha256"
    ):
        raise ValueError("projects.json SHA-256 mismatch")
    collection = ClipStudioProjects.from_dict(json_object(raw, str(path)), resolver)
    if (
        collection.dataset_id != release.dataset_id
        or release.video_id not in collection.projects
    ):
        raise ValueError(
            "projects.json does not contain the frozen benchmark's dataset/video"
        )
    project = collection.projects[release.video_id]
    match = project.match
    if match is None:
        raise ValueError(f"{release.video_id}: projects.json is missing match settings")
    timeline = sorted(
        project.clips, key=lambda clip: (clip.start_sec, clip.end_sec, clip.name)
    )
    by_name = {clip.clip.clip_name: clip for clip in release.clips}
    clip_names = {clip.name for clip in timeline}
    if clip_names != by_name.keys():
        raise ValueError(
            "projects.json clip IDs differ from V4 "
            f"(missing={sorted(by_name.keys() - clip_names)}, extra={sorted(clip_names - by_name.keys())})"
        )

    sources = tuple(source.to_dict(resolver) for source in project.sources)
    time_errors = {}
    for clip in timeline:
        v4 = by_name[clip.name]
        time_errors[clip.name] = _time_errors(v4, clip)
        _project_sources(v4, sources)

    unique, representative, held = _canonical_clips(timeline, by_name)
    views = score_views(replace(project, clips=unique))
    after, links, segments = _validate_chain(unique, match, views)
    scored = []
    for clip in timeline:
        original = representative.get(clip.name)
        hold = held.get(clip.name)
        match_key = None
        score_after = None
        if original is not None:
            v4 = by_name[original]
            match_key = f"point@{v4.global_start_sec}-{v4.global_end_sec}"
            score_after = after[original]
        elif hold is not None:
            score = clip.score
            assert (
                score is not None
                and score.before is not None
                and score.winner is not None
            )
            score_after = advance_point(match.rules, score.before, score.winner)
        scored.append(
            ScoredClip(
                by_name[clip.name],
                clip.score,
                score_after,
                match_key,
                *time_errors[clip.name],
                point_identity_hold=hold,
            )
        )
    return ScoredVideo(release, digest, match, sources, tuple(scored), links, segments)
