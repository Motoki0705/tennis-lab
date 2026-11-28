"""Join the frozen timeline to the official Clip Studio point ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any

from src.tennis_scene.analytics.ingest.input_files import json_object, sha256_value
from src.tennis_scene.analytics.ingest.v4_reader import (
    SERVE_SUBTYPES,
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
class ScoredClip:
    v4: V4Clip
    recorded_score: ClipScore | None
    score_after: ScoreCount | None
    match_point_key: str | None
    start_error_sec: float
    end_error_sec: float

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
    match_complete: bool | None
    chain_links: int
    chain_segments: int


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


def _certain_serves(clip: V4Clip) -> list[tuple[float, str | None, int | None]]:
    return [
        (
            clip.global_start_sec + event.frame / clip.clip.fps,
            event.subtype,
            event.court_side,
        )
        for event in clip.events
        if event.in_point
        and event.status == "certain"
        and event.subtype in SERVE_SUBTYPES
    ]


def _same_source_point(left: V4Clip, right: V4Clip) -> bool:
    if max(left.global_start_sec, right.global_start_sec) >= min(
        left.global_end_sec, right.global_end_sec
    ):
        return False
    if (left.global_start_sec, left.global_end_sec) == (
        right.global_start_sec,
        right.global_end_sec,
    ):
        return True
    # Both samples can be rounded by half a frame during clip export.
    tolerance = 0.5 / left.clip.fps + 0.5 / right.clip.fps
    return any(
        left_kind == right_kind
        and left_side == right_side
        and abs(left_time - right_time) <= tolerance
        for left_time, left_kind, left_side in _certain_serves(left)
        for right_time, right_kind, right_side in _certain_serves(right)
    )


def _canonical_clips(
    timeline: list[Clip], v4_clips: dict[str, V4Clip]
) -> tuple[list[Clip], dict[str, str]]:
    representative: dict[str, str] = {}
    unique: list[Clip] = []
    for clip in timeline:
        score = clip.score
        if score is None or score.status != "confirmed":
            unique.append(clip)
            continue
        v4 = v4_clips[clip.name]
        matches = [
            item
            for item in unique
            if item.score is not None
            and item.score.status == "confirmed"
            and _same_source_point(v4_clips[item.name], v4)
        ]
        if len(matches) > 1:
            raise ValueError(f"{v4.clip_id}: source timing matches multiple points")
        if matches:
            original = matches[0]
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
        else:
            unique.append(clip)
            representative[clip.name] = clip.name
    return unique, representative


def _validate_chain(
    project_clips: list[Clip], match: MatchSettings, views: dict[str, dict[str, Any]]
) -> tuple[dict[str, ScoreCount], bool | None, int, int]:
    after: dict[str, ScoreCount] = {}
    complete: bool | None = None
    match_ended = False
    links = segments = 0
    for clip in project_clips:
        score = clip.score
        if score is None or score.status != "confirmed":
            if score is None or score.status != "not_point":
                complete = None
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
        match_ended = complete = match_over(match.rules, current)
        if view["chain_available"]:
            links += 1
        else:
            segments += 1
    return after, complete, links, segments


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

    unique, representative = _canonical_clips(timeline, by_name)
    views = score_views(replace(project, clips=unique))
    after, completed, links, segments = _validate_chain(unique, match, views)
    scored = []
    for clip in timeline:
        original = representative.get(clip.name)
        match_key = None
        if original is not None:
            v4 = by_name[original]
            match_key = f"point@{v4.global_start_sec}-{v4.global_end_sec}"
        scored.append(
            ScoredClip(
                by_name[clip.name],
                clip.score,
                after.get(original) if original is not None else None,
                match_key,
                *time_errors[clip.name],
            )
        )
    return ScoredVideo(
        release, digest, match, sources, tuple(scored), completed, links, segments
    )
