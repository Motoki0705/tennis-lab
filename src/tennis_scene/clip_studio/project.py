"""Aggregate project model for dataset/video/clip editing.

One ``projects.json`` owns all independently synchronized videos in a dataset.
Each video project has its own camera sources, offsets, and temporal clips.

Sync convention: ``local_time = global_time + offset_sec``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.utils.configuration import PathResolver, PathRole
from src.utils.io import load_json, save_json_atomic
from src.utils.tennis_scoring import (
    CountPair,
    ScoreCount,
    ScoringRules,
    Side,
    SoftTennisRules,
    TennisRules,
    require_side,
    server_at,
    validate_before,
)

PROJECTS_SCHEMA_VERSION = 2
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CLIP_V1_KEYS = {"name", "start_sec", "end_sec"}
CLIP_V2_KEYS = CLIP_V1_KEYS | {"score"}
PROJECT_V1_KEYS = {"sources", "clips"}
PROJECT_V2_KEYS = PROJECT_V1_KEYS | {"match"}

ScoreStatus = Literal["auto", "confirmed", "unknown", "not_point", "multiple_points"]
BeforeSource = Literal["chain", "anchor"]
ServerSource = Literal["auto", "manual"]


def _require_exact_keys(data: dict[str, Any], expected: set[str], *, name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(data)
    if actual != expected:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _validate_identifier(value: str, *, field_name: str) -> str | None:
    if not value:
        return f"{field_name} must be non-empty"
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        return (
            f"{field_name} must start with an ASCII letter or digit and contain "
            f"only letters, digits, '.', '_' or '-', got {value!r}"
        )
    return None


@dataclass(frozen=True)
class MatchSettings:
    """One video's scoring rules and the two fixed sides."""

    side_a: str
    side_b: str
    rules: ScoringRules
    first_server: Side

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str) or not name.strip()
            for name in (self.side_a, self.side_b)
        ):
            raise ValueError("選手・ペア A / B の表示名を入力してください。")
        require_side(self.first_server, "最初のサーブ側")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchSettings:
        _require_exact_keys(data, {"sides", "rules", "first_server"}, name="match")
        sides = data["sides"]
        _require_exact_keys(sides, {"A", "B"}, name="match.sides")
        raw = data["rules"]
        if not isinstance(raw, dict):
            raise ValueError("match.rules must be an object")
        rules: ScoringRules
        if raw.get("type") == "soft_tennis":
            _require_exact_keys(raw, {"type", "match_games"}, name="match.rules")
            rules = SoftTennisRules(raw["match_games"])
        elif raw.get("type") == "tennis":
            _require_exact_keys(
                raw, {"type", "best_of", "no_ad", "final_set"}, name="match.rules"
            )
            rules = TennisRules(raw["best_of"], raw["no_ad"], raw["final_set"])
        else:
            raise ValueError(f"未対応の採点方式です: {raw.get('type')!r}")
        return cls(sides["A"], sides["B"], rules, data["first_server"])

    def to_dict(self) -> dict[str, Any]:
        rules = self.rules
        raw_rules = (
            {"type": "soft_tennis", "match_games": rules.match_games}
            if isinstance(rules, SoftTennisRules)
            else {
                "type": "tennis",
                "best_of": rules.best_of,
                "no_ad": rules.no_ad,
                "final_set": rules.final_set,
            }
        )
        return {
            "sides": {"A": self.side_a, "B": self.side_b},
            "rules": raw_rules,
            "first_server": self.first_server,
        }


def count_from_dict(data: object) -> ScoreCount:
    if not isinstance(data, dict):
        raise ValueError("開始前カウントを指定してください。")
    keys = {"games", "points"}
    if "sets" in data:
        keys.add("sets")
    _require_exact_keys(data, keys, name="score.before")

    def pair(name: str) -> CountPair:
        raw = data[name]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{name} は A / B の 2 個の整数を指定してください。")
        return raw[0], raw[1]

    return ScoreCount(
        pair("games"), pair("points"), pair("sets") if "sets" in data else None
    )


def count_to_dict(count: ScoreCount) -> dict[str, list[int]]:
    data = {"games": list(count.games), "points": list(count.points)}
    if count.sets is not None:
        data["sets"] = list(count.sets)
    return data


@dataclass(frozen=True)
class ClipScore:
    """A saved annotation, with the before-count absent for non-point clips."""

    status: ScoreStatus
    before: ScoreCount | None
    before_source: BeforeSource
    server: Side
    server_source: ServerSource
    winner: Side | None
    note: str

    def __post_init__(self) -> None:
        if self.status not in (
            "auto",
            "confirmed",
            "unknown",
            "not_point",
            "multiple_points",
        ):
            raise ValueError(f"未対応のカウント状態です: {self.status!r}")
        if (self.status == "not_point") != (self.before is None):
            raise ValueError("ポイント外のクリップ以外は開始前カウントが必須です。")
        if self.before_source not in ("chain", "anchor"):
            raise ValueError(
                "before_source は chain または anchor を指定してください。"
            )
        if self.server_source not in ("auto", "manual"):
            raise ValueError("server_source は auto または manual を指定してください。")
        require_side(self.server, "サーブ側")
        if self.status == "confirmed":
            require_side(self.winner, "得点側")
        elif self.winner is not None:
            raise ValueError("得点側を指定できるのは確定のクリップだけです。")
        if not isinstance(self.note, str):
            raise ValueError("理由・メモは文字列で入力してください。")
        if (
            self.status in ("unknown", "not_point", "multiple_points")
            and not self.note.strip()
        ):
            raise ValueError("この状態には理由を入力してください。")

    def validate(self, match: MatchSettings | None) -> None:
        if match is None:
            raise ValueError("カウントを入力するには試合設定が必要です。")
        if self.before is not None:
            validate_before(match.rules, self.before)
            if (
                self.server_source == "auto"
                and self.before.sets in (None, (0, 0))
                and self.server
                != server_at(match.rules, self.before, match.first_server)
            ):
                raise ValueError("自動サーブ側が採点ルールと一致しません。")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipScore:
        keys = {"status", "before_source", "server", "server_source", "winner", "note"}
        if isinstance(data, dict) and data.get("status") != "not_point":
            keys.add("before")
        _require_exact_keys(data, keys, name="clip.score")
        return cls(
            status=data["status"],
            before=count_from_dict(data["before"]) if "before" in data else None,
            before_source=data["before_source"],
            server=data["server"],
            server_source=data["server_source"],
            winner=data["winner"],
            note=data["note"],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "before_source": self.before_source,
            "server": self.server,
            "server_source": self.server_source,
            "winner": self.winner,
            "note": self.note,
        }
        if self.before is not None:
            data["before"] = count_to_dict(self.before)
        return data


@dataclass
class ClipSource:
    """One camera source and its offset on the video's shared timeline."""

    path: Path
    camera_id: str
    offset_sec: float = 0.0

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        return {
            "path": resolver.relative(PathRole.DATA, self.path).as_posix(),
            "camera_id": self.camera_id,
            "offset_sec": float(self.offset_sec),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], resolver: PathResolver) -> ClipSource:
        _require_exact_keys(
            data, {"path", "camera_id", "offset_sec"}, name="clip source"
        )
        raw_path = data["path"]
        if type(raw_path) is not str or not raw_path:
            raise ValueError(
                "clip source path must be a non-empty role-relative string"
            )
        return cls(
            path=resolver.resolve(PathRole.DATA, raw_path),
            camera_id=str(data["camera_id"]),
            offset_sec=float(data["offset_sec"]),
        )


@dataclass
class Clip:
    """A half-open segment ``[start_sec, end_sec)`` in one video."""

    name: str
    start_sec: float
    end_sec: float
    score: ClipScore | None = None

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def contains(self, global_sec: float) -> bool:
        return self.start_sec <= global_sec < self.end_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "score": self.score.to_dict() if self.score is not None else None,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, version: int = PROJECTS_SCHEMA_VERSION
    ) -> Clip:
        _require_exact_keys(
            data, CLIP_V1_KEYS if version == 1 else CLIP_V2_KEYS, name="clip"
        )
        raw_score = data.get("score")
        try:
            score = ClipScore.from_dict(raw_score) if raw_score is not None else None
        except ValueError as error:
            raise ValueError(f"clip '{data['name']}': {error}") from error
        return cls(
            name=str(data["name"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            score=score,
        )


@dataclass
class ClipStudioProject:
    """Editable state for one raw multi-camera video."""

    dataset_id: str = ""
    video_id: str = ""
    sources: list[ClipSource] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    match: MatchSettings | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        for value, name in (
            (self.dataset_id, "dataset_id"),
            (self.video_id, "video_id"),
        ):
            error = _validate_identifier(value, field_name=name)
            if error is not None:
                errors.append(error)
        if not self.sources:
            errors.append("project must contain at least one source")
        camera_ids = [source.camera_id for source in self.sources]
        if len(set(camera_ids)) != len(camera_ids):
            errors.append(f"camera_ids must be unique, got {camera_ids}")
        for source in self.sources:
            if not source.camera_id:
                errors.append(f"camera_id must be non-empty for source {source.path}")
            if not math.isfinite(source.offset_sec):
                errors.append(
                    f"offset_sec must be finite for {source.camera_id}, got {source.offset_sec}"
                )
        clip_names = [clip.name for clip in self.clips]
        if len(set(clip_names)) != len(clip_names):
            errors.append(f"clip names must be unique, got {clip_names}")
        for clip in self.clips:
            clip_error = _validate_identifier(clip.name, field_name="clip name")
            if clip_error is not None:
                errors.append(clip_error)
            if not (math.isfinite(clip.start_sec) and math.isfinite(clip.end_sec)):
                errors.append(f"clip '{clip.name}' has non-finite bounds")
            elif clip.end_sec <= clip.start_sec:
                errors.append(
                    f"clip '{clip.name}' must have end_sec > start_sec, "
                    f"got [{clip.start_sec}, {clip.end_sec})"
                )
            if clip.score is not None:
                try:
                    clip.score.validate(self.match)
                except ValueError as error:
                    errors.append(f"clip '{clip.name}': {error}")
        return errors

    def next_clip_name(self) -> str:
        used = {clip.name for clip in self.clips}
        index = 0
        while f"clip_{index:03d}" in used:
            index += 1
        return f"clip_{index:03d}"

    def clip_index_by_name(self, name: str) -> int:
        for index, clip in enumerate(self.clips):
            if clip.name == name:
                return index
        raise KeyError(f"clip '{name}' not found")

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        return {
            "sources": [source.to_dict(resolver) for source in self.sources],
            "clips": [clip.to_dict() for clip in self.clips],
            "match": self.match.to_dict() if self.match is not None else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        resolver: PathResolver,
        *,
        dataset_id: str,
        video_id: str,
        version: int = PROJECTS_SCHEMA_VERSION,
    ) -> ClipStudioProject:
        _require_exact_keys(
            data,
            PROJECT_V1_KEYS if version == 1 else PROJECT_V2_KEYS,
            name=f"project {video_id}",
        )
        raw_match = data.get("match")
        project = cls(
            dataset_id=dataset_id,
            video_id=video_id,
            sources=[ClipSource.from_dict(item, resolver) for item in data["sources"]],
            clips=[Clip.from_dict(item, version=version) for item in data["clips"]],
            match=MatchSettings.from_dict(raw_match) if raw_match is not None else None,
        )
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project {video_id}: {errors}")
        return project

    def save(self, path: Path, resolver: PathResolver) -> Path:
        return ClipStudioProjects.save_project(self, path, resolver)

    @classmethod
    def load(
        cls,
        path: Path,
        resolver: PathResolver,
        *,
        dataset_id: str,
        video_id: str,
    ) -> ClipStudioProject:
        projects = ClipStudioProjects.load(path, resolver)
        if projects.dataset_id != dataset_id:
            raise ValueError(
                f"projects dataset_id {projects.dataset_id!r} != {dataset_id!r}"
            )
        try:
            return projects.projects[video_id]
        except KeyError:
            raise KeyError(f"video project {video_id!r} not found in {path}") from None


@dataclass
class ClipStudioProjects:
    """All independently editable video projects in one dataset."""

    dataset_id: str
    projects: dict[str, ClipStudioProject] = field(default_factory=dict)

    def to_dict(self, resolver: PathResolver) -> dict[str, Any]:
        return {
            "version": PROJECTS_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "projects": {
                video_id: self.projects[video_id].to_dict(resolver)
                for video_id in sorted(self.projects)
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], resolver: PathResolver
    ) -> ClipStudioProjects:
        _require_exact_keys(
            data, {"version", "dataset_id", "projects"}, name="projects"
        )
        version = data["version"]
        if type(version) is not int or version not in (1, PROJECTS_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported projects version {version!r}; "
                f"expected 1 or {PROJECTS_SCHEMA_VERSION}"
            )
        dataset_id = str(data["dataset_id"])
        dataset_error = _validate_identifier(dataset_id, field_name="dataset_id")
        if dataset_error is not None:
            raise ValueError(dataset_error)
        raw_projects = data["projects"]
        if not isinstance(raw_projects, dict):
            raise ValueError("projects must be an object keyed by video_id")
        projects = {
            str(video_id): ClipStudioProject.from_dict(
                value,
                resolver,
                dataset_id=dataset_id,
                video_id=str(video_id),
                version=version,
            )
            for video_id, value in raw_projects.items()
        }
        return cls(dataset_id=dataset_id, projects=projects)

    @classmethod
    def load(cls, path: Path, resolver: PathResolver) -> ClipStudioProjects:
        projects_path = resolver.validate(PathRole.DATA, path)
        payload = load_json(projects_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{projects_path} must contain a JSON object")
        return cls.from_dict(payload, resolver)

    @classmethod
    def save_project(
        cls,
        project: ClipStudioProject,
        path: Path,
        resolver: PathResolver,
    ) -> Path:
        projects_path = resolver.validate(PathRole.DATA, path)
        errors = project.validate()
        if errors:
            raise ValueError(f"Invalid project: {errors}")
        if projects_path.exists():
            collection = cls.load(projects_path, resolver)
            if collection.dataset_id != project.dataset_id:
                raise ValueError(
                    f"projects dataset_id {collection.dataset_id!r} != "
                    f"project dataset_id {project.dataset_id!r}"
                )
        else:
            collection = cls(dataset_id=project.dataset_id)
        collection.projects[project.video_id] = project
        saved_path: Path = save_json_atomic(collection.to_dict(resolver), projects_path)
        return saved_path


__all__ = [
    "Clip",
    "ClipScore",
    "ClipSource",
    "ClipStudioProject",
    "ClipStudioProjects",
    "MatchSettings",
    "PROJECTS_SCHEMA_VERSION",
]
