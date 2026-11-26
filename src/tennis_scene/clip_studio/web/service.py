"""Serialized, revision-checked editing with atomic autosave and undo history."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject, MatchSettings
from src.tennis_scene.clip_studio.score_chain import (
    score_views,
    set_clip_score,
    set_match,
)
from src.tennis_scene.clip_studio.score_csv import import_scores
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.configuration import PathResolver
from src.utils.video import VideoInfo


class Edit(BaseModel):
    """One user operation against a specific saved revision."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    revision: int = Field(ge=0)
    action: Literal[
        "create",
        "update",
        "delete",
        "offsets",
        "undo",
        "redo",
        "set_score",
        "set_match",
        "import_scores",
    ]
    name: str | None = None
    new_name: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    offsets_sec: list[float] | None = None
    score: dict[str, Any] | None = None
    match: dict[str, Any] | None = None
    csv_text: str | None = None


class RevisionConflict(ValueError):
    """A stale browser must reload before changing the shared project."""


class Editor:
    """Save before publishing changes; failed writes never change live state."""

    HISTORY_LIMIT = 100

    def __init__(
        self,
        project: ClipStudioProject,
        infos: list[VideoInfo],
        path: Path,
        resolver: PathResolver,
        *,
        read_only: bool = False,
    ) -> None:
        ClipStudioState(project, infos)
        self.project = deepcopy(project)
        self.infos = infos
        self.path = path
        self.resolver = resolver
        self.read_only = read_only
        self.revision = 0
        self.lock = threading.RLock()
        self._undo: list[ClipStudioProject] = []
        self._redo: list[ClipStudioProject] = []

    def check_revision(self, revision: int) -> None:
        if revision != self.revision:
            raise RevisionConflict(
                "別の操作で更新されています。再読込してからやり直してください。"
            )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = ClipStudioState(self.project, self.infos)
            return {
                "revision": self.revision,
                "read_only": self.read_only,
                "dataset_id": self.project.dataset_id,
                "video_id": self.project.video_id,
                "projects_path": str(self.path),
                "sources": [
                    {
                        "camera_id": source.camera_id,
                        "source_path": str(source.path),
                        "offset_sec": source.offset_sec,
                        "fps": info.fps,
                        "frame_count": info.frame_count,
                        "duration_sec": info.frame_count / info.fps,
                        "width": info.width,
                        "height": info.height,
                    }
                    for source, info in zip(
                        self.project.sources, self.infos, strict=True
                    )
                ],
                "clips": [clip.to_dict() for clip in self.project.clips],
                "match": self.project.match.to_dict()
                if self.project.match is not None
                else None,
                "score_views": score_views(self.project),
                "extent": state.extent_sec(),
                "common": (
                    max(view.coverage_sec[0] for view in state.source_views()),
                    min(view.coverage_sec[1] for view in state.source_views()),
                ),
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
            }

    def _candidate(self, request: Edit) -> ClipStudioProject:
        if self.read_only:
            raise PermissionError("読取専用レビューではprojectを変更できません。")
        self.check_revision(request.revision)
        candidate = deepcopy(self.project)
        if request.action in {"undo", "redo"}:
            history = self._undo if request.action == "undo" else self._redo
            if not history:
                raise ValueError("取り消し／やり直しできる操作がありません。")
            candidate = deepcopy(history[-1])
        elif request.action == "offsets":
            offsets = request.offsets_sec
            if offsets is None or len(offsets) != len(candidate.sources):
                raise ValueError("全カメラのオフセットを指定してください。")
            for source, offset in zip(candidate.sources, offsets, strict=True):
                source.offset_sec = offset
        elif request.action == "delete":
            candidate.clips.pop(candidate.clip_index_by_name(request.name or ""))
        elif request.action == "set_score":
            if "score" not in request.model_fields_set:
                raise ValueError(
                    "score を指定してください。未入力に戻す場合は null を指定してください。"
                )
            set_clip_score(candidate, request.name or "", request.score)
        elif request.action == "set_match":
            if "match" not in request.model_fields_set:
                raise ValueError("match を指定してください。")
            set_match(
                candidate,
                MatchSettings.from_dict(request.match)
                if request.match is not None
                else None,
            )
        elif request.action == "import_scores":
            if request.csv_text is None:
                raise ValueError("CSV を指定してください。")
            import_scores(candidate, request.csv_text)
        else:
            if request.start_sec is None or request.end_sec is None:
                raise ValueError("開始・終了時刻を指定してください。")
            if request.action == "create":
                candidate.clips.append(
                    Clip(
                        request.new_name or candidate.next_clip_name(),
                        request.start_sec,
                        request.end_sec,
                    )
                )
            else:
                index = candidate.clip_index_by_name(request.name or "")
                clip = candidate.clips[index]
                candidate.clips[index] = replace(
                    clip,
                    name=request.new_name or clip.name,
                    start_sec=request.start_sec,
                    end_sec=request.end_sec,
                )
        errors = candidate.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return candidate

    def preview(self, request: Edit) -> dict[str, int]:
        with self.lock:
            if request.action not in {"set_score", "set_match"}:
                raise ValueError("カウントまたは試合設定の編集を指定してください。")
            candidate = self._candidate(request)
            return {
                "revision": self.revision,
                "changed_clips": sum(
                    old.score != new.score
                    for old, new in zip(
                        self.project.clips, candidate.clips, strict=True
                    )
                    if request.action != "set_score" or old.name != request.name
                ),
            }

    def edit(self, request: Edit) -> dict[str, Any]:
        with self.lock:
            candidate = self._candidate(request)
            candidate.save(self.path, self.resolver)
            if request.action == "undo":
                self._undo.pop()
                self._redo.append(self.project)
            elif request.action == "redo":
                self._redo.pop()
                self._undo.append(self.project)
            else:
                self._undo.append(self.project)
                self._undo = self._undo[-self.HISTORY_LIMIT :]
                self._redo.clear()
            self.project = candidate
            self.revision += 1
            return self.snapshot()
