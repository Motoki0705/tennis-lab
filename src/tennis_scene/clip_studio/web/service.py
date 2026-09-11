"""Serialized, revision-checked editing with atomic autosave and undo history."""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.tennis_scene.clip_studio.state import ClipStudioState
from src.utils.configuration import PathResolver
from src.utils.video import VideoInfo


class Edit(BaseModel):
    """One user operation against a specific saved revision."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    revision: int = Field(ge=0)
    action: Literal["create", "update", "delete", "offsets", "undo", "redo"]
    name: str | None = None
    new_name: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    offsets_sec: list[float] | None = None


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
    ) -> None:
        ClipStudioState(project, infos)
        self.project = deepcopy(project)
        self.infos = infos
        self.path = path
        self.resolver = resolver
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
                "dataset_id": self.project.dataset_id,
                "video_id": self.project.video_id,
                "projects_path": str(self.path),
                "sources": [
                    {
                        "camera_id": source.camera_id,
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
                "extent": state.extent_sec(),
                "common": (
                    max(view.coverage_sec[0] for view in state.source_views()),
                    min(view.coverage_sec[1] for view in state.source_views()),
                ),
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
            }

    def edit(self, request: Edit) -> dict[str, Any]:
        with self.lock:
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
            else:
                if request.start_sec is None or request.end_sec is None:
                    raise ValueError("開始・終了時刻を指定してください。")
                clip = Clip(
                    name=request.new_name
                    or (
                        candidate.next_clip_name()
                        if request.action == "create"
                        else request.name or ""
                    ),
                    start_sec=request.start_sec,
                    end_sec=request.end_sec,
                )
                if request.action == "create":
                    candidate.clips.append(clip)
                else:
                    candidate.clips[
                        candidate.clip_index_by_name(request.name or "")
                    ] = clip
            errors = candidate.validate()
            if errors:
                raise ValueError("; ".join(errors))
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
