"""Persistent state management for the annotation backend.

State is kept under a user-provided output root (e.g., ``data/tmp``) so the
workflow can be validated end-to-end without touching other folders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tools.annotation.backend.models import (
    BallAssistState,
    BallClipConfig,
    BallFrameAnnotation,
    CourtFrameAnnotation,
    CourtKeypoint,
)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class StatePaths:
    """Filesystem locations for the tool's working state."""

    root_dir: Path

    @property
    def state_dir(self) -> Path:
        return self.root_dir / "annotation_state"

    @property
    def ball_state_path(self) -> Path:
        return self.state_dir / "ball.json"

    @property
    def court_state_path(self) -> Path:
        return self.state_dir / "court.json"

    @property
    def ball_assist_path(self) -> Path:
        return self.state_dir / "ball_assist.json"


class AnnotationState:
    """Loads/saves ball and court annotations."""

    def __init__(self, paths: StatePaths) -> None:
        self._paths = paths

    def load_ball_clip_config(self) -> BallClipConfig:
        data = _read_json(self._paths.ball_state_path)
        clip = data.get("clip") or {}
        start_frame = int(clip.get("start_frame", 0))
        clip_length = int(clip.get("clip_length", 300))
        return BallClipConfig(start_frame=start_frame, clip_length=clip_length)

    def save_ball_clip_config(self, cfg: BallClipConfig) -> None:
        data = _read_json(self._paths.ball_state_path)
        data["clip"] = cfg.model_dump()
        data.setdefault("annotations", {})
        _atomic_write_json(self._paths.ball_state_path, data)

    def load_ball_annotation(self, local_idx: int) -> BallFrameAnnotation:
        data = _read_json(self._paths.ball_state_path)
        ann = (data.get("annotations") or {}).get(str(local_idx)) or {}
        try:
            return BallFrameAnnotation.model_validate(ann)
        except Exception:
            return BallFrameAnnotation()

    def save_ball_annotation(self, local_idx: int, ann: BallFrameAnnotation) -> None:
        data = _read_json(self._paths.ball_state_path)
        data.setdefault("clip", self.load_ball_clip_config().model_dump())
        annotations = data.setdefault("annotations", {})
        annotations[str(local_idx)] = ann.model_dump()
        _atomic_write_json(self._paths.ball_state_path, data)

    def delete_ball_annotation(self, local_idx: int) -> None:
        data = _read_json(self._paths.ball_state_path)
        annotations = data.get("annotations") or {}
        if str(local_idx) in annotations:
            annotations.pop(str(local_idx), None)
            data["annotations"] = annotations
            _atomic_write_json(self._paths.ball_state_path, data)

    def has_ball_annotation(self, local_idx: int) -> bool:
        data = _read_json(self._paths.ball_state_path)
        annotations = data.get("annotations") or {}
        return str(local_idx) in annotations

    def list_ball_annotated_frames(self) -> list[int]:
        data = _read_json(self._paths.ball_state_path)
        annotations = data.get("annotations") or {}
        out: list[int] = []
        for k in annotations.keys():
            try:
                out.append(int(k))
            except ValueError:
                continue
        return sorted(out)

    def load_ball_assist_state(self) -> BallAssistState | None:
        data = _read_json(self._paths.ball_assist_path)
        if not data:
            return None
        try:
            return BallAssistState.model_validate(data)
        except Exception:
            return None

    def save_ball_assist_state(self, state: BallAssistState) -> None:
        _atomic_write_json(self._paths.ball_assist_path, state.model_dump())

    def load_ball_assist_annotation(self, local_idx: int) -> BallFrameAnnotation | None:
        data = _read_json(self._paths.ball_assist_path)
        if not data:
            return None
        ann = (data.get("annotations") or {}).get(str(local_idx))
        if not ann:
            return None
        try:
            return BallFrameAnnotation.model_validate(ann)
        except Exception:
            return None

    def list_court_annotated_frames(self) -> list[int]:
        data = _read_json(self._paths.court_state_path)
        anns = data.get("annotations") or {}
        out: list[int] = []
        for k in anns.keys():
            try:
                out.append(int(k))
            except ValueError:
                continue
        return sorted(out)

    def load_court_annotation(self, frame_idx: int, num_kp: int) -> CourtFrameAnnotation:
        data = _read_json(self._paths.court_state_path)
        ann = (data.get("annotations") or {}).get(str(frame_idx)) or {}
        if not ann:
            return CourtFrameAnnotation(
                frame_idx=frame_idx,
                keypoints=[CourtKeypoint() for _ in range(num_kp)],
            )
        try:
            parsed = CourtFrameAnnotation.model_validate(ann)
        except Exception:
            parsed = CourtFrameAnnotation(
                frame_idx=frame_idx,
                keypoints=[CourtKeypoint() for _ in range(num_kp)],
            )
        if len(parsed.keypoints) < num_kp:
            parsed.keypoints = parsed.keypoints + [CourtKeypoint()] * (
                num_kp - len(parsed.keypoints)
            )
        if len(parsed.keypoints) > num_kp:
            parsed.keypoints = parsed.keypoints[:num_kp]
        parsed.frame_idx = frame_idx
        return parsed

    def save_court_annotation(self, ann: CourtFrameAnnotation) -> None:
        data = _read_json(self._paths.court_state_path)
        annotations = data.setdefault("annotations", {})
        annotations[str(ann.frame_idx)] = ann.model_dump()
        _atomic_write_json(self._paths.court_state_path, data)

    def delete_court_annotation(self, frame_idx: int) -> None:
        data = _read_json(self._paths.court_state_path)
        annotations = data.get("annotations") or {}
        if str(frame_idx) in annotations:
            annotations.pop(str(frame_idx), None)
            data["annotations"] = annotations
            _atomic_write_json(self._paths.court_state_path, data)

