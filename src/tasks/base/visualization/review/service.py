"""Task-agnostic scene service shared by the BLCS and PLCS review apps.

Subclasses supply the entity schema, the task-owned CourtKP20 validation hook,
and the fps/court-offset accessors. The base class owns path resolution,
revision checks, court/camera serialization, and the entity binary shape so the
two apps cannot drift.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import (
    PHYSICAL_COURT_TARGET_FRAME_ID,
    CourtKeypointContract,
)
from src.tasks.base.visualization.review.camera import parse_cameras
from src.tasks.base.visualization.review.catalog import DatasetCatalog
from src.tasks.base.visualization.review.court import (
    apron_polygon,
    court_edges,
    court_keypoints,
    net_geometry,
)
from src.tasks.base.visualization.review.payload import ScenePayload, pack_entity_frames
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)

DEFAULT_CAMERA_DEPTH_M = 4.0
UNITS = "m"
# Every supported form stores 3-D data and cameras in one physical court frame.
# In particular ``camera_view_v2`` only permutes per-camera CourtKP20 *labels*
# (``semantic_to_physical`` / ``canonical_from_physical``); the geometry is
# unchanged, so ``court_keypoints_3d`` is drawn as-is for all six forms.
COORDINATE_FRAME = PHYSICAL_COURT_TARGET_FRAME_ID


@dataclass(frozen=True, slots=True)
class SceneArrays:
    """Entity data reshaped to ``(slots, frames, joint_count, 3)`` plus extras."""

    frames: NDArray[np.float32]
    presence: NDArray[np.bool_] | None
    orientation: NDArray[np.float32] | None

    @property
    def slots(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[1])

    @property
    def joint_count(self) -> int:
        return int(self.frames.shape[2])


def load_json_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return document


def _npy_shape(path: Path) -> tuple[int, ...]:
    # memmap reads only the array header, so this stays cheap for big arrays.
    array = np.load(path, mmap_mode="r")
    return tuple(int(dimension) for dimension in array.shape)


def _required_key(document: dict[str, Any], key: str, *, path: Path) -> Any:
    if key not in document:
        raise ValueError(f"{path}: required key {key!r} is missing.")
    return document[key]


class DatasetSceneReviewService:
    """Read-only scene catalog and payload assembly for one task."""

    task: str = ""
    entity_kind: str = ""

    def __init__(
        self,
        data_root: str | Path,
        *,
        forms: Sequence[str] | None = None,
        camera_depth: float = DEFAULT_CAMERA_DEPTH_M,
    ) -> None:
        if not self.task or not self.entity_kind:
            raise TypeError("Subclasses must set task and entity_kind.")
        self._catalog = DatasetCatalog(data_root, self.task, forms=forms)
        self.camera_depth = float(camera_depth)

    @property
    def root(self) -> Path:
        return self._catalog.root

    # --------------------------------------------------------------- catalog

    def catalog(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "root": str(self.root),
            "forms": [form.to_dict() for form in self._catalog.forms()],
            "entity": self.entity_kind,
            "skeleton": self._catalog_skeleton(),
        }

    def scenes(self, form: str) -> dict[str, Any]:
        return {"form": form, "scenes": list(self._catalog.scenes(form))}

    def scene(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> dict[str, Any]:
        scene_path, contract = self._resolve(form, scene_id)
        current = self._catalog.revision(form, scene_id)
        if revision is not None and revision != current:
            raise RuntimeError("Scene changed on disk. Reload the catalog.")
        self._validate_scene_contract(scene_path, contract)
        meta = load_json_object(scene_path / "meta.json")
        validate_court_coordinate_normalization(meta, artifact=f"Scene {scene_path}")
        scalars = load_json_object(scene_path / "scalars.json")

        frame_count = self._frame_count(meta, scene_path)
        slots = self._entity_shape(scene_path)[0]
        if self._catalog.revision(form, scene_id) != current:
            raise RuntimeError("Scene changed on disk. Reload the catalog.")

        cameras = parse_cameras(scalars)
        offset = self._court_net_post_offset(meta)
        return {
            "task": self.task,
            "form": form,
            "dataset": f"{self.task}/{form}",
            "mode": "multi" if slots > 1 else "single",
            "scene_id": scene_id,
            "revision": current,
            "fps": self._fps(meta, scene_path),
            "frame_count": frame_count,
            "units": UNITS,
            "coordinate_frame": COORDINATE_FRAME,
            "court": self._court_document(offset),
            "cameras": [
                camera.to_dict(depth=self.camera_depth) for camera in cameras
            ],
            "entity": {
                "kind": self.entity_kind,
                "slots": slots,
                "joint_count": self._entity_joint_count(),
                "colors": list(self._entity_colors()),
                "joint_names": self._entity_joint_names(),
                "skeleton": self._entity_skeleton(),
                "frames": frame_count,
                "presence": slots > 1,
                "orientation": self._entity_has_orientation(),
            },
        }

    def buffer(self, form: str, scene_id: str, revision: str | None = None) -> bytes:
        scene_path, contract = self._resolve(form, scene_id)
        current = self._catalog.revision(form, scene_id)
        if revision is not None and revision != current:
            raise RuntimeError("Scene changed on disk. Reload the catalog.")
        self._validate_scene_contract(scene_path, contract)
        meta = load_json_object(scene_path / "meta.json")
        validate_court_coordinate_normalization(meta, artifact=f"Scene {scene_path}")
        frame_count = self._frame_count(meta, scene_path)
        arrays = self._load_entity(scene_path, frame_count=frame_count)
        data = pack_entity_frames(
            arrays.frames,
            presence=arrays.presence,
            orientation=arrays.orientation,
        )
        if self._catalog.revision(form, scene_id) != current:
            raise RuntimeError("Scene changed on disk. Reload the catalog.")
        return data

    def payload(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> ScenePayload:
        """Return the JSON document and entity binary for one scene together."""
        document = self.scene(form, scene_id, revision)
        return ScenePayload(
            document=document,
            binary=self.buffer(form, scene_id, document["revision"]),
        )

    # ------------------------------------------------------------------ hooks

    def _catalog_skeleton(self) -> dict[str, Any] | None:
        names = self._entity_joint_names()
        skeleton = self._entity_skeleton()
        if names is None or skeleton is None:
            return None
        return {"names": names, "edges": skeleton}

    def _resolve(
        self, form: str, scene_id: str
    ) -> tuple[Path, CourtKeypointContract]:
        contract = self._catalog.court_contract(form)
        return self._catalog.scene_path(form, scene_id), contract

    def _validate_scene_contract(
        self, scene_path: Path, contract: CourtKeypointContract
    ) -> None:
        raise NotImplementedError

    def _entity_joint_count(self) -> int:
        raise NotImplementedError

    def _entity_joint_names(self) -> list[str] | None:
        raise NotImplementedError

    def _entity_skeleton(self) -> list[list[int]] | None:
        raise NotImplementedError

    def _entity_colors(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _entity_has_orientation(self) -> bool:
        return False

    def _entity_frames_file(self) -> str:
        raise NotImplementedError

    def _presence_file(self) -> str:
        raise NotImplementedError

    def _orientation_file(self) -> str:
        raise NotImplementedError

    def _court_net_post_offset(self, meta: dict[str, Any]) -> float | None:
        raise NotImplementedError

    def _fps(self, meta: dict[str, Any], scene_path: Path) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------ shared code

    def _frame_count(self, meta: dict[str, Any], scene_path: Path) -> int:
        value = _required_key(meta, "num_frames", path=scene_path / "meta.json")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{scene_path / 'meta.json'}: num_frames must be a positive int."
            )
        return value

    def _court_document(self, net_post_offset_x: float | None) -> dict[str, Any]:
        keypoints = court_keypoints(net_post_offset_x)
        net = net_geometry(net_post_offset_x)
        return {
            "keypoints": keypoints.tolist(),
            "edges": [list(edge) for edge in court_edges()],
            "apron": apron_polygon().tolist(),
            "net_top": net["top"],
            "net_posts": net["posts"],
        }

    def _entity_shape(self, scene_path: Path) -> tuple[int, int]:
        """Return ``(slots, joint_count)`` from the entity array header."""
        joint_count = self._entity_joint_count()
        shape = _npy_shape(scene_path / self._entity_frames_file())
        if not shape or shape[-1] != 3:
            raise ValueError(
                f"{self._entity_frames_file()}: last axis must be 3, got {shape}."
            )
        body = shape[:-1]
        if body and body[-1] == joint_count:
            body = body[:-1]
        if len(body) == 1:
            slots = 1
        elif len(body) == 2:
            slots = int(body[1])
        else:
            raise ValueError(
                f"{self._entity_frames_file()}: unsupported shape {shape} for "
                f"joint_count={joint_count}."
            )
        if slots <= 0:
            raise ValueError(
                f"{self._entity_frames_file()}: slot count must be positive."
            )
        return slots, joint_count

    def _load_entity(self, scene_path: Path, *, frame_count: int) -> SceneArrays:
        slots, joint_count = self._entity_shape(scene_path)
        path = scene_path / self._entity_frames_file()
        raw = np.load(path)
        if raw.shape[0] != frame_count:
            raise ValueError(
                f"{path}: first axis {raw.shape[0]} must equal num_frames "
                f"{frame_count}."
            )
        frames = self._reshape_frames(raw, slots=slots, joint_count=joint_count)
        presence = None
        if slots > 1:
            presence = self._load_presence(scene_path, slots=slots, frames=frame_count)
            frames = frames.copy()
            frames[~presence] = 0.0
        orientation = None
        if self._entity_has_orientation():
            orientation = self._load_orientation(
                scene_path, slots=slots, frames=frame_count
            )
        return SceneArrays(frames=frames, presence=presence, orientation=orientation)

    def _reshape_frames(
        self, raw: NDArray[np.generic], *, slots: int, joint_count: int
    ) -> NDArray[np.float32]:
        array = np.asarray(raw, dtype=np.float32)
        frames = int(array.shape[0])
        expected: tuple[int, ...]
        if slots == 1:
            expected = (frames, 3) if joint_count == 1 else (frames, joint_count, 3)
            if array.shape != expected:
                raise ValueError(
                    f"{self._entity_frames_file()}: expected {expected}, "
                    f"got {array.shape}."
                )
            return array.reshape(1, frames, joint_count, 3)
        if joint_count == 1:
            expected = (frames, slots, 3)
            if array.shape != expected:
                raise ValueError(
                    f"{self._entity_frames_file()}: expected {expected}, "
                    f"got {array.shape}."
                )
            return np.transpose(array, (1, 0, 2)).reshape(slots, frames, 1, 3)
        expected = (frames, slots, joint_count, 3)
        if array.shape != expected:
            raise ValueError(
                f"{self._entity_frames_file()}: expected {expected}, "
                f"got {array.shape}."
            )
        return np.ascontiguousarray(np.transpose(array, (1, 0, 2, 3)))

    def _load_presence(
        self, scene_path: Path, *, slots: int, frames: int
    ) -> NDArray[np.bool_]:
        path = scene_path / self._presence_file()
        raw = np.load(path)
        if raw.shape != (frames, slots):
            raise ValueError(
                f"{path}: expected shape ({frames}, {slots}), got {raw.shape}."
            )
        return np.ascontiguousarray(np.transpose(raw.astype(bool), (1, 0)))

    def _load_orientation(
        self, scene_path: Path, *, slots: int, frames: int
    ) -> NDArray[np.float32]:
        path = scene_path / self._orientation_file()
        raw = np.asarray(np.load(path), dtype=np.float32)
        if slots == 1:
            if raw.shape != (frames, 2):
                raise ValueError(
                    f"{path}: expected shape ({frames}, 2), got {raw.shape}."
                )
            return raw.reshape(1, frames, 2)
        if raw.shape != (frames, slots, 2):
            raise ValueError(
                f"{path}: expected shape ({frames}, {slots}, 2), got {raw.shape}."
            )
        return np.ascontiguousarray(np.transpose(raw, (1, 0, 2)))


__all__ = [
    "COORDINATE_FRAME",
    "DEFAULT_CAMERA_DEPTH_M",
    "DatasetSceneReviewService",
    "SceneArrays",
    "load_json_object",
]
