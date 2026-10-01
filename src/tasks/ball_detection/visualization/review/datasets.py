"""Read-only catalog of versioned ball frame stores and the optional Web store.

Ball versions are discovered under data/ball_detection/<version>. Each scene
is a full camera clip, including unreviewed and unresolved frames. Requests
resolve only enumerated opaque IDs, never caller-supplied filesystem paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.components.web.data_access_layer.web_store import (
    WebFrameStore,
)
from src.tasks.ball_detection.data.store import (
    METADATA_FILE,
    SHARDS_DIR,
    BallFrameStore,
    shard_name,
)
from src.tasks.ball_detection.data.types import FrameLabel
from src.tasks.ball_detection.visualization.io.store_frames import StoreSceneFrames

SceneMode = Literal["temporal", "static"]

WEB_RELATIVE: Final = "tennis/web/unified"
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")
#: Scene ids are opaque to the HTTP layer; the service splits them back into
#: ``(dataset, local)`` before resolving them through the catalog.
SCENE_SEPARATOR: Final = "::"

@dataclass(frozen=True, slots=True)
class BallDatasetSpec:
    """Static description of one browsable dataset source."""

    id: str
    label: str
    relative: str
    mode: SceneMode


DATASET_SPECS: Final[tuple[BallDatasetSpec, ...]] = (
    BallDatasetSpec("web_static", "Web frames (static)", WEB_RELATIVE, "static"),
    BallDatasetSpec(
        "web_temporal", "Web sequences (temporal)", WEB_RELATIVE, "temporal"
    ),
)


class BallDatasetCatalogError(ValueError):
    """Raised when a dataset, scene, or frame reference cannot be resolved."""


class SceneFrames(Protocol):
    """Dense frame accessor for one resolved scene."""

    @property
    def mode(self) -> SceneMode:
        """Return the dataset sampling mode this scene belongs to."""
        ...

    @property
    def frames(self) -> int:
        """Number of addressable frame positions in the scene."""
        ...

    def name(self, index: int) -> str:
        """Return the display name of one frame position."""
        ...

    def original_size(self, index: int) -> tuple[int, int]:
        """Return ``(width, height)`` of the original frame."""
        ...

    def read_rgb(self, index: int) -> NDArray[np.uint8]:
        """Return the frame as an ``(H, W, 3)`` uint8 RGB array."""
        ...

    def labels(self, index: int) -> tuple[FrameLabel, ...]:
        """Return every ball annotation for one frame position."""
        ...

    def annotated(self, index: int) -> bool:
        """Return whether the frame was reviewed."""
        ...

    def supervised(self, index: int) -> bool:
        """Return whether the observed-only target is trusted for scoring."""
        ...


def _check_index(index: object, frames: int, *, context: str) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise BallDatasetCatalogError(f"{context}: frame index must be an int.")
    if index < 0 or index >= frames:
        raise BallDatasetCatalogError(
            f"{context}: frame index {index} is out of range [0, {frames - 1}]."
        )
    return index


def _require_within(path: Path, root: Path, *, what: str) -> Path:
    """Return ``path`` resolved, refusing anything outside ``root``."""
    resolved = path.resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise BallDatasetCatalogError(
            f"{what} {path} resolves outside the dataset root {root}; refusing "
            "to read it."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class WebSceneFrames:
    """Frame accessor over ordered samples of the unified web store."""

    store: WebFrameStore
    indices: tuple[int, ...]
    mode: SceneMode

    @property
    def frames(self) -> int:
        return len(self.indices)

    def _sample(self, index: int) -> int:
        checked = _check_index(index, self.frames, context="web scene")
        return self.indices[checked]

    def name(self, index: int) -> str:
        sample = self._sample(index)
        frame_index = self.store.frame_index(sample)
        if frame_index >= 0:
            return f"{frame_index:06d}.jpg"
        return f"sample_{sample:06d}.jpg"

    def original_size(self, index: int) -> tuple[int, int]:
        width, height = self.store.original_size(self._sample(index))
        return int(width), int(height)

    def read_rgb(self, index: int) -> NDArray[np.uint8]:
        bgr = self.store.decode_bgr(self._sample(index))
        return cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def labels(self, index: int) -> tuple[FrameLabel, ...]:
        return tuple(self.store.labels(self._sample(index)))

    def annotated(self, index: int) -> bool:
        """Return whether the store recorded an explicit label state.

        The unified store only persists positive and explicitly annotated
        negative samples, so every stored sample is annotated by construction.
        """
        self._sample(index)
        return True


    def supervised(self, index: int) -> bool:
        return self.annotated(index)


@dataclass(frozen=True, slots=True)
class SceneRef:
    """One catalogued scene before its frames are materialised."""

    dataset_id: str
    local_id: str
    label: str
    frames: int
    clip_id: str | None = None
    sample_indices: tuple[int, ...] = ()

    @property
    def id(self) -> str:
        """Return the opaque scene id consumed by the HTTP layer."""
        return f"{self.dataset_id}{SCENE_SEPARATOR}{self.local_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON item for the scene-list endpoint."""
        return {"id": self.id, "label": self.label, "frames": self.frames}


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """Availability summary for one dataset source."""

    spec: BallDatasetSpec
    root: Path
    available: bool
    count: int
    max_scene_frames: int
    reason: str | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload for the catalog endpoint."""
        payload: dict[str, Any] = {
            "id": self.spec.id,
            "label": self.spec.label,
            "path": str(self.root),
            "mode": self.spec.mode,
            "available": self.available,
            "count": self.count,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def _natural_key(relative: str) -> tuple[tuple[int, object], ...]:
    """Return a sort key that orders ``Clip2`` before ``Clip10``."""
    key: list[tuple[int, object]] = []
    for part in Path(relative).parts:
        stem = part.rstrip("0123456789")
        suffix = part[len(stem) :]
        key.append((0, stem))
        if suffix:
            key.append((1, int(suffix)))
    return tuple(key)


class BallDatasetCatalog:
    """Enumerate the ball-detection datasets, scenes, and frames read-only."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser()
        self._refs: dict[str, tuple[SceneRef, ...]] = {}
        self._ref_index: dict[str, dict[str, SceneRef]] = {}
        # The availability reason is part of the cache: a dataset that cannot be
        # served must keep reporting *why* on every catalog call, and only an
        # explicit ``refresh()`` may drop it.
        self._reasons: dict[str, str] = {}
        self._warnings: dict[str, tuple[str, ...]] = {}
        self._ball_stores: dict[str, BallFrameStore] = {}
        self._specs: tuple[BallDatasetSpec, ...] | None = None
        self._store: WebFrameStore | None = None
        self._store_error: str | None = None

    # -------------------------------------------------------------- refresh

    def refresh(self) -> None:
        """Drop every cached discovery result so the next call rescans disk.

        Data roots gain clips and checkpoints, so the review UI must not serve a
        permanently cached listing.  Annotations are re-read lazily by
        :meth:`resolve`, which means a refreshed catalog also drops stale labels.
        """
        self._refs.clear()
        self._ref_index.clear()
        self._reasons.clear()
        self._warnings.clear()
        self._ball_stores.clear()
        self._specs = None
        self._store = None
        self._store_error = None

    # ----------------------------------------------------------- discovery

    def specs(self) -> tuple[BallDatasetSpec, ...]:
        """Discover version directories; a malformed store stays visible with a reason."""
        if self._specs is None:
            root = self.data_root / "ball_detection"
            versions = sorted(root.iterdir()) if root.is_dir() else []
            self._specs = tuple(
                BallDatasetSpec(
                    f"store/{version.name}", f"Ball store ({version.name})",
                    f"ball_detection/{version.name}", "temporal",
                )
                for version in versions
                if version.is_dir() and (version / METADATA_FILE).is_file()
            ) + DATASET_SPECS
        return self._specs

    def root_of(self, spec: BallDatasetSpec) -> Path:
        """Return the absolute root directory of one dataset source."""
        return (self.data_root / spec.relative).resolve()

    def spec(self, dataset_id: str) -> BallDatasetSpec:
        """Return the dataset spec for ``dataset_id`` or fail loudly."""
        for spec in self.specs():
            if spec.id == dataset_id:
                return spec
        known = ", ".join(spec.id for spec in self.specs())
        raise BallDatasetCatalogError(
            f"Unknown dataset {dataset_id!r}; expected one of [{known}]."
        )

    def entries(self) -> list[DatasetEntry]:
        """Return one availability summary per dataset source."""
        entries: list[DatasetEntry] = []
        for spec in self.specs():
            refs, reason = self._scene_refs(spec)
            entries.append(
                DatasetEntry(
                    spec=spec,
                    root=self.root_of(spec),
                    available=reason is None and bool(refs),
                    count=len(refs),
                    max_scene_frames=max((ref.frames for ref in refs), default=0),
                    reason=reason,
                    warnings=self._warnings.get(spec.id, ()),
                )
            )
        return entries

    def refs(self, dataset_id: str) -> tuple[SceneRef, ...]:
        """Return the cached scene references for one dataset."""
        return self._scene_refs(self.spec(dataset_id))[0]

    def scene_ref(self, dataset_id: str, local_id: str) -> SceneRef:
        """Return one catalogued scene reference."""
        self._scene_refs(self.spec(dataset_id))
        ref = self._ref_index.get(dataset_id, {}).get(local_id)
        if ref is None:
            raise BallDatasetCatalogError(
                f"Unknown scene {local_id!r} for dataset {dataset_id!r}."
            )
        return ref

    def _scene_refs(
        self, spec: BallDatasetSpec
    ) -> tuple[tuple[SceneRef, ...], str | None]:
        cached = self._refs.get(spec.id)
        if cached is not None:
            # The reason travels with the cache so a second catalog call cannot
            # silently claim a broken dataset is fine.
            return cached, self._reasons.get(spec.id)
        if spec.id.startswith("web_"):
            refs, reason = self._web_refs(spec)
        else:
            refs, reason = self._ball_refs(spec)
        self._refs[spec.id] = refs
        self._ref_index[spec.id] = {ref.local_id: ref for ref in refs}
        if reason is not None:
            self._reasons[spec.id] = reason
        return refs, reason

    def _ball_refs(
        self, spec: BallDatasetSpec
    ) -> tuple[tuple[SceneRef, ...], str | None]:
        root = self.data_root / spec.relative
        try:
            _require_within(root, self.data_root, what="ball store")
            for name in (METADATA_FILE, "index.npz"):
                _require_within(root / name, root, what="store index")
            store = BallFrameStore(root.resolve())
            for clip in store.clips:
                _require_within(root / SHARDS_DIR / shard_name(clip.index), root, what="shard")
        except (OSError, ValueError, KeyError) as error:
            return (), str(error)
        self._ball_stores[spec.id] = store
        return tuple(
            SceneRef(
                dataset_id=spec.id, local_id=clip.clip_id,
                label=f"{clip.clip_id} [{clip.split}]", frames=clip.frame_count,
                clip_id=clip.clip_id,
            )
            for clip in sorted(store.clips, key=lambda clip: _natural_key(clip.clip_id))
        ), None

    def _open_store(self) -> WebFrameStore | None:
        if self._store is not None:
            return self._store
        if self._store_error is not None:
            return None
        root = self.root_of(self.spec("web_static"))
        try:
            self._store = WebFrameStore(root)
        except (OSError, ValueError, KeyError) as error:
            self._store_error = str(error)
            return None
        # The canonical writer uses relpath for in-place COCO images, which may
        # be siblings of unified/. Constrain these references to the configured
        # data root, not to the shard store itself.
        for relative in self._store.paths:
            candidate = Path(relative)
            if candidate.is_absolute():
                self._store_error = (
                    f"web store path {relative!r} must be a non-escaping relative path."
                )
                self._store = None
                return None
            if (
                not (root / candidate)
                .resolve()
                .is_relative_to(self.data_root.resolve())
            ):
                self._store_error = (
                    f"web store path {relative!r} must be a non-escaping reference "
                    "inside the configured data root."
                )
                self._store = None
                return None
        return self._store

    def _web_refs(
        self, spec: BallDatasetSpec
    ) -> tuple[tuple[SceneRef, ...], str | None]:
        store = self._open_store()
        if store is None:
            return (), self._store_error or "unified web store is unavailable"
        if spec.mode == "static":
            refs = tuple(
                SceneRef(
                    dataset_id=spec.id,
                    local_id=str(index),
                    label=self._web_label(store, index),
                    frames=1,
                    sample_indices=(index,),
                )
                for index in self._all_indices(store)
                if not store.temporal(index)
            )
        else:
            refs = tuple(
                SceneRef(
                    dataset_id=spec.id,
                    local_id=store.sequence_name(indices[0]),
                    label=(
                        f"{store.source_name(indices[0])}/"
                        f"{store.sequence_name(indices[0])}"
                    ),
                    frames=len(indices),
                    sample_indices=indices,
                )
                for indices in self._temporal_sequences(store)
            )
        if not refs:
            return (), f"no {spec.mode} web samples found in {self.root_of(spec)}"
        return refs, None

    @staticmethod
    def _all_indices(store: WebFrameStore) -> list[int]:
        indices: list[int] = []
        for split in SPLIT_NAMES:
            indices.extend(int(value) for value in store.split_indices(split).tolist())
        return indices

    @staticmethod
    def _web_label(store: WebFrameStore, index: int) -> str:
        frame_index = store.frame_index(index)
        suffix = f"f{frame_index}" if frame_index >= 0 else "still"
        return f"{store.source_name(index)}/{store.sequence_name(index)}/{suffix}"

    @staticmethod
    def _temporal_sequences(store: WebFrameStore) -> list[tuple[int, ...]]:
        by_sequence: dict[str, list[int]] = {}
        for index in BallDatasetCatalog._all_indices(store):
            if store.temporal(index):
                by_sequence.setdefault(store.sequence_name(index), []).append(index)
        return [
            tuple(sorted(values, key=store.frame_index))
            for _, values in sorted(by_sequence.items())
        ]

    # ---------------------------------------------------------- resolution

    def resolve(self, dataset_id: str, local_id: str) -> SceneFrames:
        """Resolve one catalogued scene to a concrete frame accessor."""
        spec = self.spec(dataset_id)
        ref = self.scene_ref(dataset_id, local_id)
        if ref.clip_id is not None:
            return StoreSceneFrames(
                self._ball_stores[dataset_id], ref.clip_id,
            )
        store = self._open_store()
        if store is None:
            raise BallDatasetCatalogError(
                self._store_error or "unified web store is unavailable"
            )
        return WebSceneFrames(store=store, indices=ref.sample_indices, mode=spec.mode)

    def iter_scene_refs(self) -> Iterator[SceneRef]:
        """Yield every scene reference across all datasets."""
        for spec in self.specs():
            yield from self._scene_refs(spec)[0]


def split_scene_id(scene: str) -> tuple[str, str]:
    """Split an opaque scene id into ``(dataset_id, local_id)``."""
    dataset_id, separator, local_id = str(scene).partition(SCENE_SEPARATOR)
    if not separator or not dataset_id or not local_id:
        raise BallDatasetCatalogError(
            f"Scene id {scene!r} must look like '<dataset>{SCENE_SEPARATOR}<scene>'."
        )
    return dataset_id, local_id


__all__ = [
    "DATASET_SPECS",
    "SCENE_SEPARATOR",
    "SPLIT_NAMES",
    "WEB_RELATIVE",
    "BallDatasetCatalog",
    "BallDatasetCatalogError",
    "BallDatasetSpec",
    "StoreSceneFrames",
    "DatasetEntry",
    "SceneFrames",
    "SceneMode",
    "SceneRef",
    "WebSceneFrames",
    "split_scene_id",
]
