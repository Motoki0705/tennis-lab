"""Read-only catalog of versioned ball frame stores.

Ball versions are discovered under data/ball_detection/<version>. Each scene
is a full camera clip, including unreviewed and unresolved frames. Requests
resolve only enumerated opaque IDs, never caller-supplied filesystem paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    METADATA_FILE,
    SHARDS_DIR,
    BallFrameStore,
    shard_name,
)
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    FrameSupervision,
    resolve_frame_supervision,
)
from src.tasks.ball_detection.data.types import FrameLabel
from src.tasks.ball_detection.visualization.io.store_frames import StoreSceneFrames

SceneMode = Literal["temporal"]

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

    def read_jpeg(self, index: int) -> bytes:
        """Return the original stored JPEG bytes."""
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
class SceneRef:
    """One catalogued scene before its frames are materialised."""

    dataset_id: str
    local_id: str
    label: str
    frames: int
    clip_id: str

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
        self._supervision: dict[str, FrameSupervision] = {}
        self._specs: tuple[BallDatasetSpec, ...] | None = None

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
        self._supervision.clear()
        self._specs = None

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
            )
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

    def store(self, dataset_id: str) -> BallFrameStore:
        """Return a discovered store, refusing unavailable datasets."""
        self.refs(dataset_id)
        if dataset_id not in self._ball_stores:
            raise BallDatasetCatalogError(self._reasons.get(dataset_id, "Store unavailable"))
        return self._ball_stores[dataset_id]

    def resolve(self, dataset_id: str, local_id: str) -> StoreSceneFrames:
        """Resolve one catalogued store clip to its frame accessor."""
        ref = self.scene_ref(dataset_id, local_id)
        store = self._ball_stores[dataset_id]
        if dataset_id not in self._supervision:
            self._supervision[dataset_id] = resolve_frame_supervision(store, OBSERVED_ONLY)
        frames = StoreSceneFrames(store, ref.clip_id, supervision=self._supervision[dataset_id])
        return frames

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
    "SCENE_SEPARATOR",
    "BallDatasetCatalog",
    "BallDatasetCatalogError",
    "BallDatasetSpec",
    "StoreSceneFrames",
    "DatasetEntry",
    "SceneFrames",
    "SceneMode",
    "SceneRef",
    "split_scene_id",
]
