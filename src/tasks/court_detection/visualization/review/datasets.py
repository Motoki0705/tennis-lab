"""Read-only dataset catalog and ground-truth previews for Court review.

Every dataset is described by the canonical typed source contract, never by a
directory-name guess: the TennisCourtDetector preset is read from its Hydra
``source`` file (including the quarantined ``QszoUKyCOHo_600`` sample) and each
synthetic scene is typed by the ``schema`` field published in its
``dataset.json``.  Records and raw samples come from ``build_court_input`` so
the review UI sees exactly the payload the training pipeline consumes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.schema import (
    court_schema_from_dataset_schema,
)
from src.tasks.court_detection.configuration import (
    CourtTargetConfig,
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtInputSpec,
    CourtRawSample,
    CourtSampleRecord,
    CourtSourceKind,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.factory import build_court_input
from src.tasks.court_detection.data.processing.targets import build_target_builder
from src.tasks.court_detection.data.target_generation.store import (
    SEGMENTATION_TARGET_SCHEMA,
    CourtDerivedTargetStore,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEMANTIC_LINE_CHANNEL_NAMES,
    SEMANTIC_LINE_TARGET_SCHEMA,
)
from src.tasks.court_detection.visualization.review.rasters import (
    line_raster,
    segmentation_raster,
    semantic_line_raster,
)
from src.utils.configuration import (
    PathResolver,
    RuntimePathRoots,
)
from src.utils.schema.court import COURT_KP_NAMES, GROUND_COURT_KP_NAMES

_TENNIS_SOURCE_PRESET = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "data"
    / "source"
    / "tennis_court_detector.yaml"
)
_SYNTHETIC_SOURCE_RELATIVE = Path("synthetic_data_generation") / "scenes"
_DERIVED_TARGET_RELATIVE = Path("court_detection") / "derived_targets"
_SCENE_SEPARATOR = "::"

# The dense schemas the review UI can render.  These are the current single
# source of truth for materialized Court targets.
DENSE_TARGET_SCHEMAS: Mapping[CourtDenseTargetKind, str] = MappingProxyType(
    {
        "seg": SEGMENTATION_TARGET_SCHEMA,
        "line": LINE_TARGET_SCHEMA,
        "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
    }
)

# The exact keypoint channel names each input implementation declares.  These
# come from the same public schema constants the input layers use, so the cheap
# catalog compatibility check cannot invent a channel order; ``layers()`` still
# revalidates the real input spec before any preview or inference runs.
_TENNIS_CHANNEL_NAMES: tuple[str, ...] = GROUND_COURT_KP_NAMES
_SYNTHETIC_V2V3_CHANNEL_NAMES: tuple[str, ...] = COURT_KP_NAMES[:14]
_SYNTHETIC_V1_CHANNEL_NAMES: tuple[str, ...] = tuple(SEMANTIC_CLASS_NAMES)


@dataclass(frozen=True, slots=True)
class CourtDatasetEntry:
    """One reviewable ``(source, scene, split)`` dataset."""

    id: str
    label: str
    path: Path
    source_kind: CourtSourceKind
    split: CourtSourceSplit
    scene_id: str | None
    published_schema: str | None
    available: bool
    reason: str | None
    count: int

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "path": str(self.path),
            "available": self.available,
            "count": self.count,
            "source_kind": self.source_kind,
            "split": self.split,
        }
        if self.published_schema is not None:
            payload["schema"] = self.published_schema
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class CourtDatasetLayers:
    """Typed layer identity resolved from one dataset's input contract."""

    keypoint_schema: str | None
    keypoint_channel_names: tuple[str, ...]
    dense_schemas: Mapping[CourtDenseTargetKind, str]


def build_path_resolver(
    *,
    project_root: Path,
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
) -> PathResolver:
    """Build the canonical role resolver for out-of-Hydra callers."""
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "checkpoint_root": str(checkpoint_root),
            "artifact_root": "assets",
            "output_root": str(output_root),
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        repository_root=project_root,
    )
    return PathResolver(roots)


def tennis_source_preset_path() -> Path:
    """Return the composed TennisCourtDetector source preset used as authority.

    The composed Hydra preset (not a directory guess) owns the annotation root,
    the split mapping, and the quarantined sample IDs, so the review catalog and
    the training pipeline share one exclusion contract.
    """
    return _TENNIS_SOURCE_PRESET


def _stat_stamp(path: Path) -> int:
    """Return a change fingerprint for one published path, or ``-1``."""
    try:
        stat = path.stat()
    except OSError:
        return -1
    return stat.st_mtime_ns


def _file_stamp(record: CourtSampleRecord) -> tuple[int, int, int, int]:
    """Return ``(image size, image mtime, annotation size, annotation mtime)``."""
    image = record.image_path.stat()
    annotation = record.annotation_path.stat()
    return (
        image.st_size,
        image.st_mtime_ns,
        annotation.st_size,
        annotation.st_mtime_ns,
    )


def _stamped_size_mtime(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=64)
def _cached_json(path_str: str, size: int, mtime_ns: int) -> object:
    """Parse one published JSON file, reusing the parse until its bytes change."""
    del size, mtime_ns
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def layer_identity(
    *,
    source_kind: CourtSourceKind,
    published_schema: str | None,
) -> CourtDatasetLayers:
    """Return the published layer identity without loading any image.

    Catalog compatibility filtering must stay cheap, so the keypoint channel
    names are read from the same constants the input layers declare and the
    dense schemas are the current materialized-target schemas.  The heavy
    canonical input is still built and revalidated before any preview or
    inference runs, so a divergence fails loudly instead of mislabeling a layer.
    """
    if source_kind == "tennis_court_detector":
        return CourtDatasetLayers(
            keypoint_schema="tennis_court_detector_kp14",
            keypoint_channel_names=_TENNIS_CHANNEL_NAMES,
            dense_schemas=DENSE_TARGET_SCHEMAS,
        )
    if published_schema is None:
        raise ValueError(
            "Synthetic Court の layer identity には publish 済み schema が必要です。"
        )
    version = court_schema_from_dataset_schema(published_schema).version.value
    channel_names = (
        _SYNTHETIC_V1_CHANNEL_NAMES
        if version == "v1"
        else _SYNTHETIC_V2V3_CHANNEL_NAMES
    )
    return CourtDatasetLayers(
        keypoint_schema=f"synthetic_court_{version}",
        keypoint_channel_names=channel_names,
        dense_schemas=DENSE_TARGET_SCHEMAS,
    )


class CourtDatasetCatalog:
    """Discover and serve Court review datasets without mutating any source."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        checkpoint_root: Path,
        output_root: Path,
        derived_target_root: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve(strict=False)
        self.data_root = data_root.resolve(strict=False)
        self.checkpoint_root = checkpoint_root.resolve(strict=False)
        self.output_root = output_root.resolve(strict=False)
        self.derived_target_root = (
            derived_target_root.resolve(strict=False)
            if derived_target_root is not None
            else (self.data_root / _DERIVED_TARGET_RELATIVE)
        )
        self.target_store = CourtDerivedTargetStore(self.derived_target_root)
        self.resolver = build_path_resolver(
            project_root=self.project_root,
            data_root=self.data_root,
            checkpoint_root=self.checkpoint_root,
            output_root=self.output_root,
        )
        self._tennis_root = self.data_root / "court"
        self._synthetic_workspace = self.data_root / _SYNTHETIC_SOURCE_RELATIVE
        self._entries: tuple[CourtDatasetEntry, ...] | None = None
        self._scan_signature: tuple[tuple[str, int], ...] | None = None
        self._inputs: dict[str, CourtInput] = {}
        self._input_errors: dict[str, str] = {}
        self._records: dict[
            tuple[str, CourtSourceSplit], tuple[CourtSampleRecord, ...]
        ] = {}
        self._record_stats: dict[
            tuple[str, CourtSourceSplit], dict[str, tuple[int, int, int, int]]
        ] = {}

    # ---------------------------------------------------------------- catalog
    def entries(self, *, refresh: bool = False) -> tuple[CourtDatasetEntry, ...]:
        signature = self._signature()
        if refresh or self._entries is None or signature != self._scan_signature:
            self._scan_signature = signature
            self._entries = tuple(self._discover())
            # A source that changed on disk must be revalidated from scratch.
            self._inputs.clear()
            self._input_errors.clear()
            self._records.clear()
            self._record_stats.clear()
        return self._entries

    def entry(self, dataset_id: str) -> CourtDatasetEntry:
        for entry in self.entries():
            if entry.id == dataset_id:
                return entry
        raise ValueError(f"未登録の Court dataset {dataset_id!r} です。")

    def warnings(self) -> list[str]:
        notes: list[str] = []
        for dataset_id, reason in sorted(self._input_errors.items()):
            notes.append(f"{dataset_id}: {reason}")
        return notes

    def records(self, dataset_id: str) -> tuple[CourtSampleRecord, ...]:
        entry = self.entry(dataset_id)
        if not entry.available:
            raise ValueError(
                f"Court dataset {dataset_id!r} は利用できません: {entry.reason}"
            )
        cache_key = self._cache_key(entry)
        key = (cache_key, entry.split)
        cached = self._records.get(key)
        if cached is None:
            cached = self._build_input(entry).records(entry.split)
            self._records[key] = cached
            self._record_stats[key] = {
                record.sample_id: _file_stamp(record) for record in cached
            }
        return cached

    def verify_sample_files(self, entry: CourtDatasetEntry, sample_id: str) -> None:
        """Fail loudly when a source file changed since the catalog was read."""
        stamps = self._record_stats.get((self._cache_key(entry), entry.split))
        if stamps is None or sample_id not in stamps:
            raise ValueError(
                f"Court sample {sample_id!r} は catalog cache に含まれません。"
                "catalog を更新してから要求してください。"
            )
        expected = stamps[sample_id]
        for record in self.records(entry.id):
            if record.sample_id == sample_id:
                current = _file_stamp(record)
                break
        else:  # pragma: no cover - guarded by the stamp membership check
            raise ValueError(f"Court sample {sample_id!r} が消えました。")
        if current != expected:
            raise ValueError(
                f"Court sample {sample_id!r} の source file が disk 上で変化しました。"
                "catalog を更新してから表示・推論してください。"
            )

    def layers(self, dataset_id: str) -> CourtDatasetLayers:
        spec = self._build_input(self.entry(dataset_id)).spec
        return CourtDatasetLayers(
            keypoint_schema=spec.keypoint_schema,
            keypoint_channel_names=spec.keypoint_channel_names,
            dense_schemas=DENSE_TARGET_SCHEMAS,
        )

    def count(self, dataset_id: str) -> int:
        entry = self.entry(dataset_id)
        if not entry.available:
            return 0
        return len(self.records(dataset_id))

    def catalog_entries(self) -> list[dict[str, object]]:
        """Refresh sources, including image changes outside the cheap signature."""
        return [entry.to_dict() for entry in self.entries(refresh=True)]

    def input_for(self, entry: CourtDatasetEntry) -> CourtInput:
        """Return the cached canonical input layer for one dataset entry."""
        return self._build_input(entry)

    # ----------------------------------------------------------------- scenes
    def scene_id(self, dataset_id: str, sample_id: str) -> str:
        return f"{dataset_id}{_SCENE_SEPARATOR}{sample_id}"

    def split_scene(self, scene: str) -> tuple[str, str]:
        dataset_id, separator, sample_id = scene.partition(_SCENE_SEPARATOR)
        if not separator or not dataset_id or not sample_id:
            raise ValueError(
                "Court の scene ID は server が解決した '<dataset>::<sample>' 形式である必要があります。"
            )
        return dataset_id, sample_id

    def sample(self, scene: str) -> tuple[CourtDatasetEntry, CourtSampleRecord]:
        dataset_id, sample_id = self.split_scene(scene)
        entry = self.entry(dataset_id)
        for record in self.records(dataset_id):
            if record.sample_id == sample_id:
                self.verify_sample_files(entry, sample_id)
                return entry, record
        raise FileNotFoundError(
            f"Court sample {sample_id!r} は dataset {dataset_id!r} に含まれません。"
        )

    def load_raw(self, scene: str) -> tuple[CourtDatasetEntry, CourtRawSample]:
        entry, record = self.sample(scene)
        return entry, self._build_input(entry).load(record)

    def image(self, scene: str) -> Image.Image:
        _, sample = self.load_raw(scene)
        return sample.image

    # -------------------------------------------------------------- internals
    def _signature(self) -> tuple[tuple[str, int], ...]:
        """Return a cheap fingerprint of every published source root.

        Ordinary scene requests reuse canonical inputs while this fingerprint
        is unchanged. Explicit catalog refresh also invalidates image stamps.
        """
        signature: list[tuple[str, int]] = []
        for path in (
            self.data_root,
            self._tennis_root,
            self._tennis_root / "data_train.json",
            self._tennis_root / "data_val.json",
            self._synthetic_workspace,
        ):
            signature.append((str(path), _stat_stamp(path)))
        if self._synthetic_workspace.is_dir():
            for scene_dir in sorted(self._synthetic_workspace.iterdir()):
                manifest = scene_dir / "datasets" / "court" / "dataset.json"
                signature.append((str(scene_dir), _stat_stamp(scene_dir)))
                signature.append((str(manifest), _stat_stamp(manifest)))
        return tuple(signature)

    def _cache_key(self, entry: CourtDatasetEntry) -> str:
        if entry.source_kind == "tennis_court_detector":
            return "tennis_court_detector"
        if entry.scene_id is None:
            raise ValueError("Synthetic Court datasets require an explicit scene ID.")
        return f"synthetic_court/{entry.scene_id}"

    def _build_input(self, entry: CourtDatasetEntry) -> CourtInput:
        key = self._cache_key(entry)
        cached = self._inputs.get(key)
        if cached is not None:
            return cached
        failure = self._input_errors.get(key)
        if failure is not None:
            raise ValueError(failure)
        try:
            layer = self._instantiate_input(entry)
        except (ValueError, FileNotFoundError, OSError, KeyError, TypeError) as error:
            message = f"{type(error).__name__}: {error}"
            self._input_errors[key] = message
            raise ValueError(message) from error
        self._inputs[key] = layer
        return layer

    def _instantiate_input(self, entry: CourtDatasetEntry) -> CourtInput:
        if entry.source_kind == "tennis_court_detector":
            return self._build_tennis_input()
        if entry.scene_id is None:
            raise ValueError("Synthetic Court datasets require an explicit scene ID.")
        return self._build_synthetic_input(entry.scene_id)

    def _build_tennis_input(self) -> CourtInput:
        preset = tennis_source_preset_path()
        if not preset.is_file():
            raise FileNotFoundError(
                f"TennisCourtDetector source preset がありません: {preset}"
            )
        document = OmegaConf.to_container(OmegaConf.load(preset), resolve=False)
        if not isinstance(document, Mapping):
            raise ValueError(
                "TennisCourtDetector source preset は mapping である必要があります。"
            )
        config = TennisCourtDetectorSourceConfig.from_mapping(
            dict(document), resolver=self.resolver
        )
        if config.root != self._tennis_root:
            raise ValueError(
                "TennisCourtDetector preset の root が review の data root と一致しません: "
                f"preset={config.root}, review={self._tennis_root}"
            )
        return build_court_input(config, target_store=self.target_store)

    def _build_synthetic_input(self, scene_id: str) -> CourtInput:
        manifest = self._synthetic_manifest(scene_id)
        published = manifest.get("schema")
        if not isinstance(published, str):
            raise ValueError(
                f"Synthetic Court scene {scene_id!r} が string schema を publish していません。"
            )
        definition = court_schema_from_dataset_schema(published)
        scope = "all_courts" if definition.version.value == "v1" else "target_court"
        config = SyntheticCourtSourceConfig.from_mapping(
            {
                "kind": "synthetic_court",
                "schema": definition.version.value,
                "court_scope": scope,
                "workspace_root": str(_SYNTHETIC_SOURCE_RELATIVE),
                "scene_ids": [scene_id],
            },
            resolver=self.resolver,
        )
        if config.workspace_root != self._synthetic_workspace:
            raise ValueError(
                "Synthetic Court の workspace が review の data root と一致しません: "
                f"config={config.workspace_root}, review={self._synthetic_workspace}"
            )
        return build_court_input(config, target_store=self.target_store)

    def _synthetic_manifest(self, scene_id: str) -> Mapping[str, object]:
        manifest_path = (
            self._synthetic_workspace / scene_id / "datasets" / "court" / "dataset.json"
        )
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Synthetic Court dataset.json が無いか通常ファイルではありません: {manifest_path}"
            )
        parsed = _cached_json(str(manifest_path), *_stamped_size_mtime(manifest_path))
        if not isinstance(parsed, Mapping):
            raise ValueError(
                f"Synthetic Court dataset.json は mapping である必要があります: {manifest_path}"
            )
        return cast("Mapping[str, object]", parsed)

    def _discover(self) -> list[CourtDatasetEntry]:
        entries: list[CourtDatasetEntry] = []
        entries.extend(self._discover_tennis())
        entries.extend(self._discover_synthetic())
        return entries

    def _discover_tennis(self) -> list[CourtDatasetEntry]:
        reason: str | None = None
        if not (self._tennis_root / "data_train.json").is_file():
            reason = (
                f"TennisCourtDetector の annotation がありません: {self._tennis_root}"
            )
        counts: dict[str, int] = {}
        if reason is None:
            try:
                counts = self._tennis_split_counts()
            except (ValueError, OSError) as error:
                reason = f"{type(error).__name__}: {error}"
            if reason is None and "train" not in counts:
                reason = "TennisCourtDetector preset に train split がありません。"
        entries = [
            CourtDatasetEntry(
                id=f"tennis_court_detector/{split}",
                label=f"TennisCourtDetector {split}",
                path=self._tennis_root,
                source_kind="tennis_court_detector",
                split=cast("CourtSourceSplit", split),
                scene_id=None,
                published_schema="tennis_court_detector_annotations_v1",
                available=reason is None and split in counts,
                reason=(
                    reason
                    if reason is not None
                    else (
                        None
                        if split in counts
                        else f"TennisCourtDetector preset に {split} split がありません。"
                    )
                ),
                count=counts.get(split, 0),
            )
            for split in ("train", "val")
        ]
        return entries

    def _tennis_split_counts(self) -> dict[str, int]:
        """Count the exact retained annotation records without opening images.

        The heavy ``records()`` build re-opens every image to hash its geometry,
        which is far too slow for a catalog request, so the count is derived from
        the same preset (including its quarantined sample IDs) and the annotation
        lengths.  ``tests/unit/tasks/court_detection/visualization/review``
        asserts this equals ``len(records())`` on the real dataset.
        """
        preset = tennis_source_preset_path()
        document = OmegaConf.to_container(OmegaConf.load(preset), resolve=False)
        if not isinstance(document, Mapping):
            raise ValueError(
                "TennisCourtDetector source preset は mapping である必要があります。"
            )
        excluded = document.get("excluded_sample_ids", ())
        if not isinstance(excluded, Sequence) or isinstance(excluded, (str, bytes)):
            raise ValueError(
                "TennisCourtDetector excluded_sample_ids は list である必要があります。"
            )
        excluded_ids = {str(value) for value in excluded}
        counts: dict[str, int] = {}
        for split, source_split in (("train", "train"), ("val", "val")):
            annotation_path = self._tennis_root / f"data_{source_split}.json"
            if not annotation_path.is_file():
                continue
            parsed = _cached_json(
                str(annotation_path), *_stamped_size_mtime(annotation_path)
            )
            if not isinstance(parsed, list):
                raise ValueError(
                    f"TennisCourtDetector {annotation_path.name} は list である必要があります。"
                )
            counts[split] = sum(
                1
                for entry in parsed
                if isinstance(entry, Mapping)
                and cast(str, entry.get("id")) not in excluded_ids
            )
        return counts

    def _discover_synthetic(self) -> list[CourtDatasetEntry]:
        if not self._synthetic_workspace.is_dir():
            return []
        entries: list[CourtDatasetEntry] = []
        for scene_dir in sorted(self._synthetic_workspace.iterdir()):
            if not scene_dir.is_dir() or scene_dir.is_symlink():
                continue
            court_dir = scene_dir / "datasets" / "court"
            if not (court_dir / "dataset.json").is_file():
                continue
            scene_id = scene_dir.name
            published: str | None = None
            reason: str | None = None
            counts: dict[str, int] = {}
            try:
                manifest = self._synthetic_manifest(scene_id)
                schema = manifest.get("schema")
                if not isinstance(schema, str):
                    raise ValueError("dataset.json does not publish a string schema.")
                definition = court_schema_from_dataset_schema(schema)
                published = schema
                if manifest.get("status") != "completed":
                    raise ValueError(
                        f"Synthetic Court scene {scene_id!r} は completed stage ではありません。"
                    )
                counts = self._synthetic_split_counts(
                    scene_id, definition.version.value
                )
            except (
                ValueError,
                FileNotFoundError,
                OSError,
                KeyError,
                TypeError,
            ) as error:
                reason = f"{type(error).__name__}: {error}"
                counts = {}
            if published is None and reason is None:
                reason = "Synthetic Court の schema を解決できませんでした。"
            for split in ("train", "val", "test"):
                available = reason is None and split in counts
                entries.append(
                    CourtDatasetEntry(
                        id=f"synthetic_court/{scene_id}/{split}",
                        label=f"Synthetic {scene_id} {split}",
                        path=court_dir,
                        source_kind="synthetic_court",
                        split=cast("CourtSourceSplit", split),
                        scene_id=scene_id,
                        published_schema=published,
                        available=available,
                        reason=None
                        if available
                        else (
                            reason
                            or f"Synthetic Court scene {scene_id!r} に "
                            f"{split} split がありません。"
                        ),
                        count=counts.get(split, 0),
                    )
                )
        return entries

    def _synthetic_split_counts(self, scene_id: str, version: str) -> dict[str, int]:
        """Count accepted samples per configured split from the published manifest.

        The split names published by each schema version are mapped explicitly
        (v1 uses ``val``; v2/v3 publish ``validation``), and the result is
        asserted against ``len(records())`` in the review unit tests so the
        cheap catalog count cannot silently drift from the canonical contract.
        """
        manifest = self._synthetic_manifest(scene_id)
        split_map = (
            {"train": "train", "val": "val", "test": "test"}
            if version == "v1"
            else {"train": "train", "validation": "val", "test": "test"}
        )
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(
                f"Synthetic Court scene {scene_id!r} は accepted sample を publish していません。"
            )
        counts = dict.fromkeys(split_map.values(), 0)
        for entry in samples:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Synthetic Court scene {scene_id!r} の sample record は mapping である必要があります。"
                )
            raw_split = entry.get("split")
            if raw_split not in split_map:
                raise ValueError(
                    f"Synthetic Court scene {scene_id!r} は未対応の split "
                    f"{raw_split!r} を publish しています。"
                )
            counts[split_map[cast(str, raw_split)]] += 1
        return counts


def keypoint_points(raw: CourtRawSample) -> list[dict[str, object]]:
    """Flatten the source keypoint channels into JSON point payloads."""
    channels = raw.keypoint_channels
    if channels is None:
        return []
    names = channels.channel_names
    coordinates = channels.points_xy
    visible = channels.point_visible
    physical = channels.physical_indices
    points: list[dict[str, object]] = []
    for channel in range(coordinates.shape[0]):
        for peak in range(coordinates.shape[1]):
            physical_index = int(physical[channel, peak])
            if physical_index < 0:
                continue
            x_coord, y_coord = coordinates[channel, peak].tolist()
            points.append(
                {
                    "x": float(x_coord),
                    "y": float(y_coord),
                    "label": names[channel],
                    "visible": bool(visible[channel, peak]),
                    "physical_index": physical_index,
                }
            )
    return points


@dataclass(frozen=True, slots=True)
class GroundTruthMasks:
    """Provenance-verified dense ground truth for one reviewed sample."""

    seg: np.ndarray | None
    line: np.ndarray | None
    semantic_line: np.ndarray | None

    def raster_payloads(self) -> list[dict[str, object]]:
        """Render the available layers as transparent-background overlays."""
        rasters: list[dict[str, object]] = []
        if self.seg is not None:
            rasters.append(segmentation_raster(self.seg).to_dict())
        if self.line is not None:
            rasters.append(line_raster(self.line > 0).to_dict())
        if self.semantic_line is not None:
            rasters.append(
                semantic_line_raster(
                    self.semantic_line, SEMANTIC_LINE_CHANNEL_NAMES
                ).to_dict()
            )
        return rasters


def load_ground_truth_masks(
    record: CourtSampleRecord,
    *,
    input_spec: CourtInputSpec,
    raw: CourtRawSample,
) -> tuple[GroundTruthMasks, list[str]]:
    """Decode each dense layer through its canonical target builder.

    A layer whose derived target is missing, stale, or provenance-mismatched is
    reported as an explicit warning for that layer only; the keypoint and RGB
    review stays available and no substitute mask is ever produced.
    """
    loaded: dict[str, np.ndarray | None] = {
        "seg": None,
        "line": None,
        "semantic_line": None,
    }
    warnings: list[str] = []
    for kind in ("seg", "line", "semantic_line"):
        dense_kind = cast("CourtDenseTargetKind", kind)
        builder = build_target_builder(
            CourtTargetConfig(
                kind=dense_kind,
                sigma_ratio=None,
                target_schema=DENSE_TARGET_SCHEMAS[dense_kind],
            ),
            input_spec=input_spec,
        )
        try:
            builder.preflight((record,))
            tensors = builder.load_dense(raw)
            tensor = tensors[dense_kind]
        except (FileNotFoundError, ValueError, OSError, KeyError) as error:
            warnings.append(
                f"{kind} の ground truth は sample {record.sample_id} で利用できません: "
                f"{error}"
            )
            continue
        array = np.asarray(tensor.detach().cpu().numpy())
        if kind == "line":
            loaded[kind] = array[0] > 0
        else:
            loaded[kind] = array
    return (
        GroundTruthMasks(
            seg=loaded["seg"],
            line=loaded["line"],
            semantic_line=loaded["semantic_line"],
        ),
        warnings,
    )


__all__ = [
    "DENSE_TARGET_SCHEMAS",
    "CourtDatasetCatalog",
    "CourtDatasetEntry",
    "CourtDatasetLayers",
    "GroundTruthMasks",
    "build_path_resolver",
    "keypoint_points",
    "layer_identity",
    "load_ground_truth_masks",
    "tennis_source_preset_path",
]
