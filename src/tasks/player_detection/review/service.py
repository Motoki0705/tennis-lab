"""Dataset facts and frame annotations, without generating labels or predictions."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.tasks.base.configuration import as_config_mapping
from src.tasks.player_detection.configuration import FrameSelectionConfig
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    BBOX_SOURCE_NAMES,
    INDEX_FILE,
    METADATA_FILE,
    SCHEMA_VERSION,
    SHARDS_DIR,
    ClipRecord,
    PlayerFrameStore,
    Split,
    shard_name,
)

FILTERS = frozenset({"all", "unresolved", "inferred", "occluded", "truncated", "excluded", "unreviewed"})


class StoreReview:
    """One immutable build; selection means default training eligibility, not QA."""

    def __init__(self, directory: Path, selection: FrameSelectionConfig) -> None:
        for name in (METADATA_FILE, INDEX_FILE, SHARDS_DIR):
            if not (directory / name).resolve().is_relative_to(directory):
                raise ValueError(f"Player store {name} escapes its version directory")
        self.store = PlayerFrameStore(directory)
        for clip in self.store.clips:
            if not (directory / SHARDS_DIR / shard_name(clip.index)).resolve().is_relative_to(directory):
                raise ValueError("Player JPEG shard escapes its version directory")
        self.selection = selection
        self.rows = {clip.clip_id: self.store.clip_frames(clip) for clip in self.store.clips}
        raw_clips = cast(list[dict[str, Any]], self.store.metadata["clips"])
        self.metadata = {str(clip["clip_id"]): clip for clip in raw_clips}
        self.reasons = [self._selection_reason(row) for row in range(len(self.store))]
        self.summaries = [self._clip_summary(clip) for clip in self.store.clips]

    def _selection_reason(self, row: int) -> str:
        store, selection = self.store, self.selection
        if selection.require_reviewed and not bool(store.frames["reviewed"][row]):
            return "unreviewed"
        instances = store.instances_of(row)
        unresolved = instances.bbox_source == BBOX_SOURCE_CODES["unresolved"]
        if selection.require_located_players and unresolved.any():
            return "unresolved_player"
        clip = store.clip_of(row)
        boxes = instances.boxes_xyxy[~unresolved].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, float(clip.width))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, float(clip.height))
        sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        return "eligible" if (sides >= selection.min_visible_box_px).any() else "no_visible_box"

    def _counts(self, rows: NDArray[np.int64]) -> dict[str, int]:
        counts: Counter[str] = Counter({name: 0 for name in BBOX_SOURCE_CODES})
        counts.update({"occluded": 0, "truncated": 0, "unreviewed": 0, "excluded": 0})
        for row in rows.tolist():
            instances = self.store.instances_of(row)
            counts.update(BBOX_SOURCE_NAMES[int(code)] for code in instances.bbox_source)
            for key in ("occluded", "truncated"):
                counts[key] += int(getattr(instances, key).sum())
            counts["unreviewed"] += int(not self.store.frames["reviewed"][row])
            counts["excluded"] += int(self.reasons[row] != "eligible")
        return dict(counts)

    def _clip_summary(self, clip: ClipRecord) -> dict[str, Any]:
        rows = self.rows[clip.clip_id]
        metadata = self.metadata[clip.clip_id]
        return {
            "id": clip.clip_id,
            "index": clip.index,
            "source_id": clip.source_id,
            "source_title": metadata["source_title"],
            "split": clip.split,
            "stored_frames": len(rows),
            "clip_frame_count": clip.frame_count,
            "unstored_frames": clip.frame_count - len(rows),
            "width": clip.width,
            "height": clip.height,
            "annotation_status": clip.annotation_status,
            "annotation_issues": metadata["annotation_issues"],
            "annotation_sha256": clip.annotation_sha256,
            "video_sha256": clip.video_sha256,
            "counts": self._counts(rows),
        }

    def summary(self) -> dict[str, Any]:
        sources: dict[str, list[str]] = {}
        for clip in self.store.clips:
            if clip.source_id not in sources:
                sources[clip.source_id] = []
            sources[clip.source_id].append(clip.split)
        leaking = [source for source, splits in sources.items() if len(set(splits)) != 1]
        return {
            "schema": SCHEMA_VERSION,
            "version": self.store.metadata["version"],
            "created_at": self.store.metadata["created_at"],
            "frames": len(self.store),
            "clips": len(self.store.clips),
            "source_videos": len(sources),
            "annotation_statuses": dict(Counter(clip.annotation_status for clip in self.store.clips)),
            "counts": self._counts(np.arange(len(self.store), dtype=np.int64)),
            "selection_reasons": dict(Counter(self.reasons)),
            "split_unit": "source_id",
            "split_source_overlap": leaking,
            "splits": {
                name: {
                    "frames": len(self.store.split_frames(cast(Split, name))),
                    "clips": sum(clip.split == name for clip in self.store.clips),
                    "sources": len({clip.source_id for clip in self.store.clips if clip.split == name}),
                }
                for name in ("train", "val", "test")
            },
            "selection": {
                "require_reviewed": self.selection.require_reviewed,
                "require_located_players": self.selection.require_located_players,
                "min_visible_box_px": self.selection.min_visible_box_px,
                "stride": "before per-split stride/sampling",
            },
        }

    def clip(self, clip_id: str) -> ClipRecord:
        if clip_id not in self.rows:
            raise FileNotFoundError(f"Unknown clip: {clip_id}")
        return self.store.clip_of(int(self.rows[clip_id][0]))

    def row(self, clip_id: str, frame_index: int) -> int:
        self.clip(clip_id)
        rows = self.rows[clip_id]
        match = rows[self.store.frames["frame_index"][rows] == frame_index]
        if match.size != 1:
            raise FileNotFoundError(f"Frame {frame_index} is not stored in {clip_id}; absence is not a negative label.")
        return int(match[0])

    def _flags(self, row: int) -> list[str]:
        instances = self.store.instances_of(row)
        flags = [name for name in ("unresolved", "inferred") if (instances.bbox_source == BBOX_SOURCE_CODES[name]).any()]
        flags.extend(name for name in ("occluded", "truncated") if getattr(instances, name).any())
        if not self.store.frames["reviewed"][row]:
            flags.append("unreviewed")
        if self.reasons[row] != "eligible":
            flags.append("excluded")
        return flags

    def timeline(self, clip_id: str) -> dict[str, Any]:
        clip = self.clip(clip_id)
        rows = self.rows[clip_id]
        summary = self.summaries[clip.index]
        return {
            **summary,
            "nominal_fps": clip.nominal_fps,
            "track_ids": list(clip.track_ids),
            "frames": [
                {"frame_index": int(self.store.frames["frame_index"][row]), "flags": self._flags(row)}
                for row in rows.tolist()
            ],
            "missing_ranges": missing_ranges(self.store.frames["frame_index"][rows], clip.frame_count),
        }

    def frame(self, clip_id: str, frame_index: int) -> dict[str, Any]:
        row = self.row(clip_id, frame_index)
        clip = self.clip(clip_id)
        instances = self.store.instances_of(row)
        annotations = []
        for track, box, source, occluded, truncated in zip(
            instances.track_index, instances.boxes_xyxy, instances.bbox_source,
            instances.occluded, instances.truncated, strict=True,
        ):
            name = BBOX_SOURCE_NAMES[int(source)]
            unresolved = name == "unresolved"
            visible = None if unresolved else [
                float(np.clip(box[0], 0, clip.width)), float(np.clip(box[1], 0, clip.height)),
                float(np.clip(box[2], 0, clip.width)), float(np.clip(box[3], 0, clip.height)),
            ]
            train_box = visible is not None and min(visible[2] - visible[0], visible[3] - visible[1]) >= self.selection.min_visible_box_px
            annotations.append({
                "track_id": clip.track_ids[int(track)],
                "bbox_source": name,
                "bbox_xyxy": None if unresolved else [float(value) for value in box],
                "visible_bbox_xyxy": visible,
                "occluded": bool(occluded),
                "truncated": bool(truncated),
                "visible_box_eligible": bool(train_box),
                "default_training_box": bool(train_box and self.reasons[row] == "eligible"),
            })
        return {
            "clip_id": clip_id,
            "frame_index": frame_index,
            "source_frame_index": int(self.store.frames["source_frame_index"][row]),
            "seconds": float(int(self.store.frames["clip_pts"][row]) * Fraction(clip.time_base)),
            "width": clip.width,
            "height": clip.height,
            "reviewed": bool(self.store.frames["reviewed"][row]),
            "is_target": bool(self.store.frames["is_target"][row]),
            "annotation_status": clip.annotation_status,
            "selection_reason": self.reasons[row],
            "annotations": annotations,
        }

    def image(self, clip_id: str, frame_index: int) -> bytes:
        row = self.row(clip_id, frame_index)
        clip = self.clip(clip_id)
        offset, length = int(self.store.frames["offset"][row]), int(self.store.frames["length"][row])
        # Return the exact existing JPEG, without encoding another data artifact.
        path: Path = self.store.directory / SHARDS_DIR / shard_name(clip.index)
        if not path.resolve().is_relative_to(self.store.directory):
            raise ValueError("Player JPEG shard escapes its version directory")
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("Stored JPEG changed or was truncated")
        return payload


def missing_ranges(indices: NDArray[np.generic], frame_count: int) -> list[list[int]]:
    """Inclusive clip-frame ranges absent from this player-only store."""
    edges = np.concatenate([[-1], indices, [frame_count]])
    return [[int(left + 1), int(right - 1)] for left, right in zip(edges[:-1], edges[1:], strict=True) if right > left + 1]


class PlayerReviewService:
    """Only catalogued version and clip identities can address dataset bytes."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        config = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs/data/default.yaml")
        self.selection = FrameSelectionConfig.from_mapping(as_config_mapping(config.selection, path="data.selection"), "data.selection")
        self.reviews: dict[str, StoreReview] = {}
        self.catalogue: list[dict[str, Any]] = []
        base = self.data_root / "player_detection"
        paths = sorted(base.iterdir()) if base.is_dir() else []
        for directory in paths:
            if not directory.is_dir() or not (directory / METADATA_FILE).is_file():
                continue
            if not directory.resolve().is_relative_to(self.data_root):
                self.catalogue.append({"id": directory.name, "available": False, "reason": "Version escapes the configured data root"})
                continue
            try:
                review = StoreReview(directory.resolve(), self.selection)
                summary = review.summary()
            except (ValueError, KeyError, OSError) as error:
                self.catalogue.append({"id": directory.name, "available": False, "reason": str(error)})
                continue
            self.reviews[directory.name] = review
            self.catalogue.append({"id": directory.name, "available": True, **summary})

    def review(self, dataset: str) -> StoreReview:
        if dataset not in self.reviews:
            raise FileNotFoundError(f"Unknown or unavailable player dataset: {dataset}")
        return self.reviews[dataset]

    def catalog(self) -> dict[str, Any]:
        return {"datasets": self.catalogue, "read_only": True, "coordinate_units": "original image pixel xyxy", "annotation_origin": "chat-annotation snapshot; bbox_source is an annotation state, not a detector probability"}

    def clips(self, dataset: str, *, search: str = "", split: str = "all", source: str = "all", flag: str = "all") -> dict[str, Any]:
        if flag not in FILTERS or split not in {"all", "train", "val", "test"}:
            raise ValueError("Unknown split or review filter")
        review = self.review(dataset)
        text = search.casefold()
        results = [
            clip for clip in review.summaries
            if (split == "all" or clip["split"] == split)
            and (source == "all" or clip["source_id"] == source)
            and (flag == "all" or clip["counts"][flag] > 0)
            and (not text or text in f"{clip['id']} {clip['source_title']} {clip['source_id']}".casefold())
        ]
        return {"total": len(results), "clips": results}
