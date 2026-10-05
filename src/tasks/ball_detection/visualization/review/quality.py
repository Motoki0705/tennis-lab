"""Stored annotation states and observed-only supervision for dataset review.

Counts come from the validated frame/instance tables, not the generated README.
Point kinds can overlap within a frame; the four supervision groups are disjoint.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    EVENT_NAMES,
    POINT_KIND_CODES,
    BallFrameStore,
    ClipRecord,
)
from src.tasks.ball_detection.data.supervision import FrameSupervision

REVIEW_FILTERS: Final = (
    "scored_positive", "scored_negative", "reference", "unreviewed",
    *POINT_KIND_CODES, "context", "segment_break",
)


class BallReviewIndex:
    """Row-aligned review masks shared by the catalog and frame endpoints."""

    def __init__(self, store: BallFrameStore, supervision: FrameSupervision) -> None:
        self.store = store
        frame_of_instance: NDArray[np.int64] = np.repeat(
            np.arange(len(store), dtype=np.int64), store.frames["inst_count"]
        )
        self.masks: dict[str, NDArray[np.bool_]] = {}
        for name, code in POINT_KIND_CODES.items():
            mask: NDArray[np.bool_] = np.zeros(len(store), dtype=np.bool_)
            mask[frame_of_instance[store.instances["point_kind"] == code]] = True
            self.masks[name] = mask
        self.masks.update(
            scored_positive=supervision.supervised & self.masks["observed"],
            scored_negative=supervision.supervised & ~self.masks["observed"],
            reference=store.frames["annotated"] & ~supervision.supervised,
            unreviewed=~store.frames["annotated"],
            context=~store.frames["is_target"],
            segment_break=store.frames["segment_break"],
        )
        self._summaries: dict[str, dict[str, Any]] = {}

    def counts(self, rows: NDArray[np.int64]) -> dict[str, int]:
        """Count frames, keeping point-kind and supervision counts distinct."""
        return {
            "frames": len(rows),
            **{name: int(mask[rows].sum()) for name, mask in self.masks.items()},
        }

    def overview(self) -> dict[str, Any]:
        """Describe the version, source/split composition, and supervision."""
        clips = self.store.clips
        all_rows: NDArray[np.int64] = np.arange(len(self.store), dtype=np.int64)
        history = self.store.metadata.get("append_history", [])
        if not isinstance(history, list):
            raise ValueError("Ball store append_history must be a list")
        sources = []
        for source in sorted({clip.source for clip in clips}):
            source_clips = [clip for clip in clips if clip.source == source]
            rows = np.flatnonzero(np.isin(
                self.store.frames["clip"], [clip.index for clip in source_clips]
            ))
            sources.append({
                "id": source, "clips": len(source_clips),
                "counts": self.counts(rows),
                "splits": self._splits(source_clips),
            })
        return {
            "version": self.store.directory.name,
            "schema": self.store.metadata["schema_version"],
            "clips": len(clips), "counts": self.counts(all_rows),
            "sources": sources, "splits": self._splits(list(clips)),
            "point_counts": {
                name: int((self.store.instances["point_kind"] == code).sum())
                for name, code in POINT_KIND_CODES.items()
            },
            "append_history": [
                {key: entry[key] for key in ("base_clips", "added_clips", "at")}
                for entry in history
            ],
        }

    @staticmethod
    def _splits(clips: list[ClipRecord]) -> dict[str, dict[str, int]]:
        return {
            split: {
                "clips": sum(clip.split == split for clip in clips),
                "frames": sum(clip.frame_count for clip in clips if clip.split == split),
                "groups": len({(clip.source, clip.group_id) for clip in clips if clip.split == split}),
            }
            for split in ("train", "val", "test")
        }

    def summary(self, clip: ClipRecord) -> dict[str, Any]:
        """Return the cached source identity and frame counts for one clip."""
        if clip.clip_id not in self._summaries:
            rows = self.store.clip_rows(clip)
            self._summaries[clip.clip_id] = {
                "source": clip.source, "split": clip.split,
                "group_id": clip.group_id, "camera_id": clip.camera_id,
                "fps": clip.fps, "source_width": clip.source_width,
                "source_height": clip.source_height,
                "counts": self.counts(rows),
            }
        return self._summaries[clip.clip_id]

    def positions(self, clip: ClipRecord) -> dict[str, list[int]]:
        """List local frame positions for explicit jumps without skipping playback."""
        rows = self.store.clip_rows(clip)
        return {
            name: np.flatnonzero(mask[rows]).tolist()
            for name, mask in self.masks.items()
        }

    def frame(self, clip: ClipRecord, index: int) -> dict[str, Any]:
        """Report the stored frame state without creating unknown coordinates."""
        row = self.store.row_of(clip, index)
        supervision = next(
            name for name in ("scored_positive", "scored_negative", "reference", "unreviewed")
            if self.masks[name][row]
        )
        return {
            "supervision": supervision,
            "point_kinds": [name for name in POINT_KIND_CODES if self.masks[name][row]],
            "context": bool(self.masks["context"][row]),
            "segment_break": bool(self.masks["segment_break"][row]),
            "event": EVENT_NAMES[int(self.store.frames["event"][row])],
            "time_seconds": float(int(self.store.frames["pts"][row]) * Fraction(clip.time_base)),
        }
