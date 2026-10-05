"""Catalog and bounded loading of explicitly supplied read-only data sources."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.utils.configuration import PathResolver

from .model import ReviewSequence
from .sources import BallPoseCampaign, CheckedFile, ComponentStore


class TrackingReviewService:
    def __init__(
        self,
        *,
        resolver: PathResolver,
        campaign: Path | None = None,
        pose_dataset: Path | None = None,
        stores: tuple[Path, ...] = (),
        reference_files: tuple[Path, ...] = (),
    ) -> None:
        if (campaign is None) != (pose_dataset is None):
            raise ValueError("--campaign and --pose-dataset must be provided together")
        if campaign is None and not stores:
            raise ValueError("Supply a saved campaign or at least one component store")
        references: dict[str, tuple[ClipLabels, CheckedFile]] = {}
        for path in reference_files:
            checked = CheckedFile.open(path)
            label = ClipLabels.load(path)
            if label.clip_id in references:
                raise ValueError(f"Multiple reference files for clip {label.clip_id}")
            references[label.clip_id] = label, checked
        self.sources: list[BallPoseCampaign | ComponentStore] = []
        if campaign is not None and pose_dataset is not None:
            self.sources.append(BallPoseCampaign(campaign, pose_dataset, resolver))
        self.sources.extend(
            ComponentStore(path, resolver, references) for path in stores
        )
        self.source_by_key = {
            key: source for source in self.sources for key in source.records
        }
        if len(self.source_by_key) != sum(
            len(source.records) for source in self.sources
        ):
            raise ValueError("Duplicate tracking sources were supplied")
        unused = set(references) - {
            record["clip_id"]
            for source in self.sources
            if isinstance(source, ComponentStore)
            for record in source.records.values()
        }
        if unused:
            raise ValueError(
                f"Reference labels have no matching explicitly supplied component store: {sorted(unused)}"
            )
        self._cache: OrderedDict[str, ReviewSequence] = OrderedDict()
        self._lock = Lock()

    def catalog(self) -> dict[str, Any]:
        return {
            "groups": [source.catalog() for source in self.sources],
            "read_only": True,
            "inference": False,
        }

    def load(self, key: str) -> ReviewSequence:
        if key not in self.source_by_key:
            raise KeyError(f"Unknown tracking sequence {key}")
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self.source_by_key[key].load(key)
                if len(self._cache) > 3:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return self._cache[key]
