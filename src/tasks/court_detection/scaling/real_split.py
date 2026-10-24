"""Immutable video-disjoint evaluation registries over the original real store."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.tasks.court_detection.configuration import (
    TennisCourtDetectorSourceConfig,
    TennisCourtEvaluationSourceConfig,
)
from src.tasks.court_detection.data.contracts import CourtSampleRecord, CourtSourceSplit
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.utils.data.image_record_store import file_sha256

EXCLUDED_SAMPLE_IDS = ("QszoUKyCOHo_600",)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def video_id(sample_id: str) -> str:
    video, separator, frame = sample_id.rpartition("_")
    if not separator or not video or not frame.isascii() or not frame.isdecimal():
        raise ValueError(f"Expected a video ID followed by _<frame>: {sample_id!r}")
    return video


def real_source(root: Path) -> TennisCourtDetectorInput:
    return TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            "tennis_court_detector",
            root,
            {"train": "train", "val": "val", "test": None},
            EXCLUDED_SAMPLE_IDS,
        )
    )


def make_registry(
    source: TennisCourtDetectorInput,
    *,
    version: str,
    seed: int,
    val_fraction: float = 0.5,
) -> dict[str, Any]:
    if not version or version != version.strip() or type(seed) is not int or seed < 0:
        raise ValueError("Registry version and nonnegative integer seed are required.")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    records = (*source.records("train"), *source.records("val"))
    videos = sorted({video_id(record.sample_id) for record in records})
    random.Random(seed).shuffle(videos)
    n_val = round(len(videos) * val_fraction)
    if not 0 < n_val < len(videos):
        raise ValueError("Each evaluation split must contain at least one video.")
    val_videos = set(videos[:n_val])
    splits: dict[str, list[str]] = {"val": [], "test": []}
    origins: dict[str, Counter[str]] = {"val": Counter(), "test": Counter()}
    video_sets: dict[str, set[str]] = {"val": set(), "test": set()}
    for record in sorted(records, key=lambda item: item.sample_id):
        video = video_id(record.sample_id)
        split = "val" if video in val_videos else "test"
        splits[split].append(record.sample_id)
        origins[split][str(record.payload["source_split"])] += 1
        video_sets[split].add(video)
    if video_sets["val"] & video_sets["test"]:
        raise ValueError("Evaluation video leakage detected.")
    payload = {
        "schema": "court_real_video_evaluation_v1",
        "version": version,
        "seed": seed,
        "val_fraction": val_fraction,
        "excluded_sample_ids": list(EXCLUDED_SAMPLE_IDS),
        "source_sha256": {
            name: file_sha256(source.root / name)
            for name in ("dataset.json", "index.npz")
        },
        "splits": splits,
        "summary": {
            split: {
                "images": len(splits[split]),
                "videos": len(video_sets[split]),
                "original_split_images": dict(sorted(origins[split].items())),
            }
            for split in splits
        },
        "video_overlap": 0,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


class TennisCourtEvaluationInput(TennisCourtDetectorInput):
    """Expose only val/test while retaining the original JPEG and sparse labels."""

    def __init__(self, config: TennisCourtEvaluationSourceConfig) -> None:
        source = real_source(config.root)
        registry = json.loads(config.registry.read_text())
        if not isinstance(registry, dict):
            raise ValueError("Evaluation registry must be an object.")
        expected = make_registry(
            source,
            version=registry["version"],
            seed=registry["seed"],
            val_fraction=registry["val_fraction"],
        )
        if registry != expected:
            raise ValueError(
                "Evaluation registry hash, source, split inventory or video separation changed."
            )
        # Reuse the original loader with immutable records remapped in memory.
        self.config = source.config
        self.image_store = source.image_store
        self.root = source.root
        self._spec = source.spec
        self.registry = registry
        lookup = {
            record.sample_id: record
            for split in source.available_splits
            for record in source.records(split)
        }
        self._records = {
            split: tuple(
                replace(lookup[sample_id], split=split)
                for sample_id in registry["splits"][split]
            )
            for split in ("val", "test")
        }

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
        if split == "train":
            raise ValueError("Real evaluation data must never be used for training.")
        return self._records[split]
