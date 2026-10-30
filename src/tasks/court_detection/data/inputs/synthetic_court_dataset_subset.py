"""Read ordered train/validation subsets of one verified source publication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.dataset.court.scale.publication import contained
from src.tasks.court_detection.configuration import (
    SyntheticCourtDatasetSubsetSourceConfig,
    SyntheticCourtDatasetVersionSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtInputSpec,
    CourtRawSample,
    CourtSampleRecord,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.inputs.synthetic_court_dataset_version import (
    SyntheticCourtDatasetVersionInput,
)
from src.tasks.court_detection.scaling.real_split import canonical_sha256
from src.utils.data.image_record_store import file_sha256


def read_subset(path: Path, split: CourtSourceSplit) -> tuple[dict[str, Any], Path]:
    document = json.loads(path.read_text())
    keys = {
        "schema",
        "source_manifest",
        "source_manifest_sha256",
        "selection_seed",
        "split",
        "samples",
    }
    if (
        not isinstance(document, dict)
        or set(document) != keys
        or document["schema"] != "court_dataset_subset_v1"
        or document["split"] != split
        or split not in {"train", "val"}
        or type(document["selection_seed"]) is not int
        or document["selection_seed"] < 0
    ):
        raise ValueError("Invalid Court dataset subset manifest or split.")
    source = contained(path.resolve(strict=True).parent, document["source_manifest"])
    if file_sha256(source) != document["source_manifest_sha256"]:
        raise ValueError("Subset source manifest hash changed.")
    rows = document["samples"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("A dataset subset requires nonempty ordered samples.")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"scene_id", "sample_id"}
            or any(
                not isinstance(v, str) or not v or v != v.strip() or ":" in v
                for v in row.values()
            )
        ):
            raise ValueError("Subset samples require explicit scene_id/sample_id.")
    return document, source


class SyntheticCourtDatasetSubsetInput:
    """Validate the pool and reject sample or trajectory leakage between splits."""

    def __init__(self, config: SyntheticCourtDatasetSubsetSourceConfig) -> None:
        self.config = config
        train, source_path = read_subset(config.manifest, "train")
        val, val_source_path = read_subset(config.validation_manifest, "val")
        if (
            source_path != val_source_path
            or train["selection_seed"] != val["selection_seed"]
        ):
            raise ValueError(
                "Train/val subsets must use the same pool and selection seed."
            )
        self.source = SyntheticCourtDatasetVersionInput(
            SyntheticCourtDatasetVersionSourceConfig(
                "synthetic_court_dataset_version", "v3", "target_court", source_path
            )
        )
        lookup = {record.sample_id: record for record in self.source.records("train")}
        self._records: dict[CourtSourceSplit, tuple[CourtSampleRecord, ...]] = {}
        seen: set[str] = set()
        groups: dict[tuple[str, str], CourtSourceSplit] = {}
        for split, document in (("train", train), ("val", val)):
            selected: list[CourtSampleRecord] = []
            for row in document["samples"]:
                identity = f"{row['scene_id']}:{row['sample_id']}"
                if identity in seen or identity not in lookup:
                    raise ValueError(f"Duplicate or unknown subset sample: {identity}")
                seen.add(identity)
                record = lookup[identity]
                group = (row["scene_id"], str(record.payload["trajectory_group_id"]))
                if group in groups and groups[group] != split:
                    raise ValueError("Train/val trajectory leakage in dataset subsets.")
                groups[group] = split
                selected.append(replace(record, split=split))
            self._records[split] = tuple(selected)
        self.subsets = {"train": train, "val": val}
        self.subset_manifest_sha256 = {
            "train": file_sha256(config.manifest),
            "val": file_sha256(config.validation_manifest),
        }
        self.subset_version = "subset-" + canonical_sha256(self.subsets)
        self.dataset_version = self.source.dataset_version
        self.version_manifest_sha256 = self.source.version_manifest_sha256
        self.store_roots = self.source.store_roots

    @property
    def spec(self) -> CourtInputSpec:
        return self.source.spec

    @property
    def available_splits(self) -> tuple[CourtSourceSplit, ...]:
        return ("train", "val")

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
        if split not in self._records:
            raise ValueError(f"Subset source has no {split!r} split.")
        return self._records[split]

    def load(self, record: CourtSampleRecord) -> CourtRawSample:
        return self.source.load(record)
