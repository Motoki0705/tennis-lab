"""Adapt a verified scale dataset version to the V3 training geometry pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
)
from src.tasks.court_detection.configuration import (
    SyntheticCourtDatasetVersionSourceConfig,
)
from src.tasks.court_detection.data.contracts import CourtSampleRecord, CourtSourceSplit
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.utils.data.image_record_store import file_sha256

_DATASET_KEYS = {
    "schema",
    "status",
    "scene_id",
    "profile",
    "scale_schema",
    "seed",
    "metadata_fields",
    "trajectory_groups",
    "rejected_samples",
    "metrics",
    "samples",
    "storage",
}


class SyntheticCourtDatasetVersionInput(SyntheticCourtInput):
    """Read only the exact accepted membership of one explicit train publication."""

    _additional_sample_keys = frozenset({"scale", "qa"})

    def __init__(self, config: SyntheticCourtDatasetVersionSourceConfig) -> None:
        super().__init__(config)

    def _load_manifests(self) -> dict[CourtSourceSplit, tuple[CourtSampleRecord, ...]]:
        from src.synthetic_data_generation.dataset.court.scale.publication import (
            contained,
            load_dataset_version,
        )

        config = cast(SyntheticCourtDatasetVersionSourceConfig, self.config)
        self.dataset_version = load_dataset_version(config.manifest, verify_images=True)
        self.version_manifest_sha256 = file_sha256(config.manifest)
        version_root = config.manifest.resolve(strict=True).parent
        roots = []
        records: list[CourtSampleRecord] = []
        global_ids: set[str] = set()
        for scene in self.dataset_version["scenes"]:
            accepted = []
            for entry in scene["stores"]:
                root = contained(version_root, entry["path"], directory=True)
                roots.append(root)
                store = open_court_store(root)
                self._image_stores[root] = store
                manifest = read_court_manifest(root, store=store)
                if set(manifest) != _DATASET_KEYS:
                    raise ValueError("Scale Court dataset metadata fields changed.")
                if manifest["seed"] != self.dataset_version["generation_seed"]:
                    raise ValueError(
                        "Scale store seed disagrees with its dataset version."
                    )
                for value in manifest["samples"]:
                    record = self._manifest_record(
                        value,
                        root=root,
                        scene_id=scene["scene_id"],
                        manifest_digest=entry["dataset_sha256"],
                    )
                    if record.split != "train" or record.sample_id in global_ids:
                        raise ValueError(
                            "Scale dataset requires unique train sample IDs."
                        )
                    global_ids.add(record.sample_id)
                    accepted.append(value["sample_id"])
                    records.append(record)
            # The shared validator checks membership against both the QA ledger and
            # candidate geometry. Keep the consumer's inventory bound to those counts.
            if len(accepted) != scene["counts"]["qa_accepted"]:
                raise ValueError(
                    "Scale training inventory disagrees with accepted count."
                )
        self.store_roots = tuple(roots)
        return {"train": tuple(records), "val": (), "test": ()}

    def _camera_source_frame_index(self, value: Mapping[str, object]) -> int:
        # Scale cameras are synthetic, while sample_index is a stable-ID prefix.
        return 0


__all__ = ["SyntheticCourtDatasetVersionInput"]
