"""Unit contracts for source-neutral dense Court target materialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtInputCapability,
    CourtInputSpec,
    CourtInstance2D,
    CourtRawSample,
    CourtSampleMetadata,
    CourtSampleRecord,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.target_generation.materializer import (
    CourtTargetMaterializer,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
    validate_derived_target,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def test_materializer_writes_both_dense_targets_below_derived_store(
    tmp_path: Path,
) -> None:
    store = CourtDerivedTargetStore(tmp_path / "derived")
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    image_points = torch.stack(
        (
            (points[:, 0] / 12.0 + 0.5) * 63.0,
            (0.5 - points[:, 1] / 26.0) * 47.0,
        ),
        dim=1,
    )
    target_specs: tuple[tuple[CourtDenseTargetKind, str], ...] = (
        ("seg", "court_cell_segmentation_v1"),
        ("line", "court_line_binary_v1"),
    )
    refs: dict[CourtDenseTargetKind, Path] = {
        kind: store.path_for(
            source_kind="tennis_court_detector",
            derived_key="train/sample",
            target_schema=schema,
        )
        for kind, schema in target_specs
    }
    record = CourtSampleRecord(
        sample_id="sample",
        split="train",
        image_path=tmp_path / "source.png",
        annotation_path=tmp_path / "source.json",
        derived_key="train/sample",
        dense_target_refs=refs,
        payload={
            "source_schema": "fixture",
            "source_sample_id": "sample",
            "source_target_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "width": 64,
            "height": 48,
        },
    )
    instance = CourtInstance2D(
        court_instance_id="court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=image_points,
        point_visible=torch.ones(14, dtype=torch.bool),
    )
    raw = CourtRawSample(
        sample_id="sample",
        image=Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)),
        keypoint_channels=None,
        court_instances=(instance,),
        dense_target_refs=refs,
        metadata=CourtSampleMetadata(
            source_kind="tennis_court_detector",
            source_schema="fixture",
            source_sample_id="sample",
            scene_id=None,
            provenance={},
        ),
    )

    class _Input:
        spec = CourtInputSpec(
            source_kind="tennis_court_detector",
            source_schema="fixture",
            capabilities=frozenset({CourtInputCapability.COURT_INSTANCES}),
        )

        available_splits: tuple[CourtSourceSplit, ...] = ("train",)

        def records(
            self, split: CourtSourceSplit
        ) -> tuple[CourtSampleRecord, ...]:
            assert split == "train"
            return (record,)

        def load(self, selected: CourtSampleRecord) -> CourtRawSample:
            assert selected is record
            return raw

    results = CourtTargetMaterializer(
        input_layer=_Input(),
        target_store=store,
    ).materialize(splits=("train",), target_kinds=("seg", "line"))

    assert [(result.target_kind, result.written) for result in results] == [
        ("seg", 1),
        ("line", 1),
    ]
    for kind, path in refs.items():
        assert path.is_file()
        metadata = json.loads(store.metadata_path(path).read_text(encoding="utf-8"))
        assert metadata["target_kind"] == kind
        assert metadata["stable_sample_id"] == "sample"
        assert metadata["source_target_sha256"] == hashlib.sha256(
            b"fixture"
        ).hexdigest()
        assert path.is_relative_to(store.root)
        validate_derived_target(
            record,
            input_spec=_Input.spec,
            target_kind=kind,
            target_schema=cast(str, metadata["schema"]),
        )

        stale = replace(
            record,
            payload={**record.payload, "source_target_sha256": "0" * 64},
        )
        with pytest.raises(ValueError, match="stale"):
            validate_derived_target(
                stale,
                input_spec=_Input.spec,
                target_kind=kind,
                target_schema=cast(str, metadata["schema"]),
            )
