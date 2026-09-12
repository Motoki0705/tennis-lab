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
from src.tasks.court_detection.data.target_generation.line import generate_line_target
from src.tasks.court_detection.data.target_generation.materializer import (
    CourtTargetMaterializer,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
    validate_derived_target,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    LINE_TARGET_SCHEMA_V1,
    LINE_TARGET_SCHEMA_V2,
    line_target_definition,
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
        ("seg", "court_cell_segmentation_single_court_v2"),
        ("line", LINE_TARGET_SCHEMA),
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
        point_in_front=torch.ones(14, dtype=torch.bool),
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

        def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
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
        assert (
            metadata["source_target_sha256"] == hashlib.sha256(b"fixture").hexdigest()
        )
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


def test_line_target_width_is_explicitly_previewable() -> None:
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    image_points = torch.stack(
        (
            (points[:, 0] / 12.0 + 0.5) * 255.0,
            (0.5 - points[:, 1] / 26.0) * 255.0,
        ),
        dim=1,
    )
    instance = CourtInstance2D(
        court_instance_id="court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=image_points,
        point_in_front=torch.ones(14, dtype=torch.bool),
        point_visible=torch.ones(14, dtype=torch.bool),
    )

    narrow = generate_line_target(
        height=256,
        width=256,
        instances=(instance,),
        line_width_metres=0.025,
        baseline_width_metres=0.05,
    )
    wide = generate_line_target(
        height=256,
        width=256,
        instances=(instance,),
        line_width_metres=0.075,
        baseline_width_metres=0.15,
    )

    assert np.count_nonzero(wide) > np.count_nonzero(narrow)


def test_current_dense_schemas_reject_multiple_selected_courts(
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
    first = CourtInstance2D(
        court_instance_id="court-a",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=image_points,
        point_in_front=torch.ones(14, dtype=torch.bool),
        point_visible=torch.ones(14, dtype=torch.bool),
    )
    second = replace(first, court_instance_id="court-b")
    path = store.path_for(
        source_kind="synthetic_court",
        derived_key="target_court/B00/sample",
        target_schema=LINE_TARGET_SCHEMA,
    )
    record = CourtSampleRecord(
        sample_id="B00:sample",
        split="train",
        image_path=tmp_path / "source.png",
        annotation_path=tmp_path / "source.json",
        derived_key="target_court/B00/sample",
        dense_target_refs={"line": path},
        payload={
            "source_schema": "fixture",
            "source_sample_id": "sample",
            "source_target_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "width": 64,
            "height": 48,
        },
    )
    raw = CourtRawSample(
        sample_id=record.sample_id,
        image=Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)),
        keypoint_channels=None,
        court_instances=(first, second),
        dense_target_refs=record.dense_target_refs,
        metadata=CourtSampleMetadata(
            source_kind="synthetic_court",
            source_schema="fixture",
            source_sample_id="sample",
            scene_id="B00",
            provenance={},
        ),
    )

    class _Input:
        spec = CourtInputSpec(
            source_kind="synthetic_court",
            source_schema="fixture",
            capabilities=frozenset({CourtInputCapability.COURT_INSTANCES}),
        )
        available_splits: tuple[CourtSourceSplit, ...] = ("train",)

        def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
            assert split == "train"
            return (record,)

        def load(self, selected: CourtSampleRecord) -> CourtRawSample:
            assert selected is record
            return raw

    with pytest.raises(ValueError, match="exactly one selected court"):
        CourtTargetMaterializer(
            input_layer=_Input(),
            target_store=store,
        ).materialize(splits=("train",), target_kinds=("line",))


def test_line_target_schemas_keep_physical_widths_immutable() -> None:
    legacy = line_target_definition(LINE_TARGET_SCHEMA_V1)
    all_court_wide = line_target_definition(LINE_TARGET_SCHEMA_V2)
    current = line_target_definition(LINE_TARGET_SCHEMA)

    assert (legacy.line_width_metres, legacy.baseline_width_metres) == (0.05, 0.10)
    assert (
        all_court_wide.line_width_metres,
        all_court_wide.baseline_width_metres,
    ) == (0.075, 0.15)
    assert (current.line_width_metres, current.baseline_width_metres) == (0.075, 0.15)
