"""Concrete Lightning hook coverage for immutable reference provenance."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.blcs.model_io import BLCSReferenceMetadata
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.plcs.model_io import PLCSReferenceMetadata
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)

ReferenceMetadata: TypeAlias = BLCSReferenceMetadata | PLCSReferenceMetadata


def _selection() -> ReferenceViewSelection:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(0.0, -12.0, 4.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(0.0, 12.0, 4.0),
            contract=contract,
        ),
    )
    return ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(
            tuple(view.camera_id for view in views)
        ),
        selected_views=views,
        reference_camera_id="camera_1",
    )


def _metadata(
    metadata_type: type[BLCSReferenceMetadata] | type[PLCSReferenceMetadata],
) -> ReferenceMetadata:
    selection = _selection()
    fields = selection.to_tensor_fields()
    reference_from_physical = fields["reference_from_physical"].unsqueeze(0)
    return metadata_type(
        selections=(selection,),
        stable_camera_id_tables=(selection.stable_camera_id_table,),
        reference_view_index=fields["reference_view_index"].unsqueeze(0),
        view_camera_ids=fields["view_camera_ids"].unsqueeze(0),
        reference_camera_id=fields["reference_camera_id"].unsqueeze(0),
        reference_from_physical=reference_from_physical,
        physical_from_reference=reference_from_physical.transpose(-1, -2),
    )


def _blcs_metadata() -> ReferenceMetadata:
    return _metadata(BLCSReferenceMetadata)


def _plcs_metadata() -> ReferenceMetadata:
    return _metadata(PLCSReferenceMetadata)


@pytest.mark.parametrize(
    ("module_type", "metadata_factory"),
    [
        (BLCSLightningModule, _blcs_metadata),
        (BLCSTrackingLightningModule, _blcs_metadata),
        (PLCSLightningModule, _plcs_metadata),
        (PLCSTrackingLightningModule, _plcs_metadata),
    ],
    ids=("blcs-ordinary", "blcs-tracking", "plcs-ordinary", "plcs-tracking"),
)
def test_reference_batches_use_frozen_metadata_aware_lightning_transfer_hook(
    module_type: type[BaseLightningModule],
    metadata_factory: Callable[[], ReferenceMetadata],
) -> None:
    metadata = metadata_factory()
    batch = metadata.to_batch_fields()
    batch.update(
        {
            "model_input": torch.ones(2),
            "reference_metadata": metadata,
            "court_reference_provenance": tuple(
                selection.provenance for selection in metadata.selections
            ),
        }
    )
    module = object.__new__(module_type)
    pl.LightningModule.__init__(module)

    moved = cast(
        "dict[str, object]",
        module._apply_batch_transfer_handler(batch, torch.device("meta"), 0),
    )

    for key in (
        "model_input",
        "reference_view_index",
        "view_camera_ids",
        "reference_camera_id",
        "reference_from_physical",
        "physical_from_reference",
    ):
        value = moved[key]
        assert isinstance(value, Tensor)
        assert value.device.type == "meta"

    moved_metadata = moved["reference_metadata"]
    assert isinstance(moved_metadata, (BLCSReferenceMetadata, PLCSReferenceMetadata))
    assert moved_metadata.reference_view_index.device.type == "cpu"
    assert moved_metadata.reference_camera_ids == metadata.reference_camera_ids
    moved_selections = moved["reference_view_selection"]
    assert isinstance(moved_selections, tuple)
    assert moved_selections[0].reference_camera_id == "camera_1"
