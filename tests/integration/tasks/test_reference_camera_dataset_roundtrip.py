"""One-scene artifact-to-model reference metadata round-trip for both tasks."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from src.tasks.base.data import (
    ReferenceViewSelection,
    StableCameraIdTable,
    validate_reference_view_batch,
)
from src.tasks.base.generate_dataset import (
    COURT_VIEW_METADATA_KEY,
    CourtKeypointArtifactMetadata,
    CourtKeypointContractMismatchError,
    build_court_view_record,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
    resolve_court_keypoint_contract,
    validate_dataset_court_keypoint_contract_documents,
)


@pytest.mark.parametrize(
    "dataset_schema_id",
    ["blcs_generated_dataset_v2", "plcs_generated_dataset_v2"],
)
def test_one_scene_camera_view_artifact_round_trips_to_model_ready_fields(
    tmp_path,
    dataset_schema_id: str,
) -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    metadata = CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id=dataset_schema_id,
    )
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(1.0, -12.0, 5.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(-1.0, 12.0, 5.0),
            contract=contract,
        ),
    )
    root_document = inject_court_keypoint_artifact_metadata(
        {"dataset_schema_id": dataset_schema_id},
        metadata,
        location="dataset/meta.json",
    )
    scene_document = inject_scene_court_keypoint_metadata(
        {"scene_id": "scene_000000", "num_cameras": 2},
        metadata,
        views,
        location="dataset/scenes/scene_000000/meta.json",
    )
    root_path = tmp_path / "meta.json"
    scene_path = tmp_path / "scenes" / "scene_000000" / "meta.json"
    scene_path.parent.mkdir(parents=True)
    root_path.write_text(json.dumps(root_document), encoding="utf-8")
    scene_path.write_text(json.dumps(scene_document), encoding="utf-8")

    loaded_root = json.loads(root_path.read_text(encoding="utf-8"))
    loaded_scene = json.loads(scene_path.read_text(encoding="utf-8"))
    artifact = validate_dataset_court_keypoint_contract_documents(
        root_metadata=loaded_root,
        scene_metadata={"scene_000000": loaded_scene},
        runtime_contract=contract,
        expected_dataset_schema_id=dataset_schema_id,
    )
    loaded_views = artifact.scenes[0].court_views
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        tuple(view.camera_id for view in loaded_views)
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=(loaded_views[1], loaded_views[0]),
        reference_camera_id="camera_1",
    )
    fields = selection.to_tensor_fields(dtype=torch.float32)
    reference_from_physical = fields["reference_from_physical"].unsqueeze(0)

    validate_reference_view_batch(
        reference_view_index=fields["reference_view_index"].unsqueeze(0),
        view_camera_ids=fields["view_camera_ids"].unsqueeze(0),
        reference_camera_id=fields["reference_camera_id"].unsqueeze(0),
        stable_camera_id_table=table,
        reference_from_physical=reference_from_physical,
        physical_from_reference=reference_from_physical.transpose(-1, -2),
        expected_device="cpu",
    )
    assert selection.reference_camera_id == "camera_1"
    assert selection.reference_view_index == 0
    assert selection.selected_camera_ids == ("camera_1", "camera_0")


def test_artifact_roundtrip_rejects_camera_record_from_mixed_schema() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    metadata = CourtKeypointArtifactMetadata.from_contract(
        contract,
        dataset_schema_id="blcs_generated_dataset_v2",
    )
    views = (
        build_court_view_record(
            camera_id="cam_0",
            camera_center_court_m=(0.0, -5.0, 3.0),
            contract=contract,
        ),
    )
    root = inject_court_keypoint_artifact_metadata({}, metadata, location="root")
    scene = inject_scene_court_keypoint_metadata(
        {}, metadata, views, location="scene"
    )
    mutated = deepcopy(scene)
    camera_records = mutated[COURT_VIEW_METADATA_KEY]
    assert isinstance(camera_records, list)
    camera_records[0]["contract_id"] = "physical_courtkp20_v1"

    with pytest.raises(CourtKeypointContractMismatchError):
        validate_dataset_court_keypoint_contract_documents(
            root_metadata=root,
            scene_metadata={"scene": mutated},
            runtime_contract=contract,
            expected_dataset_schema_id="blcs_generated_dataset_v2",
        )
