from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    MissingCourtKeypointMetadataError,
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.model_io.adapters import PLCSModelIOAdapter
from src.tasks.plcs.model_io.contracts import PLCSInputProfile
from src.tasks.plcs.model_io.court_keypoint_checkpoint import (
    prepare_plcs_checkpoint_court_keypoint_config,
    write_plcs_checkpoint_court_keypoints,
)


class _Model(nn.Module):
    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        return {}


def _ready() -> dict[str, object]:
    return {
        "human_kp": torch.full((1, 2, 3, 17, 2), 0.5),
        "court_kp": torch.full((1, 2, 3, 20, 2), 0.5),
        "human_vis": torch.ones(1, 2, 3, 17),
        "padding_mask": torch.zeros(1, 2, 3, dtype=torch.bool),
        "court_vis": torch.ones(1, 2, 3, 20),
    }


def test_checkpoint_marker_and_saved_selector_must_match_exactly() -> None:
    v2 = resolve_court_keypoint_contract("camera_view_v2")
    checkpoint: dict[str, object] = {}
    write_plcs_checkpoint_court_keypoints(checkpoint, v2)
    config = OmegaConf.create({"court_keypoints": {"selector": "camera_view_v2"}})
    resolved, restored = prepare_plcs_checkpoint_court_keypoint_config(
        checkpoint,
        config,
        v2,
    )
    assert resolved.court_keypoints.selector == "camera_view_v2"
    assert restored == v2

    wrong_config = OmegaConf.create({"court_keypoints": {"selector": "physical_v1"}})
    with pytest.raises(CourtKeypointContractMismatchError):
        prepare_plcs_checkpoint_court_keypoint_config(
            checkpoint,
            wrong_config,
            v2,
        )


def test_metadata_free_checkpoint_requires_explicit_physical_runtime() -> None:
    legacy_config = OmegaConf.create({"model": {"name": "legacy"}})
    with pytest.raises(MissingCourtKeypointMetadataError):
        prepare_plcs_checkpoint_court_keypoint_config({}, legacy_config, None)

    resolved, contract = prepare_plcs_checkpoint_court_keypoint_config(
        {},
        legacy_config,
        resolve_court_keypoint_contract("physical_v1"),
    )
    assert resolved.court_keypoints.selector == "physical_v1"
    assert contract.selector == "physical_v1"


def test_v2_direct_model_input_rejects_missing_unknown_and_mixed_context() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    adapter = PLCSModelIOAdapter(
        model_type=_Model,
        profile=PLCSInputProfile.MULTIVIEW,
        num_court_tokens=20,
        camera_index=0,
        output_rank=3,
        predict_canonical_pose=False,
        predict_auxiliary_position=False,
        court_keypoint_contract=contract,
    )
    ready = _ready()
    with pytest.raises(ValueError, match="metadata is absent"):
        adapter.build_call(ready)

    negative = build_court_view_record(
        camera_id="camera_0",
        camera_center_court_m=(0.0, -10.0, 3.0),
        contract=contract,
    )
    positive = build_court_view_record(
        camera_id="camera_1",
        camera_center_court_m=(0.0, 10.0, 3.0),
        contract=contract,
    )
    provenance = build_reference_frame_provenance(
        (negative, positive),
        reference_camera_id="camera_1",
    )
    ready["court_keypoint_metadata"] = court_keypoint_contract_document(contract)
    ready["court_reference_provenance"] = provenance
    adapter.build_call(ready)

    unknown = court_keypoint_contract_document(contract)
    unknown["court_keypoints"] = {
        **cast("dict[str, object]", unknown["court_keypoints"]),
        "contract_id": "unknown_courtkp20",
    }
    ready["court_keypoint_metadata"] = unknown
    with pytest.raises(ValueError, match="Unknown court keypoint contract ID"):
        adapter.build_call(ready)


def test_v2_direct_scene_requires_explicit_stable_reference_identity() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    adapter = PLCSModelIOAdapter(
        model_type=_Model,
        profile=PLCSInputProfile.MULTIVIEW,
        num_court_tokens=20,
        camera_index=0,
        output_rank=3,
        predict_canonical_pose=False,
        predict_auxiliary_position=False,
        court_keypoint_contract=contract,
    )
    view = build_court_view_record(
        camera_id="camera_0",
        camera_center_court_m=(0.0, -10.0, 3.0),
        contract=contract,
    )
    camera = SimpleNamespace(
        human_kp_uv=np.full((2, 17, 2), 0.5, dtype=np.float32),
        human_kp_vis=np.ones((2, 17), dtype=np.bool_),
        court_kp_uv=np.full((2, 20, 2), 0.5, dtype=np.float32),
        court_kp_vis=np.ones((2, 20), dtype=np.bool_),
        court_view=view,
    )
    scene = SimpleNamespace(
        cameras=[camera],
        court_keypoint_contract=contract,
    )

    with pytest.raises(ValueError, match="reference_camera_id"):
        adapter.prepare_scene(scene, [0])

    prepared = adapter.prepare_scene(
        scene,
        [0],
        reference_camera_id="camera_0",
    )
    assert prepared.court_reference_provenance is not None
    assert prepared.court_reference_provenance[0].reference_camera_id == "camera_0"
