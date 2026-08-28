"""Cross-task CPU integration for the exact reference-camera v2 model call."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    TrackQueryReferenceContract,
    write_track_query_reference_contract,
)
from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.model_io import (
    TrackQueryBoundModelIO,
    compose_blcs_track_query_model_io,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.model_io import PLCSTrackingBoundModelIO, build_plcs_model_io

_BLCS_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()
_PLCS_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def _selections() -> tuple[ReferenceViewSelection, ReferenceViewSelection]:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(2.0, -8.0, 4.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(-2.0, 8.0, 4.0),
            contract=contract,
        ),
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(
        tuple(view.camera_id for view in views)
    )
    return (
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=views,
            reference_camera_id="camera_0",
        ),
        ReferenceViewSelection.create(
            stable_camera_id_table=table,
            selected_views=views,
            reference_camera_id="camera_1",
        ),
    )


def _reference_fields(selector_mode: str) -> dict[str, object]:
    selections = _selections()
    matrices = torch.tensor(
        [selection.provenance.reference_from_physical for selection in selections],
        dtype=torch.float32,
    )
    result: dict[str, object] = {
        "reference_view_selection": selections,
        "stable_camera_id_table": tuple(
            selection.stable_camera_id_table for selection in selections
        ),
        "reference_view_index": torch.tensor([0, 1], dtype=torch.int64),
        "view_camera_ids": torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
        "reference_camera_id": torch.tensor([0, 1], dtype=torch.int64),
        "reference_from_physical": matrices,
        "physical_from_reference": matrices.transpose(-1, -2),
        "court_reference_provenance": tuple(
            selection.provenance for selection in selections
        ),
    }
    write_track_query_reference_contract(
        result,
        TrackQueryReferenceContract.reference_v2(
            ReferenceSelectorMode(selector_mode)
        ),
    )
    return result


def _blcs_binding(profile: str) -> TrackQueryBoundModelIO:
    with initialize_config_dir(config_dir=str(_BLCS_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_queries=2",
                "model.num_stages=4",
                "model.dropout=0.0",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
                "model.cswa.backend=reference",
            ],
        )
    return compose_blcs_track_query_model_io(config)


def _plcs_binding(profile: str) -> PLCSTrackingBoundModelIO:
    with initialize_config_dir(config_dir=str(_PLCS_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_queries=2",
                "model.num_stages=4",
                "model.dropout=0.0",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
                "model.cswa.backend=reference",
            ],
        )
    return cast(
        "PLCSTrackingBoundModelIO",
        build_plcs_model_io(PLCSTrainingConfig.from_config(config)),
    )


@pytest.mark.parametrize(
    ("task", "profile", "selector_mode"),
    [
        ("blcs", "track_query_reference", "reference"),
        ("blcs", "track_query_ablation_d_v2_selector", "reference"),
        ("blcs", "track_query_ablation_d_v2_selector_zero", "selector_zero"),
        ("plcs", "track_query_reference", "reference"),
        ("plcs", "track_query_ablation_d_v2_selector", "reference"),
        ("plcs", "track_query_ablation_d_v2_selector_zero", "selector_zero"),
    ],
)
def test_v2_normal_and_d_use_exact_six_input_cpu_forward_backward(
    task: str,
    profile: str,
    selector_mode: str,
) -> None:
    torch.manual_seed(801)
    reference = _reference_fields(selector_mode)
    binding: TrackQueryBoundModelIO | PLCSTrackingBoundModelIO
    if task == "blcs":
        binding = _blcs_binding(profile)
        batch: dict[str, object] = {
            "ball_uv": torch.rand(2, 2, 2, 2, 2, requires_grad=True),
            "ball_vis": torch.ones(2, 2, 2, 2, dtype=torch.bool),
            "court_kp": torch.rand(2, 2, 2, 14, 2, requires_grad=True),
            "court_vis": torch.ones(2, 2, 2, 14, dtype=torch.bool),
            "padding_mask": torch.zeros(2, 2, 2, dtype=torch.bool),
            **reference,
        }
    else:
        binding = _plcs_binding(profile)
        court_contract = resolve_court_keypoint_contract("camera_view_v2")
        batch = {
            "human_kp": torch.rand(2, 2, 2, 2, 17, 2, requires_grad=True),
            "human_vis": torch.ones(2, 2, 2, 2, 17, dtype=torch.bool),
            "court_kp": torch.rand(2, 2, 2, 14, 2, requires_grad=True),
            "court_vis": torch.ones(2, 2, 2, 14, dtype=torch.bool),
            "padding_mask": torch.zeros(2, 2, 2, dtype=torch.bool),
            "court_keypoint_metadata": court_keypoint_contract_document(
                court_contract
            ),
            **reference,
        }

    forward_parameters = tuple(inspect.signature(binding.model.forward).parameters)
    assert len(forward_parameters) == 6
    assert forward_parameters[-1] == "reference_view_index"
    call = binding.build_call(batch)
    assert tuple(call.kwargs)[-1] == "reference_view_index"
    assert len(call.kwargs) == 6
    output = cast("dict[str, Tensor]", binding.execute_call(call))
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert any(
        parameter.grad is not None
        for parameter in binding.model.parameters()
        if parameter.requires_grad
    )
