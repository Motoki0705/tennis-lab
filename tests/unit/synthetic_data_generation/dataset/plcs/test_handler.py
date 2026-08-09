"""Production-device policy tests for the PLCS stage boundary."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court_assignment import CourtAssignment
from src.synthetic_data_generation.dataset.plcs.execution import PLCSExecutionBackend
from src.synthetic_data_generation.dataset.plcs.handler import (
    PLCSObjectRequest,
    PLCSStageParameters,
    _build_scene_inventory,
    _validate_execution_backend,
    _validate_staged_court_inventory,
)
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.timeline import PLCSObjectTrack
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)
from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    MotionCategory,
    PLCSMotionClip,
)


def _parameters(model_root: Path, *, device: str) -> PLCSStageParameters:
    return PLCSStageParameters(
        seed=7,
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        split="train",
        scene_splits={"B00": "train"},
        objects=(
            PLCSObjectRequest(
                category=MotionCategory.RUNNING,
                start_frame=0,
                anchor_position_court_m=(0.0, 0.0, 0.0),
                yaw_radians=0.0,
            ),
        ),
        smplh_model_root=model_root,
        gaussian_count=32,
        smplh_batch_size=8,
        device=device,
    )


def test_production_parameters_reject_cpu_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit CUDA"):
        _parameters(tmp_path, device="cpu")

    configured = _parameters(tmp_path, device="cuda:0")
    assert configured.smplh_batch_size == 8


def test_handler_parameters_reject_mode_cardinality_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two objects"):
        PLCSStageParameters(
            seed=7,
            production_mode=PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
            split="train",
            scene_splits={"B00": "train"},
            objects=(
                PLCSObjectRequest(
                    category=MotionCategory.RUNNING,
                    start_frame=0,
                    anchor_position_court_m=(0.0, 0.0, 0.0),
                    yaw_radians=0.0,
                ),
            ),
            smplh_model_root=tmp_path,
            gaussian_count=32,
            smplh_batch_size=8,
            device="cuda:0",
        )


def test_handler_builds_one_object_full_source_inventory(tmp_path: Path) -> None:
    source = tmp_path / "running.npz"
    source.touch()
    clip = PLCSMotionClip.from_amass_arrays(
        source_path=source,
        category="running",
        gender="neutral",
        fps=30.0,
        poses=np.zeros((5, 156), dtype=np.float64),
        trans=np.zeros((5, 3), dtype=np.float64),
        betas=np.zeros(16, dtype=np.float64),
    )
    court = CourtInstance(
        court_instance_id="court-000",
        candidate_id="candidate-000",
        scene_from_court=RigidTransform.identity(),
        court_from_scene=RigidTransform.identity(),
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.01},
    )
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-10.0, -20.0, -2.0, 10.0, 20.0, 10.0),
        primary_court_instance_id="court-000",
    )
    inventory = _build_scene_inventory(
        dataset_scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        assignments=(
            CourtAssignment(
                scene_id="B00",
                split="train",
                court_instance_id="court-000",
                candidate_id="candidate-000",
                selection_seed=7,
            ),
        ),
        layout=layout,
        tracks=(
            PLCSObjectTrack(
                object_id="player-001",
                instance_id=1,
                asset_id="avatar-001",
                clip=clip,
                start_frame=0,
                anchor_position_court_m=(0.0, 0.0, 0.0),
                yaw_radians=0.0,
            ),
        ),
        required_motion_categories=frozenset({"running"}),
    )

    timeline = inventory.scenes[0].timeline
    assert timeline.production_mode is PLCSProductionMode.SINGLE_OBJECT
    assert timeline.frame_count == clip.frame_count
    assert inventory.aggregate_source_frame_count == clip.frame_count
    assert inventory.aggregate_global_frame_count == clip.frame_count


class _ExplicitTestCPUBackend:
    execution_device = "test-cpu-oracle"
    torch_device = torch.device("cpu")
    cuda_peak_bytes = 0
    background_upload_count = 0


def test_cpu_backend_marker_requires_explicit_nonproduction_budget() -> None:
    backend = cast(PLCSExecutionBackend, _ExplicitTestCPUBackend())

    _validate_execution_backend(
        backend,
        configured_device="test-cpu-oracle",
        require_cuda=False,
    )
    with pytest.raises(ValueError, match="non-CUDA test budget"):
        _validate_execution_backend(
            backend,
            configured_device="test-cpu-oracle",
            require_cuda=True,
        )


def test_staged_manifest_rejects_missing_accepted_court(tmp_path: Path) -> None:
    courts = []
    for index, x_value in enumerate((0.0, 30.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x_value
        transform = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index:03d}",
                candidate_id=f"candidate-{index:03d}",
                scene_from_court=transform,
                court_from_scene=transform.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.01},
            )
        )
    layout = MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-10.0, -20.0, -2.0, 40.0, 20.0, 10.0),
        primary_court_instance_id="court-000",
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    binding = TargetCourtBinding(
        court_instance_id="court-000",
        candidate_id="candidate-000",
        scene_from_court=courts[0].scene_from_court,
        selection_seed=7,
    )
    (staging / "dataset.json").write_text(
        json.dumps(
            {
                "metadata": {"accepted_court_instance_ids": ["court-000"]},
                "target_courts": [binding.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not cover"):
        _validate_staged_court_inventory(staging, layout=layout)
