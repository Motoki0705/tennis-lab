"""Cross-domain CPU checks for shared camera, court, and timeline contracts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import yaml

from src.synthetic_data_generation.dataset.camera_profiles import (
    CameraProfileConfig,
    assert_projection_equivalent,
    sample_camera_rig,
)
from src.synthetic_data_generation.dataset.continuity import (
    TimelineFrameRecord,
    validate_frame_continuity,
)
from src.synthetic_data_generation.dataset.court_assignment import (
    assign_courts_balanced,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)
from src.utils.paths import PROJECT_ROOT


def _layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((0.0, 30.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        scene_from_court = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.01},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -30.0, -2.0, 50.0, 30.0, 20.0),
        primary_court_instance_id="court-0",
    )


def _profile(name: str) -> CameraProfileConfig:
    path = (
        PROJECT_ROOT
        / "src/synthetic_data_generation/configs/camera"
        / f"{name}.yaml"
    )
    return CameraProfileConfig.from_mapping(yaml.safe_load(path.read_text()))


def test_config_owned_camera_profiles_are_deterministic_and_projection_equivalent() -> None:
    court = _layout().court("court-1")
    default = _profile("default")
    broadcast = _profile("broadcast")

    first = sample_camera_rig(default, seed=695, court=court)
    repeated = sample_camera_rig(default, seed=695, court=court)
    broadcast_rig = sample_camera_rig(broadcast, seed=695, court=court)

    assert len(first.cameras) == 6
    assert len(broadcast_rig.cameras) == 2
    assert [item.to_metadata() for item in first.cameras] == [
        item.to_metadata() for item in repeated.cameras
    ]
    points = np.asarray(
        [[-4.0, -8.0, 0.0], [0.0, 0.0, 1.0], [4.0, 8.0, 0.0]],
        dtype=np.float64,
    )
    for camera in first.cameras:
        assert_projection_equivalent(camera, court, points, atol=1.0e-6)


def test_blcs_and_plcs_share_balanced_target_court_and_continuity_authority() -> None:
    layout = _layout()
    splits = {
        **{f"train-{index}": "train" for index in range(5)},
        **{f"validation-{index}": "validation" for index in range(4)},
        **{f"test-{index}": "test" for index in range(3)},
    }

    blcs = assign_courts_balanced(splits, layout=layout, seed=695)
    plcs = assign_courts_balanced(splits, layout=layout, seed=695)

    assert blcs == plcs
    counts = Counter(item.court_instance_id for item in blcs)
    assert set(counts) == {"court-0", "court-1"}
    assert max(counts.values()) - min(counts.values()) <= 1
    records = tuple(
        TimelineFrameRecord(
            frame_index=frame,
            chunk_index=frame // 2,
            track_id=domain,
            present=True,
            source_frame_index=frame,
            camera_id="camera-0",
            label_id=f"{domain}-{frame}",
            court_instance_id="court-0",
        )
        for domain in ("ball", "player")
        for frame in range(5)
    )

    report = validate_frame_continuity(records, frame_count=5)

    assert report.frame_count == 5
    assert report.track_count == 2
    mismatched = list(records)
    mismatched[-1] = TimelineFrameRecord(
        frame_index=4,
        chunk_index=2,
        track_id="player",
        present=True,
        source_frame_index=4,
        camera_id="camera-0",
        label_id="player-4",
        court_instance_id="court-1",
    )
    with pytest.raises(ValueError, match="target-court binding changed"):
        validate_frame_continuity(mismatched, frame_count=5)


def test_unknown_camera_field_fails_closed() -> None:
    payload = yaml.safe_load(
        (
            PROJECT_ROOT
            / "src/synthetic_data_generation/configs/camera/broadcast.yaml"
        ).read_text()
    )
    payload["python_default"] = True
    with pytest.raises(ValueError, match="unknown=.*python_default"):
        CameraProfileConfig.from_mapping(payload)
