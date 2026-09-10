"""Real-clip boundary contracts, independent of detector/model execution."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.base.data.track_query_reference import (
    ReferenceViewSelection,
    StableCameraIdTable,
)
from src.tasks.base.generate_dataset.court_view import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.plcs.model_io.axial_reference import PLCSAxialReferenceIOAdapter
from src.tasks.plcs.model_io.contracts import PLCSInputProfile
from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.reference_pipeline.observations import import_ball, sha256
from src.tennis_scene.reference_pipeline.reconstruction import restore_frames
from src.tennis_scene.reference_pipeline.reference import reference_metadata
from src.tennis_scene.schema import SceneResult


def ball_fixture(tmp_path: Path) -> Path:
    clip = tmp_path / "clip"
    (clip / "outsource").mkdir(parents=True)
    (clip / "media").mkdir()
    (clip / "clip.json").write_text(
        json.dumps(
            {
                "camera_ids": ["cam0", "cam1", "cam2"],
                "video_paths": [f"media/cam{i}.mp4" for i in range(3)],
                "num_frames": 4,
                "width": 100,
                "height": 50,
            }
        )
    )
    for i in range(3):
        video = clip / f"media/cam{i}.mp4"
        video.write_bytes(f"video{i}".encode())
        frames = [
            {
                "frame_index": j,
                "status": s,
                "center_px": None if j == 3 else {"x": 20.0, "y": 10.0},
            }
            for j, s in enumerate(
                ["observed", "interpolated", "occlusion_estimated", "unresolved"]
            )
        ]
        (clip / f"outsource/cam{i}_annotations.json").write_text(
            json.dumps(
                {
                    "schema_version": "video_ball_annotation.v2",
                    "source": {
                        "width": 100,
                        "height": 50,
                        "frame_count": 4,
                        "sha256": sha256(video),
                    },
                    "frames": frames,
                }
            )
        )
    return clip


def test_external_ball_includes_estimates_but_zeros_unresolved(tmp_path: Path) -> None:
    clip = ball_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    import_ball(clip, out)
    result = json.loads((out / "ball_detection_result.json").read_text())
    assert result["visibility"][0] == [True, True, True, False]
    np.testing.assert_allclose(result["ball_uv"][0], [[0.2, 0.2]] * 3 + [[0, 0]])
    metadata = json.loads((out / "ball_import.metadata.json").read_text())
    assert metadata["status"][1][2] == "occlusion_estimated"
    assert metadata["score_semantics"].endswith("not detector confidence")


def test_external_ball_rejects_wrong_video(tmp_path: Path) -> None:
    clip = ball_fixture(tmp_path)
    (clip / "media/cam1.mp4").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="video mismatch"):
        import_ball(clip, tmp_path)


def test_external_ball_rejects_missing_frame(tmp_path: Path) -> None:
    clip = ball_fixture(tmp_path)
    p = clip / "outsource/cam0_annotations.json"
    d = json.loads(p.read_text())
    d["frames"].pop()
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="frame IDs"):
        import_ball(clip, tmp_path)


def test_pose_only_scene_archive_has_no_fake_smpl(tmp_path: Path) -> None:
    pose: np.ndarray = np.ones((2, 3, 17, 3), np.float32)
    scene = SceneResult(
        3,
        30.0,
        100,
        50,
        np.zeros((3, 3, 14, 2), np.float32),
        np.ones((3, 3, 14), np.float32),
        np.zeros((2, 3, 3), np.float32),
        np.zeros((2, 3), np.float32),
        player_canonical_pose=pose,
        player_kp_3d=pose,
        metadata={"smpl_available": False},
    )
    p = tmp_path / "scene.npz"
    save_scene_result(scene, p)
    restored = load_scene_result(p)
    assert restored.smpl_body_pose is None
    np.testing.assert_array_equal(restored.player_canonical_pose, pose)
    with np.load(p, allow_pickle=False) as arrays:
        assert "smpl_body_pose" not in arrays.files


def test_multiplayer_reference_metadata_reaches_numpy_adapter() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = tuple(
        build_court_view_record(
            camera_id=f"cam{i}",
            camera_center_court_m=(0.0, -20.0 if i < 2 else 20.0, 3.0),
            contract=contract,
        )
        for i in range(3)
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(("cam0", "cam1", "cam2"))
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table, selected_views=views, reference_camera_id="cam1"
    )
    metadata = reference_metadata(selection, 2, "plcs")
    document: dict[str, object] = {}
    write_model_artifact_court_keypoint_contract(document, contract)
    adapter = PLCSAxialReferenceIOAdapter(
        model_type=torch.nn.Module,
        profile=PLCSInputProfile.MULTIVIEW,
        num_court_tokens=14,
        camera_index=0,
        output_rank=3,
        predict_canonical_pose=True,
        predict_auxiliary_position=False,
        court_keypoint_contract=contract,
    )
    prepared = adapter.prepare_multiview_observations(
        human_kp=np.full((2, 3, 4, 17, 2), 0.5, np.float32),
        court_kp=np.full((3, 4, 14, 2), 0.5, np.float32),
        human_vis=np.ones((2, 3, 4, 17), bool),
        court_vis=np.ones((3, 4, 14), bool),
        padding_mask=np.zeros((2, 3, 4), bool),
        court_keypoint_metadata=document,
        court_reference_provenance=tuple(x.provenance for x in metadata.selections),
    )
    assert prepared.call.kwargs["reference_view_index"].tolist() == [1, 1]


def test_restore_time_including_final_half_step() -> None:
    values = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    result = restore_frames(values, np.array([0, 2, 4]), 6)
    np.testing.assert_allclose(result[:, 0], [0, 1, 2, 3, 4, 4])


def test_manual_court_rejects_missing_points_instead_of_fabricating_visibility(
    tmp_path: Path,
) -> None:
    from src.tennis_scene.reference_pipeline.observations import court_homographies

    clip = ball_fixture(tmp_path)
    (clip / "annotations").mkdir()
    data = {
        "keypoints": np.full((3, 4, 14, 2), 0.5).tolist(),
        "visibility": np.ones((3, 4, 14)).tolist(),
        "frame_indices": list(range(4)),
    }
    data["visibility"][1][0][2] = 0
    (clip / "annotations/manual_court_kp_result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="all 14 points visible"):
        court_homographies(clip)
