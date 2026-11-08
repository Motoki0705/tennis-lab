"""Saved v2 scene -> portable mesh/pose/timeline inputs, without source videos."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SCENE_MASK_FIELDS, SCENE_REASON_FIELDS, SceneResult
from src.tennis_scene.scripts.export_player_view_cameras import main as camera_main
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2
from src.tennis_scene.visualization.player_view import (
    BUNDLE_SCHEMA,
    PlayerViewBundleConfig,
    export_player_view_bundle,
)
from src.tennis_scene.visualization.player_view.__main__ import main
from src.tennis_scene.visualization.player_view.blender import (
    PlayerViewRenderConfig,
    prepare_blender_shot,
)
from src.tennis_scene.visualization.player_view.camera_bundle import (
    CAMERA_SCHEMA,
    export_head_cameras,
    load_head_cameras,
)
from src.tennis_scene.visualization.player_view.head_camera import HeadCameraConfig
from src.tennis_scene.visualization.player_view.video import (
    publish_player_views,
    validate_render,
)
from src.utils.checksum import dual_sha256

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Inputs:
    scene: Path
    faces: Path
    regressor: Path
    model: Path


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    players, frames, views = 2, 7, 2
    valid = np.array(
        [
            [False, True, True, False, True, True, True],
            [True, False, True, True, True, False, True],
        ]
    )
    root_valid = valid.copy()
    root_valid[0, 3] = True  # A valid root/heading does not revive a missing mesh.
    ball_valid = np.array([True, True, False, True, True, False, True])
    local = np.tile(
        [[1, 2, 3], [3, 2, 3], [1, 6, 3], [1, 2, 9]], (players, frames, 1, 1)
    ).astype(np.float32)
    local[0] *= 1.5
    local[1] *= 0.75
    orient: np.ndarray = np.zeros((players, frames, 3), np.float32)
    orient[0, :, 2] = np.pi / 2
    orient[1, :, 0] = np.pi / 2
    pose = (
        np.arange(players * frames * 63, dtype=np.float32).reshape(players, frames, 63)
        / 1000
    )
    position = np.array([[[5, -3, 2]] * frames, [[-4, 8, 1]] * frames], np.float32)
    yaw: np.ndarray = np.zeros((players, frames), np.float32)
    yaw[0] = np.pi / 2
    for value in (local, orient, pose):
        value[~valid] = 0
    position[~root_valid] = 0
    yaw[~root_valid] = 0
    ball = np.stack(
        [np.arange(frames), np.arange(frames) * 2, np.ones(frames)], -1
    ).astype(np.float32)
    ball[~ball_valid] = 0
    scene = SceneResult(
        num_frames=frames,
        fps=59.94,
        width=64,
        height=48,
        court_kp=np.zeros((views, frames, 14, 2), np.float32),
        court_vis=np.ones((views, frames, 14), np.float32),
        player_position=position,
        player_yaw=yaw,
        smpl_body_pose=pose,
        smpl_global_orient=orient,
        smpl_betas=np.array([[0.1] * 10, [0.2] * 10], np.float32),
        smpl_vertices_local=local,
        player_kp_3d=np.zeros((players, frames, 17, 3), np.float32),
        human_kp_2d=np.zeros((players, views, frames, 17, 2), np.float32),
        human_kp_vis=np.zeros((players, views, frames, 17), np.float32),
        ball_3d=ball,
        ball_uv=np.zeros((views, frames, 2), np.float32),
        ball_vis=np.ones((views, frames), bool),
        player_track_ids=np.array([7, 42], np.int32),
        player_observed=root_valid.copy(),
        player_valid=root_valid.copy(),
        player_heading_valid=root_valid.copy(),
        player_smpl_valid=valid.copy(),
        player_kp_3d_vis=np.zeros((players, frames, 17), bool),
        ball_3d_valid=ball_valid,
        player_rejection_code=(~root_valid).astype(np.uint8),
        player_kp_3d_rejection_code=np.ones((players, frames, 17), np.uint8),
        ball_rejection_code=(~ball_valid).astype(np.uint8),
        metadata={
            "scene_schema_version": 2,
            "camera_ids": ["cam0", "cam1"],
            "video_paths": ["/unavailable/cam0.mp4", "/unavailable/cam1.mp4"],
            "court_reference": {"reference_id": "cam0"},
            "body_placement": {"players": {"7": {"scale": 1.5}, "42": {"scale": 0.75}}},
        },
    )
    scene_path = tmp_path / "scene.npz"
    save_scene_result(scene, scene_path)
    faces = tmp_path / "faces.npz"
    regressor = tmp_path / "regressor.pt"
    model = tmp_path / "neutral.npz"
    np.savez(faces, f=np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    joint_regressor: np.ndarray = np.zeros((24, 4), np.float32)
    joint_regressor[:, 0] = 1
    joint_regressor[15] = [0, 0, 1, 0]
    torch.save(torch.from_numpy(joint_regressor), regressor)
    np.savez(
        model,
        v_template=np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32),
        J_regressor=joint_regressor[:22],
        shapedirs=np.ones((4, 3, 10), np.float32) / 10,
        kintree_table=np.array(
            [
                [
                    -1,
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    9,
                    9,
                    12,
                    13,
                    14,
                    16,
                    17,
                    18,
                    19,
                ],
                range(22),
            ],
            np.int32,
        ),
    )
    return Inputs(scene_path, faces, regressor, model)


def _export(
    inputs: Inputs,
    output: Path,
    *,
    source: Path | None = None,
    config: PlayerViewBundleConfig | None = None,
) -> Path:
    manifest_path: Path = export_player_view_bundle(
        scene_path=inputs.scene if source is None else source,
        smpl_faces_path=inputs.faces,
        smpl_joint_regressor_path=inputs.regressor,
        smplx_model_path=inputs.model,
        output_dir=output,
        config=PlayerViewBundleConfig(frames=(1, 6)) if config is None else config,
    )
    return manifest_path


def test_blender_input_binds_saved_ids_timeline_masks_and_poses(
    inputs: Inputs, tmp_path: Path
) -> None:
    bundle, cameras, shot = (tmp_path / name for name in ("bundle", "cameras", "shot"))
    _export(inputs, bundle)
    export_head_cameras(
        bundle_dir=bundle, output_dir=cameras, config=HeadCameraConfig()
    )
    before = _snapshot(bundle), _snapshot(cameras)
    prepare_blender_shot(
        bundle_dir=bundle,
        cameras_dir=cameras,
        output_dir=shot,
        config=PlayerViewRenderConfig(),
    )
    manifest = json.loads((shot / "manifest.json").read_text())
    assert manifest["frames"] == {"start": 1, "stop": 6, "source_num_frames": 7}
    assert manifest["fps"] == 59.94
    assert [p["player_id"] for p in manifest["players"]] == [7, 42]
    np.testing.assert_array_equal(np.load(shot / "source_frames.npy"), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        np.load(shot / "player_smpl_valid.npy"),
        [[True, True, False, True, True], [False, True, True, True, False]],
    )
    for index, identity in enumerate((7, 42)):
        saved = load_head_cameras(cameras, player_id=identity)
        with np.load(shot / f"camera_{index}.npz") as arrays:
            np.testing.assert_array_equal(arrays["valid"], saved.series.camera_valid)
            np.testing.assert_allclose(
                arrays["location"], saved.series.camera_to_court[:, :3, 3]
            )
            # Camera-to-world axes: saved CV right/down/forward -> Blender right/up/back.
            np.testing.assert_allclose(
                arrays["rotation"],
                saved.series.camera_to_court[:, :3, :3] * np.array([1, -1, -1]),
            )
    assert before == (_snapshot(bundle), _snapshot(cameras))
    assert manifest["policy"]["human_acceptance"] == "pending"
    with pytest.raises(FileExistsError):
        prepare_blender_shot(
            bundle_dir=bundle,
            cameras_dir=cameras,
            output_dir=shot,
            config=PlayerViewRenderConfig(),
        )


@pytest.mark.parametrize("mismatch", ["scene", "timeline", "id", "pc2"])
def test_blender_input_rejects_mismatched_saved_inputs(
    inputs: Inputs, tmp_path: Path, mismatch: str
) -> None:
    bundle, cameras, shot = (tmp_path / name for name in ("bundle", "cameras", "shot"))
    _export(inputs, bundle)
    export_head_cameras(
        bundle_dir=bundle, output_dir=cameras, config=HeadCameraConfig()
    )
    camera_manifest_path = cameras / "manifest.json"
    if mismatch == "scene":
        values = np.load(bundle / "ball_positions.npy")
        values[0, 0] += 1
        np.save(bundle / "ball_positions.npy", values)
        # Ball is not used to estimate cameras, so shot preparation validates it directly.
        values = np.load(bundle / "player_smpl_valid.npy")
        values[0, 0] = False
        np.save(bundle / "player_smpl_valid.npy", values)
    elif mismatch == "timeline":
        manifest = json.loads(camera_manifest_path.read_text())
        manifest["fps"] = 30
        camera_manifest_path.write_text(json.dumps(manifest))
    elif mismatch == "id":
        manifest = json.loads(camera_manifest_path.read_text())
        manifest["players"][0]["player_id"] = 42
        camera_manifest_path.write_text(json.dumps(manifest))
    else:
        with (bundle / "player_1.pc2").open("ab") as handle:
            handle.write(b"x")
    with pytest.raises(ValueError, match="digest|timeline|IDs"):
        prepare_blender_shot(
            bundle_dir=bundle,
            cameras_dir=cameras,
            output_dir=shot,
            config=PlayerViewRenderConfig(),
        )
    assert not shot.exists()
    assert not list(tmp_path.glob(".shot-*"))


@pytest.fixture
def shot_reports(inputs: Inputs, tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic render audit/PNGs exercise publication, not Blender rendering."""
    bundle, cameras, shot, renders = (
        tmp_path / name for name in ("bundle", "cameras", "shot", "renders")
    )
    _export(inputs, bundle)
    export_head_cameras(
        bundle_dir=bundle,
        output_dir=cameras,
        config=HeadCameraConfig(width=128, height=72),
    )
    prepare_blender_shot(
        bundle_dir=bundle,
        cameras_dir=cameras,
        output_dir=shot,
        config=PlayerViewRenderConfig(),
    )
    valid = np.load(shot / "player_smpl_valid.npy")
    for p, identity in enumerate((7, 42)):
        directory = renders / f"player_{identity}"
        (directory / "frames").mkdir(parents=True)
        saved = load_head_cameras(cameras, player_id=identity)
        rows = []
        for index, source_frame in enumerate(range(1, 6)):
            ok = bool(saved.series.camera_valid[index])
            transform = saved.series.camera_to_court[index].copy()
            transform[:3, :3] *= np.array([1, -1, -1])
            rows.append(
                {
                    "output_index": index,
                    "source_frame": source_frame,
                    "cache_index": index,
                    "camera_valid": ok,
                    "camera_reason": int(saved.series.camera_rejection_code[index]),
                    "camera_matrix_world": transform.tolist() if ok else None,
                    "player_hidden": [
                        not (ok and valid[q, index] and p != q) for q in (0, 1)
                    ],
                    "ball_hidden": not (ok and source_frame in (1, 3, 4)),
                    "ball_position": [source_frame, 2 * source_frame, 1]
                    if source_frame in (1, 3, 4)
                    else None,
                    "pc2_max_error_m": [0, 0],
                }
            )
            frame: np.ndarray = np.full(
                (72, 128, 3), 25 + index * 20 if ok else 0, np.uint8
            )
            assert cv2.imwrite(str(directory / "frames" / f"{index:05d}.png"), frame)
        report = {
            "schema": "tennis_scene_player_view_render_v1",
            "complete": True,
            "package_sha256": dual_sha256(shot / "manifest.json"),
            "player_id": identity,
            "engine": "CYCLES",
            "gpu_devices": [{"name": "synthetic audit fixture"}],
            "motion_blur": False,
            "material": {
                name: {"alpha": 1, "emission_strength": 0}
                for name in ("OpaqueGrayPlayer", "OpaqueGreenBall")
            },
            "frames": rows,
        }
        (directory / "render.json").write_text(json.dumps(report))
    return shot, renders


def test_video_publication_preserves_all_source_frames_and_missing_camera_slots(
    shot_reports: tuple[Path, Path], tmp_path: Path
) -> None:
    shot, renders = shot_reports
    output = tmp_path / "videos"
    result = publish_player_views(package=shot, renders=renders, output=output)
    manifest = json.loads(result.read_text())
    assert manifest["player_ids"] == [7, 42]
    assert manifest["source_frames"] == [1, 2, 3, 4, 5]
    assert manifest["camera_missing_frames"] == {"7": [3], "42": [1, 5]}
    assert manifest["human_acceptance"] == "pending"
    for value in manifest["verification"].values():
        assert value["decoded_frames"] == 5
        assert value["fps"] == pytest.approx(59.94, abs=1e-6)
    assert manifest["verification"]["comparison.mp4"]["width"] == 256
    capture = cv2.VideoCapture(str(output / "comparison.mp4"))
    ok, first = capture.read()
    capture.release()
    assert ok
    # Source frame 1: ID7 is valid/gray, ID42 has a blank missing-camera slot.
    # Independent pixel evidence also catches swapped panels or skipped slots.
    assert float(first[60:70, 50:80].mean()) == pytest.approx(25, abs=3)
    assert float(first[60:70, 178:208].mean()) < 3
    with pytest.raises(FileExistsError):
        publish_player_views(package=shot, renders=renders, output=output)


@pytest.mark.parametrize(
    "fault",
    ["frame", "mask", "camera", "ball", "opacity", "incomplete", "missing-material"],
)
def test_publication_rejects_incorrect_blender_audit(
    shot_reports: tuple[Path, Path], fault: str
) -> None:
    shot, renders = shot_reports
    path = renders / "player_7" / "render.json"
    report = json.loads(path.read_text())
    if fault == "frame":
        report["frames"][0]["source_frame"] += 1
    elif fault == "mask":
        report["frames"][0]["player_hidden"][0] = False
    elif fault == "camera":
        report["frames"][0]["camera_matrix_world"][0][3] += 1
    elif fault == "ball":
        report["frames"][0]["ball_position"][0] += 1
    elif fault == "opacity":
        report["material"]["OpaqueGrayPlayer"]["alpha"] = 0.5
    elif fault == "incomplete":
        report["complete"] = False
    else:
        report["material"].pop("OpaqueGreenBall")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_render(shot, renders / "player_7", 7)


@pytest.mark.parametrize("fault", ["truncated", "missing-png", "bad-png"])
def test_video_publication_refuses_missing_frames_and_cleans_failed_encoding(
    shot_reports: tuple[Path, Path], tmp_path: Path, fault: str
) -> None:
    shot, renders = shot_reports
    output = tmp_path / "videos"
    if fault == "truncated":
        for identity in (7, 42):
            path = renders / f"player_{identity}" / "render.json"
            report = json.loads(path.read_text())
            report["frames"].pop()
            path.write_text(json.dumps(report))
            (path.parent / "frames" / "00004.png").unlink()
    elif fault == "missing-png":
        (renders / "player_7" / "frames" / "00001.png").unlink()
    else:
        (renders / "player_7" / "frames" / "00001.png").write_bytes(b"broken PNG")
    with pytest.raises(ValueError, match="entire input timeline|PNG"):
        publish_player_views(package=shot, renders=renders, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".videos-*"))


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): dual_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _store(inputs: Inputs, root: Path) -> ClipStore:
    scene = load_scene_result(inputs.scene)
    store = ClipStore(root, {"clip": "fixture"})
    codec = ArtifactCodec(SceneResult)
    body = store.publish(
        "body_placement",
        scene,
        codec,
        schema="placed_bodies",
        version=1,
        identity={"revision": 1},
        dependencies={},
        provenance={},
    )
    assembled = store.publish(
        "scene_assembly",
        scene,
        codec,
        schema="scene_result",
        version=2,
        identity={"revision": 1},
        dependencies={"body": body},
        provenance={},
    )
    export_scene(store, scene, assembled)
    return store


def test_saved_scene_bundle_preserves_ids_frames_masks_and_placed_geometry(
    inputs: Inputs, tmp_path: Path
) -> None:
    before = _snapshot(tmp_path)
    output = tmp_path / "bundle"
    manifest = json.loads(_export(inputs, output).read_text())
    assert {k: dual_sha256(tmp_path / k) for k in before} == before
    assert manifest["schema"] == BUNDLE_SCHEMA
    assert (manifest["num_frames"], manifest["fps"]) == (5, 59.94)
    assert manifest["frames"] == {"start": 1, "stop": 6, "source_num_frames": 7}
    assert [
        (p["player_id"], p["index"], p["source_index"], p["scale"])
        for p in manifest["players"]
    ] == [(7, 0, 0, 1.5), (42, 1, 1, 0.75)]
    assert manifest["coordinates"]["units"] == "m"
    assert not (output / "plate").exists()
    assert not list(output.glob(".*"))
    np.testing.assert_array_equal(
        np.load(output / "source_frames.npy"), [1, 2, 3, 4, 5]
    )
    np.testing.assert_allclose(
        np.load(output / "source_time_seconds.npy"), np.arange(1, 6) / 59.94
    )
    np.testing.assert_array_equal(
        np.load(output / "player_smpl_valid.npy"),
        [[True, True, False, True, True], [False, True, True, True, False]],
    )
    np.testing.assert_array_equal(
        np.load(output / "ball_3d_valid.npy"), [True, False, True, True, False]
    )
    np.testing.assert_array_equal(
        np.load(output / "ball_positions.npy"),
        [[1, 2, 1], [0, 0, 0], [3, 6, 1], [4, 8, 1], [0, 0, 0]],
    )
    mesh, start, rate = read_pc2(output / "player_0.pc2")
    assert (start, rate) == (0, 1)
    # Independent hand calculation: inverse encoded Z rotation, Y->Z, yaw,
    # root-centering and translation. The scale is already in the local mesh.
    np.testing.assert_allclose(
        mesh[0], [[5, -3, 2], [5, -3, -1], [5, 3, 2], [14, -3, 2]], atol=2e-6
    )
    assert (mesh[2] == 0).all()  # An isolated gap is not filled.
    assert np.load(output / "player_valid.npy")[0, 2]
    other, _, _ = read_pc2(output / "player_1.pc2")
    np.testing.assert_allclose(
        other[1], [[-4, 8, 1], [-2.5, 8, 1], [-4, 11, 1], [-4, 8, 5.5]], atol=2e-6
    )
    assert (other[[0, 4]] == 0).all()
    np.testing.assert_allclose(
        np.load(output / "body_to_court_rotation.npy")[1, 1], np.eye(3), atol=1e-6
    )
    np.testing.assert_allclose(
        np.load(output / "smpl_root_local.npy")[0, 0], [1.5, 3, 4.5]
    )
    scene = load_scene_result(inputs.scene)
    for field in (
        *SCENE_MASK_FIELDS,
        *SCENE_REASON_FIELDS,
        "smpl_body_pose",
        "smpl_global_orient",
        "player_position",
        "player_yaw",
    ):
        value = getattr(scene, field)
        np.testing.assert_array_equal(
            np.load(output / f"{field}.npy"),
            value[1:6] if field.startswith("ball_") else value[:, 1:6],
        )
    np.testing.assert_array_equal(np.load(output / "smpl_betas.npy"), scene.smpl_betas)
    np.testing.assert_allclose(np.load(output / "smplx_rest_joints.npy")[0], [1, 2, 3])
    np.testing.assert_allclose(np.load(output / "smplx_joint_shapedirs.npy")[0], 0.1)
    assert np.load(output / "smplx_parents.npy")[0] == -1
    existing = _snapshot(output)
    with pytest.raises(FileExistsError):
        _export(inputs, output)
    assert _snapshot(output) == existing


@pytest.mark.parametrize("index_input", [False, True], ids=["store", "scene-json"])
def test_cli_resolves_only_adopted_scene_and_selects_ids(
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    index_input: bool,
) -> None:
    store = _store(inputs, tmp_path / "store")
    before = _snapshot(store.root)
    source = store.index_path if index_input else store.root
    output = tmp_path / "bundle"
    selected_ids = ["42", "7"] if index_input else ["42"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "player-view",
            "--input",
            str(source),
            "--output",
            str(output),
            "--smpl-faces",
            str(inputs.faces),
            "--smpl-joint-regressor",
            str(inputs.regressor),
            "--smplx-model",
            str(inputs.model),
            "--frames",
            "2",
            "7",
            "--player-ids",
            *selected_ids,
        ],
    )
    main()
    assert capsys.readouterr().out.strip() == str(output / "manifest.json")
    manifest = json.loads((output / "manifest.json").read_text())
    expected_players = [(42, 0, 1), (7, 1, 0)] if index_input else [(42, 0, 1)]
    assert [
        (p["player_id"], p["index"], p["source_index"]) for p in manifest["players"]
    ] == expected_players
    assert manifest["source"]["store_index"]["sha256"] == dual_sha256(store.index_path)
    expected_betas = [[0.2] * 10, [0.1] * 10] if index_input else [[0.2] * 10]
    np.testing.assert_allclose(np.load(output / "smpl_betas.npy"), expected_betas)
    assert _snapshot(store.root) == before


@pytest.mark.parametrize(
    "invalid",
    ["unknown-id", "v1", "missing-scale", "duplicate-ids", "missing-pose", "frames"],
)
def test_rejects_invalid_inputs_before_publication(
    inputs: Inputs, tmp_path: Path, invalid: str
) -> None:
    scene = load_scene_result(inputs.scene)
    config = PlayerViewBundleConfig()
    if invalid == "unknown-id":
        config = PlayerViewBundleConfig(player_ids=(1,))
    elif invalid == "v1":
        scene.metadata.pop("scene_schema_version")
        for field in (*SCENE_MASK_FIELDS, *SCENE_REASON_FIELDS):
            setattr(scene, field, None)
    elif invalid == "missing-scale":
        scene.metadata["body_placement"]["players"]["7"].pop("scale")
    elif invalid == "duplicate-ids":
        scene.player_track_ids = np.array([7, 7], np.int32)
    elif invalid == "missing-pose":
        scene.smpl_betas = None
    elif invalid == "frames":
        config = PlayerViewBundleConfig(frames=(1, 8))
    save_scene_result(scene, inputs.scene)
    before = _snapshot(tmp_path)
    output = tmp_path / "bundle"
    expected_error = {
        "unknown-id": "Unknown player IDs",
        "v1": "requires SceneResult v2",
        "missing-scale": "requires a recorded positive",
        "duplicate-ids": "unique non-negative player_track_ids",
        "missing-pose": "requires smpl_betas",
        "frames": "exceed the source scene timeline",
    }[invalid]
    with pytest.raises(ValueError, match=expected_error):
        _export(inputs, output, config=config)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("stale", [False, True], ids=["incomplete", "stale-lineage"])
def test_store_does_not_search_for_another_export(
    inputs: Inputs, tmp_path: Path, stale: bool
) -> None:
    store = _store(inputs, tmp_path / "store")
    index = json.loads(store.index_path.read_text())
    old_export = index["exports"]["scene"]
    scene = load_scene_result(inputs.scene)
    store.publish(
        "body_placement",
        scene,
        ArtifactCodec(SceneResult),
        schema="placed_bodies",
        version=1,
        identity={"revision": 2},
        dependencies={},
        provenance={},
    )
    if stale:
        index = json.loads(store.index_path.read_text())
        index["exports"]["scene"] = old_export
        store.index_path.write_text(json.dumps(index))
    before = _snapshot(store.root)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="superseded|no completed scene export"):
        _export(inputs, output, source=store.root)
    assert not output.exists()
    assert _snapshot(store.root) == before


def test_failed_writer_never_publishes_and_removes_staging(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    before = _snapshot(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        assert not output.exists()
        raise OSError("Injected disk failure")

    monkeypatch.setattr(
        "src.tennis_scene.visualization.player_view.bundle.write_pc2", fail
    )
    with pytest.raises(OSError, match="disk failure"):
        _export(inputs, output)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".bundle-*"))


def test_head_camera_cli_preserves_selected_ids_timeline_and_isolated_gaps(
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, output = tmp_path / "bundle", tmp_path / "cameras"
    _export(
        inputs, bundle, config=PlayerViewBundleConfig(frames=(1, 6), player_ids=(42, 7))
    )
    before = _snapshot(bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "head-camera",
            "--bundle",
            str(bundle),
            "--output",
            str(output),
            "--face-offset-head-m",
            "0",
            "0",
            "0",
        ],
    )
    camera_main()
    assert _snapshot(bundle) == before
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == CAMERA_SCHEMA
    assert [p["player_id"] for p in manifest["players"]] == [42, 7]
    assert manifest["players"][0]["rejection_counts"] == {
        "VALID": 3,
        "MESH_INVALID": 2,
        "HEAD_POSE_NONFINITE": 0,
        "PLACEMENT_ROTATION_INVALID": 0,
        "HEAD_POSITION_NONFINITE": 0,
    }
    assert manifest["source_bundle"]["path"] == "../bundle"
    assert (
        manifest["source_bundle"]["files"]["manifest.json"] == before["manifest.json"]
    )
    for player_id, mask, valid_index, center in [
        (42, [False, True, True, True, False], 1, [-4, 11, 1]),
        (7, [True, True, False, True, True], 0, [5, 3, 2]),
    ]:
        saved = load_head_cameras(output, player_id=player_id)
        assert saved.player_id == player_id and saved.fps == 59.94
        np.testing.assert_array_equal(saved.source_frames, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(saved.source_time_seconds, np.arange(1, 6) / 59.94)
        np.testing.assert_array_equal(saved.series.camera_valid, mask)
        np.testing.assert_allclose(
            saved.series.pinhole(valid_index).center, center, atol=1e-6
        )
        missing_index = mask.index(False)
        with pytest.raises(ValueError, match="MESH_INVALID"):
            saved.series.pinhole(missing_index)
        if player_id == 7:
            # The source root is valid at this mesh gap. Preserve its reason 0
            # separately; the head camera must still be rejected as MESH_INVALID.
            assert saved.source_player_rejection_code[missing_index] == 0
            assert saved.series.camera_rejection_code[missing_index] == 1
    with pytest.raises(ValueError, match="exactly one player ID"):
        load_head_cameras(output, player_id=99)
    # Consumers do not need the original scene or even the source bundle.
    bundle.rename(tmp_path / "moved-bundle")
    assert load_head_cameras(output, player_id=42).series.camera_valid.sum() == 3


@pytest.mark.parametrize("corrupt", ["zero-missing-pose", "false-valid-mask"])
def test_camera_reader_never_treats_missing_as_origin_or_fixed_camera(
    inputs: Inputs,
    tmp_path: Path,
    corrupt: str,
) -> None:
    bundle, output = tmp_path / "bundle", tmp_path / "cameras"
    _export(inputs, bundle)
    export_head_cameras(bundle_dir=bundle, output_dir=output, config=HeadCameraConfig())
    path = output / "player_0.npz"
    with np.load(path) as data:
        values = {name: data[name] for name in data.files}
    if corrupt == "zero-missing-pose":
        values["camera_to_court"][2] = np.eye(4)
    else:
        values["camera_valid"][2] = True
    np.savez(path, **values)
    with pytest.raises(ValueError, match="inconsistent validity"):
        load_head_cameras(output, player_id=7)


def test_camera_export_refuses_existing_output_and_cleans_failed_staging(
    inputs: Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, output = tmp_path / "bundle", tmp_path / "cameras"
    _export(inputs, bundle)
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        export_head_cameras(
            bundle_dir=bundle, output_dir=output, config=HeadCameraConfig()
        )
    assert sentinel.read_text() == "preserve"
    sentinel.unlink()
    output.rmdir()
    before = _snapshot(bundle)
    with pytest.raises(ValueError, match="outside the read-only"):
        export_head_cameras(
            bundle_dir=bundle, output_dir=bundle / "cameras", config=HeadCameraConfig()
        )
    assert _snapshot(bundle) == before

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("Injected camera archive failure")

    monkeypatch.setattr(
        "src.tennis_scene.visualization.player_view.camera_bundle.np.savez", fail
    )
    with pytest.raises(OSError, match="archive failure"):
        export_head_cameras(
            bundle_dir=bundle, output_dir=output, config=HeadCameraConfig()
        )
    assert not output.exists() and not list(tmp_path.glob(".cameras-*"))
    assert _snapshot(bundle) == before
