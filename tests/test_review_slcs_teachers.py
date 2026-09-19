from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.tasks.slcs.data.quality import QualityConfig
from src.tennis_scene.dataset_pipeline import teacher_review
from src.tennis_scene.dataset_pipeline.teacher_review import (
    contact_sheet,
    label_masks,
    select_frames,
    trajectory_plot,
    validate_scene,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.scripts import render_reconstruction_review as cli
from src.utils.configuration.paths import PathResolver, RuntimePathRoots


def fixture(tmp_path: Path) -> tuple[SceneResult, ClipManifest]:
    t = 9
    cameras = ("a", "b", "c")
    scene = SceneResult(
        num_frames=t,
        fps=30,
        width=64,
        height=48,
        court_kp=np.zeros((3, t, 20, 2), np.float32),
        court_vis=np.ones((3, t, 20), np.float32),
        player_position=np.zeros((1, t, 3), np.float32),
        player_yaw=np.zeros((1, t), np.float32),
        ball_3d=np.zeros((t, 3), np.float32),
        ball_uv=np.zeros((3, t, 2), np.float32),
        ball_vis=np.ones((3, t), bool),
        human_kp_2d=np.zeros((1, 3, t, 17, 2), np.float32),
        human_kp_vis=np.ones((1, 3, t, 17), np.float32),
        metadata={
            "reference": {
                "camera_ids": list(cameras),
                "camera_fits": [
                    {"R": np.eye(3).tolist(), "t": [0, 0, 1], "K": np.eye(3).tolist()}
                    for _ in cameras
                ],
            },
            "label_quality": {
                "schema_version": 1,
                "is_ground_truth": False,
                "ball_weight": [1, 0, 1, 1, 1, 1, 1, 1, 1],
                "player_weight": [[1] * t],
            },
        },
    )
    clip = ClipManifest(
        tmp_path,
        "test",
        "v/c",
        "v",
        "c",
        30,
        t,
        64,
        48,
        cameras,
        tuple(f"{c}.avi" for c in cameras),
        (),
    )
    return scene, clip


def test_selection_uses_same_positive_weight_mask_and_frame_axis(
    tmp_path: Path,
) -> None:
    raw, clip = fixture(tmp_path)
    refined = deepcopy(raw)
    assert raw.ball_3d is not None and refined.ball_3d is not None
    raw.ball_3d[1, 0] = 1000  # rejected by teacher mask
    raw.ball_3d[3, 0] = 100
    refined.ball_3d[5, 0] = 90
    refined.player_position[0, 7, 0] = 10
    masks = label_masks(refined, QualityConfig(0.3, 1, 1, 0.5))
    selected = select_frames(raw, refined, masks)
    assert "max_raw_ball_residual_on_positive_teacher_weight" in selected[3]
    assert "max_refined_ball_residual_on_positive_teacher_weight" in selected[5]
    assert "minimum_total_teacher_weight" in selected[1]
    assert "max_raw_to_refined_root_displacement" in selected[7]
    assert "max_refined_root_frame_step" in selected[7]
    validate_scene(refined, clip)


def test_validation_and_zero_weight(tmp_path: Path) -> None:
    scene, clip = fixture(tmp_path)
    masks = label_masks(scene, QualityConfig(0.3, 1, 1, 0.5))
    assert not masks["ball_label_valid"][1]
    scene.metadata["reference"]["camera_fits"][0]["t"] = [[0, 0, 1]]
    with pytest.raises(ValueError, match="reference camera t"):
        validate_scene(scene, clip)
    scene.metadata["reference"]["camera_fits"][0]["t"] = [0, 0, 1]
    scene.metadata["reference"]["camera_ids"].reverse()
    with pytest.raises(ValueError, match="camera order"):
        validate_scene(scene, clip)


@pytest.mark.filterwarnings("error:You passed both c and facecolor.*:UserWarning")
def test_cpu_fixture_images(tmp_path: Path) -> None:
    scene, clip = fixture(tmp_path)
    for camera in clip.camera_ids:
        writer = cv2.VideoWriter(
            str(clip.media_path(camera, must_exist=False)),
            cv2.VideoWriter.fourcc(*"MJPG"),
            30,
            (64, 48),
        )
        assert writer.isOpened()
        try:
            for _ in range(scene.num_frames):
                writer.write(np.zeros((48, 64, 3), np.uint8))
        finally:
            writer.release()
    masks = label_masks(scene, QualityConfig(0.3, 1, 1, 0.5))
    contact_sheet(
        clip,
        scene,
        scene,
        masks,
        {0: ["test"], 1: ["unsupported"]},
        tmp_path / "sheet.png",
    )
    trajectory_plot(scene, scene, masks, tmp_path / "trajectory.png")
    for name in ("sheet.png", "trajectory.png"):
        image = cv2.imread(str(tmp_path / name))
        assert image is not None and image.shape[0] > 100 and image.shape[1] > 100


def output_resolver(tmp_path: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            output_root=tmp_path / "custom_results",
            checkpoint_root=tmp_path / "ckpt",
            artifact_root=tmp_path / "artifacts",
            cache_root=tmp_path / "cache",
            external_asset_root=tmp_path / "third_party",
        )
    )


@pytest.mark.parametrize(
    "fragment",
    [
        "tennis_scene/visualize/../run",
        "tennis_scene/visualize/exp/..",
        "tennis_scene/visualize/exp/../../outside",
        "/tennis_scene/visualize/exp/run",
        "tennis_scene/analyze/exp/run",
        "tennis_scene/visualize/exp",
        "tennis_scene/visualize//run",
    ],
)
def test_invalid_output_fragment(tmp_path: Path, fragment: str) -> None:
    with pytest.raises(ValueError):
        cli.resolve_review_output(output_resolver(tmp_path), fragment)


def test_custom_output_and_symlink_escape(tmp_path: Path) -> None:
    resolver = output_resolver(tmp_path)
    expected = tmp_path / "custom_results/tennis_scene/visualize/exp/run"
    assert (
        cli.resolve_review_output(resolver, "tennis_scene/visualize/exp/run")
        == expected
    )
    expected.parent.parent.mkdir(parents=True)
    expected.parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        cli.resolve_review_output(resolver, "tennis_scene/visualize/exp/run")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = cli.resolve_review_output(
        output_resolver(tmp_path), "tennis_scene/visualize/exp/run"
    )
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        teacher_review.review(
            tmp_path / "nonexistent",
            tmp_path / "runs",
            output,
            ["v/c"],
            QualityConfig(0.3, 1, 1, 0.5),
        )
    assert list(output.iterdir()) == []


def test_automatic_cli_custom_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(teacher_review, "review", lambda *args: calls.append(args))
    monkeypatch.setattr(
        "sys.argv",
        [
            "review",
            "--dataset-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(tmp_path / "custom_results"),
            "--output",
            "tennis_scene/visualize/exp/run",
            "--clip",
            "v/a",
            "--clip",
            "v/b",
        ],
    )
    cli.main()
    assert calls[0][:4] == (
        tmp_path / "data",
        tmp_path / "runs",
        tmp_path / "custom_results/tennis_scene/visualize/exp/run",
        ["v/a", "v/b"],
    )


def test_legacy_cli_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    index = SimpleNamespace(
        clips=[SimpleNamespace(video_id="v", clip_id="v/a")],
        clip_dir=lambda record: tmp_path / "clip",
    )
    monkeypatch.setattr(cli.SLCSDataIndex, "load", lambda _: index)

    def render(*args: object, **kwargs: object) -> tuple[Path, Path]:
        calls.append((*args, kwargs))
        return tmp_path / "image", tmp_path / "receipt"

    monkeypatch.setattr(cli, "render_review", render)
    monkeypatch.setattr(
        "sys.argv",
        [
            "review",
            "--data-root",
            str(tmp_path / "data"),
            "--dataset",
            "dataset",
            "--video",
            "v",
            "--clip",
            "v/a",
            "--frames",
            "1",
            "3",
            "--output-root",
            str(tmp_path / "custom_results"),
            "--output",
            "tennis_scene/visualize/exp/run",
        ],
    )
    cli.main()
    assert calls[0] == (
        tmp_path / "clip",
        tmp_path / "custom_results/tennis_scene/visualize/exp/run",
        {"frames": [1, 3], "cameras": None},
    )
