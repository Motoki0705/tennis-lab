"""CPU contracts for diagnostic paths, dispatch, media and held-out evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.submodules.models import Pose2DRequest, Pose2DResult
from src.tennis_scene.dataset_pipeline.diagnostics import configuration, media
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.scripts import render_reconstruction_review as review
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def configured(name: str, tmp_path: Path) -> DictConfig:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(config_name=name)
    cfg.paths.project_root = str(PROJECT_ROOT)
    for role in ("data", "checkpoint", "artifact", "output", "cache", "external_asset"):
        cfg.paths[f"{role}_root"] = str(tmp_path / role)
    cfg.output = "tennis_scene/evaluate/diagnostic/run"
    return cfg


def test_vitpose_config_resolves_roles_and_rejects_reuse(tmp_path: Path) -> None:
    cfg = configured("benchmark_vitpose_precision", tmp_path)
    cfg.clip = "dataset/videos/v/clips/c"
    cfg.observations = "tennis_scene/precompute/observations/run/v/c"
    request = configuration.ViTPoseBenchmarkConfig.from_config(cfg)
    assert request.clip == tmp_path / "data/dataset/videos/v/clips/c"
    assert (
        request.checkpoint
        == tmp_path
        / "external_asset/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    )
    assert request.frames == 128 and request.people == (0, 1)
    assert request.head.num_deconv_filters == (256, 256)
    request.output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        configuration.validate_vitpose(cfg)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frames", 0),
        ("warmup_frames", 129),
        ("cameras", ["../escape"]),
        ("people", [-1]),
        ("device", "cpu"),
        ("crop_enlarge", float("nan")),
        ("output", "tennis_scene/evaluate/../escape"),
        ("clip", "/outside"),
    ],
)
def test_invalid_benchmark_config(tmp_path: Path, key: str, value: object) -> None:
    cfg = configured("benchmark_vitpose_precision", tmp_path)
    cfg.clip, cfg.observations = "dataset/clip", "tennis_scene/precompute/obs/run"
    cfg[key] = value
    with pytest.raises((ValueError, TypeError)):
        configuration.validate_vitpose(cfg)
    assert not (tmp_path / "output").exists()


def test_refinement_recipe_preserves_view_mapping(tmp_path: Path) -> None:
    cfg = configured("evaluate_refinement", tmp_path)
    cfg.scene, cfg.court = (
        "tennis_scene/generate/run/scene.npz",
        "tennis_scene/precompute/run/court.npz",
    )
    request = configuration.RefinementEvaluationConfig.from_config(cfg)
    assert request.view_half_turns == (False, False, True)
    assert request.settings.player_root_view_support == "hips_and_shoulders"
    cfg.overrides = ["coordinate_mode=physical"]
    assert configuration.RefinementEvaluationConfig.from_config(
        cfg
    ).view_half_turns == (False,)


def test_roots_resolve_worktree_symlinks_and_forbid_hidden_recipe_roots(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "shared"
    actual.mkdir()
    link = tmp_path / "worktree_data"
    link.symlink_to(actual, target_is_directory=True)
    cfg = configured("evaluate_refinement", tmp_path)
    cfg.paths.data_root = str(link)
    cfg.scene, cfg.court = (
        "tennis_scene/generate/x/scene.npz",
        "tennis_scene/precompute/x/court.npz",
    )
    assert configuration.resolver_for(cfg).roots.data_root == actual
    cfg.overrides = ["paths.data_root=/somewhere"]
    with pytest.raises(ValueError, match="diagnostic paths"):
        configuration.validate_refinement(cfg)


def test_court_probe_preserves_checkpoint_and_crop_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = configured("probe_meiji_court", tmp_path)
    builds = []

    def runtime(build: DictConfig) -> SimpleNamespace:
        builds.append(build)
        return SimpleNamespace(
            clip_ids=("video_000/clip_000",),
            output=tmp_path / "output/tennis_scene/evaluate/diagnostic/run",
        )

    monkeypatch.setattr(configuration.DatasetBuildConfig, "from_config", runtime)
    request = configuration.CourtProbeConfig.from_config(cfg)
    assert request.samples == 9 and request.margins == (0.25, 0.5, 0.75, 1.0)
    assert request.checkpoints == (
        "court_detection/multiscale-depth3-local-rtx/logs/version_4/checkpoints/court-detection-epoch=17.ckpt",
    )
    assert list(builds[0].clip_ids) == ["video_000/clip_000"]
    assert builds[0].paths.external_asset_root == str(tmp_path / "external_asset")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset-root", "/datasets", "--run-root", "/runs", "--frames", "0"],
        ["--dataset-root", "/datasets", "--run-root", "/runs", "--mode", "frames"],
        [
            "--data-root",
            "/datasets",
            "--dataset",
            "sample",
            "--video",
            "v",
            "--frames",
            "0",
            "--run-root",
            "/runs",
        ],
        [
            "--data-root",
            "/datasets",
            "--dataset",
            "sample",
            "--video",
            "v",
            "--frames",
            "-1",
        ],
        ["--dataset-root", "relative", "--run-root", "/runs"],
    ],
)
def test_review_rejects_conflicting_or_invalid_modes(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        review.main(
            [
                *arguments,
                "--output-root",
                "/results",
                "--output",
                "tennis_scene/visualize/exp/run",
                "--clip",
                "v/c",
            ]
        )
    assert error.value.code == 2


@pytest.fixture
def diagnostic_clip(tmp_path: Path) -> ClipManifest:
    (tmp_path / "media").mkdir()
    (tmp_path / "media/cam0.mp4").touch()
    return ClipManifest(
        clip_dir=tmp_path,
        dataset_id="diagnostic",
        clip_id="video_000/clip_000",
        video_id="video_000",
        clip_name="clip_000",
        fps=30.0,
        num_frames=3,
        width=4,
        height=2,
        camera_ids=("cam0",),
        video_paths=("media/cam0.mp4",),
        cameras=({"camera_id": "cam0"},),
    )


@pytest.mark.parametrize("fps", [0.0, float("nan"), 60.0])
def test_media_rejects_invalid_or_mismatched_fps_and_releases(
    diagnostic_clip: ClipManifest, monkeypatch: pytest.MonkeyPatch, fps: float
) -> None:
    released = []
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda prop: fps,
        release=lambda: released.append(True),
    )
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda path: capture)
    with pytest.raises(ValueError, match="FPS"):
        media.sample_frames(diagnostic_clip, "cam0", np.asarray([0]))
    assert released == [True]


def test_media_samples_requested_frames(
    diagnostic_clip: ClipManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = []
    props = {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 4,
        cv2.CAP_PROP_FRAME_HEIGHT: 2,
        cv2.CAP_PROP_FRAME_COUNT: 3,
    }
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=props.__getitem__,
        release=lambda: None,
        set=lambda prop, index: positions.append(index),
        read=lambda: (True, np.zeros((2, 4, 3), np.uint8)),
    )
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda path: capture)
    result = media.sample_frames(diagnostic_clip, "cam0", np.asarray([0, 2]))
    assert positions == [0, 2] and len(result) == 2


@pytest.mark.parametrize("standard_name_present", [False, True])
def test_vitpose_infers_the_same_manifest_media_that_was_validated(
    diagnostic_clip: ClipManifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standard_name_present: bool,
) -> None:
    from src.tennis_scene.dataset_pipeline.diagnostics import vitpose

    standard = diagnostic_clip.media_path("cam0")
    if standard_name_present:
        standard.write_bytes(b"different, unselected video")
    else:
        standard.unlink()
    clip = replace(diagnostic_clip, video_paths=("media/custom_cam0.mkv",))
    selected = clip.clip_dir / clip.video_paths[0]
    selected.write_bytes(b"selected manifest video")
    monkeypatch.setattr(vitpose.ClipManifest, "load", lambda path: clip)
    validated: list[Path] = []
    props = {
        cv2.CAP_PROP_FPS: clip.fps,
        cv2.CAP_PROP_FRAME_WIDTH: clip.width,
        cv2.CAP_PROP_FRAME_HEIGHT: clip.height,
        cv2.CAP_PROP_FRAME_COUNT: clip.num_frames,
    }

    def capture(path: str) -> SimpleNamespace:
        validated.append(Path(path))
        return SimpleNamespace(
            isOpened=lambda: True, get=props.__getitem__, release=lambda: None
        )

    monkeypatch.setattr(media.cv2, "VideoCapture", capture)
    cfg = configured("benchmark_vitpose_precision", tmp_path)
    cfg.clip, cfg.observations = "dataset/clip", "tennis_scene/precompute/obs/run"
    cfg.cameras, cfg.people = ["cam0"], [0]
    cfg.frames, cfg.warmup_frames = 3, 1
    args = replace(
        configuration.ViTPoseBenchmarkConfig.from_config(cfg), clip=clip.clip_dir
    )
    args.checkpoint.parent.mkdir(parents=True)
    args.checkpoint.write_bytes(b"checkpoint fixture")
    args.observations.mkdir(parents=True)
    np.savez(
        args.observations / "cam0_people.npz",
        boxes=np.tile(np.asarray([0, 0, 2, 2], dtype=np.float32), (1, 3, 1)),
    )
    predictions: list[tuple[str, Path, int]] = []
    model = SimpleNamespace(precision="float32", load=lambda: None, unload=lambda: None)

    def predict(request: Pose2DRequest) -> Pose2DResult:
        predictions.append(
            (model.precision, Path(request.video_path), len(request.bbx_xys))
        )
        return Pose2DResult(torch.zeros((len(request.bbx_xys), 17, 3)))

    model.predict = predict
    monkeypatch.setattr(vitpose, "ViTPosePose2D", lambda *args, **kwargs: model)
    monkeypatch.setattr(vitpose.torch.cuda, "synchronize", lambda: None)
    vitpose.benchmark(args)
    assert validated == [selected]
    assert predictions == [
        (precision, selected, frames)
        for precision in ("float32", "bfloat16")
        for frames in (1, 3)
    ]
    assert (args.output / "metrics.json").is_file()
    if standard_name_present:
        assert standard.read_bytes() == b"different, unselected video"


def test_blcs_fixed_seed_strict_load_and_test_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.blcs.evaluation import real
    from src.tasks.blcs.evaluation.configuration import RealEvaluationConfig

    calls: list[object] = []
    roots = RuntimePathRoots(
        **{
            f"{role}_root": tmp_path
            for role in (
                "project",
                "data",
                "checkpoint",
                "artifact",
                "output",
                "cache",
                "external_asset",
            )
        }
    )
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = tmp_path / "dataset/provenance.json"
    provenance.parent.mkdir()
    provenance.write_text("{}")
    module = SimpleNamespace(
        on_load_checkpoint=lambda value: calls.append("contract"),
        load_state_dict=lambda state, strict: calls.append((state, strict)),
    )
    monkeypatch.setattr(
        real,
        "compose_blcs_training",
        lambda cfg, generator_config: SimpleNamespace(
            lightning_module=module, datamodule="fixed-split"
        ),
    )
    monkeypatch.setattr(
        real.torch, "load", lambda *args, **kwargs: {"state_dict": "weights"}
    )
    monkeypatch.setattr(
        real.pl, "seed_everything", lambda seed, workers: calls.append((seed, workers))
    )

    def trainer(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(test=lambda module, datamodule: [{"test/position": 1.0}])

    monkeypatch.setattr(real.pl, "Trainer", trainer)
    cfg = OmegaConf.create({"run": {"seed": 42}, "data": {"scene_dir": "dataset"}})
    request = RealEvaluationConfig(
        cfg, PathResolver(roots), checkpoint, tmp_path / "evaluation", "cpu"
    )
    real.evaluate(request)
    assert calls[:3] == [(42, True), "contract", ("weights", True)]
    trainer_options = calls[3]
    assert isinstance(trainer_options, dict)
    assert trainer_options["enable_checkpointing"] is False
    assert trainer_options["precision"] == "32-true"
    report = json.loads((request.output / "evaluation.json").read_text())
    assert report["metrics"] == {"test/position": 1.0}
    with pytest.raises(FileExistsError):
        real.evaluate(request)


def test_blcs_evaluation_configuration_is_training_recipe_with_explicit_checkpoint(
    tmp_path: Path,
) -> None:
    from src.tasks.blcs.evaluation.configuration import RealEvaluationConfig

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="evaluate_real",
            overrides=["evaluation.checkpoint=blcs/selected.ckpt"],
        )
    cfg.paths.output_root = str(tmp_path / "results")
    request = RealEvaluationConfig.from_config(cfg)
    assert request.training.run.seed == 42
    assert request.training.data.scene_dir == "blcs/meiji_geometry_replay_v1"
    assert request.training.data.num_workers == 0
    assert "evaluation" not in request.training
    assert request.output.is_relative_to(tmp_path / "results/blcs/evaluate")


def test_refinement_failure_keeps_evidence_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tennis_scene.dataset_pipeline.diagnostics import refinement

    cfg = configured("evaluate_refinement", tmp_path)
    cfg.scene, cfg.court = (
        "tennis_scene/generate/x/scene.npz",
        "tennis_scene/precompute/x/court.npz",
    )
    request = configuration.RefinementEvaluationConfig.from_config(cfg)
    request.scene.parent.mkdir(parents=True)
    request.scene.write_bytes(b"source scene")
    request.court.parent.mkdir(parents=True)
    np.savez(request.court, homographies=np.eye(3)[None])
    scene = SimpleNamespace(refined=False)
    calls = []
    monkeypatch.setattr(refinement, "load_scene_result", lambda path: scene)

    def quality(
        value: SimpleNamespace, matrices: np.ndarray, turns: list[bool]
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        calls.append((value.refined, turns))
        return {"refined": value.refined}, {"mask": np.ones(1)}

    monkeypatch.setattr(refinement, "evaluate_reconstruction", quality)

    def refine(
        value: SimpleNamespace, matrices: np.ndarray, settings: object
    ) -> dict[str, object]:
        value.refined = True
        return {"coverage": 0}

    monkeypatch.setattr(refinement, "refine_scene", refine)
    monkeypatch.setattr(
        refinement,
        "save_scene_result",
        lambda value, path: path.write_bytes(b"refined"),
    )

    def reject(evidence: object, settings: object) -> None:
        raise ValueError("insufficient coverage")

    monkeypatch.setattr(refinement, "check_label_coverage", reject)
    with pytest.raises(ValueError, match="insufficient coverage"):
        refinement.evaluate(request)
    assert calls == [(False, [False, False, True]), (True, [False, False, True])]
    assert request.scene.read_bytes() == b"source scene"
    report = json.loads((request.output / "metrics.json").read_text())
    assert report["before"] == {"refined": False} and report["after"] == {
        "refined": True
    }
    assert report["evidence"] == {"coverage": 0}
