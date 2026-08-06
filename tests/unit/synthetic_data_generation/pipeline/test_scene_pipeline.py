from __future__ import annotations

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]
import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.pipeline import PipelineRequest, run_scene_pipeline
from src.synthetic_data_generation.pipeline.config import (
    ScenePipelineConfig,
    ScenePipelineRunConfig,
)
from src.synthetic_data_generation.pipeline.court_evidence import (
    LineObservation,
    evaluate,
)
from src.synthetic_data_generation.pipeline.stages import Stage, Target
from src.synthetic_data_generation.pipeline.workspace import (
    SceneWorkspace,
    WorkspaceLock,
)
from src.utils.configuration import PathResolver, RuntimePathRoots


def test_hydra_run_boundary_builds_a_strict_pipeline_request() -> None:
    runtime = ScenePipelineRunConfig.from_config(
        OmegaConf.create(
            {
                "scene_id": "B00",
                "input_video": None,
                "pipeline_config": (
                    "src/synthetic_data_generation/configs/pipeline/scene.yaml"
                ),
                "from_stage": "alignment",
                "targets": ["court", "blcs", "plcs"],
                "nht_from_stage": "frames",
            }
        )
    )

    assert runtime.scene_id == "B00"
    assert runtime.input_video is None
    assert runtime.targets == ("court", "blcs", "plcs")
    assert runtime.pipeline_config.is_file()


def test_hydra_run_boundary_rejects_ingest_without_input_video() -> None:
    with pytest.raises(ValueError, match="requires input_video"):
        ScenePipelineRunConfig.from_config(
            OmegaConf.create(
                {
                    "scene_id": "B00",
                    "input_video": None,
                    "pipeline_config": (
                        "src/synthetic_data_generation/configs/pipeline/scene.yaml"
                    ),
                    "from_stage": "ingest",
                    "targets": ["court"],
                    "nht_from_stage": "frames",
                }
            )
        )


def _config(tmp_path: Path, *, samples: int = 2) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f"""\
schema: tennis_scene_pipeline_config_v1
seed: 17
roots:
  project_root: {tmp_path}
  data_root: {tmp_path / "data"}
  checkpoint_root: {tmp_path / "ckpt"}
  artifact_root: {tmp_path / "artifacts"}
  output_root: {tmp_path / "outputs"}
  cache_root: {tmp_path / "cache"}
  external_asset_root: {tmp_path / "external"}
nht:
  reconstruct_command: [fake-reconstruct]
  render_command: [fake-render]
  config: null
  working_directory: null
  environment: {{}}
alignment:
  minimum_ground_points: 20
  minimum_ground_support_fraction: 0.05
  minimum_positive_camera_fraction: 0.75
  holdout_fraction: 0.20
  evidence:
    mode: sparse_control
    maximum_views: 5
    maximum_image_size: 64
    maximum_pixels_per_view: 100
    minimum_line_brightness: 0.55
    maximum_line_saturation: 0.30
    minimum_local_contrast: 0.08
    minimum_projected_pixels_per_view: 20
    raster_size: 64
    optimizer_iterations: 5
    optimizer_population_size: 4
    minimum_fit_template_score: 0.08
    line_inlier_distance_m: 0.50
    minimum_holdout_view_fraction: 0.50
    minimum_holdout_inlier_fraction: 0.15
datasets:
  samples_per_domain: {samples}
"""
    )
    return path


def _publish_scene(workspace: Path, scene_id: str) -> None:
    export = workspace / "export"
    (export / "images").mkdir(parents=True)
    (export / "model/ckpts").mkdir(parents=True)
    (export / "model/runtime.json").write_text("{}")
    (export / "model/ckpts/ckpt.pt").write_bytes(b"checkpoint")
    cameras = []
    for index, x in enumerate((-2.0, -1.0, 1.0, 2.0, 3.0)):
        image_name = f"frame_{index:06d}.png"
        Image.new("RGB", (64, 48), (20 + index, 30, 40)).save(
            export / "images" / image_name
        )
        pose = np.eye(4)
        pose[:3, 3] = [x, 3.0, -5.0 + index]
        cameras.append(
            {
                "camera_id": Path(image_name).stem,
                "image": f"images/{image_name}",
                "width": 64,
                "height": 48,
                "intrinsics": {
                    "model": "PINHOLE",
                    "distortion_model": "NONE",
                    "params": [50.0, 50.0, 32.0, 24.0],
                    "matrix": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
                },
                "camera_to_scene": pose.tolist(),
                "source_frame_index": index,
                "time_seconds": float(index),
                "split": "train",
                "source_image_processing": {
                    "source_resolution": [64, 48],
                    "crop_xywh": [0, 0, 64, 48],
                    "undistorted": True,
                    "data_factor": 1,
                },
                "diagnostics": {
                    "sfm_camera_id": 1,
                    "sfm_camera_to_world": pose.tolist(),
                },
                "group": "default",
            }
        )
    (export / "cameras.json").write_text(
        json.dumps(
            {
                "schema": "nht_standard_cameras_v1",
                "camera_coordinate_convention": "x-right, y-down, z-forward",
                "transform_semantics": "camera_to_scene",
                "cameras": cameras,
            }
        )
    )
    x, z = np.meshgrid(np.linspace(-6, 6, 30), np.linspace(-13, 13, 40))
    ground = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    elevated = np.column_stack(
        [
            np.linspace(-3, 3, 100),
            np.linspace(0.5, 2.0, 100),
            np.linspace(-5, 5, 100),
        ]
    )
    points = np.column_stack(
        [np.vstack([ground, elevated]), np.full((len(ground) + len(elevated), 3), 0.5)]
    ).astype(np.float32)
    np.save(export / "points_scene.npy", points)
    scene = {
        "schema": "nht_standard_scene_v1",
        "scene_id": scene_id,
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "scene_coordinate_convention": "right-handed canonical scene",
        "pixel_coordinate_convention": "top-left pixel centres",
        "image_resolution_semantics": "full exported image pixels",
        "camera_count": len(cameras),
        "cameras": "cameras.json",
        "point_cloud": {"path": "points_scene.npy", "shape": list(points.shape)},
        "image_root": "images",
        "model_root": "model",
        "scene_from_sfm": np.eye(4).tolist(),
        "sfm_from_scene": np.eye(4).tolist(),
        "renderer": {
            "command": "nht-render",
            "checkpoint": "model/ckpts/ckpt.pt",
            "runtime_config": "model/runtime.json",
        },
        "capabilities": ["nht_rendering_model"],
    }
    (export / "scene.json").write_text(json.dumps(scene))
    stages = {
        name: {"status": "completed"}
        for name in (
            "frames",
            "preprocess",
            "sfm",
            "sfm_selection",
            "nht_training",
            "scene_export",
            "reconstruction_report",
        )
    }
    (workspace / "run.json").write_text(
        json.dumps(
            {
                "schema": "nht_pipeline_run_v1",
                "scene_id": scene_id,
                "status": "completed",
                "stages": stages,
            }
        )
    )


def _fake_process(assertion=None):
    def run(command, *, working_directory, environment):
        del working_directory, environment
        if "--workspace" in command:
            workspace = Path(command[command.index("--workspace") + 1])
            if assertion is not None:
                assertion(workspace, command)
            _publish_scene(workspace, command[command.index("--scene-id") + 1])
            return
        output = Path(command[command.index("--output") + 1])
        scene = json.loads(Path(command[command.index("--scene") + 1]).read_text())
        camera_ids = [
            command[index + 1]
            for index, token in enumerate(command)
            if token == "--camera-id"
        ]
        output.mkdir(parents=True)
        records = []
        for camera_id in camera_ids:
            root = output / camera_id
            root.mkdir()
            np.save(root / "rgb.npy", np.zeros((48, 64, 3), dtype=np.float32))
            np.save(root / "alpha.npy", np.ones((48, 64, 1), dtype=np.float32))
            np.save(root / "depth.npy", np.ones((48, 64, 1), dtype=np.float32))
            records.append(
                {
                    "camera_id": camera_id,
                    "width": 64,
                    "height": 48,
                    "rgb": f"{camera_id}/rgb.npy",
                    "alpha": f"{camera_id}/alpha.npy",
                    "depth": f"{camera_id}/depth.npy",
                }
            )
        (output / "render.json").write_text(
            json.dumps(
                {
                    "schema": "nht_render_result_v1",
                    "scene_schema": scene["schema"],
                    "scene_id": scene["scene_id"],
                    "renders": records,
                }
            )
        )

    return run


def _request(
    tmp_path: Path, config: Path, video: Path | None, stage: Stage
) -> PipelineRequest:
    return PipelineRequest(
        scene_id="B00",
        config_path=config,
        repository_root=tmp_path,
        input_video=video,
        from_stage=stage,
        targets=(Target.COURT, Target.BLCS, Target.PLCS),
        nht_from_stage="sfm" if stage is Stage.RECONSTRUCTION else "frames",
    )


def test_video_to_alignment_and_all_domain_datasets(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )

    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    run = json.loads(run_path.read_text())
    root = run_path.parent

    assert run["status"] == "completed"
    assert run["source_video"] == "source/video.mp4"
    jsonschema.Draft202012Validator(
        json.loads(
            (
                Path(__file__).resolve().parents[4]
                / "src/synthetic_data_generation/schemas/run.schema.json"
            ).read_text()
        ),
        format_checker=jsonschema.FormatChecker(),
    ).validate(run)
    assert run["stages"]["reconstruction"]["summary"]["nht_run"] == (
        "reconstruction/run.json"
    )
    assert json.loads((root / "alignment/alignment.json").read_text())["accepted"]
    assert (root / "alignment/ground-line-map.npz").is_file()
    assert (root / "alignment/ground-line-preview.png").is_file()
    assert (root / "alignment/court-geometry.json").is_file()
    assert (root / "alignment/diagnostics/fit-holdout.json").is_file()
    with np.load(
        root / "alignment/ground-line-map.npz", allow_pickle=False
    ) as line_map:
        assert line_map["mean_probability"].ndim == 2
        assert line_map["mean_probability"].dtype == np.float32
        assert line_map["view_count"].dtype == np.uint16
    for domain in ("court", "blcs", "plcs"):
        dataset = json.loads((root / f"datasets/{domain}/dataset.json").read_text())
        assert dataset["sample_count"] == 2
        assert dataset["renderer_boundary"]["scene"] == (
            "reconstruction/export/scene.json"
        )
        if domain != "court":
            for sample in dataset["samples"]:
                mask = np.load(
                    root / f"datasets/{domain}" / sample["instance_mask"],
                    allow_pickle=False,
                )
                assert mask.dtype == np.uint8
                assert mask.shape == (48, 64)
                assert sample["visible_instance_pixels"] == int(np.count_nonzero(mask))
    assert (root / "report/index.html").is_file()
    report = (root / "report/index.html").read_text()
    assert report.count("<tr><td>") == 3
    assert all(f"<td>{domain}</td>" in report for domain in ("court", "blcs", "plcs"))


def test_alignment_rerun_preserves_reconstruction(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )
    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    root = run_path.parent
    nht_before = (root / "reconstruction/run.json").read_bytes()
    attempts_before = json.loads(run_path.read_text())["stages"]["reconstruction"][
        "attempts"
    ]

    run_scene_pipeline(_request(tmp_path, config, None, Stage.ALIGNMENT))

    assert (root / "reconstruction/run.json").read_bytes() == nht_before
    run = json.loads(run_path.read_text())
    assert run["stages"]["reconstruction"]["attempts"] == attempts_before
    assert run["stages"]["alignment"]["attempts"] == 2


def test_sfm_rerun_unpublishes_every_downstream_before_subprocess(
    monkeypatch, tmp_path
) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )
    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    root = run_path.parent

    def assert_invalidated(workspace, command):
        assert command[command.index("--from-stage") + 1] == "sfm"
        assert not (workspace / "export").exists()
        assert not (root / "alignment").exists()
        assert not (root / "datasets").exists()
        assert not (root / "report").exists()

    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(assert_invalidated),
    )
    run_scene_pipeline(_request(tmp_path, config, None, Stage.RECONSTRUCTION))
    assert json.loads(run_path.read_text())["status"] == "completed"


def test_workspace_resolver_separates_scene_ids_and_lock_refuses_parallel(
    tmp_path,
) -> None:
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": str(tmp_path),
            "data_root": str(tmp_path / "data"),
            "checkpoint_root": str(tmp_path / "ckpt"),
            "artifact_root": str(tmp_path / "artifacts"),
            "output_root": str(tmp_path / "outputs"),
            "cache_root": str(tmp_path / "cache"),
            "external_asset_root": str(tmp_path / "external"),
        },
        repository_root=tmp_path.resolve(),
    )
    resolver = PathResolver(roots)
    b00 = SceneWorkspace.resolve(resolver, "B00")
    b01 = SceneWorkspace.resolve(resolver, "B01")
    assert b00.root != b01.root
    with WorkspaceLock(b00):
        try:
            WorkspaceLock(b00).__enter__()
        except RuntimeError as error:
            assert "locked by live process" in str(error)
        else:
            raise AssertionError("Expected live workspace lock rejection")


def test_pipeline_has_no_nht_or_colmap_internal_dependency() -> None:
    source_root = (
        Path(__file__).resolve().parents[4] / "src/synthetic_data_generation/pipeline"
    )
    combined = "\n".join(path.read_text() for path in sorted(source_root.glob("*.py")))
    for forbidden in (
        "import nht_pipeline",
        "from nht_pipeline",
        "import pycolmap",
        "from pycolmap",
        "sfm/model",
        "state_dict",
        "ckpt_",
    ):
        assert forbidden not in combined


def test_failed_reconstruction_is_recorded_and_stale_export_stays_unpublished(
    monkeypatch, tmp_path
) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )
    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    root = run_path.parent

    def fail_reconstruction(command, *, working_directory, environment):
        del command, working_directory, environment
        print("synthetic NHT failure marker")
        raise RuntimeError("NHT process exited with status 17")

    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        fail_reconstruction,
    )
    with pytest.raises(RuntimeError, match="status 17"):
        run_scene_pipeline(_request(tmp_path, config, None, Stage.RECONSTRUCTION))

    run = json.loads(run_path.read_text())
    record = run["stages"]["reconstruction"]
    assert run["status"] == "failed"
    assert record["status"] == "failed"
    assert record["error"] == {
        "category": "stage_failure",
        "message": "NHT process exited with status 17",
    }
    assert not (root / "reconstruction/export").exists()
    assert not (root / "alignment").exists()
    assert not (root / "datasets").exists()
    assert not (root / "report").exists()
    assert (
        "synthetic NHT failure marker"
        in (root / "logs/reconstruction/attempt-2.log").read_text()
    )


def test_interrupted_alignment_and_staging_are_recovered(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )
    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    root = run_path.parent
    run = json.loads(run_path.read_text())
    run["status"] = "running"
    run["stages"]["alignment"]["status"] = "running"
    run_path.write_text(json.dumps(run))
    interrupted = root / ".staging/alignment/alignment"
    interrupted.mkdir(parents=True)
    (interrupted / "partial.json").write_text("{}")
    (root / ".pipeline.lock").write_text(
        json.dumps(
            {
                "schema": "tennis_scene_workspace_lock_v1",
                "pid": 999_999_999,
                "scene_id": "B00",
            }
        )
    )

    run_scene_pipeline(_request(tmp_path, config, None, Stage.ALIGNMENT))

    recovered = json.loads(run_path.read_text())
    assert recovered["status"] == "completed"
    assert recovered["stages"]["alignment"]["status"] == "completed"
    assert recovered["stages"]["alignment"]["attempts"] == 2
    assert not (root / ".staging").exists()


def test_alignment_config_change_invalidates_alignment_and_descendants(
    monkeypatch, tmp_path
) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        _fake_process(),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", _fake_process()
    )
    run_path = run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))
    root = run_path.parent
    reconstruction_before = (root / "reconstruction/run.json").read_bytes()
    run_before = json.loads(run_path.read_text())

    config.write_text(
        config.read_text().replace(
            "minimum_ground_support_fraction: 0.05",
            "minimum_ground_support_fraction: 0.04",
        )
    )
    run_scene_pipeline(_request(tmp_path, config, None, Stage.REPORT))

    run = json.loads(run_path.read_text())
    assert (root / "reconstruction/run.json").read_bytes() == reconstruction_before
    assert (
        run["stages"]["reconstruction"]["attempts"]
        == run_before["stages"]["reconstruction"]["attempts"]
    )
    assert run["stages"]["alignment"]["attempts"] == 2
    for stage in ("court_dataset", "blcs_dataset", "plcs_dataset", "report"):
        assert run["stages"][stage]["attempts"] == 2


def test_dataset_rejects_non_finite_renderer_output(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    healthy_process = _fake_process()

    def corrupt_render(command, *, working_directory, environment):
        healthy_process(
            command,
            working_directory=working_directory,
            environment=environment,
        )
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            camera_id = command[command.index("--camera-id") + 1]
            rgb_path = output / camera_id / "rgb.npy"
            rgb = np.load(rgb_path, allow_pickle=False)
            rgb[0, 0, 0] = np.nan
            np.save(rgb_path, rgb)

    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.orchestrator.run_process",
        healthy_process,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.pipeline.datasets.run_process", corrupt_render
    )

    with pytest.raises(ValueError, match="contains non-finite values"):
        run_scene_pipeline(_request(tmp_path, config, video, Stage.INGEST))


def test_holdout_fraction_excludes_views_where_target_court_is_not_evaluable(
    tmp_path,
) -> None:
    config = ScenePipelineConfig.load(_config(tmp_path), tmp_path).alignment.evidence
    on_line = np.column_stack(
        [
            np.linspace(-5.0, 5.0, 30),
            np.zeros(30),
            np.full(30, -11.885),
        ]
    )
    outside_court = on_line + np.asarray([100.0, 0.0, 100.0])
    observations = [
        LineObservation(
            camera_id="visible",
            points_scene=on_line,
            points_uv=on_line[:, [0, 2]],
            scores=np.ones(30, dtype=np.float32),
            selected_pixel_count=30,
        ),
        LineObservation(
            camera_id="other-court",
            points_scene=outside_court,
            points_uv=outside_court[:, [0, 2]],
            scores=np.ones(30, dtype=np.float32),
            selected_pixel_count=30,
        ),
    ]

    metrics = evaluate(observations, np.eye(4), config)

    assert metrics["view_count"] == 2
    assert metrics["evaluable_view_count"] == 1
    assert metrics["accepted_view_count"] == 1
    assert metrics["accepted_view_fraction"] == 1.0
    assert metrics["by_view"]["other-court"]["evaluable"] is False
