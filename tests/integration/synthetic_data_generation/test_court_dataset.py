"""CPU integration of config, alignment layout, planning, sampling, and assignment."""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentPartitions,
    AlignmentResult,
    CandidateAlignment,
    MetricSceneAdapter,
    PartitionAssessment,
    PartitionMetrics,
    PartitionThresholds,
)
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court import assembler as court_assembler
from src.synthetic_data_generation.dataset.court.assembler import (
    assemble_court_dataset,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlan
from src.synthetic_data_generation.dataset.court.performance import (
    CourtPerformanceEvidence,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import CourtNHTRenderer
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    COURT_SEMANTIC_MANIFEST_PATH,
    require_equal_court_semantic_manifests,
)
from src.synthetic_data_generation.dataset.runtime import PerformanceTimer
from src.synthetic_data_generation.reconstruction.scene_export import (
    NHT_ALPHA_OUTPUT_SEMANTICS,
    NHT_DEPTH_OUTPUT_SEMANTICS,
    NHT_IMAGE_RESOLUTION_SEMANTICS,
    NHT_PIXEL_COORDINATE_CONVENTION,
    NHT_RGB_OUTPUT_SEMANTICS,
    NHT_SCENE_COORDINATE_CONVENTION,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderClient
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.io import load_json


def test_court_domain_resolves_production_quantities_and_balanced_courts() -> None:
    layout = _layout()
    configuration = CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(
                Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
            ),
            resolve=True,
        )
    )
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=_captured_cameras(),
        layout=layout,
        configuration=configuration,
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
    )
    assert len(plan.groups) >= 24
    assert 2_000 <= plan.proposal_count <= 5_000
    assert max(group.maximum_adjacent_step_m for group in plan.groups) <= 1.05
    global_counts = Counter(
        group.target_court.court_instance_id for group in plan.groups
    )
    assert set(global_counts) == {court.court_instance_id for court in layout.courts}
    assert max(global_counts.values()) - min(global_counts.values()) <= 1
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for group in plan.groups:
        by_split[group.split.value][group.target_court.court_instance_id] += 1
    assert all(
        max(counts.values()) - min(counts.values()) <= 1
        for counts in by_split.values()
    )
    assert all(
        group.trajectory.center_court_instance_id is None
        or group.trajectory.center_court_instance_id
        == group.target_court.court_instance_id
        for group in plan.groups
    )


def test_same_seed_public_renderer_runs_publish_equal_semantic_manifests(
    tmp_path: Path,
) -> None:
    executable = _write_fake_nht_render(tmp_path / "bin/nht-render")
    first_plan, first_manifest, first_root = _execute_court_render(
        tmp_path / "repeat-a",
        executable=executable,
        rgb_value=0.2,
        verify_attempt_local_reuse=True,
    )
    second_plan, second_manifest, second_root = _execute_court_render(
        tmp_path / "repeat-b",
        executable=executable,
        rgb_value=0.8,
    )

    assert first_plan.to_dict() == second_plan.to_dict()
    require_equal_court_semantic_manifests(first_manifest, second_manifest)
    assert first_manifest == second_manifest
    first_dataset = _json_mapping(load_json(first_root / "dataset.json"))
    second_dataset = _json_mapping(load_json(second_root / "dataset.json"))
    first_performance = CourtPerformanceEvidence.from_dict(
        load_json(first_root / "diagnostics/performance.json")
    )
    second_performance = CourtPerformanceEvidence.from_dict(
        load_json(second_root / "diagnostics/performance.json")
    )
    assert first_manifest["trajectory_groups"] == first_dataset["trajectory_groups"]
    assert first_manifest["counts"] == second_manifest["counts"]
    assert first_performance.post_render_rejected_sample_count > 0
    assert (
        first_performance.post_render_rejected_sample_count
        == second_performance.post_render_rejected_sample_count
    )
    assert first_performance.fresh_rendered_sample_count == (
        first_performance.renderable_sample_count
    )
    assert first_performance.retained_nht_array_bytes == 0
    _assert_no_operational_manifest_fields(first_manifest)

    first_record = _first_accepted_record(first_dataset)
    second_record = _first_accepted_record(second_dataset)
    first_rgb = np.load(_record_path(first_root, first_record, "rgb"), allow_pickle=False)
    second_rgb = np.load(
        _record_path(second_root, second_record, "rgb"), allow_pickle=False
    )
    assert not np.array_equal(first_rgb, second_rgb)
    _assert_repeat_semantic_mutations_fail(first_manifest)


def _layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((-8.0, 8.0)):
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
                holdout_metrics={"rms_error_m": 0.02},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id="court-0",
    )


def _captured_cameras() -> tuple[SceneCamera, ...]:
    result = []
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    ):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        result.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=64,
                height=48,
                intrinsics=(
                    100.0,
                    0.0,
                    31.5,
                    0.0,
                    100.0,
                    23.5,
                    0.0,
                    0.0,
                    1.0,
                ),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(result)


def _configuration() -> CourtDatasetConfiguration:
    return CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(
                Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
            ),
            resolve=True,
        )
    )


def _execute_court_render(
    workspace: Path,
    *,
    executable: Path,
    rgb_value: float,
    verify_attempt_local_reuse: bool = False,
) -> tuple[CourtDatasetPlan, dict[str, object], Path]:
    alignment = _alignment()
    scene_path = _write_standard_scene(workspace, _render_cameras())
    renderer = CourtNHTRenderer(
        executable=executable,
        client=NHTRenderClient(),
        environment={
            "FAKE_NHT_REJECT_FIRST": "1",
            "FAKE_NHT_RGB_VALUE": str(rgb_value),
        },
        timeout_seconds=180.0,
    )
    scene = renderer.preflight(scene_path)
    configuration = _configuration()
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=scene.cameras,
        layout=alignment.layout,
        configuration=configuration,
        metric_adapter=alignment.metric_adapter,
    )
    dataset_root = workspace / "datasets/court"
    dataset_root.mkdir(parents=True)
    attempt_root = dataset_root / "_attempt"
    timer = PerformanceTimer()
    result = renderer.render(
        plan=plan,
        scene=scene,
        attempt_root=attempt_root,
        attempt_token="repeat-attempt",
        alignment=alignment,
    )
    reused_result = None
    if verify_attempt_local_reuse:
        reused_result = renderer.render(
            plan=plan,
            scene=scene,
            attempt_root=attempt_root,
            attempt_token="repeat-attempt",
            alignment=alignment,
        )
        assert reused_result.samples == result.samples
        assert reused_result.nht_invocations == 0
        assert reused_result.nht_complete_array_scans == 0
        assert reused_result.retained_nht_array_bytes == 0
    report = assemble_court_dataset(
        dataset_root,
        plan=plan,
        layout=alignment.layout,
        metric_adapter=alignment.metric_adapter,
        render_result=result,
        configuration=configuration,
        attempt_root=attempt_root,
        performance_timer=timer,
    )
    if reused_result is not None:
        reused_evidence_root = workspace / "reused-performance-evidence"
        (reused_evidence_root / "diagnostics").mkdir(parents=True)
        reused_evidence = court_assembler._write_performance_evidence(
            reused_evidence_root,
            timer=PerformanceTimer(),
            render_result=reused_result,
            proposal_count=report.proposal_count,
            accepted_frame_count=report.accepted_frame_count,
            rejected_frame_count=report.rejected_frame_count,
            accepted_staged_complete_array_scans=report.accepted_frame_count,
            post_render_rejected_staged_complete_array_scans=(
                report.performance.post_render_rejected_sample_count
            ),
            budget=configuration.performance,
            visible_by_class=report.performance.visible_points_by_class,
        )
        assert reused_evidence.fresh_run_complete_array_scan_requirement == (
            report.performance.fresh_run_complete_array_scan_requirement
        )
        assert reused_evidence.complete_array_scan_budget_capacity == (
            report.performance.complete_array_scan_budget_capacity
        )
        assert reused_evidence.metrics.complete_array_scans < (
            report.performance.metrics.complete_array_scans
        )
    manifest = _json_mapping(
        load_json(dataset_root / COURT_SEMANTIC_MANIFEST_PATH)
    )
    return plan, manifest, dataset_root


def _alignment() -> AlignmentResult:
    thresholds = PartitionThresholds(
        minimum_camera_count=1,
        minimum_correspondence_count=3,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    policy = AlignmentAcceptancePolicy(fit=thresholds, holdout=thresholds)
    fit = PartitionAssessment.evaluate(_partition_metrics("captured-0"), thresholds)
    holdout = PartitionAssessment.evaluate(
        _partition_metrics("captured-1"), thresholds
    )
    candidates = []
    for index, x in enumerate((-8.0, 8.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        scene_from_court = RigidTransform.from_matrix(matrix)
        candidates.append(
            CandidateAlignment(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit=fit,
                holdout=holdout,
            )
        )
    candidate_tuple = tuple(candidates)
    return AlignmentResult(
        partitions=AlignmentPartitions(
            fit_camera_ids=("captured-0",),
            holdout_camera_ids=("captured-1",),
        ),
        policy=policy,
        candidates=candidate_tuple,
        layout=MultiCourtLayout(
            courts=tuple(candidate.to_court_instance() for candidate in candidate_tuple),
            complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
            primary_court_instance_id="court-0",
        ),
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
    )


def _partition_metrics(camera_id: str) -> PartitionMetrics:
    return PartitionMetrics(
        camera_ids=(camera_id,),
        correspondence_count=3,
        inlier_count=3,
        inlier_fraction=1.0,
        rms_error_m=0.0,
        q95_error_m=0.0,
        maximum_error_m=0.0,
    )


def _render_cameras() -> tuple[SceneCamera, ...]:
    cameras = []
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    ):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        cameras.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=16,
                height=12,
                intrinsics=(25.0, 0.0, 7.5, 0.0, 25.0, 5.5, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(cameras)


def _write_standard_scene(workspace: Path, cameras: tuple[SceneCamera, ...]) -> Path:
    export_root = workspace / "reconstruction/export"
    image_root = export_root / "images"
    model_root = export_root / "model/ckpts"
    image_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    camera_records = []
    for camera in cameras:
        image_name = f"{camera.camera_id}.png"
        Image.new("RGB", (camera.width, camera.height)).save(image_root / image_name)
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        camera_records.append(
            {
                "camera_id": camera.camera_id,
                "source_frame_index": camera.source_frame_index,
                "time_seconds": float(camera.source_frame_index),
                "split": "train",
                "image": f"images/{image_name}",
                "width": camera.width,
                "height": camera.height,
                "intrinsics": {
                    "model": "PINHOLE",
                    "distortion_model": "NONE",
                    "params": [
                        float(intrinsics[0, 0]),
                        float(intrinsics[1, 1]),
                        float(intrinsics[0, 2]),
                        float(intrinsics[1, 2]),
                    ],
                    "matrix": intrinsics.tolist(),
                },
                "camera_to_scene": camera.camera_to_scene.matrix().tolist(),
                "source_image_processing": {
                    "source_resolution": [camera.width, camera.height],
                    "crop_xywh": [0, 0, camera.width, camera.height],
                    "undistorted": True,
                    "data_factor": 1,
                },
                "diagnostics": {
                    "sfm_camera_id": camera.source_frame_index + 1,
                    "sfm_camera_to_world": camera.camera_to_scene.matrix().tolist(),
                },
                "group": "default",
            }
        )
    runtime_config = {
        "schema": "nht_runtime_config_v1",
        "camera_model": "pinhole",
        "pose_opt": False,
        "primitive_type": "3dgs",
        "antialiased": False,
        "packed": False,
        "tile_size": 16,
        "with_ut": True,
        "with_eval3d": True,
        "near_plane": 0.01,
        "far_plane": 100.0,
        "deferred_opt_feature_dim": 48,
        "deferred_opt_enable_view_encoding": True,
        "deferred_opt_view_encoding_type": "sh",
        "deferred_mlp_hidden_dim": 128,
        "deferred_mlp_num_layers": 3,
        "deferred_opt_sh_degree": 3,
        "deferred_opt_sh_scale": 3.0,
        "deferred_opt_fourier_num_freqs": 4,
        "deferred_opt_center_ray_encoding": False,
        "deferred_decode_activation": "sigmoid",
        "post_processing": None,
    }
    (export_root / "model/runtime-config.json").write_text(
        json.dumps(runtime_config), encoding="utf-8"
    )
    (model_root / "model.pt").write_bytes(b"fake-public-model")
    np.save(
        export_root / "points_scene.npy",
        np.asarray([[0.0, 0.0, 0.0, 1.0, 0.5, 0.0]], dtype=np.float32),
    )
    identity = np.eye(4, dtype=np.float64).tolist()
    (export_root / "cameras.json").write_text(
        json.dumps(
            {
                "schema": "nht_standard_cameras_v1",
                "camera_coordinate_convention": "x-right, y-down, z-forward",
                "transform_semantics": (
                    "camera_to_scene maps homogeneous camera coordinates to scene coordinates"
                ),
                "cameras": camera_records,
            }
        ),
        encoding="utf-8",
    )
    scene = {
        "schema": "nht_standard_scene_v1",
        "scene_id": "B00",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "scene_coordinate_convention": NHT_SCENE_COORDINATE_CONVENTION,
        "pixel_coordinate_convention": NHT_PIXEL_COORDINATE_CONVENTION,
        "image_resolution_semantics": NHT_IMAGE_RESOLUTION_SEMANTICS,
        "camera_count": len(cameras),
        "cameras": "cameras.json",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": [1, 6],
            "dtype": "float32",
            "columns": ["x", "y", "z", "red", "green", "blue"],
            "color_range": [0.0, 1.0],
        },
        "image_root": "images",
        "model_root": "model",
        "scene_from_sfm": identity,
        "sfm_from_scene": identity,
        "normalization": {
            "applied": True,
            "camera_similarity": identity,
            "principal_axis_alignment": identity,
            "upside_down_correction": identity,
        },
        "renderer": {
            "command": "nht-render",
            "model": "model",
            "runtime_config": "model/runtime-config.json",
            "checkpoint": "model/ckpts/model.pt",
            "outputs": {
                "rgb": NHT_RGB_OUTPUT_SEMANTICS,
                "alpha": NHT_ALPHA_OUTPUT_SEMANTICS,
                "depth": NHT_DEPTH_OUTPUT_SEMANTICS,
            },
        },
        "sfm_summary": {},
        "nht_training_summary": {},
        "capabilities": ["nht_rendering_model"],
    }
    scene_path = export_root / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return scene_path


def _write_fake_nht_render(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    interpreter = Path(sys.executable)
    path.write_text(
        f"""#!{interpreter}
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--cameras", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
request = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
rgb_value = float(os.environ["FAKE_NHT_RGB_VALUE"])
reject_first = os.environ.get("FAKE_NHT_REJECT_FIRST") == "1"
previews = {{}}
records = []
for camera_index, camera in enumerate(request["cameras"]):
    camera_id = camera["camera_id"]
    width = camera["width"]
    height = camera["height"]
    frame = output / camera_id
    frame.mkdir()
    np.save(frame / "rgb.npy", np.full((height, width, 3), rgb_value, dtype=np.float32))
    alpha_value = 0.0 if reject_first and camera_index == 0 else 1.0
    np.save(
        frame / "alpha.npy",
        np.full((height, width, 1), alpha_value, dtype=np.float32),
    )
    np.save(frame / "depth.npy", np.ones((height, width, 1), dtype=np.float32))
    key = (width, height)
    if key not in previews:
        rgb_buffer = io.BytesIO()
        alpha_buffer = io.BytesIO()
        Image.new("RGB", key).save(rgb_buffer, format="PNG")
        Image.new("L", key, color=255).save(alpha_buffer, format="PNG")
        previews[key] = (rgb_buffer.getvalue(), alpha_buffer.getvalue())
    rgb_preview, alpha_preview = previews[key]
    (frame / "rgb.png").write_bytes(rgb_preview)
    (frame / "alpha.png").write_bytes(alpha_preview)
    records.append({{
        "camera_id": camera_id,
        "request_source": "arbitrary",
        "width": width,
        "height": height,
        "rgb": f"{{camera_id}}/rgb.npy",
        "rgb_preview": f"{{camera_id}}/rgb.png",
        "alpha": f"{{camera_id}}/alpha.npy",
        "alpha_preview": f"{{camera_id}}/alpha.png",
        "depth": f"{{camera_id}}/depth.npy",
    }})
(output / "render.json").write_text(json.dumps({{
    "schema": "nht_render_result_v1",
    "scene_schema": "nht_standard_scene_v1",
    "scene_id": scene["scene_id"],
    "coordinate_space": "canonical NHT scene space",
    "export_validation": {{}},
    "renders": records,
}}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path.resolve()


def _assert_repeat_semantic_mutations_fail(manifest: dict[str, object]) -> None:
    def mutate_sample_id(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["sample_id"] = "sample-mutated"

    def mutate_split(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["split"] = "validation"

    def mutate_camera_pose(payload: dict[str, object]) -> None:
        camera = _json_mapping(_manifest_sample(payload)["camera"])
        transform = cast(list[object], camera["camera_to_scene"])
        transform[3] = cast(float, transform[3]) + 1.0

    def mutate_disposition(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["disposition"] = "rejected"

    def mutate_class_visibility(payload: dict[str, object]) -> None:
        projection = _json_mapping(
            _manifest_sample(payload)["semantic_projection"]
        )
        courts = cast(list[object], projection["courts"])
        court = _json_mapping(courts[0])
        counts = _json_mapping(court["renderer_visible_points_by_class"])
        class_name = next(iter(counts))
        counts[class_name] = cast(int, counts[class_name]) + 1

    for mutation in (
        mutate_sample_id,
        mutate_split,
        mutate_camera_pose,
        mutate_disposition,
        mutate_class_visibility,
    ):
        mutated = copy.deepcopy(manifest)
        mutation(mutated)
        with pytest.raises(ValueError, match="not exactly equal"):
            require_equal_court_semantic_manifests(manifest, mutated)


def _manifest_sample(manifest: dict[str, object]) -> dict[str, object]:
    samples = cast(list[object], manifest["samples"])
    for sample in samples:
        record = _json_mapping(sample)
        if record.get("disposition") == "accepted":
            return record
    raise AssertionError("Court semantic manifest has no accepted sample.")


def _first_accepted_record(dataset: dict[str, object]) -> dict[str, object]:
    samples = cast(list[object], dataset["samples"])
    if not samples:
        raise AssertionError("Court integration fixture produced no accepted samples.")
    return _json_mapping(samples[0])


def _record_path(root: Path, record: dict[str, object], field: str) -> Path:
    value = record[field]
    if not isinstance(value, str):
        raise TypeError(f"Court record {field} must be a path string.")
    return root / value


def _json_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("Expected a string-keyed JSON object.")
    return value


def _assert_no_operational_manifest_fields(value: object) -> None:
    forbidden = {
        "directory",
        "rgb",
        "rgb_preview",
        "alpha",
        "alpha_preview",
        "depth",
        "labels",
        "image_path",
        "wall_seconds",
        "generated_bytes",
        "published_bytes",
    }
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for item in value.values():
            _assert_no_operational_manifest_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_operational_manifest_fields(item)
