"""Compact BLCS rendering tests with an explicit test-only CPU oracle."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSCompactDatasetReader,
    assemble_blcs_dataset,
    validate_blcs_dataset,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSNHTRenderer,
    CUDABLCSForegroundCompositor,
    build_blcs_sample_metadata,
)
from src.synthetic_data_generation.dataset.blcs.timeline import (
    BLCSTrajectoryPlan,
    build_blcs_plans,
)
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    DatasetPerformanceBudget,
    ForegroundDeltaBatch,
    PerformanceTimer,
    RenderSampleKey,
    sparse_delta_from_composite,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderArrays,
    NHTRenderCommandRequest,
    NHTRenderRecord,
    NHTRenderResult,
)


@dataclass
class _ExplicitCPUOracle:
    """Independent dense NumPy reference; never selected by production code."""

    execution_device: str = "test-cpu-oracle"
    cuda_peak_bytes: int = 0

    def compose(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        backgrounds: Mapping[str, BackgroundArrays],
        ball_radius_m: float,
    ) -> Iterator[ForegroundDeltaBatch]:
        for chunk in plan.chunks:
            deltas = []
            metadata = []
            for frame_index in chunk.frame_indices:
                for camera_index, sampled in enumerate(plan.camera_rig.cameras):
                    camera = sampled.scene_camera
                    background = backgrounds[camera.camera_id]
                    rgb = np.array(background.rgb, copy=True)
                    alpha = np.array(background.alpha, copy=True)
                    depth = np.array(background.depth, copy=True)
                    labels: NDArray[np.int32] = np.zeros(
                        (camera.height, camera.width), dtype=np.int32
                    )
                    focal = float(camera.intrinsics[0])
                    for object_index in range(plan.source.object_count):
                        if not plan.geometric_visible[
                            frame_index, camera_index, object_index
                        ]:
                            continue
                        centre = plan.camera_uv[frame_index, camera_index, object_index]
                        object_depth = float(
                            plan.camera_depth[frame_index, camera_index, object_index]
                        )
                        radius = max(
                            1, int(round(focal * ball_radius_m / object_depth))
                        )
                        x_min = max(0, int(math.floor(centre[0] - radius)))
                        x_max = min(
                            camera.width,
                            int(math.ceil(centre[0] + radius + 1)),
                        )
                        y_min = max(0, int(math.floor(centre[1] - radius)))
                        y_max = min(
                            camera.height,
                            int(math.ceil(centre[1] + radius + 1)),
                        )
                        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                        disc = (xx - centre[0]) ** 2 + (
                            yy - centre[1]
                        ) ** 2 <= radius**2
                        local_depth = depth[y_min:y_max, x_min:x_max, 0]
                        visible = disc & (
                            (local_depth <= 0.0) | (object_depth <= local_depth)
                        )
                        rgb[y_min:y_max, x_min:x_max][visible] = (1.0, 0.85, 0.0)
                        alpha[y_min:y_max, x_min:x_max, 0][visible] = 1.0
                        local_depth[visible] = object_depth
                        labels[y_min:y_max, x_min:x_max][visible] = object_index + 1
                    delta = sparse_delta_from_composite(
                        key=RenderSampleKey(frame_index, camera.camera_id),
                        background=background,
                        rgb=rgb,
                        alpha=alpha,
                        depth=depth,
                        instance_ids=labels,
                    )
                    deltas.append(delta)
                    metadata.append(
                        build_blcs_sample_metadata(
                            plan=plan,
                            source_frame_index=frame_index,
                            camera_index=camera_index,
                            chunk_index=chunk.chunk_index,
                            delta=delta,
                        )
                    )
            yield ForegroundDeltaBatch(
                chunk_id=f"chunk-{chunk.chunk_index:06d}",
                deltas=tuple(deltas),
                metadata=tuple(metadata),
            )


class _FakeNHTClient:
    def __init__(self) -> None:
        self.requests: list[NHTRenderCommandRequest] = []

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment=None,
        timeout_seconds=None,
    ) -> NHTRenderResult:
        del environment, timeout_seconds
        self.requests.append(request)
        assert request.arbitrary_cameras is not None
        request.output_directory.mkdir(parents=True, exist_ok=False)
        records = []
        for camera in request.arbitrary_cameras.cameras:
            root = request.output_directory / camera.camera_id
            root.mkdir()
            rgb_path = root / "rgb.npy"
            alpha_path = root / "alpha.npy"
            depth_path = root / "depth.npy"
            np.save(
                rgb_path,
                np.full((camera.height, camera.width, 3), 0.25, np.float32),
                allow_pickle=False,
            )
            np.save(
                alpha_path,
                np.ones((camera.height, camera.width, 1), np.float32),
                allow_pickle=False,
            )
            np.save(
                depth_path,
                np.full((camera.height, camera.width, 1), 100.0, np.float32),
                allow_pickle=False,
            )
            record = NHTRenderRecord(
                camera_id=camera.camera_id,
                request_source="arbitrary",
                width=camera.width,
                height=camera.height,
                rgb_path=rgb_path,
                rgb_preview_path=root / "unused.png",
                alpha_path=alpha_path,
                alpha_preview_path=root / "unused.png",
                depth_path=depth_path,
            )
            record._bind_arrays(
                NHTRenderArrays(
                    rgb=np.load(rgb_path, allow_pickle=False),
                    alpha=np.load(alpha_path, allow_pickle=False),
                    depth=np.load(depth_path, allow_pickle=False),
                )
            )
            records.append(record)
        return NHTRenderResult(
            scene_id="B00",
            output_directory=request.output_directory,
            records=tuple(records),
        )


def _budget() -> DatasetPerformanceBudget:
    return DatasetPerformanceBudget(
        maximum_wall_seconds=60.0,
        maximum_published_bytes=1024**3,
        maximum_published_fraction_of_dense_reference=0.2,
        maximum_nht_invocations=3,
        maximum_background_cache_misses=18,
        maximum_complete_array_scans_per_sample=1,
        maximum_batch_frames=2,
        execution_device="test-cpu-oracle",
        require_cuda=False,
    )


def test_compact_attempt_reuses_backgrounds_and_materializes_on_demand(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plans = build_blcs_plans(
        tuple(
            blcs_trajectory_factory(f"trajectory-{index}", frame_count=3)
            for index in range(3)
        ),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=695,
        chunk_size_frames=2,
    )
    client = _FakeNHTClient()
    renderer = BLCSNHTRenderer(
        assets=blcs_assets,
        client=client,  # type: ignore[arg-type]
        executable="nht-render",
        environment={},
        timeout_seconds=60.0,
        execution_device="test-cpu-oracle",
        maximum_batch_frames=2,
        test_cpu_oracle=_ExplicitCPUOracle(),
    )
    output = tmp_path / ".transactions" / "blcs_dataset" / "snapshot"
    output.mkdir(parents=True)
    scene_path = tmp_path / "reconstruction" / "export" / "scene.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}\n", encoding="utf-8")
    timer = PerformanceTimer()
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(np.eye(4))

    attempt = renderer.render(
        plans=plans,
        scene_path=scene_path,
        samples_directory=output / "samples",
        metric_adapter=adapter,
        attempt_token="attempt-1",
    )
    assembled = assemble_blcs_dataset(
        output,
        plans=plans,
        metric_adapter=adapter,
        render_attempt=attempt,
        performance_timer=timer,
        performance_budget=_budget(),
    )
    validated = validate_blcs_dataset(output)

    assert len(client.requests) == 3
    assert attempt.nht_invocations == 3
    assert attempt.background_cache_misses == 18
    assert assembled.performance.execution_device == "test-cpu-oracle"
    assert assembled.performance.published_bytes == sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    assert (
        assembled.performance.generated_bytes >= assembled.performance.published_bytes
    )
    assert (
        assembled.performance.published_bytes
        <= 0.2 * assembled.performance.dense_reference_bytes
    )
    assert assembled.performance.complete_array_scans == len(assembled.sample_records)
    assert len(validated.sample_records) == 3 * 3 * 6
    assert not list(output.rglob("frame-*"))
    assert not list(output.rglob("camera-*.json"))
    assert len(list(output.rglob("foreground.npz"))) == 6
    assert len(list(output.rglob("chunk.json"))) == 6
    assert not list(output.rglob("shard.json"))

    first = validated.sample_records[0]
    reader = BLCSCompactDatasetReader(output)
    logical = reader.materialize(
        trajectory_id=first.trajectory_id,
        source_frame_index=first.source_frame_index,
        camera_id=first.camera_id,
    )
    assert logical.render.rgb.shape == (24, 32, 3)
    assert logical.render.instance_ids.dtype == np.int32
    assert logical.semantic_arrays["ball_uv"].shape == (1, 2)
    assert logical.metadata["target_court"] in {"court-0", "court-1"}
    all_views = reader.materialize_all_views(first.trajectory_id)
    assert all_views.ball_uv.shape == (6, 3, 1, 2)
    assert all_views.court_kp.shape == (6, 20, 2)
    assert all_views.index.camera_ids == tuple(
        record.camera_id
        for record in validated.sample_records
        if record.trajectory_id == first.trajectory_id
        and record.source_frame_index == 0
    )

    first_chunk_marker = next(output.rglob("chunk.json"))
    first_chunk_marker.unlink()
    with pytest.raises(FileNotFoundError, match="chunk.json"):
        validate_blcs_dataset(output)


def test_production_compositor_has_no_cpu_fallback() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        CUDABLCSForegroundCompositor(device="cpu", maximum_batch_frames=2)
    with pytest.raises(ValueError, match="requires explicit CUDA"):
        BLCSNHTRenderer(
            assets=object(),  # type: ignore[arg-type]
            client=object(),  # type: ignore[arg-type]
            executable="nht-render",
            environment={},
            timeout_seconds=1.0,
            execution_device="cpu",
            maximum_batch_frames=2,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bounded_cuda_compositor_matches_explicit_cpu_oracle(
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=3),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=9,
        chunk_size_frames=2,
    )[0]
    backgrounds = {
        sampled.scene_camera.camera_id: BackgroundArrays(
            camera_id=sampled.scene_camera.camera_id,
            rgb=np.full((24, 32, 3), 0.25, dtype=np.float32),
            alpha=np.ones((24, 32, 1), dtype=np.float32),
            depth=np.full((24, 32, 1), 100.0, dtype=np.float32),
        )
        for sampled in plan.camera_rig.cameras
    }
    oracle = tuple(
        _ExplicitCPUOracle().compose(
            plan=plan,
            backgrounds=backgrounds,
            ball_radius_m=blcs_assets.ball_radius_m,
        )
    )
    cuda_compositor = CUDABLCSForegroundCompositor(
        device="cuda:0", maximum_batch_frames=2
    )
    cuda = tuple(
        cuda_compositor.compose(
            plan=plan,
            backgrounds=backgrounds,
            ball_radius_m=blcs_assets.ball_radius_m,
        )
    )

    assert len(cuda) == len(oracle) == 2
    for actual_batch, expected_batch in zip(cuda, oracle, strict=True):
        assert actual_batch.chunk_id == expected_batch.chunk_id
        assert actual_batch.metadata == expected_batch.metadata
        for actual, expected in zip(
            actual_batch.deltas, expected_batch.deltas, strict=True
        ):
            np.testing.assert_array_equal(actual.pixel_indices, expected.pixel_indices)
            np.testing.assert_array_equal(actual.rgb, expected.rgb)
            np.testing.assert_array_equal(actual.alpha, expected.alpha)
            np.testing.assert_array_equal(actual.depth, expected.depth)
            np.testing.assert_array_equal(actual.instance_ids, expected.instance_ids)
    assert cuda_compositor.cuda_peak_bytes > 0
