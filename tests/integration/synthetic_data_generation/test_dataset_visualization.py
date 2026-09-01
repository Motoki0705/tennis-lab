"""CPU integration tests for streaming visualization and MP4 publication."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.renderer as renderer_module
from src.synthetic_data_generation.dataset.blcs.contracts import BLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_CLASS_NAMES_V2,
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.runtime import (
    LogicalRenderSample,
    RenderSampleKey,
)
from src.synthetic_data_generation.visualization import (
    VISUALIZATION_METADATA_SCHEMA,
    CourtOverlayConfiguration,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
    visualize_dataset,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSSourceFrame,
    CourtSourceFrame,
    PLCSSourceFrame,
)
from src.utils.video.reader import probe_video_info

_V1_SIDECAR_KEYS = {
    "schema",
    "domain",
    "dataset_schema",
    "dataset_scene_id",
    "selection",
    "frame_count",
    "source_frame_order",
    "source_width",
    "source_height",
    "width",
    "height",
    "padding",
    "output_fps",
    "source_fps",
    "history_frames",
    "video",
}


def _root(tmp_path: Path, domain: str) -> Path:
    root = tmp_path / "scenes" / "scene-0" / "datasets" / domain
    root.mkdir(parents=True)
    return root


def _semantic_court_overlay() -> CourtOverlayConfiguration:
    return CourtOverlayConfiguration(
        mode=CourtOverlayMode.SEMANTIC,
        color_rgb=(255, 96, 32),
        background_color_rgb=(0, 0, 0),
        opacity=0.55,
        depth_epsilon_m=0.02,
        near_plane_m=0.05,
        maximum_cells=1_000_000,
        maximum_surface_faces=4_000_000,
        maximum_projected_pixels=100_000_000,
    )


def _logical_render(frame_index: int) -> LogicalRenderSample:
    rgb: NDArray[np.float32] = np.zeros((96, 128, 3), dtype=np.float32)
    rgb[..., 1] = 0.15 + frame_index * 0.1
    instance_ids: NDArray[np.int32] = np.zeros((96, 128), dtype=np.int32)
    instance_ids.reshape(-1)[:24] = 1
    return LogicalRenderSample(
        key=RenderSampleKey(frame_index, "camera-0"),
        rgb=rgb,
        alpha=np.ones((96, 128, 1), dtype=np.float32),
        depth=np.ones((96, 128, 1), dtype=np.float32),
        instance_ids=instance_ids,
    )


def _court_projection() -> dict[str, object]:
    names = (
        "doubles_left",
        "doubles_right",
        "singles_left",
        "singles_right",
        "service_left",
        "service_right",
        "service_t",
    )
    return {
        "courts": [
            {
                "court_instance_id": "court-0",
                "coverage_mode": "full",
                "classes": [
                    {
                        "class_id": class_id,
                        "class_name": name,
                        "renderer_visible": True,
                        "points": [
                            {
                                "uv": [20.0 + class_id * 5, 60.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                            {
                                "uv": [20.0 + class_id * 5, 80.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                        ],
                    }
                    for class_id, name in enumerate(names)
                ],
            }
        ]
    }


def _court_projection_v2() -> dict[str, object]:
    return {
        "courts": [
            {
                "court_instance_id": "court-0",
                "coverage_mode": "full",
                "classes": [
                    {
                        "class_id": class_id,
                        "class_name": name,
                        "renderer_visible": True,
                        "points": [
                            {
                                "physical_index": class_id,
                                "uv": [15.0 + class_id * 6.0, 65.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            }
                        ],
                    }
                    for class_id, name in enumerate(COURT_SEMANTIC_CLASS_NAMES_V2)
                ],
            }
        ]
    }


class _FakeCourt:
    dataset_schema = "canonical_court_dataset_v1"
    dataset_scene_id = "scene-0"
    width = 128
    height = 96
    frame_order = tuple(
        {
            "sample_id": f"sample-{index}",
            "view_id": "view-0",
            "trajectory_frame_index": index,
        }
        for index in range(3)
    )

    def __init__(self, root: Path, *, trajectory_id: str) -> None:
        assert root.name == "court"
        assert trajectory_id == "orbit-0"

    def frames(self) -> Iterator[CourtSourceFrame]:
        for index in range(3):
            rgb: NDArray[np.float32] = np.zeros((96, 128, 3), dtype=np.float32)
            rgb[..., 0] = 0.1 * index
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=f"sample-{index}",
                view_id="view-0",
                trajectory_frame_index=index,
                projection=_court_projection(),
            )


class _FakeCourtV2(_FakeCourt):
    dataset_schema = "canonical_court_dataset_v2"

    def frames(self) -> Iterator[CourtSourceFrame]:
        for index in range(3):
            rgb: NDArray[np.float32] = np.zeros((96, 128, 3), dtype=np.float32)
            rgb[..., 0] = 0.1 * index
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=f"sample-{index}",
                view_id="view-0",
                trajectory_frame_index=index,
                projection=_court_projection_v2(),
                schema_version=CourtDatasetSchemaVersion.V2,
            )


class _FakeCourtV3(_FakeCourtV2):
    dataset_schema = "canonical_court_dataset_v3"

    def frames(self) -> Iterator[CourtSourceFrame]:
        for frame in super().frames():
            yield CourtSourceFrame(
                rgb=frame.rgb,
                sample_id=frame.sample_id,
                view_id=frame.view_id,
                trajectory_frame_index=frame.trajectory_frame_index,
                projection=frame.projection,
                schema_version=CourtDatasetSchemaVersion.V3,
            )


class _FakeBLCS:
    dataset_schema = BLCS_DATASET_SCHEMA
    dataset_scene_id = "scene-0"
    width = 128
    height = 96
    source_fps = 60.0
    object_ids = ("ball-0",)
    court_kp: NDArray[np.float32] = np.zeros((20, 2), dtype=np.float32)
    court_vis: NDArray[np.bool_] = np.zeros((20,), dtype=np.bool_)
    frame_order = tuple(
        {"source_frame_index": index, "global_frame_index": index} for index in range(3)
    )

    def __init__(self, root: Path, *, logical_scene_id: str, camera_id: str) -> None:
        assert root.name == "blcs"
        assert logical_scene_id == "logical-0"
        assert camera_id == "camera-0"

    def frames(self) -> Iterator[BLCSSourceFrame]:
        for index in range(3):
            yield BLCSSourceFrame(
                render=_logical_render(index),
                source_frame_index=index,
                global_frame_index=index,
                metadata={
                    "objects": [
                        {
                            "object_id": "ball-0",
                            "instance_id": 1,
                            "present": True,
                            "geometric_visible": True,
                            "rendered_visible": True,
                        }
                    ],
                    "semantic_arrays": {
                        "ball_uv": [[40.0 + index * 8, 62.0]],
                        "present": [True],
                        "geometric_visible": [True],
                        "rendered_visible": [True],
                        "instance_ids": [1],
                    },
                },
            )


class _FakePLCS:
    dataset_schema = PLCS_DATASET_SCHEMA
    dataset_scene_id = "scene-0"
    width = 128
    height = 96
    object_ids = ("person-0",)
    frame_order = tuple({"frame_index": index} for index in range(3))

    def __init__(self, root: Path, *, logical_scene_id: str, camera_id: str) -> None:
        assert root.name == "plcs"
        assert logical_scene_id == "logical-0"
        assert camera_id == "camera-0"

    def frames(self) -> Iterator[PLCSSourceFrame]:
        for index in range(3):
            keypoints: NDArray[np.float32] = np.zeros((1, 17, 2), dtype=np.float32)
            keypoints[0, 5] = (0.4, 0.4)
            keypoints[0, 6] = (0.6, 0.4)
            visible: NDArray[np.bool_] = np.zeros((1, 17), dtype=np.bool_)
            visible[0, 5:7] = True
            yield PLCSSourceFrame(
                render=_logical_render(index),
                frame_index=index,
                label={
                    "objects": [
                        {
                            "object_id": "person-0",
                            "instance_id": 1,
                            "present": True,
                            "visible_pixel_count": 24,
                        }
                    ]
                },
                human_kp=keypoints,
                human_vis=visible,
                court_kp=np.zeros((20, 2), dtype=np.float32),
                court_vis=np.zeros((20,), dtype=np.bool_),
                present=np.ones((1,), dtype=np.bool_),
            )


class _FailingCourt(_FakeCourt):
    def frames(self) -> Iterator[CourtSourceFrame]:
        frames = super().frames()
        yield next(frames)
        raise ValueError("corrupt source frame")


class _OddSizedCourt(_FakeCourt):
    width = 127
    height = 95

    def frames(self) -> Iterator[CourtSourceFrame]:
        for index in range(3):
            rgb: NDArray[np.float32] = np.zeros((95, 127, 3), dtype=np.float32)
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=f"sample-{index}",
                view_id="view-0",
                trajectory_frame_index=index,
                projection=_court_projection(),
            )


class _ConcurrentVideoWriter:
    barrier: threading.Barrier | None = None

    def __init__(self, path: Path, *, fps: float, crf: int) -> None:
        del fps, crf
        self.path = path
        self.frame_count = 0

    def __enter__(self) -> _ConcurrentVideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.path.write_text(threading.current_thread().name, encoding="utf-8")
        barrier = type(self).barrier
        assert barrier is not None
        barrier.wait(timeout=10.0)

    def write_frame(self, frame: NDArray[np.uint8]) -> None:
        assert frame.dtype == np.uint8
        self.frame_count += 1


def test_court_orbit_streams_exact_manifest_sequence_to_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _FakeCourt)
    output = tmp_path / "court.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    result = visualize_dataset(request)

    assert probe_video_info(output).frame_count == 3
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["source_frame_order"] == list(_FakeCourt.frame_order)
    assert payload["selection"]["trajectory_id"] == "orbit-0"
    assert set(payload) == _V1_SIDECAR_KEYS
    assert payload["schema"] == VISUALIZATION_METADATA_SCHEMA


@pytest.mark.parametrize(
    ("source", "dataset_schema"),
    [
        (_FakeCourtV2, "canonical_court_dataset_v2"),
        (_FakeCourtV3, "canonical_court_dataset_v3"),
    ],
)
def test_court_singleton_overlay_streams_to_mp4_without_v1_reshaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: type[_FakeCourtV2],
    dataset_schema: str,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", source)
    output = tmp_path / f"court-{dataset_schema}.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    result = visualize_dataset(request)

    assert probe_video_info(output).frame_count == 3
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["dataset_schema"] == dataset_schema
    assert payload["frame_count"] == 3


def test_odd_canonical_dimensions_are_explicitly_padded_for_yuv420(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _OddSizedCourt)
    output = tmp_path / "odd-court.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    result = visualize_dataset(request)

    info = probe_video_info(output)
    assert (info.width, info.height) == (128, 96)
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert (payload["source_width"], payload["source_height"]) == (127, 95)
    assert (payload["width"], payload["height"]) == (128, 96)
    assert payload["padding"] == {"bottom": 1, "right": 1}


def test_stream_failure_leaves_no_partial_video_or_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _FailingCourt)
    old_video_staging = tmp_path / ".failed.staging.mp4"
    old_metadata_staging = tmp_path / ".failed.staging.json"
    old_video_staging.write_bytes(b"another invocation")
    old_metadata_staging.write_bytes(b"another invocation")
    output = tmp_path / "failed.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    with pytest.raises(ValueError, match="corrupt source frame"):
        visualize_dataset(request)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert old_video_staging.read_bytes() == b"another invocation"
    assert old_metadata_staging.read_bytes() == b"another invocation"
    assert not tuple(tmp_path.glob(".failed.mp4.*.staging*"))


def test_concurrent_publication_has_one_exclusive_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _FakeCourt)
    monkeypatch.setattr(renderer_module, "VideoWriter", _ConcurrentVideoWriter)
    _ConcurrentVideoWriter.barrier = threading.Barrier(2)
    output = tmp_path / "concurrent.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="visualizer") as pool:
        futures = [pool.submit(visualize_dataset, request) for _ in range(2)]
        results = []
        failures = []
        for future in futures:
            try:
                results.append(future.result())
            except FileExistsError as error:
                failures.append(error)

    assert len(results) == 1
    assert len(failures) == 1
    assert output.read_text(encoding="utf-8").startswith("visualizer")
    assert (
        json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))[
            "frame_count"
        ]
        == 3
    )
    assert not tuple(tmp_path.glob(".concurrent.mp4.*.staging*"))


def test_late_racer_is_not_overwritten_or_removed_during_owned_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _FakeCourt)
    output = tmp_path / "raced.mp4"
    metadata_path = output.with_suffix(".json")
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )
    original_publish = renderer_module._publish_exclusively

    def publish_with_late_racer(staged: Path, target: Path) -> None:
        if target == output:
            metadata_path.unlink()
            metadata_path.write_bytes(b"racer metadata")
            output.write_bytes(b"racer video")
        original_publish(staged, target)

    monkeypatch.setattr(
        renderer_module,
        "_publish_exclusively",
        publish_with_late_racer,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        visualize_dataset(request)

    assert output.read_bytes() == b"racer video"
    assert metadata_path.read_bytes() == b"racer metadata"
    assert not tuple(tmp_path.glob(".raced.mp4.*.staging*"))


def test_video_publication_failure_rolls_back_owned_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(renderer_module, "CourtVisualizationSource", _FakeCourt)
    output = tmp_path / "publication-failed.mp4"
    request = DatasetVisualizationRequest(
        domain=DatasetVisualizationDomain.COURT,
        dataset_root=_root(tmp_path, "court"),
        output_video=output,
        trajectory_id="orbit-0",
        logical_scene_id=None,
        camera_id=None,
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )
    original_publish = renderer_module._publish_exclusively

    def fail_video_publication(staged: Path, target: Path) -> None:
        if target == output:
            raise OSError("simulated final video link failure")
        original_publish(staged, target)

    monkeypatch.setattr(
        renderer_module,
        "_publish_exclusively",
        fail_video_publication,
    )

    with pytest.raises(OSError, match="simulated final video link failure"):
        visualize_dataset(request)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not tuple(tmp_path.glob(".publication-failed.mp4.*.staging*"))


@pytest.mark.parametrize(
    ("domain", "fake_name", "fake_type"),
    [
        (DatasetVisualizationDomain.BLCS, "BLCSVisualizationSource", _FakeBLCS),
        (DatasetVisualizationDomain.PLCS, "PLCSVisualizationSource", _FakePLCS),
    ],
)
def test_compact_view_streams_three_frames_to_mp4_and_deterministic_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: DatasetVisualizationDomain,
    fake_name: str,
    fake_type: type[_FakeBLCS] | type[_FakePLCS],
) -> None:
    monkeypatch.setattr(renderer_module, fake_name, fake_type)
    output = tmp_path / f"{domain.value}.mp4"
    request = DatasetVisualizationRequest(
        domain=domain,
        dataset_root=_root(tmp_path, domain.value),
        output_video=output,
        trajectory_id=None,
        logical_scene_id="logical-0",
        camera_id="camera-0",
        fps=12.0,
        crf=20,
        history_frames=3,
        court_overlay=_semantic_court_overlay(),
    )

    result = visualize_dataset(request)

    info = probe_video_info(output)
    assert info.frame_count == 3
    assert (info.width, info.height) == (128, 96)
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["domain"] == domain.value
    assert payload["frame_count"] == 3
    assert payload["selection"] == {
        "camera_id": "camera-0",
        "logical_scene_id": "logical-0",
        "trajectory_id": None,
    }
    assert set(payload) == _V1_SIDECAR_KEYS
    assert payload["schema"] == VISUALIZATION_METADATA_SCHEMA
    assert "created_at" not in payload
    assert not tuple(tmp_path.glob(f".{domain.value}.mp4.*.staging*"))
