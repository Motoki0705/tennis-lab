"""Strict fake of the public composed NHT file boundary for CPU tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.rendering.nht import (
    NHTComposedChunkRecord,
    NHTComposedRenderClient,
    NHTComposedRenderCommandRequest,
    NHTComposedRenderResult,
    NHTRenderArrays,
    NHTRenderRecord,
    NHTRenderResult,
)


class FakeComposedNHTClient(NHTComposedRenderClient):
    """Publish deterministic joint pixels without emulating Gaussian rasterization."""

    def __init__(self, *, scene_id: str = "B00", cuda_peak_bytes: int = 4096) -> None:
        super().__init__()
        self.scene_id = scene_id
        self.cuda_peak_bytes = cuda_peak_bytes
        self.requests: list[NHTComposedRenderCommandRequest] = []

    def validate_scene(self, scene_path: Path) -> SimpleNamespace:
        """Return only the scene identity needed by stage preflight."""
        del scene_path
        return SimpleNamespace(scene_id=self.scene_id)

    def render_composed(
        self,
        request: NHTComposedRenderCommandRequest,
        *,
        environment: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> NHTComposedRenderResult:
        """Materialize a structurally real public result with one joint pixel per view."""
        del environment, timeout_seconds
        self.requests.append(request)
        cameras = request.base.arbitrary_cameras
        camera_path = request.base.arbitrary_request_path
        if cameras is None or camera_path is None:
            raise ValueError("Fake composed NHT requires arbitrary cameras.")
        cameras.write(camera_path)
        output = request.base.output_directory
        output.mkdir(parents=True, exist_ok=False)

        background_root = output / "background"
        background_root.mkdir()
        background_records: list[NHTRenderRecord] = []
        for camera in cameras.cameras:
            camera_root = background_root / camera.camera_id
            camera_root.mkdir()
            rgb = np.full((camera.height, camera.width, 3), 0.1, dtype=np.float32)
            alpha = np.ones((camera.height, camera.width, 1), dtype=np.float32)
            depth = np.full((camera.height, camera.width, 1), 100.0, dtype=np.float32)
            rgb_path = camera_root / "rgb.npy"
            alpha_path = camera_root / "alpha.npy"
            depth_path = camera_root / "depth.npy"
            np.save(rgb_path, rgb, allow_pickle=False)
            np.save(alpha_path, alpha, allow_pickle=False)
            np.save(depth_path, depth, allow_pickle=False)
            record = NHTRenderRecord(
                camera_id=camera.camera_id,
                request_source="arbitrary",
                width=camera.width,
                height=camera.height,
                rgb_path=rgb_path,
                rgb_preview_path=camera_root / "rgb.png",
                alpha_path=alpha_path,
                alpha_preview_path=camera_root / "alpha.png",
                depth_path=depth_path,
            )
            record._bind_arrays(NHTRenderArrays(rgb=rgb, alpha=alpha, depth=depth))
            background_records.append(record)
        background = NHTRenderResult(
            scene_id=self.scene_id,
            output_directory=background_root,
            records=tuple(background_records),
        )

        payload = json.loads(
            request.composition_request_path.read_text(encoding="utf-8")
        )
        timeline_path = request.composition_request_path.parent / payload["timeline"][
            "tensors"
        ]
        with np.load(timeline_path, allow_pickle=False) as archive:
            present = np.array(archive["present"], copy=True)
        object_count = int(payload["timeline"]["object_count"])
        chunks_root = output / "chunks"
        chunks_root.mkdir()
        chunks: list[NHTComposedChunkRecord] = []
        for chunk in payload["timeline"]["chunks"]:
            chunk_index = int(chunk["chunk_index"])
            chunk_id = f"chunk-{chunk_index:06d}"
            frame_indices = tuple(int(value) for value in chunk["frame_indices"])
            chunk_root = chunks_root / chunk_id
            chunk_root.mkdir()
            arrays_path = chunk_root / "composed.npz"
            arrays = _joint_sparse_arrays(
                frame_indices=frame_indices,
                camera_count=len(cameras.cameras),
                width=cameras.cameras[0].width,
                height=cameras.cameras[0].height,
                present=present,
            )
            np.savez(arrays_path, **arrays)
            chunks.append(
                NHTComposedChunkRecord(
                    chunk_id=chunk_id,
                    frame_indices=frame_indices,
                    camera_ids=tuple(camera.camera_id for camera in cameras.cameras),
                    sample_count=len(frame_indices) * len(cameras.cameras),
                    pixel_count=int(arrays["offsets"][-1]),
                    arrays_path=arrays_path,
                    width=cameras.cameras[0].width,
                    height=cameras.cameras[0].height,
                    object_count=object_count,
                )
            )
        return NHTComposedRenderResult(
            scene_id=self.scene_id,
            output_directory=output,
            background=background,
            chunks=tuple(chunks),
            appearance_model="direct_linear_rgb",
            rasterization="joint_3dgs_eval3d_transmittance_v1",
            cuda_peak_bytes=self.cuda_peak_bytes,
        )


def _joint_sparse_arrays(
    *,
    frame_indices: tuple[int, ...],
    camera_count: int,
    width: int,
    height: int,
    present: NDArray[np.bool_],
) -> dict[str, NDArray[np.generic]]:
    frame_values: list[int] = []
    camera_values: list[int] = []
    pixels: list[int] = []
    rgb: list[tuple[float, float, float]] = []
    alpha: list[float] = []
    depth: list[float] = []
    instance_ids: list[int] = []
    offsets = [0]
    for frame_index in frame_indices:
        active = np.flatnonzero(present[frame_index])
        for camera_index in range(camera_count):
            frame_values.append(frame_index)
            camera_values.append(camera_index)
            if len(active):
                pixels.append((frame_index + camera_index) % (width * height))
                rgb.append((0.72, 0.92, 0.08))
                alpha.append(0.97)
                depth.append(8.0 + frame_index)
                instance_ids.append(int(active[0]) + 1)
            offsets.append(len(pixels))
    return {
        "frame_indices": np.asarray(frame_values, dtype=np.int64),
        "camera_indices": np.asarray(camera_values, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "pixel_indices": np.asarray(pixels, dtype=np.int32),
        "rgb": np.asarray(rgb, dtype=np.float32).reshape(-1, 3),
        "alpha": np.asarray(alpha, dtype=np.float32),
        "depth": np.asarray(depth, dtype=np.float32),
        "instance_ids": np.asarray(instance_ids, dtype=np.int32),
    }


__all__ = ["FakeComposedNHTClient"]
