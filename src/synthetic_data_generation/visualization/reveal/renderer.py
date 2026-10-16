"""Render the staged reveal: timeline -> NHT states -> composited frames."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.rendering.nht import (
    NHT_SPLAT_ORDER_LARGEST_FIRST,
    NHTAppearanceControls,
    NHTAppearancePolicy,
    NHTRenderClient,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.visualization.reveal.compositor import (
    gaussian_layer,
    over,
    to_uint8,
)
from src.synthetic_data_generation.visualization.reveal.contracts import (
    REVEAL_PHASE_ORDER,
    SCENE_REVEAL_METADATA_SCHEMA,
    RevealMode,
    SceneRevealRequest,
    SceneRevealResult,
)
from src.synthetic_data_generation.visualization.reveal.layers import (
    ProjectedCameras,
    ProjectedPoints,
    draw_cameras,
    draw_points,
    project_cameras,
    project_points,
)
from src.synthetic_data_generation.visualization.reveal.sources import (
    crop_points_to_court_box,
    load_reveal_sources,
    radial_reveal_order,
)
from src.synthetic_data_generation.visualization.reveal.timeline import (
    NHTAppearanceState,
    RevealFrameState,
    build_timeline,
    frame_index_at,
    phase_frame_counts,
)
from src.synthetic_data_generation.visualization.reveal.viewpoint import (
    RevealCamera,
    build_reveal_camera,
)
from src.utils.video.writer import VideoWriter

NHTLayer = tuple[NDArray[np.float32], NDArray[np.float32]]


def selected_frame_indices(
    request: SceneRevealRequest, frame_count: int
) -> tuple[int, ...]:
    """Return every frame (video) or the frames at the requested still times."""
    if request.mode is RevealMode.VIDEO:
        return tuple(range(frame_count))
    return tuple(
        sorted(
            {
                frame_index_at(seconds, request.output.fps, frame_count)
                for seconds in request.still_times_seconds
            }
        )
    )


def ordered_nht_states(
    timeline: Sequence[RevealFrameState], indices: Sequence[int]
) -> tuple[NHTAppearanceState, ...]:
    """Return the distinct NHT states needed by ``indices``, in first-use order."""
    states: dict[NHTAppearanceState, None] = {}
    for index in indices:
        if timeline[index].needs_nht:
            states.setdefault(timeline[index].nht, None)
    return tuple(states)


class _NHTStateRenderer:
    """Render distinct NHT states in bounded ``nht-render`` invocations."""

    def __init__(
        self,
        request: SceneRevealRequest,
        camera: RevealCamera,
        scene_path: Path,
        policy: NHTAppearancePolicy,
        states: tuple[NHTAppearanceState, ...],
    ) -> None:
        self._request = request
        self._camera = camera
        self._scene_path = scene_path
        self._policy = policy
        self._states = states
        self._position = {state: index for index, state in enumerate(states)}
        self._client = NHTRenderClient()
        self._cache: dict[NHTAppearanceState, NHTLayer] = {}
        self._invocations = 0

    def layer(self, state: NHTAppearanceState) -> NHTLayer:
        """Return RGB and alpha for one state, rendering its chunk on first use."""
        if state not in self._cache:
            start = self._position[state]
            chunk = self._states[start : start + self._request.nht.states_per_invocation]
            self._cache = self._render(chunk)
        return self._cache[state]

    def _render(self, chunk: tuple[NHTAppearanceState, ...]) -> dict[NHTAppearanceState, NHTLayer]:
        root = self._request.work_directory / "nht"
        name = f"chunk-{self._invocations:05d}"
        self._invocations += 1
        cameras = tuple(
            self._camera.nht_render_camera(
                f"state-{self._position[state]:05d}",
                NHTAppearanceControls(
                    splat_progress=state.splat_progress,
                    view_dependence=state.view_dependence,
                ),
            )
            for state in chunk
        )
        executable = self._request.nht.executable
        command = NHTRenderCommandRequest(
            scene_path=self._scene_path,
            output_directory=root / name,
            arbitrary_cameras=NHTRenderRequest(cameras, appearance_policy=self._policy),
            arbitrary_request_path=root / f"{name}-request.json",
            executable=executable if executable == "nht-render" else Path(executable),
        )
        result = self._client.render(command)
        layers: dict[NHTAppearanceState, NHTLayer] = {}
        for state, camera in zip(chunk, cameras, strict=True):
            arrays = result.record(camera.camera_id).load_arrays()
            layers[state] = (np.array(arrays.rgb), np.array(arrays.alpha))
        shutil.rmtree(root / name)
        (root / f"{name}-request.json").unlink()
        return layers

    @property
    def invocation_count(self) -> int:
        """Return how many ``nht-render`` processes were started."""
        return self._invocations


class _FrameCompositor:
    """Composite timeline states with cached procedural layers."""

    def __init__(
        self,
        request: SceneRevealRequest,
        points: ProjectedPoints,
        cameras: ProjectedCameras,
        nht: _NHTStateRenderer,
    ) -> None:
        self._request = request
        self._points = points
        self._cameras = cameras
        self._nht = nht
        self._camera_layer: tuple[float, NHTLayer] | None = None
        self._point_layer: tuple[float, NHTLayer] | None = None
        style = request.style
        self._background = np.broadcast_to(
            np.asarray(style.background_rgb, dtype=np.float32),
            (request.output.height, request.output.width, 3),
        )

    def frame(self, state: RevealFrameState) -> NDArray[np.uint8]:
        """Return one encoded-ready RGB frame."""
        style = self._request.style
        frame = np.array(self._background)
        if state.needs_nht:
            rgb, alpha = self._nht.layer(state.nht)
            layer_rgb, layer_alpha = gaussian_layer(
                state, rgb, alpha, style.gaussian_point_rgb
            )
            frame = over(frame, layer_rgb, layer_alpha, state.gaussian_opacity)
        if state.sfm_opacity > 0.0 and state.sfm_fraction > 0.0:
            if self._point_layer is None or self._point_layer[0] != state.sfm_fraction:
                self._point_layer = (
                    state.sfm_fraction,
                    draw_points(self._points, state.sfm_fraction),
                )
            frame = over(frame, *self._point_layer[1], state.sfm_opacity)
        if state.camera_opacity > 0.0 and state.camera_fraction > 0.0:
            if self._camera_layer is None or self._camera_layer[0] != state.camera_fraction:
                self._camera_layer = (
                    state.camera_fraction,
                    draw_cameras(
                        self._cameras,
                        state.camera_fraction,
                        rgb=style.camera_rgb,
                        line_width_px=style.camera_line_width_px,
                    ),
                )
            frame = over(frame, *self._camera_layer[1], state.camera_opacity)
        return to_uint8(frame)


def _prepare_work_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or path.is_symlink() or any(path.iterdir())):
        raise FileExistsError(
            f"work_directory must be absent or an empty directory: {path}"
        )
    (path / "nht").mkdir(parents=True, exist_ok=True)


def _write_video(
    request: SceneRevealRequest,
    frames: Iterator[NDArray[np.uint8]],
) -> Path:
    target = request.output_video
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.stem}.partial.mp4")
    try:
        with VideoWriter(partial, fps=request.output.fps, crf=request.output.crf) as writer:
            for frame in frames:
                writer.write_frame(frame)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def render_scene_reveal(request: SceneRevealRequest) -> SceneRevealResult:
    """Render and publish the staged reveal described by ``request``."""
    sources = load_reveal_sources(request.scene_root)
    camera = build_reveal_camera(
        request.viewpoint, sources.layout, sources.adapter, request.output
    )
    units_per_metre = float(sources.adapter.nht_scene_units_per_metre)
    scene_from_court = sources.scene_from_court(request.viewpoint.anchor_court_id)
    points = crop_points_to_court_box(
        sources.scene.points_scene, scene_from_court, request.style.sfm_crop_court_m
    )
    target_scene = scene_from_court @ np.append(request.viewpoint.target_court_m, 1.0)
    projected_points = project_points(
        camera,
        points,
        radial_reveal_order(points, target_scene[:3]),
        radius_px=request.style.sfm_point_radius_px,
    )
    projected_cameras = project_cameras(
        camera,
        sources.capture_cameras,
        frustum_depth_scene=request.style.camera_frustum_depth_m * units_per_metre,
        frustum_stride=request.style.camera_frustum_stride,
    )
    policy = NHTAppearancePolicy(
        point_radius=request.nht.point_radius_m * units_per_metre,
        splat_order=NHT_SPLAT_ORDER_LARGEST_FIRST,
        splat_stagger=request.nht.splat_stagger,
    )

    timeline = build_timeline(request.durations, request.output.fps)
    indices = selected_frame_indices(request, len(timeline))
    states = ordered_nht_states(timeline, indices)
    _prepare_work_directory(request.work_directory)
    nht = _NHTStateRenderer(request, camera, sources.scene_path, policy, states)
    compositor = _FrameCompositor(request, projected_points, projected_cameras, nht)

    video_path: Path | None = None
    still_paths: list[Path] = []
    if request.mode is RevealMode.VIDEO:
        video_path = _write_video(
            request, (compositor.frame(timeline[index]) for index in indices)
        )
    else:
        directory = request.stills_directory
        directory.mkdir(parents=True, exist_ok=True)
        for index in indices:
            path = directory / f"frame-{index:05d}-{timeline[index].phase.value}.png"
            Image.fromarray(compositor.frame(timeline[index])).save(path)
            still_paths.append(path)
    shutil.rmtree(request.work_directory)

    metadata = {
        "schema": SCENE_REVEAL_METADATA_SCHEMA,
        "scene_id": sources.scene.scene_id,
        "mode": request.mode.value,
        "video": None if video_path is None else video_path.name,
        "stills": [path.name for path in still_paths],
        "frame_count": len(timeline),
        "rendered_frame_count": len(indices),
        "fps": request.output.fps,
        "phases": [
            {"phase": phase.value, "seconds": seconds, "frames": frames}
            for phase, seconds, frames in zip(
                REVEAL_PHASE_ORDER,
                request.durations.seconds,
                phase_frame_counts(request.durations, request.output.fps),
                strict=True,
            )
        ],
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "intrinsics": camera.intrinsics.tolist(),
            "camera_to_nht_scene": camera.camera_to_scene.tolist(),
        },
        "viewpoint": asdict(request.viewpoint),
        "style": asdict(request.style),
        "nht": {
            "appearance_policy": policy.to_dict(),
            "distinct_states": len(states),
            "invocations": nht.invocation_count,
        },
        "sfm_points": {
            "exported": int(len(sources.scene.points_scene)),
            "cropped": int(len(points)),
        },
        "capture_cameras": len(sources.capture_cameras),
    }
    request.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    request.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return SceneRevealResult(
        video_path=video_path,
        still_paths=tuple(still_paths),
        metadata_path=request.metadata_path,
        frame_count=len(indices),
        nht_state_count=len(states),
    )
