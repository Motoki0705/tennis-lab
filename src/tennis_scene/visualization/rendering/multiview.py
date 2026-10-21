"""CPU composition of synchronized source views and an existing 3D renderer.

Inputs must already share frame zero and a time base. This module checks their
metadata; it does not estimate synchronization, resample inputs, or fill gaps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.utils.video.reader import RandomAccessVideoReader, probe_video_info

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.schema import SceneResult
    from src.tennis_scene.visualization.rendering.tennis_scene_renderer import (
        TennisSceneRenderer,
    )


def sampled_frame_indices(
    num_frames: int, *, start: int, stop: int, step: int
) -> tuple[int, ...]:
    """Select source indices once for every panel; stop is exclusive."""
    if step < 1 or not 0 <= start < stop <= num_frames:
        raise ValueError("Require step >= 1 and 0 <= start < stop <= num_frames")
    return tuple(range(start, stop, step))


class MultiviewSceneRenderer:
    """Stack source cameras on the left and the same-frame 3D scene on the right.

    ``videos`` explicitly binds each scene camera ID to its source file, so
    relocated inputs do not depend on historical absolute metadata paths.
    Missing SMPL/ball reconstruction, mismatched video metadata and failed
    decoding are errors. Per-frame validity and trajectory gaps are handled by
    ``TennisSceneRenderer`` unchanged. Use as a context manager to close readers.
    """

    def __init__(
        self,
        scene: SceneResult,
        renderer: TennisSceneRenderer,
        videos: Mapping[str, Path],
        *,
        width: int = 960,
    ) -> None:
        if scene.schema_version != 2:
            raise ValueError("Multiview rendering requires SceneResult v2 masks")
        camera_ids = scene.metadata.get("camera_ids")
        if (
            not isinstance(camera_ids, list)
            or not 3 <= len(camera_ids) <= 5
            or any(not isinstance(camera, str) for camera in camera_ids)
            or len(set(camera_ids)) != len(camera_ids)
            or set(camera_ids) != set(videos)
        ):
            raise ValueError("Bind exactly the scene's 3 to 5 distinct camera_ids")
        if width < 480 or width % 3:
            raise ValueError("width must be >= 480 and divisible by 3")
        if (
            renderer.style.player_representation != "smpl"
            or scene.smpl_vertices_local is None
            or scene.smpl_global_orient is None
            or scene.player_smpl_valid is None
            or not scene.player_smpl_valid.any()
            or scene.ball_3d is None
            or scene.ball_3d_valid is None
            or not scene.ball_3d_valid.any()
        ):
            raise ValueError("Require SMPL rendering and valid SMPL/ball reconstruction")
        for camera in camera_ids:
            info = probe_video_info(videos[camera])
            if (
                (info.width, info.height) != (scene.width, scene.height)
                or info.frame_count != scene.num_frames
                or not math.isclose(info.fps, scene.fps, rel_tol=0, abs_tol=1e-3)
            ):
                raise ValueError(f"{camera}: video dimensions/FPS/frame count differ from scene: {info}")

        self.scene = scene
        self.renderer = renderer
        self.readers = [RandomAccessVideoReader(videos[camera]) for camera in camera_ids]
        self.panel_width = width // 3
        self.panel_height = round(self.panel_width * scene.height / scene.width)
        body_height = self.panel_height * len(camera_ids)
        self.width, self.height = width, body_height + 80
        self.figure = Figure(figsize=(width / 100, self.height / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self.figure.set_facecolor(renderer.theme.figure_color)
        self.figure.text(.015, 1 - 28 / self.height, "SYNCHRONIZED INPUT", color="#E8E8E8", size=12, weight="bold")
        self.figure.text(1 / 3 + .025, 1 - 28 / self.height, "3D RECONSTRUCTION", color="#E8E8E8", size=14, weight="bold")
        self.figure.text(1 / 3 + .025, 12 / self.height, "Court / SMPL players / Ball trajectory", color="#BAC6CE", size=10)
        self.clock = self.figure.text(.015, 12 / self.height, "", color="#BAC6CE", size=10)

        self.images = []
        blank = np.zeros((self.panel_height, self.panel_width, 3), np.uint8)
        for row, camera in enumerate(camera_ids):
            bottom = (32 + body_height - (row + 1) * self.panel_height) / self.height
            axis = self.figure.add_axes((0, bottom, 1 / 3, self.panel_height / self.height))
            axis.set_axis_off()
            self.images.append(axis.imshow(blank, interpolation="nearest", aspect="auto"))
            axis.text(.025, .94, camera, transform=axis.transAxes, va="top", color="white", size=10,
                      bbox={"facecolor": "#101418", "alpha": .8, "edgecolor": "none", "pad": 3})
        self.axis_3d = self.figure.add_axes(
            (1 / 3, 32 / self.height, 2 / 3, body_height / self.height), projection="3d"
        )
        self.closed = False

    def render(self, frame_index: int) -> NDArray[np.uint8]:
        """Return an RGB image; every camera and 3D uses ``frame_index``."""
        if self.closed:
            raise RuntimeError("Multiview renderer is closed")
        if not 0 <= frame_index < self.scene.num_frames:
            raise ValueError(f"Source frame out of range: {frame_index}")
        for reader, image in zip(self.readers, self.images, strict=True):
            bgr = reader.read(frame_index)
            small = cv2.resize(bgr, (self.panel_width, self.panel_height), interpolation=cv2.INTER_AREA)
            image.set_data(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self.axis_3d.clear()
        self.renderer.render_frame_3d(self.scene, frame_index, ax=self.axis_3d)
        self.clock.set_text(f"{frame_index / self.scene.fps:05.2f} s  |  frame {frame_index}")
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()

    def close(self) -> None:
        """Release all sources and the figure; safe to call repeatedly."""
        for reader in self.readers:
            reader.close()
        self.figure.clear()
        self.closed = True

    def __enter__(self) -> MultiviewSceneRenderer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
