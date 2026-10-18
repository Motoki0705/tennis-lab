"""Inputs of the reveal: public NHT export and accepted court geometry.

Only two scene artifacts are read:

* ``reconstruction/export/scene.json`` (validated public NHT export): capture
  cameras and the coloured SfM point cloud, in canonical NHT scene units.
* ``alignment/alignment.json``: accepted courts and the metric/NHT similarity.

The reveal places a camera and crops points; it does not consume the alignment
line evidence, so it reads ``alignment.json`` directly instead of validating the
complete alignment owner inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
    validate_standard_scene_export,
)
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera


@dataclass(frozen=True, slots=True)
class RevealSources:
    """Validated scene inputs, all geometry in canonical NHT scene units."""

    scene: StandardSceneExport
    layout: MultiCourtLayout
    adapter: MetricSceneAdapter
    capture_cameras: tuple[SceneCamera, ...]

    @property
    def scene_path(self) -> Path:
        """Return the validated public ``scene.json``."""
        return Path(self.scene.scene_path)

    def scene_from_court(self, court_id: str) -> NDArray[np.float64]:
        """Return the similarity mapping court metres into NHT scene units."""
        courts = {court.court_instance_id: court for court in self.layout.courts}
        if court_id not in courts:
            raise ValueError(f"Unknown court {court_id!r}; accepted: {sorted(courts)}.")
        similarity = np.asarray(
            self.adapter.nht_scene_from_metric_scene, dtype=np.float64
        ).reshape(4, 4)
        return similarity @ courts[court_id].scene_from_court.matrix()


def load_reveal_sources(scene_root: Path) -> RevealSources:
    """Validate the public export and the accepted court layout of one scene."""
    scene = validate_standard_scene_export(
        scene_root / "reconstruction" / "export" / "scene.json"
    )
    alignment = load_alignment_result(scene_root / "alignment" / "alignment.json")
    accepted = {
        candidate.court_instance_id
        for candidate in alignment.candidates
        if candidate.accepted
    }
    layout_ids = {court.court_instance_id for court in alignment.layout.courts}
    if not layout_ids <= accepted:
        raise ValueError("A rejected court candidate is present in the layout.")
    cameras = tuple(sorted(scene.cameras, key=lambda camera: camera.source_frame_index))
    if not cameras:
        raise ValueError("The scene export contains no capture cameras.")
    return RevealSources(
        scene=scene,
        layout=alignment.layout,
        adapter=alignment.metric_adapter,
        capture_cameras=cameras,
    )


def crop_points_to_court_box(
    points_scene: NDArray[np.float32],
    scene_from_court: NDArray[np.float64],
    box_court_m: tuple[float, float, float, float, float, float],
) -> NDArray[np.float32]:
    """Keep ``(N, 6)`` XYZRGB points whose court-frame position is inside the box."""
    points = np.asarray(points_scene)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("points_scene must have shape (N, 6).")
    court_from_scene = np.linalg.inv(scene_from_court)
    court = points[:, :3].astype(np.float64) @ court_from_scene[:3, :3].T
    court += court_from_scene[:3, 3]
    lower = np.asarray(box_court_m[0::2])
    upper = np.asarray(box_court_m[1::2])
    inside = np.all((court >= lower) & (court <= upper), axis=1)
    if not inside.any():
        raise ValueError("The SfM crop box contains no points.")
    return np.ascontiguousarray(points[inside])


def radial_reveal_order(
    points_scene: NDArray[np.float32], centre_scene: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Order points by distance from ``centre_scene`` (nearest first, stable)."""
    distances = np.linalg.norm(
        np.asarray(points_scene)[:, :3].astype(np.float64) - centre_scene, axis=1
    )
    return np.argsort(distances, kind="stable")
