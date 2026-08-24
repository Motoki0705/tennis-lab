"""BLCS dataset adapters for the shared CourtKP20 frame contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.tasks.base.data.scene_dataset import Scene
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    DatasetCourtKeypointContract,
    align_court_keypoints_to_reference,
    build_physical_court_provenance,
    build_reference_frame_provenance,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    validate_blcs_dataset_court_keypoints as validate_blcs_artifact,
)


@dataclass(frozen=True, slots=True)
class BLCSSampleCourtFrame:
    """Validated selected views plus one reversible target-frame provenance."""

    selected_views: tuple[CourtViewRecord, ...]
    reference_view: CourtViewRecord | None
    provenance: CourtReferenceFrameProvenance


def _scene_paths(scene_dir: Path, split_file: Path) -> list[Path]:
    split_path = split_file if split_file.is_absolute() else scene_dir / split_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    scenes_base = scene_dir / "scenes"
    if not scenes_base.is_dir():
        scenes_base = scene_dir
    return [
        scenes_base / name
        for raw in split_path.read_text(encoding="utf-8").splitlines()
        if (name := raw.strip())
    ]


def validate_blcs_dataset_court_keypoints(
    *,
    scene_dir: str | Path,
    split_file: str | Path,
    contract: CourtKeypointContract,
) -> DatasetCourtKeypointContract:
    """Validate root/scene/camera headers before any task array is consumed."""
    root = Path(scene_dir)
    paths = _scene_paths(root, Path(split_file))
    return validate_blcs_artifact(
        root,
        contract,
        scene_paths=paths,
    )


def court_views_by_scene(
    result: DatasetCourtKeypointContract,
) -> dict[str, tuple[CourtViewRecord, ...]]:
    """Index validated ordered camera records by stable scene directory name."""
    return {record.scene_id: record.court_views for record in result.scenes}


def resolve_blcs_sample_court_frame(
    *,
    scene: Scene,
    selected_camera_indices: tuple[int, ...],
    court_views: tuple[CourtViewRecord, ...],
    contract: CourtKeypointContract,
    rng: np.random.Generator,
    training: bool,
) -> BLCSSampleCourtFrame:
    """Resolve stable reference identity independently of selected view ordering."""
    if contract.selector == PHYSICAL_V1_SELECTOR:
        selected = (
            tuple(court_views[index] for index in selected_camera_indices)
            if court_views
            else ()
        )
        return BLCSSampleCourtFrame(
            selected_views=selected,
            reference_view=None,
            provenance=build_physical_court_provenance(),
        )
    if len(court_views) != scene.num_cameras:
        raise ValueError(
            f"{scene.path}: camera-view-v2 metadata has {len(court_views)} records "
            f"for {scene.num_cameras} persisted cameras."
        )
    selected = tuple(court_views[index] for index in selected_camera_indices)
    stable_ids = tuple(sorted(view.camera_id for view in selected))
    if training:
        reference_id = stable_ids[int(rng.integers(0, len(stable_ids)))]
    else:
        reference_id = stable_ids[0]
    provenance = build_reference_frame_provenance(
        selected,
        reference_camera_id=reference_id,
    )
    assert provenance.reference_camera_local_index is not None
    return BLCSSampleCourtFrame(
        selected_views=selected,
        reference_view=selected[provenance.reference_camera_local_index],
        provenance=provenance,
    )


def align_blcs_court_array(
    value: np.ndarray,
    *,
    source_view: CourtViewRecord | None,
    frame: BLCSSampleCourtFrame,
    keypoint_axis: int,
) -> np.ndarray:
    """Align a persisted camera-local Court array before consumer truncation."""
    if frame.reference_view is None:
        return value
    if source_view is None:
        raise ValueError("camera-view-v2 Court alignment requires camera metadata.")
    aligned = align_court_keypoints_to_reference(
        value,
        source_view,
        frame.reference_view,
        keypoint_axis=keypoint_axis,
    )
    if not isinstance(aligned, np.ndarray):
        raise TypeError("BLCS numpy Court alignment returned a non-array.")
    return aligned


__all__ = [
    "BLCSSampleCourtFrame",
    "align_blcs_court_array",
    "court_views_by_scene",
    "resolve_blcs_sample_court_frame",
    "validate_blcs_dataset_court_keypoints",
]
