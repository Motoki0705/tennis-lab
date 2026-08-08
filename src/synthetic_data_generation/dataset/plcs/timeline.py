"""Complete single- and multi-object PLCS compositor timelines."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianDeformationKind,
    GaussianForegroundComposition,
    GaussianFrame,
    GaussianInstance,
    GaussianSceneObject,
    GaussianTransform,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.production import (
    PLCSProductionMode,
    validate_plcs_production_contract,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip

FloatArray: TypeAlias = NDArray[np.float64]

_SMPLH_TO_COURT = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class PLCSObjectTrack:
    """One stable avatar identity and its full source-frame interval."""

    object_id: str
    instance_id: int
    asset_id: str
    clip: PLCSMotionClip
    start_frame: int
    anchor_position_court_m: tuple[float, float, float]
    yaw_radians: float

    def __post_init__(self) -> None:
        for name, value in (("object_id", self.object_id), ("asset_id", self.asset_id)):
            if not value or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed identifier.")
        if isinstance(self.instance_id, bool) or self.instance_id <= 0:
            raise ValueError("instance_id must be a positive integer.")
        if isinstance(self.start_frame, bool) or self.start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer.")
        anchor = np.asarray(self.anchor_position_court_m, dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError(
                "anchor_position_court_m must contain three finite values."
            )
        if not np.isclose(anchor[2], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("Avatar court anchors must lie on z=0 ground.")
        if abs(anchor[0]) > 4.115 or abs(anchor[1]) > 11.885:
            raise ValueError("Avatar court anchor lies outside the singles court.")
        if not math.isfinite(self.yaw_radians):
            raise ValueError("yaw_radians must be finite.")
        object.__setattr__(
            self,
            "anchor_position_court_m",
            cast(tuple[float, float, float], tuple(float(value) for value in anchor)),
        )

    @property
    def stop_frame(self) -> int:
        """Return the exclusive global end of this unsliced source clip."""
        return self.start_frame + int(self.clip.frame_count)


@dataclass(frozen=True, slots=True)
class PLCSFrameEntry:
    """One track's explicit presence and source mapping on a global frame."""

    object_id: str
    instance_id: int
    frame_index: int
    present: bool
    source_frame_index: int | None
    scene_from_asset: GaussianTransform | None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if self.present != (self.source_frame_index is not None):
            raise ValueError("present must agree with source_frame_index availability.")
        if self.present != (self.scene_from_asset is not None):
            raise ValueError("present must agree with scene_from_asset availability.")
        if self.source_frame_index is not None and self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "instance_id": self.instance_id,
            "frame_index": self.frame_index,
            "present": self.present,
            "source_frame_index": self.source_frame_index,
            "scene_from_asset": (
                self.scene_from_asset.to_dict()
                if self.scene_from_asset is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PLCSGlobalFrame:
    """Every declared track on one compositor frame, including absences."""

    frame_index: int
    entries: tuple[PLCSFrameEntry, ...]

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not self.entries:
            raise ValueError("A global frame needs a non-negative index and tracks.")
        if any(entry.frame_index != self.frame_index for entry in self.entries):
            raise ValueError("Frame entries disagree with their global frame index.")
        object_ids = [entry.object_id for entry in self.entries]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("A global frame contains duplicate object identities.")


@dataclass(frozen=True, slots=True)
class PLCSGlobalTimeline:
    """The entire compositor interval and target-court transform authority."""

    scene_id: str
    production_mode: PLCSProductionMode
    target_court: TargetCourtBinding
    tracks: tuple[PLCSObjectTrack, ...]
    frames: tuple[PLCSGlobalFrame, ...]

    def __post_init__(self) -> None:
        if _PORTABLE_ID.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a portable identifier.")
        if not self.tracks or not self.frames:
            raise ValueError("A PLCS timeline requires tracks and global frames.")
        validate_plcs_production_contract(
            mode=self.production_mode,
            configured_motion_categories=(
                track.clip.category.value for track in self.tracks
            ),
            object_motion_categories=(
                track.clip.category.value for track in self.tracks
            ),
            object_start_frames=(track.start_frame for track in self.tracks),
        )
        if tuple(frame.frame_index for frame in self.frames) != tuple(
            range(len(self.frames))
        ):
            raise ValueError("PLCS global frame indices must exactly equal 0..T-1.")
        object_ids = [track.object_id for track in self.tracks]
        instance_ids = [track.instance_id for track in self.tracks]
        if len(object_ids) != len(set(object_ids)) or len(instance_ids) != len(
            set(instance_ids)
        ):
            raise ValueError("PLCS track object and instance IDs must be unique.")
        expected_ids = tuple(object_ids)
        for frame in self.frames:
            if tuple(entry.object_id for entry in frame.entries) != expected_ids:
                raise ValueError("Every global frame must preserve stable track order.")
        expected_count = max(track.stop_frame for track in self.tracks)
        if len(self.frames) != expected_count:
            raise ValueError(
                "PLCS timeline is not the full multi-object global interval."
            )
        for track in self.tracks:
            entries = [
                frame.entries[index]
                for frame in self.frames
                for index, candidate in enumerate(frame.entries)
                if candidate.object_id == track.object_id
            ]
            present_sources = [
                entry.source_frame_index for entry in entries if entry.present
            ]
            if present_sources != list(range(track.clip.frame_count)):
                raise ValueError(
                    f"Track {track.object_id!r} does not retain every source frame in order."
                )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def mode(self) -> str:
        return self.production_mode.persisted_timeline_mode

    def to_foreground_composition(
        self,
        *,
        assets: tuple[GaussianAsset, ...],
    ) -> GaussianForegroundComposition:
        """Build the shared semantic foreground-only frame composition."""
        asset_ids = {asset.asset_id for asset in assets}
        expected_asset_ids = {track.asset_id for track in self.tracks}
        if asset_ids != expected_asset_ids:
            raise ValueError(
                "Timeline Gaussian assets differ; "
                f"missing={sorted(expected_asset_ids - asset_ids)}, "
                f"unexpected={sorted(asset_ids - expected_asset_ids)}."
            )
        objects = tuple(
            GaussianSceneObject(
                object_id=track.object_id,
                instance_id=track.instance_id,
                asset_id=track.asset_id,
                deformation_kind=GaussianDeformationKind.ARTICULATED,
            )
            for track in self.tracks
        )
        frames = tuple(
            GaussianFrame(
                frame_index=frame.frame_index,
                instances=tuple(
                    GaussianInstance(
                        object_id=entry.object_id,
                        source_frame_index=cast(int, entry.source_frame_index),
                        scene_from_asset=cast(
                            GaussianTransform, entry.scene_from_asset
                        ),
                    )
                    for entry in frame.entries
                    if entry.present
                ),
            )
            for frame in self.frames
        )
        return GaussianForegroundComposition(
            scene_id=self.scene_id,
            composition_id=f"{self.scene_id}-plcs",
            assets=assets,
            objects=objects,
            frames=frames,
        )


@dataclass(frozen=True, slots=True)
class PLCSLogicalScene:
    """One complete compositor timeline assigned wholly to one target court."""

    split: str
    timeline: PLCSGlobalTimeline

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("PLCS logical-scene split is unsupported.")
        if not isinstance(self.timeline, PLCSGlobalTimeline):
            raise TypeError("PLCS logical scene requires a global timeline.")


@dataclass(frozen=True, slots=True)
class PLCSSceneInventory:
    """Balanced collection of intact logical scenes for one dataset workspace."""

    dataset_scene_id: str
    scenes: tuple[PLCSLogicalScene, ...]
    accepted_court_instance_ids: tuple[str, ...]
    required_motion_categories: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not self.dataset_scene_id.strip()
            or self.dataset_scene_id != self.dataset_scene_id.strip()
        ):
            raise ValueError("dataset_scene_id must be a non-empty trimmed string.")
        scenes = tuple(self.scenes)
        accepted = tuple(self.accepted_court_instance_ids)
        if not scenes:
            raise ValueError("PLCS scene inventory must contain logical scenes.")
        scene_ids = tuple(scene.timeline.scene_id for scene in scenes)
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("PLCS logical scene IDs must be unique.")
        if (
            not accepted
            or len(accepted) != len(set(accepted))
            or any(not value.strip() for value in accepted)
        ):
            raise ValueError("Accepted PLCS court IDs must be non-empty and unique.")
        if len(scenes) < len(accepted):
            raise ValueError(
                "PLCS logical scene inventory cannot cover every accepted court."
            )
        required = frozenset(self.required_motion_categories)
        if not required or any(not value.strip() for value in required):
            raise ValueError("Required PLCS motion categories must be explicit.")

        expected_signature = _track_inventory_signature(scenes[0].timeline)
        expected_mode = scenes[0].timeline.production_mode
        for scene in scenes:
            if scene.timeline.production_mode is not expected_mode:
                raise ValueError(
                    "Every PLCS logical scene must use one production mode."
                )
            if _track_inventory_signature(scene.timeline) != expected_signature:
                raise ValueError(
                    "Every PLCS logical scene must retain the same complete source "
                    "motion inventory."
                )
            categories = {track.clip.category.value for track in scene.timeline.tracks}
            if categories != required:
                raise ValueError(
                    "PLCS logical scene motion categories differ from the required "
                    "complete inventory."
                )

        accepted_set = set(accepted)
        court_counts = Counter(
            scene.timeline.target_court.court_instance_id for scene in scenes
        )
        unknown = set(court_counts).difference(accepted_set)
        if unknown:
            raise ValueError(
                f"PLCS logical scenes reference unknown courts: {sorted(unknown)}."
            )
        if set(court_counts) != accepted_set:
            missing = sorted(accepted_set.difference(court_counts))
            raise ValueError(
                f"PLCS logical scenes do not use every accepted court: {missing}."
            )
        values = [court_counts[court_id] for court_id in accepted]
        if max(values) - min(values) > 1:
            raise ValueError("PLCS logical-scene court count difference exceeds one.")

        split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for scene in scenes:
            split_counts[scene.split][
                scene.timeline.target_court.court_instance_id
            ] += 1
        for split, counts in split_counts.items():
            split_values = [counts[court_id] for court_id in accepted]
            if max(split_values) - min(split_values) > 1:
                raise ValueError(
                    f"PLCS logical-scene court assignment is imbalanced in {split!r}."
                )

        object.__setattr__(self, "scenes", scenes)
        object.__setattr__(self, "accepted_court_instance_ids", accepted)
        object.__setattr__(self, "required_motion_categories", required)

    @property
    def scene_count(self) -> int:
        """Return the number of complete logical timelines."""
        return len(self.scenes)

    @property
    def aggregate_global_frame_count(self) -> int:
        """Return the exact sum of complete per-scene compositor frames."""
        return sum(scene.timeline.frame_count for scene in self.scenes)

    @property
    def aggregate_source_frame_count(self) -> int:
        """Return every retained source-motion frame across all logical scenes."""
        return sum(
            track.clip.frame_count
            for scene in self.scenes
            for track in scene.timeline.tracks
        )

    @property
    def target_courts(self) -> tuple[TargetCourtBinding, ...]:
        """Return one canonical binding for every accepted court in layout order."""
        by_id: dict[str, TargetCourtBinding] = {}
        for scene in self.scenes:
            binding = scene.timeline.target_court
            previous = by_id.setdefault(binding.court_instance_id, binding)
            if previous != binding:
                raise ValueError(
                    "PLCS logical scenes disagree on one court's canonical binding."
                )
        return tuple(by_id[court_id] for court_id in self.accepted_court_instance_ids)


def _track_inventory_signature(
    timeline: PLCSGlobalTimeline,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            track.object_id,
            track.instance_id,
            track.asset_id,
            track.clip.source_path,
            track.clip.category.value,
            track.clip.gender,
            track.clip.fps,
            track.clip.frame_count,
            track.start_frame,
            track.anchor_position_court_m,
            track.yaw_radians,
        )
        for track in timeline.tracks
    )


def build_global_timeline(
    *,
    scene_id: str,
    production_mode: PLCSProductionMode,
    target_court: TargetCourtBinding,
    tracks: tuple[PLCSObjectTrack, ...],
) -> PLCSGlobalTimeline:
    """Construct every global frame without truncating any object clip."""
    if not tracks:
        raise ValueError("At least one PLCS object track is required.")
    if min(track.start_frame for track in tracks) != 0:
        raise ValueError(
            "At least one PLCS object track must begin at global frame zero."
        )
    frame_count = max(track.stop_frame for track in tracks)
    frames: list[PLCSGlobalFrame] = []
    for frame_index in range(frame_count):
        entries = tuple(
            _frame_entry(track, frame_index=frame_index, target_court=target_court)
            for track in tracks
        )
        frames.append(PLCSGlobalFrame(frame_index=frame_index, entries=entries))
    return PLCSGlobalTimeline(
        scene_id=scene_id,
        production_mode=production_mode,
        target_court=target_court,
        tracks=tracks,
        frames=tuple(frames),
    )


def _frame_entry(
    track: PLCSObjectTrack,
    *,
    frame_index: int,
    target_court: TargetCourtBinding,
) -> PLCSFrameEntry:
    source_index = frame_index - track.start_frame
    if source_index < 0 or source_index >= track.clip.frame_count:
        return PLCSFrameEntry(
            object_id=track.object_id,
            instance_id=track.instance_id,
            frame_index=frame_index,
            present=False,
            source_frame_index=None,
            scene_from_asset=None,
        )
    root_relative = (
        track.clip.root_translation_m[source_index] - track.clip.root_translation_m[0]
    ).astype(np.float64, copy=False)
    yaw = np.asarray(
        (
            (math.cos(track.yaw_radians), -math.sin(track.yaw_radians), 0.0),
            (math.sin(track.yaw_radians), math.cos(track.yaw_radians), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    displacement_court = yaw @ (_SMPLH_TO_COURT @ root_relative)
    anchor = np.asarray(track.anchor_position_court_m, dtype=np.float64)
    position_court = anchor + displacement_court
    if abs(position_court[0]) > 4.115 or abs(position_court[1]) > 11.885:
        raise ValueError(
            f"Track {track.object_id!r} frame {source_index} leaves the singles court."
        )
    court_from_asset = np.eye(4, dtype=np.float64)
    court_from_asset[:3, :3] = yaw @ _SMPLH_TO_COURT
    court_from_asset[:3, 3] = position_court
    scene_from_asset = target_court.scene_from_court.matrix() @ court_from_asset
    return PLCSFrameEntry(
        object_id=track.object_id,
        instance_id=track.instance_id,
        frame_index=frame_index,
        present=True,
        source_frame_index=source_index,
        scene_from_asset=GaussianTransform(
            scale=1.0,
            rigid=RigidTransform.from_matrix(scene_from_asset),
        ),
    )


__all__ = [
    "PLCSFrameEntry",
    "PLCSGlobalFrame",
    "PLCSGlobalTimeline",
    "PLCSLogicalScene",
    "PLCSObjectTrack",
    "PLCSSceneInventory",
    "build_global_timeline",
]
