"""Manual player association for multi-camera tennis scene inputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.io import load_json, save_json
from src.utils.video import read_video_frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.pipeline.model_io.gvhmr import GVHMRResult
    from src.utils.video import VideoInfo

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlayerAssociationConfig:
    """Configuration for manual player association."""

    source: Literal["execute", "load"]
    mode: Literal["manual_ui"]
    initial_frame_index: int
    reference_camera: str | int
    save_result: bool
    output_path: Path
    load_path: Path | None

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "PlayerAssociation source='load' requires load_path; execute forbids it"
            )


@dataclass
class PlayerAssociationSegment:
    """A temporal assignment segment with frame interval [start_frame, end_frame)."""

    start_frame: int
    end_frame: int
    assignments: NDArray[np.int32]  # (P, N), local GVHMR player axis per camera

    def to_dict(self) -> dict:
        """Convert segment to JSON-serializable dict."""
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "assignments": self.assignments.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerAssociationSegment:
        """Create segment from dict."""
        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            assignments=np.asarray(data["assignments"], dtype=np.int32),
        )


@dataclass
class PlayerAssociationResult:
    """Manual player association result.

    Attributes:
        camera_ids: Camera identifiers aligned to N.
        canonical_player_ids: Stable player IDs aligned to output P.
        segments: Temporal assignments. Each assignment is shaped (P, N) and maps
            canonical player/camera to a local GVHMR player axis index.
        reference_camera: Camera ID used for SMPL arrays without a camera axis.
    """

    camera_ids: list[str]
    canonical_player_ids: NDArray[np.int32]
    segments: list[PlayerAssociationSegment]
    reference_camera: str

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "camera_ids": self.camera_ids,
            "canonical_player_ids": self.canonical_player_ids.tolist(),
            "segments": [segment.to_dict() for segment in self.segments],
            "reference_camera": self.reference_camera,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerAssociationResult:
        """Create result from dict."""
        return cls(
            camera_ids=[str(camera_id) for camera_id in data["camera_ids"]],
            canonical_player_ids=np.asarray(
                data["canonical_player_ids"],
                dtype=np.int32,
            ),
            segments=[
                PlayerAssociationSegment.from_dict(segment)
                for segment in data["segments"]
            ],
            reference_camera=str(data["reference_camera"]),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved player association result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PlayerAssociationResult:
        """Load result from JSON file."""
        return cls.from_dict(load_json(path))

    def reference_camera_index(self) -> int:
        """Return the reference camera index."""
        try:
            return self.camera_ids.index(self.reference_camera)
        except ValueError as exc:
            raise ValueError(
                f"reference_camera={self.reference_camera!r} is not in camera_ids"
            ) from exc

    def validate(
        self,
        *,
        num_frames: int,
        local_player_counts: Sequence[int],
    ) -> tuple[bool, list[str]]:
        """Validate association coverage and local player assignments."""
        errors: list[str] = []
        num_cameras = len(self.camera_ids)
        num_players = int(self.canonical_player_ids.shape[0])

        if self.canonical_player_ids.ndim != 1:
            errors.append(
                "canonical_player_ids must have shape (P,), "
                f"got {self.canonical_player_ids.shape}"
            )
            num_players = 0
        if len(local_player_counts) != num_cameras:
            errors.append(
                "local_player_counts length must match camera_ids length, "
                f"got {len(local_player_counts)} and {num_cameras}"
            )
        if not self.segments:
            errors.append("segments must not be empty")
            return False, errors
        if self.reference_camera not in self.camera_ids:
            errors.append(
                f"reference_camera={self.reference_camera!r} is not in camera_ids"
            )

        expected_start = 0
        for segment_index, segment in enumerate(self.segments):
            if segment.start_frame != expected_start:
                errors.append(
                    f"segment {segment_index} must start at {expected_start}, "
                    f"got {segment.start_frame}"
                )
            if segment.end_frame <= segment.start_frame:
                errors.append(
                    f"segment {segment_index} must have end_frame > start_frame"
                )
            if segment.assignments.shape != (num_players, num_cameras):
                errors.append(
                    f"segment {segment_index} assignments must have shape "
                    f"{(num_players, num_cameras)}, got {segment.assignments.shape}"
                )
                expected_start = segment.end_frame
                continue
            for camera_index in range(num_cameras):
                assigned = segment.assignments[:, camera_index]
                if np.any(assigned < 0):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} has "
                        "negative local player index"
                    )
                if camera_index < len(local_player_counts) and np.any(
                    assigned >= int(local_player_counts[camera_index])
                ):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} local "
                        "player index is out of range"
                    )
                if len(np.unique(assigned)) != len(assigned):
                    errors.append(
                        f"segment {segment_index} camera {camera_index} assigns "
                        "the same local player to multiple canonical players"
                    )
            expected_start = segment.end_frame

        if expected_start != num_frames:
            errors.append(
                f"segments must cover [0, {num_frames}), got end {expected_start}"
            )
        return len(errors) == 0, errors


@dataclass
class PlayerAssociationApplied:
    """GVHMR arrays aligned by a player association result."""

    human_kp_2d: NDArray[np.float32]  # (P, N, T, 17, 2), normalized
    human_kp_vis: NDArray[np.float32]  # (P, N, T, 17)
    smpl_body_pose: NDArray[np.float32]  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32]  # (P, T, 3)
    smpl_betas: NDArray[np.float32]  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None  # (P, T, V, 3)
    track_ids: NDArray[np.int32]  # (P,)
    track_ids_by_camera: list[NDArray[np.int32]]  # length N


class PlayerAssociationModule(BasePipelineModule):
    """Manual player association module."""

    def __init__(self, config: PlayerAssociationConfig) -> None:
        self.config = config

    def load(self) -> None:
        """No model state is required."""

    @property
    def is_loaded(self) -> bool:
        """Return True because this module has no model state."""
        return True

    def process(
        self,
        *,
        gvhmr_results: Sequence[GVHMRResult],
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        camera_ids: Sequence[str],
    ) -> PlayerAssociationResult:
        """Create or load player association for GVHMR results."""
        local_player_counts = [result.human_kp_2d.shape[0] for result in gvhmr_results]
        num_frames = int(gvhmr_results[0].human_kp_2d.shape[1])
        camera_ids = [str(camera_id) for camera_id in camera_ids]

        if self.config.source == "load":
            load_path = self.config.load_path
            if load_path is None:
                raise RuntimeError("Validated load source is missing load_path")
            if not load_path.is_file():
                raise FileNotFoundError(
                    f"Player association artifact not found: {load_path}"
                )
            LOGGER.info(f"Loading player association result from {load_path}")
            result = PlayerAssociationResult.load(load_path)
            self._validate_or_raise(
                result,
                num_frames=num_frames,
                local_player_counts=local_player_counts,
            )
            return result

        if len(camera_ids) == 1:
            result = self._process_single_camera_identity(
                camera_ids=camera_ids,
                num_frames=num_frames,
                local_player_counts=local_player_counts,
            )
            self._validate_or_raise(
                result,
                num_frames=num_frames,
                local_player_counts=local_player_counts,
            )
            if self.config.save_result:
                result.save(self.config.output_path)
            return result

        result = self._process_manual_ui(
            gvhmr_results=gvhmr_results,
            video_paths=video_paths,
            video_infos=video_infos,
            camera_ids=camera_ids,
            num_frames=num_frames,
            local_player_counts=local_player_counts,
        )
        self._validate_or_raise(
            result,
            num_frames=num_frames,
            local_player_counts=local_player_counts,
        )

        if self.config.save_result:
            result.save(self.config.output_path)
        return result

    def apply(
        self,
        *,
        gvhmr_results: Sequence[GVHMRResult],
        video_infos: Sequence[VideoInfo],
        association: PlayerAssociationResult,
    ) -> PlayerAssociationApplied:
        """Apply association and return canonical player/camera arrays."""
        local_player_counts = [result.human_kp_2d.shape[0] for result in gvhmr_results]
        num_frames = int(gvhmr_results[0].human_kp_2d.shape[1])
        self._validate_or_raise(
            association,
            num_frames=num_frames,
            local_player_counts=local_player_counts,
        )
        return apply_player_association(
            gvhmr_results=gvhmr_results,
            video_infos=video_infos,
            association=association,
        )

    def _validate_or_raise(
        self,
        result: PlayerAssociationResult,
        *,
        num_frames: int,
        local_player_counts: Sequence[int],
    ) -> None:
        is_valid, errors = result.validate(
            num_frames=num_frames,
            local_player_counts=local_player_counts,
        )
        if not is_valid:
            raise ValueError(f"Invalid player association result: {errors}")

    def _process_single_camera_identity(
        self,
        *,
        camera_ids: Sequence[str],
        num_frames: int,
        local_player_counts: Sequence[int],
    ) -> PlayerAssociationResult:
        """Create explicit identity association for a single-camera pipeline run."""
        num_players = int(local_player_counts[0])
        if num_players <= 0:
            raise ValueError("GVHMR did not produce any players")

        reference_camera = self._resolve_reference_camera(camera_ids)
        LOGGER.info(
            "single camera -> identity association: "
            f"camera={camera_ids[0]}, players={num_players}"
        )
        return PlayerAssociationResult(
            camera_ids=list(camera_ids),
            canonical_player_ids=np.arange(num_players, dtype=np.int32),
            segments=[
                PlayerAssociationSegment(
                    start_frame=0,
                    end_frame=num_frames,
                    assignments=np.arange(num_players, dtype=np.int32).reshape(
                        num_players,
                        1,
                    ),
                )
            ],
            reference_camera=reference_camera,
        )

    def _process_manual_ui(
        self,
        *,
        gvhmr_results: Sequence[GVHMRResult],
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        camera_ids: Sequence[str],
        num_frames: int,
        local_player_counts: Sequence[int],
    ) -> PlayerAssociationResult:
        """Collect temporal player assignments with a lightweight manual UI."""
        num_players = min(local_player_counts)
        if num_players <= 0:
            raise ValueError("GVHMR did not produce any players")

        reference_camera = self._resolve_reference_camera(camera_ids)
        segments = [
            PlayerAssociationSegment(
                start_frame=0,
                end_frame=num_frames,
                assignments=np.stack(
                    [np.arange(num_players, dtype=np.int32) for _ in camera_ids],
                    axis=1,
                ),
            )
        ]
        current_frame = min(
            max(int(self.config.initial_frame_index), 0),
            max(num_frames - 1, 0),
        )

        print(
            "Player association manual UI. Commands: "
            "n/p frame, s <frame> split, a <seg> <player> <cam> <local>, "
            "save, q"
        )
        window_name = "Player Association"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                image = self._draw_association_frame(
                    video_paths=video_paths,
                    video_infos=video_infos,
                    gvhmr_results=gvhmr_results,
                    camera_ids=camera_ids,
                    segments=segments,
                    frame_index=current_frame,
                )
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
                command = input("player-association> ").strip()
                if command in {"q", "quit", "save"}:
                    break
                if command in {"n", "next"}:
                    current_frame = min(current_frame + 1, num_frames - 1)
                    continue
                if command in {"p", "prev"}:
                    current_frame = max(current_frame - 1, 0)
                    continue
                parts = command.split()
                if len(parts) == 2 and parts[0] in {"s", "split"}:
                    segments = self._split_segments(segments, int(parts[1]))
                    continue
                if len(parts) == 5 and parts[0] in {"a", "assign"}:
                    segment_index = int(parts[1])
                    player_index = int(parts[2])
                    camera_index = self._parse_camera(parts[3], camera_ids)
                    local_player_index = int(parts[4])
                    segments[segment_index].assignments[player_index, camera_index] = (
                        local_player_index
                    )
                    continue
                print(f"Unknown command: {command}")
        finally:
            cv2.destroyWindow(window_name)

        return PlayerAssociationResult(
            camera_ids=list(camera_ids),
            canonical_player_ids=np.arange(num_players, dtype=np.int32),
            segments=segments,
            reference_camera=reference_camera,
        )

    def _resolve_reference_camera(self, camera_ids: Sequence[str]) -> str:
        reference = self.config.reference_camera
        if isinstance(reference, int):
            return str(camera_ids[reference])
        reference_text = str(reference)
        if reference_text.isdigit():
            return str(camera_ids[int(reference_text)])
        return reference_text

    @staticmethod
    def _parse_camera(value: str, camera_ids: Sequence[str]) -> int:
        if value.isdigit():
            return int(value)
        return list(camera_ids).index(value)

    @staticmethod
    def _split_segments(
        segments: Sequence[PlayerAssociationSegment],
        frame_index: int,
    ) -> list[PlayerAssociationSegment]:
        """Split the segment containing frame_index."""
        new_segments: list[PlayerAssociationSegment] = []
        inserted = False
        for segment in segments:
            if segment.start_frame < frame_index < segment.end_frame:
                new_segments.append(
                    PlayerAssociationSegment(
                        start_frame=segment.start_frame,
                        end_frame=frame_index,
                        assignments=segment.assignments.copy(),
                    )
                )
                new_segments.append(
                    PlayerAssociationSegment(
                        start_frame=frame_index,
                        end_frame=segment.end_frame,
                        assignments=segment.assignments.copy(),
                    )
                )
                inserted = True
            else:
                new_segments.append(segment)
        if not inserted:
            LOGGER.warning(f"No segment was split at frame {frame_index}")
        return new_segments

    def _draw_association_frame(
        self,
        *,
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        gvhmr_results: Sequence[GVHMRResult],
        camera_ids: Sequence[str],
        segments: Sequence[PlayerAssociationSegment],
        frame_index: int,
    ) -> NDArray[np.uint8]:
        """Draw all cameras with GVHMR labels for manual association."""
        segment_index = self._find_segment_index(segments, frame_index)
        segment = segments[segment_index]
        frames: list[NDArray[np.uint8]] = []
        for camera_index, video_path in enumerate(video_paths):
            packet = read_video_frame(video_path, frame_index)
            frame = packet.frame.copy()
            result = gvhmr_results[camera_index]
            for local_player_index in range(result.human_kp_2d.shape[0]):
                kp = result.human_kp_2d[local_player_index, frame_index]
                vis = result.human_kp_vis[local_player_index, frame_index]
                self._draw_player_overlay(
                    frame,
                    keypoints=kp,
                    visibility=vis,
                    label=self._player_label(
                        segment=segment,
                        camera_index=camera_index,
                        local_player_index=local_player_index,
                        track_id=result.track_ids[local_player_index],
                    ),
                )
            cv2.putText(
                frame,
                (
                    f"{camera_ids[camera_index]} frame={frame_index} "
                    f"segment={segment_index} size="
                    f"{video_infos[camera_index].width}x{video_infos[camera_index].height}"
                ),
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)
        return np.concatenate(frames, axis=1)

    @staticmethod
    def _find_segment_index(
        segments: Sequence[PlayerAssociationSegment],
        frame_index: int,
    ) -> int:
        for index, segment in enumerate(segments):
            if segment.start_frame <= frame_index < segment.end_frame:
                return index
        raise ValueError(f"No association segment contains frame {frame_index}")

    @staticmethod
    def _player_label(
        *,
        segment: PlayerAssociationSegment,
        camera_index: int,
        local_player_index: int,
        track_id: int,
    ) -> str:
        canonical_matches = np.where(
            segment.assignments[:, camera_index] == local_player_index
        )[0]
        canonical_text = (
            f"P{int(canonical_matches[0])}" if canonical_matches.size else "unassigned"
        )
        return f"{canonical_text} local={local_player_index} track={track_id}"

    @staticmethod
    def _draw_player_overlay(
        frame: NDArray[np.uint8],
        *,
        keypoints: NDArray[np.float32],
        visibility: NDArray[np.float32],
        label: str,
    ) -> None:
        visible = visibility > 0
        if not visible.any():
            return
        points = keypoints[visible]
        x0, y0 = points.min(axis=0).astype(int)
        x1, y1 = points.max(axis=0).astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x0, max(y0 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def apply_player_association(
    *,
    gvhmr_results: Sequence[GVHMRResult],
    video_infos: Sequence[VideoInfo],
    association: PlayerAssociationResult,
) -> PlayerAssociationApplied:
    """Apply temporal association to per-camera GVHMR outputs."""
    num_players = int(association.canonical_player_ids.shape[0])
    num_cameras = len(association.camera_ids)
    first_result = gvhmr_results[0]
    num_frames = int(first_result.human_kp_2d.shape[1])

    human_kp_2d: NDArray[np.float32] = np.zeros(
        (num_players, num_cameras, num_frames, 17, 2),
        dtype=np.float32,
    )
    human_kp_vis: NDArray[np.float32] = np.zeros(
        (num_players, num_cameras, num_frames, 17),
        dtype=np.float32,
    )

    reference_camera_index = association.reference_camera_index()
    reference_result = gvhmr_results[reference_camera_index]
    smpl_body_pose: NDArray[np.float32] = np.zeros(
        (num_players, num_frames, 63),
        dtype=np.float32,
    )
    smpl_global_orient: NDArray[np.float32] = np.zeros(
        (num_players, num_frames, 3),
        dtype=np.float32,
    )
    smpl_betas: NDArray[np.float32] = np.zeros((num_players, 10), dtype=np.float32)
    smpl_vertices_local: NDArray[np.float32] | None = None
    if reference_result.smpl_vertices_local is not None:
        vertex_shape = reference_result.smpl_vertices_local.shape[2:]
        smpl_vertices_local = np.zeros(
            (num_players, num_frames, *vertex_shape),
            dtype=np.float32,
        )

    for segment in association.segments:
        frame_slice = slice(segment.start_frame, segment.end_frame)
        for player_index in range(num_players):
            for camera_index, video_info in enumerate(video_infos):
                local_player_index = int(
                    segment.assignments[player_index, camera_index]
                )
                local_human = (
                    gvhmr_results[camera_index]
                    .human_kp_2d[
                        local_player_index,
                        frame_slice,
                    ]
                    .copy()
                )
                local_human[..., 0] /= video_info.width
                local_human[..., 1] /= video_info.height
                human_kp_2d[player_index, camera_index, frame_slice] = local_human
                human_kp_vis[player_index, camera_index, frame_slice] = gvhmr_results[
                    camera_index
                ].human_kp_vis[local_player_index, frame_slice]

            reference_local_player = int(
                segment.assignments[player_index, reference_camera_index]
            )
            smpl_body_pose[player_index, frame_slice] = reference_result.smpl_body_pose[
                reference_local_player,
                frame_slice,
            ]
            smpl_global_orient[player_index, frame_slice] = (
                reference_result.smpl_global_orient[
                    reference_local_player,
                    frame_slice,
                ]
            )
            if (
                smpl_vertices_local is not None
                and reference_result.smpl_vertices_local is not None
            ):
                smpl_vertices_local[player_index, frame_slice] = (
                    reference_result.smpl_vertices_local[
                        reference_local_player,
                        frame_slice,
                    ]
                )

    first_segment = association.segments[0]
    track_ids = association.canonical_player_ids.astype(np.int32)
    for player_index in range(num_players):
        reference_local_player = int(
            first_segment.assignments[player_index, reference_camera_index]
        )
        smpl_betas[player_index] = reference_result.smpl_betas[reference_local_player]

    track_ids_by_camera = [
        result.track_ids.astype(np.int32) for result in gvhmr_results
    ]

    return PlayerAssociationApplied(
        human_kp_2d=human_kp_2d,
        human_kp_vis=human_kp_vis,
        smpl_body_pose=smpl_body_pose,
        smpl_global_orient=smpl_global_orient,
        smpl_betas=smpl_betas,
        smpl_vertices_local=smpl_vertices_local,
        track_ids=track_ids,
        track_ids_by_camera=track_ids_by_camera,
    )
