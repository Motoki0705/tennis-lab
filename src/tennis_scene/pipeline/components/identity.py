"""Cross-camera player identities: the ``player_association`` node (#933).

The node runs ``src.tasks.player_association.association.associate`` on the
camera-local tracks (their boxes, not the pose) of the calibrated cameras,
placed on the court by the side ``court_side`` decided. When the association
config has an ``appearance`` section, the Re-ID embeddings of sampled crops of
the source videos are added. Pixel thresholds (the footpoint border and the
crop sampling) are configured at 1920x1080 and scaled to the source, as for
``court_side``.

A player ID is given per frame, so one track can carry two people across an
identity switch. A clip the association cannot decide stops with
``ReconstructionUnavailable("player_association_<reason>")`` and every score
up to the stop; it never falls back to a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
    embed_tracks,
)
from src.tasks.player_association.association.associate import (
    AssociationConfig,
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import (
    SIDE_PORT,
    CourtSideOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable

PLAYER_ASSOCIATION = "player_association"
# Version 3: per-frame player IDs (a track may switch people) and the association evidence.
IDENTITIES_PORT = InputPort("person_identities", 3)


@dataclass(frozen=True)
class PlayerIdentitiesOutput:
    """Clip-global player ID of every camera-local track, frame by frame.

    ``camera_ids`` are the calibrated cameras in calibration order. Row ``v`` of
    ``local_track_ids`` repeats the tracker IDs of that camera's pose carriers
    (``-1`` for padding) so a join can verify the carrier order it was built on.
    ``player_ids[v, d, t]`` is the player carrier ``d`` observes at frame ``t``,
    or ``-1`` (not a player, or not observed). ``diagnostics`` holds the
    association evidence: segments, pair scores, identities, the selection of
    the players and the decision margins.
    """

    camera_ids: tuple[str, ...]
    local_track_ids: NDArray[np.int64]  # (V, D)
    player_ids: NDArray[np.int64]  # (V, D, T)
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        views = len(self.camera_ids)
        if views < 1 or len(set(self.camera_ids)) != views or any(not c for c in self.camera_ids):
            raise ValueError("Player identities require unique nonempty camera IDs")
        tracks, players = self.local_track_ids, self.player_ids
        if not isinstance(tracks, np.ndarray) or tracks.dtype != np.int64 or tracks.ndim != 2 or len(tracks) != views:
            raise ValueError("local_track_ids must be int64 (V, D) aligned to camera_ids")
        if not isinstance(players, np.ndarray) or players.dtype != np.int64 or players.ndim != 3 or players.shape[:2] != tracks.shape:
            raise ValueError("player_ids must be int64 (V, D, T) aligned to local_track_ids")
        if ((tracks < 0)[:, :, None] & (players >= 0)).any():
            raise ValueError("A padding carrier cannot carry a player ID")
        if (players < -1).any():
            raise ValueError("Player IDs are nonnegative, or -1 for no player")
        for row in players:
            for player in np.unique(row[row >= 0]):
                if ((row == player).sum(0) > 1).any():
                    raise ValueError("One camera cannot observe the same player through two tracks in one frame")


@dataclass(frozen=True)
class PlayerAssociationInput:
    source: ClipSource
    calibration: CourtCalibrationOutput
    side: CourtSideOutput
    tracks: tuple[PersonTrackingOutput, ...]  # every source camera, in source order


def player_association_io(camera_ids: tuple[str, ...]) -> ComponentIO[PlayerAssociationInput, PlayerIdentitiesOutput]:
    """The node's contract: calibration, the decided side and the tracks of every source camera."""
    return ComponentIO(PLAYER_ASSOCIATION, PlayerAssociationInput, PlayerIdentitiesOutput,
        {"calibration": InputPort("local_court_calibration"), "side": SIDE_PORT,
         **{f"tracks_{c}": InputPort("selected_player_tracks", 2) for c in camera_ids}},
        IDENTITIES_PORT.schema, IDENTITIES_PORT.version)


class PlayerAssociationModule:
    """``encoder`` builds the appearance encoder; it is required exactly when ``config.appearance`` is set.

    A disabled module (person observations off) publishes identities without
    players and runs nothing.
    """

    def __init__(self, camera_ids: tuple[str, ...], config: AssociationConfig, *, sampling: CropSamplingConfig,
                 encoder: Callable[[], AppearanceEncoder] | None, device: str, enabled: bool) -> None:
        if (config.appearance is None) != (encoder is None):
            raise ValueError("An appearance encoder is required exactly when the association config scores appearance")
        self.config, self.sampling, self.encoder, self.device, self.enabled = config, sampling, encoder, device, enabled
        self.io = player_association_io(camera_ids)

    def _appearances(self, source: ClipSource, tracks: list[PersonTrackingOutput], sampling: CropSamplingConfig
                     ) -> tuple[list[tuple[TrackAppearance, ...]] | None, dict[str, Any]]:
        if self.encoder is None or self.config.appearance is None:
            return None, {"enabled": False}
        name = self.config.appearance.encoder
        encoder = self.encoder()
        if encoder.name != name:
            raise ValueError(f"The appearance encoder {encoder.name!r} is not the configured {name!r}")
        appearances, record = [], {}
        try:
            for camera in tracks:
                appearance, samples = embed_tracks(source.video(camera.camera_id).path, camera.boxes_xyxy, camera.observed,
                                                   source.size, encoder, sampling)
                appearances.append(appearance)
                record[camera.camera_id] = [{"track_id": int(track), "samples": len(sample.frames), "rejected": sample.rejected}
                                            for track, sample in zip(camera.track_ids.tolist(), samples, strict=True)]
        finally:
            del encoder
            release_inference_memory(self.device)
        return appearances, {"enabled": True, "encoder": name, "sampling": asdict(sampling), "tracks": record}

    def process(self, inputs: PlayerAssociationInput) -> PlayerIdentitiesOutput:
        source, views = inputs.source, inputs.calibration.calibration.views
        camera_ids = tuple(view.camera.camera_id for view in views)
        if inputs.side.camera_ids != camera_ids:
            raise ValueError("The court side was decided for other cameras than the calibrated ones")
        if tuple(t.camera_id for t in inputs.tracks) != source.camera_ids or any(t.observed.shape[1] != source.num_frames for t in inputs.tracks):
            raise ValueError("Association requires the tracks of every source camera on the source timeline")
        by_camera = {t.camera_id: t for t in inputs.tracks}
        carriers = max((len(t.track_ids) for t in inputs.tracks), default=0)
        local: NDArray[np.int64] = np.full((len(views), carriers), -1, np.int64)
        players: NDArray[np.int64] = np.full((len(views), carriers, source.num_frames), -1, np.int64)
        for row, camera in enumerate(camera_ids):
            local[row, :len(by_camera[camera].track_ids)] = by_camera[camera].track_ids
        if not self.enabled:
            return PlayerIdentitiesOutput(camera_ids, local, players, {"enabled": False})

        scale = source.pixel_threshold_scale
        config = replace(self.config, footpoints=replace(self.config.footpoints, bottom_border_px=self.config.footpoints.bottom_border_px * scale))
        sampling = replace(self.sampling, min_height_px=self.sampling.min_height_px * scale, border_px=self.sampling.border_px * scale)
        tracks = [by_camera[camera] for camera in camera_ids]
        appearances, appearance_record = self._appearances(source, tracks, sampling)
        turns = dict(zip(inputs.side.camera_ids, inputs.side.view_half_turns, strict=True))
        cameras = [CameraTracks(view.camera.half_turned(turns[view.camera.camera_id]), source.size, t.track_ids, t.boxes_xyxy, t.observed,
                                None if appearances is None else appearances[row]) for row, (view, t) in enumerate(zip(views, tracks, strict=True))]
        try:
            association = associate(cameras, source.fps, config)
        except AssociationUndecided as undecided:
            raise ReconstructionUnavailable(f"player_association_{undecided.reason}", str(undecided), diagnostics={
                "association": undecided.diagnostics, "appearance": appearance_record}) from undecided
        for row, (t, ids) in enumerate(zip(tracks, association.player_ids, strict=True)):
            players[row, :len(t.track_ids)] = np.where(t.observed, ids, -1)
        return PlayerIdentitiesOutput(camera_ids, local, players, {
            "enabled": True, "view_half_turns": list(inputs.side.view_half_turns), "pixel_threshold_scale": scale,
            "appearance": appearance_record, "association": association.diagnostics})
