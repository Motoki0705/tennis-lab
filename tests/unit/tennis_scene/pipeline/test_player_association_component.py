"""The ``player_association`` node: side-resolved cameras in, per-frame player IDs or an explicit stop out."""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.court_side.hypothesis import HypothesisScore
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationInput,
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from tests.unit.tasks.player_association.test_association import (
    CAMERAS,
    FAR,
    FPS,
    FRAMES,
    FULL,
    NEAR,
    SIZE,
    SPECTATOR,
    _config,
    _tracks,
)

IDS = tuple(camera.camera_id for camera in CAMERAS)
# cam2's calibration is its camera-local court: half-turned relative to the reference.
TURNS = (False, False, True)


def _source(cameras: tuple[str, ...] = IDS) -> ClipSource:
    return ClipSource("clip", tuple(SourceVideo(c, Path(f"{c}.mp4"), c * 20, FRAMES, FPS, *SIZE) for c in cameras))


def _calibration() -> CourtCalibrationOutput:
    views = tuple(SimpleNamespace(camera=camera.half_turned(turn), source_index=index)
                  for index, (camera, turn) in enumerate(zip(CAMERAS, TURNS, strict=True)))
    return cast(CourtCalibrationOutput, SimpleNamespace(calibration=SimpleNamespace(views=views)))


def _side(turns: tuple[bool, ...] = TURNS) -> CourtSideOutput:
    hypotheses = [HypothesisScore((False, *rest), 1., 0., 10) for rest in itertools.product((False, True), repeat=2) if (False, *rest) != turns]
    return CourtSideOutput(IDS, "cam0", turns, (HypothesisScore(turns, .1, .9, 10), *hypotheses), .5, 10)


def _tracking(camera_index: int, tracks: list[tuple[int, list[tuple[int, int, np.ndarray, int]]]]) -> PersonTrackingOutput:
    built = _tracks(CAMERAS[camera_index], tracks, with_appearance=False)
    return PersonTrackingOutput(IDS[camera_index], built.track_ids, built.boxes_xyxy.astype(np.float32), built.observed,
                                tuple((int(t),) for t in built.track_ids), ())


def _singles(cam0: list[tuple[int, list[tuple[int, int, np.ndarray, int]]]] | None = None) -> tuple[PersonTrackingOutput, ...]:
    return (_tracking(0, cam0 or [(7, [(*FULL, NEAR, 0)]), (3, [(*FULL, FAR, 1)]), (9, [(*FULL, SPECTATOR, 2)])]),
            _tracking(1, [(1, [(*FULL, FAR, 1)]), (2, [(*FULL, NEAR, 0)])]),
            _tracking(2, [(4, [(*FULL, NEAR, 0)]), (8, [(*FULL, FAR, 1)])]))


def _module(**changes: Any) -> PlayerAssociationModule:
    arguments: dict[str, Any] = {"sampling": CropSamplingConfig(), "encoder": None, "device": "cpu", "enabled": True}
    return PlayerAssociationModule(IDS, _config(appearance=None), **{**arguments, **changes})


def _run(module: PlayerAssociationModule, tracks: tuple[PersonTrackingOutput, ...], side: CourtSideOutput | None = None) -> PlayerIdentitiesOutput:
    return module.process(PlayerAssociationInput(_source(), _calibration(), side or _side(), tracks))


def test_players_get_per_frame_ids_on_the_carrier_order_of_every_camera() -> None:
    tracks = _singles()
    identities = _run(_module(), tracks)
    assert identities.camera_ids == IDS
    # Carriers are padded to the largest camera; padding never carries a player.
    assert identities.local_track_ids.tolist() == [[7, 3, 9], [1, 2, -1], [4, 8, -1]]
    assert identities.player_ids.shape == (3, 3, FRAMES)
    assert [sorted({int(p) for p in row}) for row in identities.player_ids[0]] == [[0], [1], [-1]]
    assert [sorted({int(p) for p in row}) for row in identities.player_ids[2]] == [[0], [1], [-1]]
    assert identities.diagnostics["view_half_turns"] == list(TURNS) and identities.diagnostics["appearance"] == {"enabled": False}
    assert identities.diagnostics["association"]["min_player_margin"] >= 1.


def test_unobserved_frames_carry_no_player_and_an_identity_switch_splits_a_track() -> None:
    # cam0 track 3 follows the spectator, then (from frame 60) the far player, whose own track 6 ends there.
    tracks = _singles([(7, [(*FULL, NEAR, 0)]), (6, [(0, 60, FAR, 1)]), (3, [(0, 60, SPECTATOR, 2), (60, FRAMES, FAR, 1)])])
    identities = _run(_module(), tracks)
    rows = dict(zip(identities.local_track_ids[0].tolist(), identities.player_ids[0], strict=True))
    assert set(rows[6][:60].tolist()) == {1} and set(rows[6][60:].tolist()) == {-1}
    assert set(rows[3][:55].tolist()) == {-1} and set(rows[3][65:].tolist()) == {1}


def test_the_decided_side_places_the_camera_local_courts() -> None:
    # Left unturned, cam2's camera-local court puts its near player on the far half: its tracks swap players.
    # The node trusts the side it is given; deciding it is court_side's job.
    identities = _run(_module(), _singles(), _side((False, False, False)))
    assert [sorted({int(p) for p in row}) for row in identities.player_ids[2][:2]] == [[1], [0]]
    assert [sorted({int(p) for p in row}) for row in identities.player_ids[0][:2]] == [[0], [1]]


def test_an_undecidable_clip_stops_with_the_reason_and_the_evidence() -> None:
    # Only the near player is tracked: the far side has no player.
    tracks = tuple(_tracking(v, [(1, [(*FULL, NEAR, 0)])]) for v in range(3))
    with pytest.raises(ReconstructionUnavailable) as stopped:
        _run(_module(), tracks)
    assert stopped.value.reason == "player_association_player_not_found"
    assert stopped.value.diagnostics["association"]["identities"] and stopped.value.diagnostics["appearance"] == {"enabled": False}


def test_disabled_people_publish_no_players() -> None:
    empty = tuple(PersonTrackingOutput(c, np.empty(0, np.int64), np.empty((0, FRAMES, 4), np.float32), np.empty((0, FRAMES), bool), (), ())
                  for c in IDS)
    identities = _run(_module(enabled=False), empty)
    assert identities.player_ids.shape == (3, 0, FRAMES) and identities.diagnostics == {"enabled": False}


def test_inputs_and_encoders_must_agree_with_the_declaration() -> None:
    with pytest.raises(ValueError, match="exactly when"):
        _module(encoder=lambda: None)
    with pytest.raises(ValueError, match="exactly when"):
        PlayerAssociationModule(IDS, _config(), sampling=CropSamplingConfig(), encoder=None, device="cpu", enabled=True)
    named = PlayerAssociationModule(IDS, _config(), sampling=CropSamplingConfig(), device="cpu", enabled=True,
                                    encoder=lambda: cast(Any, SimpleNamespace(name="other", input_size=(256, 128))))
    with pytest.raises(ValueError, match="not the configured"):
        _run(named, _singles())
    with pytest.raises(ValueError, match="other cameras"):
        _run(_module(), _singles(), _side_for(("cam0", "cam1", "cam9")))
    with pytest.raises(ValueError, match="every source camera"):
        _run(_module(), _singles()[:2])


def _side_for(cameras: tuple[str, ...]) -> CourtSideOutput:
    side = _side()
    return CourtSideOutput(cameras, cameras[0], side.view_half_turns, side.hypotheses, side.margin, side.frames)


def test_the_output_contract_rejects_two_boxes_of_one_player_and_players_on_padding() -> None:
    tracks = np.array([[1, 2], [3, -1]], np.int64)
    players: np.ndarray = np.full((2, 2, 4), -1, np.int64)
    PlayerIdentitiesOutput(("a", "b"), tracks, players, {})
    both = players.copy()
    both[0, :, 1] = 0  # two tracks of camera a carry player 0 at frame 1
    with pytest.raises(ValueError, match="two tracks in one frame"):
        PlayerIdentitiesOutput(("a", "b"), tracks, both, {})
    handoff = players.copy()
    handoff[0, 0, :2], handoff[0, 1, 2:] = 0, 0  # one player moving from track 1 to track 2 is fine
    PlayerIdentitiesOutput(("a", "b"), tracks, handoff, {})
    padded = players.copy()
    padded[1, 1, 0] = 0
    with pytest.raises(ValueError, match="padding"):
        PlayerIdentitiesOutput(("a", "b"), tracks, padded, {})
    with pytest.raises(ValueError, match="aligned"):
        PlayerIdentitiesOutput(("a", "b"), tracks, players[:, :, 0], {})
