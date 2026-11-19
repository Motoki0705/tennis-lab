"""The ``player_association`` node: side-resolved cameras in, per-frame player IDs or an explicit stop out."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.court_side.hypothesis import HypothesisScore
from src.tasks.person_tracking.sequence import CLIP_ENCODER, TrackEvidence
from src.tasks.player_association.appearance import sampling
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tennis_scene.pipeline.components import person_features
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.identity import (
    PlayerAssociationInput,
    PlayerAssociationModule,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.input_assembly.reconstruction import (
    PlayerAssociationInputAssembler,
)
from src.utils.video import FramePacket
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
from tests.unit.tennis_scene.pipeline.components.test_association_appearance import (
    PixelEncoder,
)
from tests.unit.tennis_scene.pipeline.config_factories import fake_encoder_spec

IDS = tuple(camera.camera_id for camera in CAMERAS)
# cam2's calibration is its camera-local court: half-turned relative to the reference.
TURNS = (False, False, True)


def _source(cameras: tuple[str, ...] = IDS) -> ClipSource:
    return ClipSource("clip", tuple(SourceVideo(c, Path(f"{c}.mp4"), c * 20, FRAMES, FPS, *SIZE) for c in cameras))


def _calibration(scale: float = 1.) -> CourtCalibrationOutput:
    views = tuple(SimpleNamespace(camera=replace(camera.half_turned(turn), intrinsic=np.diag([scale, scale, 1.]) @ camera.intrinsic), source_index=index)
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
    return cast(PlayerIdentitiesOutput, module.process(PlayerAssociationInput(_source(), _calibration(), side or _side(), tracks)))


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
                                    encoder=fake_encoder_spec(lambda: cast(Any, SimpleNamespace(name="other", input_size=(256, 128)))))
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


def test_saved_crops_preserve_association_and_rejected_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the same camera geometry at one fifth of the pixel resolution.
    width, height = SIZE[0] // 5, SIZE[1] // 5
    source = ClipSource('clip', tuple(replace(video, width=width, height=height) for video in _source().videos))
    calibration = _calibration(.2)
    tracks = [replace(track, boxes_xyxy=track.boxes_xyxy / 5) for track in _singles()]
    tracks[0].boxes_xyxy[2, 1] = tracks[0].boxes_xyxy[0, 1]  # A duplicated/overlapped observation.
    encoder = PixelEncoder()
    spec = fake_encoder_spec(lambda: encoder)
    producer = person_features.PersonAppearanceFeaturesModule(spec, capture_frame_sha256=True)
    track_boxes = {track.camera_id: track for track in tracks}
    person_channels = {'cam0': [20, 170, 90], 'cam1': [170, 20], 'cam2': [20, 170]}

    def reader(path: Path, *, max_frames: int) -> Any:
        track = track_boxes[path.stem]
        for frame in range(max_frames):
            image = np.zeros((height, width, 3), np.uint8)
            for box, color in zip(track.boxes_xyxy[:, frame], person_channels[track.camera_id], strict=True):
                x1, y1, x2, y2 = np.rint(box).astype(int)
                image[y1:y2, x1:x2] = [color, frame + 1, 80]
            yield FramePacket(frame, image, (width, height))

    monkeypatch.setattr(sampling, 'OpenCVVideoFrameReader', reader)
    monkeypatch.setattr(person_features, 'OpenCVVideoFrameReader', reader)
    artifacts: dict[str, Any] = {'calibration': calibration, 'side': _side()}
    for video, track in zip(source.videos, tracks, strict=True):
        count, frames = track.observed.shape
        rows: np.ndarray = np.arange(count * frames, dtype=np.int64).reshape(frames, count) * 2 + 3
        # Reverse the detection order relative to carrier order. Keep the tracked
        # embeddings empty to ensure association reads the C5 rows.
        track_evidence = TrackEvidence(rows[:, ::-1].T, None, np.zeros((count, frames, 1280), np.float32),
                                       np.zeros_like(track.observed), CLIP_ENCODER)
        selection = replace(track, evidence=track_evidence)
        detections = PersonDetectionOutput(video.camera_id, np.arange(frames + 1, dtype=np.int64) * count,
            track.boxes_xyxy[::-1].transpose(1, 0, 2).reshape(-1, 4), np.ones(count * frames, np.float32), rows.ravel())
        artifacts[f'tracks_{video.camera_id}'] = SimpleNamespace(tracks=selection)
        artifacts[f'detections_{video.camera_id}'] = detections
        artifacts[f'appearance_{video.camera_id}'] = producer.process(person_features.PersonAppearanceFeaturesInput(video, detections))

    config = _config()
    assert config.appearance is not None
    config = replace(config, appearance=replace(config.appearance, encoder=CLIP_ENCODER))
    options: dict[str, Any] = dict(sampling=CropSamplingConfig(min_height_px=20., max_samples=5), device='cpu', enabled=True)
    context = AssemblyContext(source)
    encoder.calls.clear()
    fresh = PlayerAssociationModule(IDS, config, encoder=spec, **options).process(
        PlayerAssociationInputAssembler().assemble(context, artifacts))
    assert encoder.calls == [15, 10, 10]
    saved = PlayerAssociationModule(IDS, config, encoder=replace(spec, factory=lambda: pytest.fail('all crops matched')),
                                    reuse_clip_embeddings=True, **options).process(
        PlayerAssociationInputAssembler(reuse_clip_embeddings=True).assemble(context, artifacts))

    np.testing.assert_array_equal(saved.local_track_ids, fresh.local_track_ids)
    np.testing.assert_array_equal(saved.player_ids, fresh.player_ids)
    np.testing.assert_array_equal(saved.player_ids[0, :, 0], [0, 1, -1])
    assert saved.diagnostics['association'] == fresh.diagnostics['association']
    assert any('appearance' in pair for pair in saved.diagnostics['association']['pairs'])
    fresh_record, saved_record = (item.diagnostics['appearance'] for item in (fresh, saved))
    assert fresh_record['recomputed_samples'] == saved_record['reused_samples'] == 35
    assert fresh_record['reused_samples'] == saved_record['recomputed_samples'] == 0
    for camera in IDS:
        for original, reused in zip(fresh_record['tracks'][camera], saved_record['tracks'][camera], strict=True):
            assert original['recomputed_frames']['reuse_disabled'] == reused['reused_frames']
            assert {k: original[k] for k in ('track_id', 'samples', 'rejected')} == {k: reused[k] for k in ('track_id', 'samples', 'rejected')}
    assert saved_record['tracks']['cam0'][0]['rejected']['overlapped'] == 1
