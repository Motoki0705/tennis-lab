"""Association on synthetic three-camera scenes whose truth is known by construction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.tasks.player_association.appearance.affinity import AppearanceAffinityConfig
from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import (
    AMBIGUOUS_ASSOCIATION,
    AMBIGUOUS_PLAYERS,
    PLAYER_NOT_FOUND,
    AssociationConfig,
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    association_config,
    load_association_config,
)
from src.tasks.player_association.geometry.affinity import GeometryAffinityConfig
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_association.geometry.region import PlayRegionConfig
from src.tasks.player_association.geometry.switches import SwitchConfig
from src.utils.geometry.triangulation import PinholeCamera

FPS = 30.
FRAMES = 120
SIZE = (1920, 1080)


def _camera(camera_id: str, center: tuple[float, float, float]) -> PinholeCamera:
    position = np.asarray(center, float)
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(camera_id, np.array([[1100., 0, 960], [0, 1100, 540], [0, 0, 1]]), rotation, -rotation @ position)


CAMERAS = (_camera("cam0", (0., -34., 9.)), _camera("cam1", (9., -32., 7.)), _camera("cam2", (-2., 34., 8.)))


def _path(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    path: np.ndarray = np.linspace(start, end, FRAMES)
    return path


def _box(camera: PinholeCamera, ground: np.ndarray) -> np.ndarray:
    feet = camera.project(np.column_stack((ground, np.zeros(len(ground)))))[0]
    head = camera.project(np.column_stack((ground, np.full(len(ground), 1.8))))[0]
    width = .22 * (feet[:, 1] - head[:, 1])
    box: np.ndarray = np.column_stack((feet[:, 0] - width, head[:, 1], feet[:, 0] + width, feet[:, 1]))
    return box


def _embedding(person: int, dimension: int = 8) -> np.ndarray:
    vector: np.ndarray = np.zeros(dimension, np.float32)
    vector[person % dimension] = 1.
    return vector


def _tracks(camera: PinholeCamera, tracks: list[tuple[int, list[tuple[int, int, np.ndarray, int]]]],
            with_appearance: bool = True) -> CameraTracks:
    """``tracks``: (track id, [(start, end, ground path (T, 2), person), ...]) with the person used for appearance."""
    boxes = np.zeros((len(tracks), FRAMES, 4))
    observed: np.ndarray = np.zeros((len(tracks), FRAMES), bool)
    appearance = []
    for row, (_, pieces) in enumerate(tracks):
        frames, vectors = [], []
        for start, end, ground, person in pieces:
            boxes[row, start:end] = _box(camera, ground[start:end])
            observed[row, start:end] = True
            sampled = np.arange(start, end, 10)
            frames.append(sampled)
            vectors.append(np.repeat(_embedding(person)[None], len(sampled), 0))
        appearance.append(TrackAppearance(np.concatenate(frames).astype(np.int64), np.concatenate(vectors)))
    assert (boxes[observed][:, 3] < SIZE[1] - 4).all() and (boxes[observed][:, :2] > 0).all(), "synthetic box leaves the image"
    return CameraTracks(camera, SIZE, np.asarray([track for track, _ in tracks], np.int64), boxes, observed,
                        tuple(appearance) if with_appearance else None)


def _config(**changes: object) -> AssociationConfig:
    base = AssociationConfig(
        players_per_side=1, footpoints=FootpointConfig(4.), switches=SwitchConfig(.25, 3.),
        geometry=GeometryAffinityConfig(sigma_m=.8, area_m2=539.3, full_evidence_s=1., max_abs_score=10.),
        appearance=AppearanceAffinityConfig("synthetic", slope=10., center=.5, max_abs_score=4.),
        continuity_score=2., min_segment_s=.25, max_handoff_overlap_s=.2, region=PlayRegionConfig(2.5, 5.),
        min_presence_fraction=.25, max_runner_up_ratio=.5, min_margin=1., max_undecided_segment_s=1.)
    return replace(base, **changes)  # type: ignore[arg-type]


FAR = _path((-1., 12.5), (2., 13.5))  # player on the y > 0 side
NEAR = _path((1., -13.), (-2., -11.))  # player on the y < 0 side
SPECTATOR = _path((1., 19.), (1.2, 19.))  # behind the far fence, outside the play region
FULL = (0, FRAMES)


def _singles_scene(with_appearance: bool = True) -> list[CameraTracks]:
    return [
        _tracks(CAMERAS[0], [(7, [(*FULL, NEAR, 0)]), (3, [(*FULL, FAR, 1)]), (9, [(*FULL, SPECTATOR, 2)])], with_appearance),
        _tracks(CAMERAS[1], [(1, [(*FULL, FAR, 1)]), (2, [(*FULL, NEAR, 0)]), (5, [(*FULL, SPECTATOR, 2)])], with_appearance),
        _tracks(CAMERAS[2], [(4, [(*FULL, NEAR, 0)]), (8, [(*FULL, FAR, 1)])], with_appearance),
    ]


def _ids(result: object) -> dict[str, dict[int, set[int]]]:
    """Observed-frame player IDs per camera and track."""
    out: dict[str, dict[int, set[int]]] = {}
    for camera_id, tracks, ids in zip(result.camera_ids, result.track_ids, result.player_ids, strict=True):  # type: ignore[attr-defined]
        out[camera_id] = {int(track): set(ids[row].tolist()) for row, track in enumerate(tracks)}
    return out


def test_singles_players_are_matched_and_the_spectator_is_excluded() -> None:
    result = associate(_singles_scene(), FPS, _config())
    ids = _ids(result)
    # Player 0 is on the y < 0 side, player 1 on the y > 0 side, regardless of the tracker's IDs.
    assert ids == {"cam0": {7: {0}, 3: {1}, 9: {-1}}, "cam1": {1: {1}, 2: {0}, 5: {-1}}, "cam2": {4: {0}, 8: {1}}}
    spectator = [item for item in result.diagnostics["identities"] if item["side"] == 0]
    assert spectator and spectator[0]["presence_frames"] == 0
    assert result.diagnostics["min_player_margin"] >= 1.


def test_geometry_alone_matches_the_singles_scene() -> None:
    result = associate(_singles_scene(with_appearance=False), FPS, _config(appearance=None))
    assert _ids(result)["cam2"] == {4: {0}, 8: {1}}


def test_appearance_is_required_when_configured() -> None:
    with pytest.raises(ValueError, match="no track appearance"):
        associate(_singles_scene(with_appearance=False), FPS, _config())


def test_an_identity_switch_is_cut_and_both_parts_are_placed() -> None:
    # cam0 track 3 follows the spectator, then (from frame 60) the far player; cam0 track 6 had the far player until then.
    scene = _singles_scene()
    scene[0] = _tracks(CAMERAS[0], [(7, [(*FULL, NEAR, 0)]), (6, [(0, 60, FAR, 1)]), (3, [(0, 60, SPECTATOR, 2), (60, FRAMES, FAR, 1)])])
    result = associate(scene, FPS, _config())
    cam0 = dict(zip(result.track_ids[0].tolist(), result.player_ids[0], strict=True))
    assert set(cam0[3][:55].tolist()) == {-1} and set(cam0[3][65:].tolist()) == {1}
    assert set(cam0[6][:60].tolist()) == {1}
    cuts = [s for s in result.diagnostics["segments"] if s["camera"] == "cam0" and s["track_id"] == 3]
    assert len(cuts) == 2 and abs(cuts[1]["start"] - 60) <= 2


def test_a_short_handoff_overlap_keeps_one_box_per_camera_and_player() -> None:
    scene = _singles_scene()
    scene[1] = _tracks(CAMERAS[1], [(1, [(0, 64, FAR, 1)]), (11, [(60, FRAMES, FAR, 1)]), (2, [(*FULL, NEAR, 0)])])
    result = associate(scene, FPS, _config())
    rows = dict(zip(result.track_ids[1].tolist(), result.player_ids[1], strict=True))
    # Track 1 has more footpoints (64 vs 60), so it keeps the shared frames 60..63 and track 11 yields them.
    assert set(rows[1].tolist()) == {1} and set(rows[11][64:].tolist()) == {1} and set(rows[11][60:64].tolist()) == {-1}
    (handoff,) = result.diagnostics["handoff_frames"]
    assert handoff["player_id"] == 1 and handoff["frames"] == 4


def _doubles_scene(same_clothes: bool, partners_apart: bool, with_appearance: bool = True) -> list[CameraTracks]:
    gap = 3. if partners_apart else .1
    far_left, far_right = _path((-gap, 12.), (-gap, 13.)), _path((gap, 12.), (gap, 13.))
    near_left, near_right = _path((-3., -12.), (-3., -11.)), _path((3., -12.), (3., -11.))
    people = [(near_left, 0), (near_right, 1), (far_left, 2), (far_right, 3)]
    look = [0, 0, 0, 0] if same_clothes else [0, 1, 2, 3]
    scene = []
    for view, camera in enumerate(CAMERAS):
        order = np.roll(np.arange(4), view)  # tracker IDs differ per camera
        scene.append(_tracks(camera, [(10 + int(k), [(*FULL, people[int(k)][0], look[int(k)])]) for k in order], with_appearance))
    return scene


def _doubles_partition(result: object) -> set[frozenset[tuple[str, int]]]:
    groups: dict[int, set[tuple[str, int]]] = {}
    for camera_id, tracks, ids in zip(result.camera_ids, result.track_ids, result.player_ids, strict=True):  # type: ignore[attr-defined]
        for row, track in enumerate(tracks.tolist()):
            (player,) = set(ids[row].tolist())
            groups.setdefault(player, set()).add((camera_id, track))
    return {frozenset(group) for group in groups.values()}


def _truth(scene: list[CameraTracks]) -> set[frozenset[tuple[str, int]]]:
    return {frozenset((camera.camera.camera_id, 10 + person) for camera in scene) for person in range(4)}


def test_doubles_in_identical_clothes_are_resolved_by_geometry() -> None:
    scene = _doubles_scene(same_clothes=True, partners_apart=True)
    result = associate(scene, FPS, _config(players_per_side=2))
    assert _doubles_partition(result) == _truth(scene)


def test_partners_standing_together_are_resolved_by_appearance() -> None:
    scene = _doubles_scene(same_clothes=False, partners_apart=False)
    result = associate(scene, FPS, _config(players_per_side=2))
    assert _doubles_partition(result) == _truth(scene)


def test_partners_standing_together_without_appearance_stop_as_ambiguous() -> None:
    scene = _doubles_scene(same_clothes=True, partners_apart=False)
    with pytest.raises(AssociationUndecided) as stopped:
        associate(scene, FPS, _config(players_per_side=2))
    assert stopped.value.reason == AMBIGUOUS_ASSOCIATION
    pairs = stopped.value.diagnostics["ambiguous_pairs"]
    assert pairs and all(pair["margin"] < 1. for pair in pairs)
    # The stop keeps every score: the tied decisions can be inspected in the artifact.
    assert stopped.value.diagnostics["pairs"] and stopped.value.diagnostics["identities"]


def test_a_missing_player_stops() -> None:
    scene = [_tracks(camera, [(1, [(*FULL, NEAR, 0)])]) for camera in CAMERAS]
    with pytest.raises(AssociationUndecided) as stopped:
        associate(scene, FPS, _config())
    assert stopped.value.reason == PLAYER_NOT_FOUND


def test_two_equally_present_people_on_one_singles_side_stop() -> None:
    other = _path((-4., 10.), (-4., 11.))
    scene = [_tracks(camera, [(1, [(*FULL, NEAR, 0)]), (2, [(*FULL, FAR, 1)]), (3, [(*FULL, other, 2)])]) for camera in CAMERAS]
    with pytest.raises(AssociationUndecided) as stopped:
        associate(scene, FPS, _config())
    assert stopped.value.reason == AMBIGUOUS_PLAYERS
    assert stopped.value.diagnostics["selection"]["1"]["runner_up"] is not None


def test_the_default_config_loads_and_rejects_unknown_or_missing_fields(tmp_path: Path) -> None:
    config = load_association_config(players_per_side=2)
    assert config.appearance is not None and config.players_per_side == 2
    assert load_association_config(players_per_side=1, overrides={"appearance": None}).appearance is None
    values = yaml.safe_load(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError, match="missing"):
        association_config({key: value for key, value in values.items() if key != "min_margin"}, players_per_side=1)
    with pytest.raises(ValueError, match="unknown"):
        association_config({**values, "geometry": {**values["geometry"], "sigma": 1.}}, players_per_side=1)
    with pytest.raises(ValueError, match="unknown fields"):
        load_association_config(players_per_side=1, overrides={"sigma_m": 1.})
    # The game format belongs to the clip, never to the method's file.
    with pytest.raises(ValueError, match="unknown"):
        association_config({**values, "players_per_side": 1}, players_per_side=1)
    with pytest.raises(ValueError, match="singles"):
        load_association_config(players_per_side=3)
