"""Known-H multi-frame contracts; no reference or learned output is the oracle."""

import hashlib
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.tasks.court_detection.geometry.paint_observations import PaintConfig
from src.tasks.court_detection.geometry.video_shared.candidates import (
    CourtSeed,
    read_frozen_candidates,
)
from src.tasks.court_detection.geometry.video_shared.endpoints import measure_endings
from src.tasks.court_detection.geometry.video_shared.fit import (
    PaintGroup,
    SharedFit,
    SharedPaintConfig,
    choose_common_solution,
    fit_common_court,
)
from src.tasks.court_detection.geometry.video_shared.observations import (
    clustered_initializers,
    measure_frame,
)

# Literal court coordinates and independent homogeneous projection keep the
# oracle separate from the extraction, correspondence and fitting code.
WORLD = np.array([[-5.485, 11.885], [5.485, 11.885], [-5.485, -11.885], [5.485, -11.885],
                  [-4.115, 11.885], [-4.115, -11.885], [4.115, 11.885], [4.115, -11.885],
                  [-4.115, 6.4], [4.115, 6.4], [-4.115, -6.4], [4.115, -6.4], [0, 6.4], [0, -6.4]])
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
H = np.array([[25., 2., 320.], [1., -12., 240.], [.001, -.001, 1.]])
CONFIG = SharedPaintConfig()


def project(matrix, xy=WORLD):
    transformed = np.c_[xy, np.ones(len(xy))] @ matrix.T
    return transformed[:, :2] / transformed[:, 2, None]


def observations(*, lines=range(9), shifts=(-2., 0., 2.), counts=(24, 24, 24)):
    groups = []
    for frame, (shift, count) in enumerate(zip(shifts, counts, strict=True)):
        matrix = H.copy()
        matrix[0] += shift * matrix[2]
        for line in lines:
            a, b = WORLD[np.array(EDGES[line])]
            xy = a + np.linspace(.05, .95, count)[:, None] * (b - a)
            groups.append(PaintGroup(f'video/clip_{frame}', frame, line, project(matrix, xy), ('ended', 'ended')))
    return tuple(groups)


def test_common_fit_balances_frames_and_preserves_all_residuals():
    initial = H.copy()
    initial[0, 2] += 4
    groups = observations(counts=(240, 24, 24))
    result = fit_common_court(initial, groups, (640, 480), CONFIG)
    assert result.matrix is not None, result.reasons
    assert np.max(np.linalg.norm(project(result.matrix) - project(H), axis=1)) < .3
    residuals = result.diagnostic['group_residuals']
    assert len(residuals) == 27
    assert {row['clip_id'] for row in residuals} == {'video/clip_0', 'video/clip_1', 'video/clip_2'}
    assert max(row['p95_px'] for row in residuals) > 1.5


@pytest.mark.parametrize('groups', [(), observations(lines=(0, 1, 2, 3, 4)),
                                     observations(shifts=(0.,), counts=(24,))])
def test_missing_semantic_or_temporal_support_has_no_initial_h_fallback(groups):
    result = fit_common_court(H, groups, (640, 480), CONFIG)
    assert result.matrix is None and result.fitted_matrix is None
    assert 'insufficient_independent_line_support' in result.reasons


def test_duplicate_measurements_cannot_create_temporal_support():
    group = observations()[0]
    with pytest.raises(ValueError, match='Duplicate'):
        fit_common_court(H, (group, group), (640, 480), CONFIG)


def test_failed_solve_and_inconsistent_epoch_are_explicit():
    initial = H.copy()
    initial[0, 2] += 9
    failed = fit_common_court(initial, observations(), (640, 480), replace(CONFIG, max_nfev=1))
    assert failed.matrix is None and failed.reasons == ('shared_optimization_failed',)
    changed = fit_common_court(H, observations(shifts=(0., 0., 30.)), (640, 480), CONFIG)
    assert changed.matrix is None and 'inconsistent_common_court' in changed.reasons
    assert len(changed.diagnostic['group_residuals']) == 27


def test_distinct_valid_courts_are_not_resolved_by_score_or_input_order():
    other = H.copy()
    other[0, 2] += 20
    a = SharedFit(H, H, (), {'max_group_p95_px': .1})
    b = SharedFit(other, other, (), {'max_group_p95_px': 2.})
    for order in ((a, b), (b, a)):
        chosen = choose_common_solution(order, CONFIG)
        assert chosen.matrix is None and chosen.reasons == ('ambiguous_line_assignment',)


def test_initializer_clustering_is_stable_and_does_not_count_majority_votes():
    other = H.copy()
    other[0, 2] += 20
    a = CourtSeed('a', 'video/a', 0, H, False)
    b = CourtSeed('b', 'video/b', 0, other, True)
    c = CourtSeed('c', 'video/c', 0, H, True)
    for order in ((a, b, c), (c, b, a)):
        seeds, diagnostic = clustered_initializers(order, CONFIG)
        assert [seed.candidate_id for seed in seeds] == ['a', 'b']
        assert diagnostic['assignments'] == {'a': 'a', 'b': 'b', 'c': 'a'}


def seed_manifest(tmp_path):
    path = tmp_path / 'candidates.json'
    data = {'schema': 'court_seed_manifest_v1', 'dataset_id': 'set', 'video_id': 'video',
            'camera_id': 'left', 'size_wh': [640, 480], 'provenance': {'model': 'frozen'},
            'candidates': [{'candidate_id': 'a', 'clip_id': 'video/clip', 'frame_index': 0,
                            'matrix': H.tolist(), 'model_accepted': False, 'region_xyxy': [13, 17, 600, 440],
                            'source_sha256': 'a' * 64}]}
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), data


def test_frozen_rejected_seed_has_one_crop_transform_and_checked_digest(tmp_path):
    path, sha, _ = seed_manifest(tmp_path)
    options = dict(sha256=sha, dataset_id='set', video_id='video', camera_id='left',
                   size_wh=(640, 480), clip_ids=('video/clip',))
    seeds = read_frozen_candidates(path, **options)
    assert not seeds[0].model_accepted
    np.testing.assert_allclose(project(seeds[0].matrix), project(H) + [13, 17])
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='digest changed'):
        read_frozen_candidates(path, **options)


@pytest.mark.parametrize('changes', [{'camera_id': 'right'}, {'video_id': 'other'}, {'size_wh': (480, 640)},
                                     {'clip_ids': ('video/absent',)}, {'clip_ids': ('video/clip', 'video/clip')}])
def test_frozen_scope_mismatch_never_selects_another_camera(tmp_path, changes):
    path, sha, _ = seed_manifest(tmp_path)
    options = dict(sha256=sha, dataset_id='set', video_id='video', camera_id='left',
                   size_wh=(640, 480), clip_ids=('video/clip',))
    with pytest.raises(ValueError):
        read_frozen_candidates(path, **(options | changes))


def raster(frame, lines=range(9)):
    image: np.ndarray = np.full((480, 640, 3), [40, 90, 55], np.uint8)
    xy = project(H)
    for i in lines:
        a, b = xy[np.array(EDGES[i])]
        cv2.line(image, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (235, 235, 235), 3, cv2.LINE_AA)
    cv2.line(image, (140, 215), (505, 225), (255, 255, 255), 3, cv2.LINE_AA)  # static net, never averaged away
    cv2.rectangle(image, (190 + frame * 70, 200), (210 + frame * 70, 290), (25, 35, 40), -1)
    return image


def test_multiple_rgb_frames_recover_known_h_with_static_net_and_moving_occlusion():
    config = replace(CONFIG, paint=replace(PaintConfig(), ransac_trials=100, sample_spacing_px=12.))
    initial = H.copy()
    initial[0, 2] += 3
    measurements = [measure_frame(raster(frame), initial, clip_id=f'video/{frame}', frame_index=frame, config=config) for frame in range(3)]
    groups = tuple(group for frame in measurements for group in frame.groups)
    result = fit_common_court(initial, groups, (640, 480), config)
    assert result.matrix is not None, [frame.diagnostic for frame in measurements]
    assert np.max(np.linalg.norm(project(result.matrix) - project(H), axis=1)) < 1.5


def test_rgb_two_transverse_lines_cannot_certify_static_net_identity():
    config = replace(CONFIG, paint=replace(PaintConfig(), ransac_trials=100, sample_spacing_px=12.))
    measurements = [measure_frame(raster(frame, lines=(0, 1, 2, 3, 4)), H,
                                  clip_id=f'video/{frame}', frame_index=frame, config=config) for frame in range(3)]
    result = fit_common_court(H, tuple(g for m in measurements for g in m.groups), (640, 480), config)
    assert result.matrix is None


def test_visible_ground_end_is_not_the_end_of_the_search_window():
    image = raster(0)
    status, _ = measure_endings(image, H, 0, PaintConfig())
    assert status == ('ended', 'ended')
    start, end = project(H)[[0, 1]]
    delta = end - start
    cv2.line(image, tuple(np.rint(start - .2 * delta).astype(int)),
             tuple(np.rint(end + .2 * delta).astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
    status, _ = measure_endings(image, H, 0, PaintConfig())
    assert status == ('continues', 'continues')
    missing, _ = measure_endings(np.full_like(image, 80), H, 0, PaintConfig())
    assert missing == ('unobservable', 'unobservable')


def test_algebraic_fit_needs_independent_endpoint_evidence():
    groups = tuple(replace(g, endings=('unobservable', 'unobservable')) for g in observations())
    result = fit_common_court(H, groups, (640, 480), CONFIG)
    assert result.fitted_matrix is not None and result.matrix is None
    assert 'unverified_finite_line_identity' in result.reasons
