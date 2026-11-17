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
from src.tasks.court_detection.geometry.video_shared.consensus import merge_measurements
from src.tasks.court_detection.geometry.video_shared.drift import inspect_drift
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


def test_failed_solve_and_a_displaced_epoch_remain_explicit():
    initial = H.copy()
    initial[0, 2] += 9
    failed = fit_common_court(initial, observations(), (640, 480), replace(CONFIG, max_nfev=1))
    assert failed.matrix is None and failed.reasons == ('shared_optimization_failed',)
    changed = fit_common_court(H, observations(shifts=(0., 0., 30.)), (640, 480), CONFIG)
    assert changed.matrix is not None
    assert changed.diagnostic['requires_clip_drift_inspection']
    assert changed.diagnostic['large_common_residual_groups']
    assert len(changed.diagnostic['group_residuals']) == 27
    moved_frame = tuple(g for g in observations(shifts=(0., 0., 30.)) if g.frame_index == 2)
    assert inspect_drift(changed.matrix, moved_frame, (640, 480), CONFIG)['status'] == 'failed'


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
    image[100:430, 410:610] //= 2  # fixed shadow crosses visible ground markings
    cv2.rectangle(image, (190 + frame * 70, 200), (210 + frame * 70, 290), (25, 35, 40), -1)
    cv2.circle(image, (200 + frame * 70, 194), 9, (25, 35, 40), -1)
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
    measurements = [measure_frame(raster(frame, lines=(0, 1, 2, 3, 4, 5)), H,
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
    assert result.matrix is None
    assert result.diagnostic['unverified_transverse_groups']
    assert 'unverified_finite_line_identity' in result.reasons


@pytest.mark.parametrize('shift,expected', [(0., 'ok'), (5., 'warning'), (11., 'failed')])
def test_drift_measures_known_pixel_motion_without_replacing_the_common_h(shift, expected):
    # Use full-HD coordinates so the agreed 3/10/25px units apply directly.
    enlargement = np.diag([3., 2.25, 1.])
    shared = enlargement @ H
    original = shared.copy()
    groups = tuple(replace(g, points=g.points * [3, 2.25] + [shift, 0]) for g in observations(shifts=(0.,), counts=(24,)))
    result = inspect_drift(shared, groups, (1920, 1080), CONFIG)
    assert result['status'] == expected, result
    assert abs(result['mean_displacement_px'] - shift) < .01
    residuals = result['common_line_residuals']
    assert len(residuals) == 9 and all(row['used_by_local_fit'] for row in residuals)
    assert .95 * shift - .01 <= max(row['p95_px'] for row in residuals) <= shift + .01
    np.testing.assert_array_equal(shared, original)


def test_missing_drift_observations_are_failed_not_zero_motion():
    result = inspect_drift(H, (), (640, 480), CONFIG)
    assert result['status'] == 'failed' and result['reason'] == 'drift_unobservable'
    assert result['mean_displacement_px'] is None and result['max_displacement_px'] is None
    assert result['common_line_residuals'] == []


def test_unverified_stripe_cannot_reduce_drift_uncertainty():
    shared = np.diag([3., 2.25, 1.]) @ H
    groups = tuple(replace(g, points=g.points * [3, 2.25] + [5., 0.])
                   for g in observations(shifts=(0.,), counts=(24,)))
    without = inspect_drift(shared, groups[1:], (1920, 1080), CONFIG)
    bad = replace(groups[0], endings=('unobservable', 'unobservable'), points=groups[0].points + [0, 30])
    with_unverified = inspect_drift(shared, (bad, *groups[1:]), (1920, 1080), CONFIG)
    assert with_unverified['status'] == without['status'] == 'warning'
    assert with_unverified['max_visible_projection_std_px'] == without['max_visible_projection_std_px']
    assert with_unverified['mean_displacement_std_px'] == without['mean_displacement_std_px']
    assert not with_unverified['common_line_residuals'][0]['used_by_local_fit']


def test_local_projective_motion_can_fail_maximum_while_mean_stays_below_ten():
    shared = np.diag([3., 2.25, 1.]) @ H
    corners = project(shared)[:4]
    corners[1] += [30., 0.]
    moved = cv2.getPerspectiveTransform(WORLD[:4].astype(np.float32), corners.astype(np.float32)).astype(float)
    groups = []
    for index, (a, b) in enumerate(EDGES):
        xy = WORLD[a] + np.linspace(.05, .95, 24)[:, None] * (WORLD[b] - WORLD[a])
        groups.append(PaintGroup('video/clip', 0, index, project(moved, xy), ('ended', 'ended')))
    expected = np.linalg.norm(project(moved) - project(shared), axis=1)
    assert expected.mean() < 10 and expected.max() > 25
    result = inspect_drift(shared, tuple(groups), (1920, 1080), CONFIG)
    assert result['reason'] == 'camera_drift_exceeds_tolerance'
    np.testing.assert_allclose([result['mean_displacement_px'], result['max_displacement_px']],
                               [expected.mean(), expected.max()], atol=.01)


def test_one_wrong_line_is_not_reported_as_coherent_camera_motion():
    shared = np.diag([3., 2.25, 1.]) @ H
    groups = tuple(replace(g, points=g.points * [3, 2.25] + ([0, 30] if g.line_index == 0 else [0, 0]))
                   for g in observations(shifts=(0.,), counts=(24,)))
    result = inspect_drift(shared, groups, (1920, 1080), CONFIG)
    assert result['reason'] == 'drift_unobservable'


def test_repeated_initializers_share_measurements_without_manufacturing_support():
    groups = observations(shifts=(0., 0., 0.))
    sparse = observations(shifts=(0., 0., 0.), counts=(12, 12, 12))
    result = merge_measurements(H, (groups, sparse, groups), (640, 480), CONFIG)
    assert result.matrix is not None, result.reasons
    assert np.linalg.norm(project(result.matrix) - project(H), axis=1).max() < .001
    assert result.diagnostic['merged_groups'] == 27
    one_frame = tuple(g for g in groups if g.frame_index == 0)
    rejected = merge_measurements(H, (one_frame,) * 4, (640, 480), CONFIG)
    assert rejected.matrix is None


def test_different_static_stripes_are_not_averaged_into_a_shared_court():
    groups = observations(shifts=(0., 0., 0.))
    displaced = tuple(replace(g, points=g.points + [0, 12]) if g.line_index == 0 else g for g in groups)
    result = merge_measurements(H, (groups, displaced), (640, 480), CONFIG)
    assert result.matrix is None and result.reasons == ('ambiguous_observed_stripes',)


def test_baseline_and_one_service_need_a_finite_centre_segment_for_order():
    ambiguous = observations(lines=(1, 2, 3, 4, 5, 7))
    result = fit_common_court(H, ambiguous, (640, 480), CONFIG)
    assert result.matrix is None and result.reasons == ('unverified_service_order',)
    supported = fit_common_court(H, observations(lines=(1, 2, 3, 4, 5, 7, 8)), (640, 480), CONFIG)
    assert supported.matrix is not None


def test_same_stripe_with_edge_outliers_is_not_a_semantic_alternative():
    clean = observations(shifts=(0., 0., 0.))
    noisy = []
    for group in clean:
        points = group.points.copy()
        if group.line_index == 0:
            points[[7, 17]] += [0, 5]
        noisy.append(replace(group, points=points))
    result = merge_measurements(H, (clean, tuple(noisy)), (640, 480), CONFIG)
    assert result.matrix is not None, result.reasons
    assert np.linalg.norm(project(result.matrix) - project(H), axis=1).max() < 1.


def test_estimator_checks_frozen_rgb_before_any_new_measurement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.tennis_scene.pipeline.video_court import estimator

    path, _, document = seed_manifest(tmp_path)
    metadata = {'source_frames': {'video/clip/0': {'media_sha256': 'f' * 64, 'rgb_sha256': '0' * 64}}}
    document['provenance'] = metadata
    path.write_text(json.dumps(document))
    monkeypatch.setattr(estimator, 'read_video_frame', lambda *args: SimpleNamespace(original_size=(640, 480), frame=np.zeros((480, 640, 3), np.uint8)))

    def forbidden(*args, **kwargs):
        raise AssertionError('RGB provenance was not checked')

    monkeypatch.setattr(estimator, 'measure_frame', forbidden)
    with pytest.raises(ValueError, match='source RGB'):
        estimator.estimate_camera(camera='left', size_wh=(640, 480), sources={'video/clip': {'sha256': 'f' * 64}},
                                  paths={'video/clip': tmp_path / 'video.mp4'}, seeds_path=path,
                                  seeds_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed_metadata=metadata,
                                  dataset_id='set', video_id='video', request={'seed_clip_ids': None, 'fit_clip_ids': None})


def test_bootstrap_is_an_initializer_and_semantic_adoption_uses_all_fit_frames(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.tasks.court_detection.geometry.video_shared.observations import FramePaint
    from src.tennis_scene.pipeline.video_court import estimator

    path, _, document = seed_manifest(tmp_path)
    rgb: np.ndarray = np.zeros((480, 640, 3), np.uint8)
    metadata = {'source_frames': {'video/clip/0': {'media_sha256': 'f' * 64, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}}}
    document['provenance'] = metadata
    document['candidates'][0]['region_xyxy'] = [0, 0, 640, 480]
    path.write_text(json.dumps(document))
    monkeypatch.setattr(estimator, 'read_video_frame', lambda *args: SimpleNamespace(original_size=(640, 480), frame=rgb))

    def measured(image, initial, *, clip_id, frame_index, config):
        # Bootstrap has baseline/service but no centre segment. Later frames
        # reveal the centre line independently in each clip and fix its order.
        lines = (1, 2, 3, 4, 5, 7) if frame_index == 1 else tuple(range(9)) if frame_index == 4 else (1, 2, 3, 4, 5, 7, 8)
        groups = tuple(replace(g, clip_id=clip_id, frame_index=frame_index)
                       for g in observations(lines=lines, shifts=(0.,), counts=(24,)))
        return FramePaint(groups, {'reason': 'measured', 'chosen_homography': H.tolist()})

    monkeypatch.setattr(estimator, 'measure_frame', measured)
    clips = ('video/clip', 'video/clip_b', 'video/clip_c')
    result = estimator.estimate_camera(camera='left', size_wh=(640, 480),
        sources={clip: {'sha256': 'f' * 64, 'num_frames': 5} for clip in clips},
        paths={clip: tmp_path / 'video.mp4' for clip in clips}, seeds_path=path,
        seeds_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed_metadata=metadata,
        dataset_id='set', video_id='video', request={'seed_clip_ids': ['video/clip'], 'fit_clip_ids': None,
        'bootstrap_clip_ids': None, 'fit_numerators': [1, 2, 3], 'diagnostic_numerators': [4], 'denominator': 4, 'workers': 1})
    assert result['status'] == 'accepted', result['diagnostics']['reasons']
    np.testing.assert_allclose(project(np.array(result['matrix'])), project(H), atol=.01)
    assert all(row['used_as_initializer'] and row['reasons'] for row in result['diagnostics']['bootstrap'])
    assert all(row['status'] == 'ok' for row in result['clips'].values())
