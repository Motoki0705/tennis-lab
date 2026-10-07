from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.dataset_statistics.aggregation import aggregate
from src.tasks.ball_detection.dataset_statistics.configuration import StatisticsConfig
from src.tasks.ball_detection.dataset_statistics.contracts import (
    Measurements,
    PoseInput,
    Samples,
)
from src.tasks.ball_detection.dataset_statistics.inputs import read_clip
from src.tasks.ball_detection.dataset_statistics.metrics import (
    gaps,
    interpolation,
    motion,
    pose,
    windows,
)
from src.tasks.ball_detection.dataset_statistics.pipeline import (
    compute_clip_statistics,
    compute_dataset_statistics,
)
from src.tasks.ball_detection.dataset_statistics.scopes import selections
from src.tasks.ball_detection.dataset_statistics.summaries import summarize
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def fixture_clip(tmp_path, labels=None):
    if labels is None:
        labels = [frame(i, ball(xy=(float(i + 10), 20.))) for i in range(80)]
    root = write_store_clip(tmp_path / 'ball', 'clip', labels, size=(200, 160))
    store = BallFrameStore(root)
    return store, read_clip(store, store.clips[0], {}, tmp_path, None)


def test_exact_quantiles_and_empty_semantics():
    empty = summarize(Samples(np.array([np.nan]), 'px', 3))
    assert empty['mean'] is None and empty['p95'] is None
    assert empty['n'] == 0 and empty['missing'] == 3
    result = summarize(Samples(np.arange(101.), 'px', 101))
    assert (result['mean'], result['median'], result['p5'], result['p95']) == (50, 50, 5, 95)


def test_pooled_samples_and_equal_clip_rates_are_not_interchanged():
    a, b = Measurements(), Measurements()
    a.sample('speed', [0], 'px/s')
    b.sample('speed', [10] * 9, 'px/s')
    a.rates['observed'] = (0, 1)
    b.rates['observed'] = (9, 9)
    result = aggregate([a, b])
    assert result['pooled']['distributions']['speed']['mean'] == 9
    assert result['between_clips']['mean/speed']['mean'] == 5
    assert result['pooled']['rates']['observed']['value'] == .9
    assert result['between_clips']['rates/observed']['mean'] == .5


def test_annotation_states_preserve_visibility_reference_and_multiple_balls(tmp_path):
    labels = [replace(frame(0, ball()), is_target=False), frame(1, ball('unresolved', None)),
              frame(2, ball(), ball(track='b002')), frame(3), frame(4, annotated=False)]
    store, data = fixture_clip(tmp_path, labels)
    s = annotation_states(store, store.clips[0])
    assert s.located.tolist() == [1, 0, 2, 0, 0]
    assert s.evidence.tolist() == [False, True, False, False, False]
    assert not s.supervision.any()
    assert np.isnan(s.xy[2]).all()
    report, _ = compute_clip_statistics(data, StatisticsConfig.shipped())
    counts = report['scopes']['clip']['counts']
    assert counts['frames'] == 5 and counts['instances'] == 4
    assert report['scopes']['clip']['rates']['instances/observed']['value'] == .75


def test_gap_scopes_and_annotation_boundaries_are_separate(tmp_path):
    labels = [frame(i) if 20 <= i < 60 else frame(i, ball()) for i in range(80)]
    labels[40] = replace(labels[40], segment_break=True)
    _, data = fixture_clip(tmp_path, labels)
    all_frames: NDArray[np.bool_] = np.ones(80, bool)
    metrics = gaps.compute(data, all_frames)
    assert metrics.samples['gaps/coordinate_gap/raw/frames'].values.tolist() == [40]
    assert metrics.samples['gaps/coordinate_gap/bounded/frames'].values.tolist() == [20, 20]
    scope = all_frames.copy()
    scope[30:50] = False
    assert gaps.compute(data, scope).samples['gaps/coordinate_gap/raw/frames'].values.tolist() == [10, 10]


def test_motion_uses_pts_and_excludes_breaks_missing_and_separate_scope_runs(tmp_path):
    labels = [frame(i, ball(xy=(float(i + 10 + (40 if i >= 40 else 0)), 20.))) for i in range(80)]
    labels[40] = replace(labels[40], segment_break=True)
    _, data = fixture_clip(tmp_path, labels)
    scope: NDArray[np.bool_] = np.ones(80, bool)
    scope[20:30] = False
    m = motion.compute(data, scope, StatisticsConfig.shipped())
    np.testing.assert_allclose(m.samples['motion/observed/speed_px'].values, 30)
    assert len(m.samples['motion/observed/speed_px'].values) == 67
    assert sum(count for _, _, count in m.views['motion/observed/transitions']) == 67


def test_interpolation_runs_and_endpoint_support_are_not_fabricated(tmp_path):
    labels = [frame(i, ball('interpolated' if 20 < i < 25 else 'observed')) for i in range(80)]
    _, data = fixture_clip(tmp_path, labels)
    m = interpolation.compute(data, np.ones(80, bool))
    assert m.samples['interpolation/run_frames'].values.tolist() == [4]
    assert m.counts['interpolation/endpoint_pairs'] == 0
    assert m.views['interpolation/availability'] == 'unsupported_source'


def test_multiple_strides_cover_unique_frames_and_count_repeated_exposures(tmp_path):
    labels = [frame(i, ball()) if 10 <= i < 70 else frame(i) for i in range(80)]
    _, data = fixture_clip(tmp_path, labels)
    cfg = StatisticsConfig.shipped()
    result = windows.compute(data, selections(data, cfg), cfg, None)
    assert result.counts['windows/stride_1/count'] == 29
    assert result.counts['windows/stride_16/count'] == 3
    assert result.counts['windows/stride_1/unique_frames'] == 60
    assert result.counts['windows/stride_1/frame_occurrences'] == 29 * 32
    assert result.views['windows/stride_16/position_supervision'] == [3] * 32
    assert summarize(result.samples['windows/stride_16/pose_flagged'])['mean'] is None


def with_pose(data, jump=False):
    n = data.clip.frame_count
    points = np.zeros((n, 1, 17, 2), float)
    points[..., 0] = np.arange(n)[:, None, None] + np.arange(17)[None, None, :] + 20
    points[..., 1] = np.arange(17)[None, None, :] + 30
    if jump:
        points[40, 0, 9, 0] += 80
    boxes = np.tile([0., 0., 150., 100.], (n, 1, 1))
    pose_data = PoseInput(points, np.full((n, 1, 17), 1.2), np.ones((n, 1), bool), boxes,
                          ('player_1',), np.zeros((n, 1), np.int64))
    return replace(data, pose=pose_data, pose_reason=None)


def test_pose_translation_is_distinct_from_a_single_joint_teleport(tmp_path):
    _, data = fixture_clip(tmp_path)
    cfg = StatisticsConfig.shipped()
    normal = pose.signals(with_pose(data), cfg)
    assert normal is not None
    np.testing.assert_allclose(normal.relative_speed[1:], 0)
    assert not normal.flags.any()
    jumped = pose.signals(with_pose(data, jump=True), cfg)
    assert jumped is not None
    assert jumped.flags[40, 0, 9] and jumped.flags[41, 0, 9]
    assert not jumped.flags[40, 0, 10]
    m = pose.compute(with_pose(data, jump=True), np.ones(80, bool), cfg, jumped)
    assert m.samples['pose/left_wrist/score'].values.min() == 1.2
    assert any(f['frame'] == 40 and f['joint'] == 'left_wrist' for f in m.findings)
    scope: NDArray[np.bool_] = np.zeros(80, bool)
    scope[40:41] = True
    assert pose.compute(with_pose(data, jump=True), scope, cfg, jumped).counts['pose/flagged_joint_frames'] == 0


def test_dataset_report_identity_json_and_source_split_groups(tmp_path):
    store, _ = fixture_clip(tmp_path)
    report = compute_dataset_statistics(store, ['clip'], StatisticsConfig.shipped(), project_root=tmp_path)
    assert set(report['groups']) == {'all', 'source/tracknet', 'split/train', 'source_split/tracknet/train'}
    assert report['groups']['all']['clip']['pooled']['counts']['frames'] == 80
    assert report['groups']['all']['selected']['pooled']['counts']['frames'] == 80
    assert report['raw_annotation_availability'] == {'unsupported_source': 1}
    assert len(report['identity']['hashes']['index.npz']) == 64


@pytest.mark.parametrize('field,value', [('strides',[0]), ('strides',[1,1]), ('scope_stride',2), ('grid_x',0), ('pose_jump_primary',float('nan'))])
def test_invalid_settings_fail_explicitly(field,value):
    raw=StatisticsConfig.shipped().to_dict()
    raw[field]=value
    with pytest.raises(ValueError):
        StatisticsConfig.from_mapping(raw)


def test_pose_recovery_does_not_bridge_separate_scope_regions(tmp_path):
    _, data = fixture_clip(tmp_path)
    data = with_pose(data)
    p = data.pose
    observed = p.observed.copy()
    observed[38:41] = False
    data = replace(data, pose=replace(p, observed=observed))
    config = StatisticsConfig.shipped()
    signals = pose.signals(data, config)
    assert signals is not None
    assert len(signals.recovery) == 1
    scope: NDArray[np.bool_] = np.ones(80, bool)
    assert pose.compute(data, scope, config, signals).samples['pose/recovery_step'].values.size == 1
    scope[38:40] = False
    assert pose.compute(data, scope, config, signals).samples['pose/recovery_step'].values.size == 0


def test_occlusion_location_is_matched_per_ball_not_across_instances(tmp_path):
    from src.tasks.ball_detection.dataset_statistics.metrics import (
        annotations as metrics,
    )
    from src.tasks.ball_detection.generate_dataset.frame_store.clip import BallInstance
    _, data = fixture_clip(tmp_path, [frame(0, BallInstance('hidden', 'unresolved', None, True), ball())])
    result = metrics.compute(data, np.ones(1, bool))
    assert result.rates['annotation/occluded_located'] == (0, 1)
    assert result.rates['annotation/occluded_unlocated'] == (1, 1)


def test_pose_recovery_uses_only_joints_valid_at_both_endpoints(tmp_path):
    _, data = fixture_clip(tmp_path)
    data = with_pose(data)
    p = data.pose
    assert p is not None
    points, scores, observed = p.points.copy(), p.scores.copy(), p.observed.copy()
    observed[38:41] = False
    scores[37] = scores[41] = .1
    points[41, 0, :, 0] += 1000
    data = replace(data, pose=replace(p, points=points, scores=scores, observed=observed))
    config = replace(StatisticsConfig.shipped(), pose_min_score=.5)
    result = pose.signals(data, config)
    assert result is not None and result.recovery == ()

    # Low-score outliers at either endpoint cannot dominate the recovery median.
    scores[37, 0, 9] = scores[41, 0, 9] = 1.0
    scores[41, 0, 10] = 1.0  # Only the later endpoint is valid for this joint.
    points[41, 0, 9, 0] -= 1000
    result = pose.signals(data, config)
    assert result is not None
    assert len(result.recovery) == 1
    assert result.recovery[0][:3] == (37, 41, 0)
    assert result.recovery[0][3] == pytest.approx(.04)
