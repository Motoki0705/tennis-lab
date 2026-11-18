"""Execution policy must not turn missing motion evidence into a measurement."""

from copy import deepcopy

import pytest

from src.tennis_scene.pipeline.video_court.policy import apply_drift_policy


def measured_clip():
    return {'status': 'failed', 'reason': 'drift_unobservable', 'frames': [
        {'frame_index': 3, 'status': 'ok', 'reason': None, 'mean_displacement_px': .7},
        {'frame_index': 7, 'status': 'failed', 'reason': 'drift_unobservable',
         'geometry_reasons': ['non_similarity_or_conflicting_line'], 'mean_displacement_px': None,
         'max_displacement_px': None, 'source': {'rgb': 'fixed'}},
        # A large estimate without reliable observation is not a measured failure.
        {'frame_index': 11, 'status': 'failed', 'reason': 'drift_unobservable',
         'geometry_reasons': ['uncertain_local_projection'], 'mean_displacement_px': 18.,
         'max_displacement_px': 35., 'max_visible_projection_std_px': 8.}]}


@pytest.mark.parametrize('policy,expected', [('warn', 'warning'), ('fail', 'failed')])
def test_unobservable_policy_keeps_source_values_reasons_and_original_status(policy, expected):
    original = measured_clip()
    before = deepcopy(original)
    result = apply_drift_policy(original, policy, common_status='accepted')
    assert original == before
    assert result['status'] == expected and result['measurement_status'] == 'failed'
    assert result['unobservable_policy'] == policy
    assert result['frames'][0]['status'] == 'ok'
    assert [f['status'] for f in result['frames'][1:]] == [expected, expected]
    assert result['frames'][1]['reason'] == 'drift_unobservable'
    assert result['frames'][1]['geometry_reasons'] == ['non_similarity_or_conflicting_line']
    assert result['frames'][1]['mean_displacement_px'] is None
    assert result['frames'][2]['mean_displacement_px'] == 18.
    result['frames'][1]['source']['rgb'] = 'changed copy'
    assert original == before


@pytest.mark.parametrize('mean,maximum', [(11., 12.), (9., 26.)])
def test_reliable_tolerance_failure_dominates_unobservable_warnings(mean, maximum):
    report = measured_clip()
    report['frames'].append({'frame_index': 19, 'status': 'failed', 'reason': 'camera_drift_exceeds_tolerance',
                             'mean_displacement_px': mean, 'max_displacement_px': maximum,
                             'max_visible_projection_std_px': 1.})
    result = apply_drift_policy(report, 'warn', common_status='accepted')
    assert result['status'] == 'failed' and result['worst_frame_index'] == 19
    assert result['reason'] == 'camera_drift_exceeds_tolerance'


def test_missing_common_h_and_unrecognized_failure_are_not_reclassified():
    report = measured_clip()
    assert apply_drift_policy(report, 'warn', common_status='failed')['status'] == 'failed'
    report['frames'][1]['reason'] = 'source_missing'
    result = apply_drift_policy(report, 'warn', common_status='accepted')
    assert result['status'] == 'failed' and result['reason'] == 'source_missing'


def test_explicit_line_conflict_and_reapplication_preserve_measurement_identity():
    report = measured_clip()
    report['frames'][1]['reason'] = 'non_similarity_or_conflicting_line'
    warn = apply_drift_policy(report, 'warn', common_status='accepted')
    assert warn['frames'][1]['status'] == 'warning'
    assert apply_drift_policy(warn, 'warn', common_status='accepted') == warn
    assert apply_drift_policy(warn, 'fail', common_status='accepted') == apply_drift_policy(report, 'fail', common_status='accepted')


def test_unknown_policy_does_not_choose_a_default():
    with pytest.raises(ValueError, match='warn or fail'):
        apply_drift_policy(measured_clip(), 'ignore', common_status='accepted')
