"""Admission must fail on incomplete data, changed cohorts or multiple factors."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion.cohort import reference_validation
from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import (
    assert_combined_candidate,
    assert_single_factor,
    preflight,
)
from src.utils.paths import PROJECT_ROOT


def test_single_factor_rejects_unrelated_optimizer_or_sampler_changes() -> None:
    control = yaml.safe_load((PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/training_dev_anchored_t128.yaml').read_text())
    candidate = deepcopy(control)
    candidate['loss']['physics'] *= 10
    assert_single_factor(candidate, control, 'physics_weight')
    candidate['steps'] *= 2
    with pytest.raises(ValueError, match='beyond'):
        assert_single_factor(candidate, control, 'physics_weight')
    candidate = deepcopy(control)
    candidate['expected_counts'] = {'train': 512, 'val': 64, 'test': 64}
    assert_single_factor(candidate, control, 'training_rallies')
    candidate['learning_rate'] *= 2
    with pytest.raises(ValueError, match='beyond'):
        assert_single_factor(candidate, control, 'training_rallies')


def test_reference_validation_is_exact_and_rejects_changed_missing_or_duplicate_hashes() -> None:
    reference: dict[str, Any] = {'status': 'complete', 'read_rallies': [{'rally_id': 'val-00000', 'npz_sha256': 'abc'}],
                 'config': {'expected_counts': {'val': 1}}}
    records = [{'rally_id': 'val-00000', 'split': 'val', 'npz_sha256': 'abc'},
               {'rally_id': 'val-00001', 'split': 'val', 'npz_sha256': 'def'}]
    assert reference_validation(records, reference) == {'val-00000'}
    for invalid in (records[1:], [dict(records[0], npz_sha256='changed')]):
        with pytest.raises(ValueError, match='mismatch'):
            reference_validation(invalid, reference)
    reference['read_rallies'] = [reference['read_rallies'][0]] * 2
    with pytest.raises(ValueError, match='duplicated'):
        reference_validation(records, reference)


@pytest.mark.parametrize('status', ['running', 'failed'])
def test_incomplete_generation_stops_before_reading_training_inputs(tmp_path: Path, status: str) -> None:
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema': 'ball_refiner_3d.synthetic.v2', 'status': status}))
    plan: dict[str, Any] = {key: str(tmp_path / key) for key in ('control_manifest', 'config', 'generation_plan', 'preflight_output', 'output')}
    plan['dataset'] = str(tmp_path)
    plan['pinned_files'] = {plan[key]: 'unread' for key in ('control_manifest', 'config', 'generation_plan')}
    with pytest.raises(ValueError, match='complete'):
        preflight(plan)
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize(('filename', 'factor'), [('training_pilot512_t128.yaml', 'training_rallies'),
                                                 ('training_physics10_t128.yaml', 'physics_weight')])
def test_committed_trial_configs_are_single_factor(filename: str, factor: str) -> None:
    directory = PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d'
    control = yaml.safe_load((directory / 'training_dev_anchored_t128.yaml').read_text())
    candidate = yaml.safe_load((directory / filename).read_text())
    assert_single_factor(candidate, control, factor)


def test_combined_candidate_is_explicit_and_rejects_any_extra_factor() -> None:
    directory = PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d'
    control = yaml.safe_load((directory / 'training_dev_anchored_t128.yaml').read_text())
    candidate = yaml.safe_load((directory / 'training_pilot512_physics10_t128.yaml').read_text())
    assert_combined_candidate(candidate, control)
    # It must never be admitted/described as either single-factor trial.
    for factor in ('training_rallies', 'physics_weight'):
        with pytest.raises(ValueError, match='beyond'):
            assert_single_factor(candidate, control, factor)
    for key, value in (('steps', 16), ('seed', 937), ('validation_frames', None)):
        changed = deepcopy(candidate)
        changed[key] = value
        with pytest.raises(ValueError, match='beyond'):
            assert_combined_candidate(changed, control)
    for weight in (0.0001, 0.002):
        changed = deepcopy(candidate)
        changed['loss']['physics'] = weight
        with pytest.raises(ValueError, match='10x'):
            assert_combined_candidate(changed, control)
    for key in ('train', 'val', 'test'):
        changed = deepcopy(candidate)
        changed['expected_counts'][key] += 1
        with pytest.raises(ValueError, match='counts'):
            assert_combined_candidate(changed, control)


def test_reprojection_trial_preserves_combined_base_and_rejects_extra_factors() -> None:
    directory = PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d'
    control = yaml.safe_load((directory / 'training_pilot512_physics10_t128.yaml').read_text())
    candidate = yaml.safe_load((directory / 'training_pilot512_physics10_repro3_t128.yaml').read_text())
    assert_single_factor(candidate, control, 'reprojection_weight')
    for field, value in (('physics', 0.0001), ('event', 0.2)):
        changed = deepcopy(candidate)
        changed['loss'][field] = value
        with pytest.raises(ValueError, match='beyond'):
            assert_single_factor(changed, control, 'reprojection_weight')
    for option, setting in (('validation_frames', None), ('steps', 16), ('seed', 937)):
        changed = deepcopy(candidate)
        changed[option] = setting
        with pytest.raises(ValueError, match='beyond'):
            assert_single_factor(changed, control, 'reprojection_weight')
    for weight in (0.01, 0.02, 0.1):
        changed = deepcopy(candidate)
        changed['loss']['reprojection'] = weight
        with pytest.raises(ValueError, match='3x'):
            assert_single_factor(changed, control, 'reprojection_weight')


def test_chained_validation_requires_complete_explicit_subset_partition() -> None:
    records = [{'rally_id': f'val-{i:05d}', 'split': 'val', 'npz_sha256': str(i)} for i in range(3)]
    reference: dict[str, Any] = {
        'status': 'complete', 'config': {'expected_counts': {'val': 3}},
        'read_rallies': [records[0]],
        'validation_reference': {'rallies': ['val-00000'], 'unused_val_rallies': ['val-00001', 'val-00002']}}
    assert reference_validation(records, reference) == {'val-00000'}
    for key, value in (
        ('rallies', ['val-00001']), ('rallies', ['val-00000', 'val-00000']),
        ('unused_val_rallies', ['val-00001']), ('unused_val_rallies', ['val-00000', 'val-00002']),
        ('unused_val_rallies', ['val-00001', 'val-00001']),
        ('unused_val_rallies', ['val-00001', 'val-99999']), ('unused_val_rallies', None),
    ):
        broken = deepcopy(reference)
        broken['validation_reference'][key] = value
        with pytest.raises(ValueError, match='subset'):
            reference_validation(records, broken)
    broken = deepcopy(reference)
    del broken['validation_reference']
    with pytest.raises(ValueError, match='incomplete'):
        reference_validation(records, broken)
    changed = deepcopy(records)
    changed[0]['npz_sha256'] = 'changed'
    with pytest.raises(ValueError, match='mismatch'):
        reference_validation(changed, reference)


def test_head_input_trial_preserves_every_other_factor_and_explicit_opt_in() -> None:
    directory = PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d'
    control = yaml.safe_load((directory / 'training_pilot512_physics10_t128.yaml').read_text())
    candidate = yaml.safe_load((directory / 'training_pilot512_physics10_head_context_t128.yaml').read_text())
    original = deepcopy(control)
    assert_single_factor(candidate, control, 'position_head_input')
    assert control == original  # Comparison cannot mutate the recorded control.
    explicit_control = deepcopy(control)
    explicit_control['model']['position_head_input'] = 'temporal'
    assert_single_factor(candidate, explicit_control, 'position_head_input')
    for path, value in (
        (('model', 'width'), 256), (('model', 'layers'), 3), (('loss', 'reprojection'), .03),
        (('loss', 'physics'), .0001), (('seed',), 937), (('updates',), 10000),
        (('learning_rate',), .001), (('expected_counts', 'train'), 64),
        (('validation_frames',), None), (('steps',), 16),
    ):
        changed = deepcopy(candidate)
        target = changed if len(path) == 1 else changed[path[0]]
        target[path[-1]] = value
        with pytest.raises(ValueError, match='beyond'):
            assert_single_factor(changed, control, 'position_head_input')
    for setting in ('temporal', 'invalid', None):
        changed = deepcopy(candidate)
        changed['model']['position_head_input'] = setting
        with pytest.raises(ValueError, match='Head-input'):
            assert_single_factor(changed, control, 'position_head_input')
    # A head change must also be rejected under a loss-only declaration.
    candidate['loss']['reprojection'] *= 3
    with pytest.raises(ValueError, match='beyond'):
        assert_single_factor(candidate, control, 'reprojection_weight')
