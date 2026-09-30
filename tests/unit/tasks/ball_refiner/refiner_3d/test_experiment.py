"""Admission must fail on incomplete data, changed cohorts or multiple factors."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion.cohort import reference_validation
from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import (
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
