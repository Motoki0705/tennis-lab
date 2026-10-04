"""A bank comparison must reject unrelated configuration, seed, or mask changes."""
from copy import deepcopy
from typing import Any

import pytest

from src.tasks.ball_refiner.refiner_3d.condition_audit import (
    IDENTITY_METADATA,
    compare_condition_reports,
)


def reports() -> tuple[dict[str, Any], dict[str, Any]]:
    before: dict[str, Any] = {
        'status': 'complete', 'audit_only': False, 'samples_each_threshold_and_volume': 512,
        'audit': {'plan': {'seed': 936, 'degradation': {'status': 'old', 'components': 125,
            'calibration': {'bank': 'a', 'bank_sha256': 'a', 'report': 'a.json', 'report_sha256': 'a', 'block_frames': 16}}},
            'rallies': [{**dict.fromkeys(IDENTITY_METADATA, 1), 'rally_id': 'train-00000'}]},
        'array_audits': [{'rally_id': 'train-00000', 'identity_array_hashes': {'truth': 'x', 'occlusion': 'y'}}],
        'condition_metrics': {'train_val': {'nll': 1}}, 'baselines': {},
    }
    after = deepcopy(before)
    after['audit']['plan']['degradation']['status'] = 'anchored'
    after['audit']['plan']['degradation']['calibration'].update(bank='b', bank_sha256='b', report='b.json', report_sha256='b')
    return before, after


def test_only_bank_change_is_allowed_without_mutating_reports() -> None:
    before, after = reports()
    saved = deepcopy((before, after))
    result = compare_condition_reports(before, after)
    assert result['test_arrays_read'] == 0
    assert (before, after) == saved


@pytest.mark.parametrize('changed', ['seed', 'block', 'truth', 'mask', 'samples', 'unfinished', 'unscored', 'rally'])
def test_unpaired_comparison_is_rejected(changed: str) -> None:
    before, after = reports()
    if changed == 'seed':
        after['audit']['rallies'][0]['seed'] += 1
    elif changed == 'block':
        after['audit']['plan']['degradation']['calibration']['block_frames'] += 1
    elif changed in ('truth', 'mask'):
        after['array_audits'][0]['identity_array_hashes']['truth' if changed == 'truth' else 'occlusion'] = 'changed'
    elif changed == 'samples':
        after['samples_each_threshold_and_volume'] += 1
    elif changed == 'unfinished':
        after['status'] = 'running'
    elif changed == 'unscored':
        after['audit_only'] = True
    else:
        after['audit']['rallies'].clear()
    with pytest.raises(ValueError):
        compare_condition_reports(before, after)
