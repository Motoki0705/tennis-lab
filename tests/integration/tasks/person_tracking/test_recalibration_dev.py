"""Protect the pre-dev publication boundary and abstention denominators."""
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def dev(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('association_recalibration_dev')


def test_dev_requires_exact_pushed_config_and_evidence(dev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    git('init', '-b', 'fixture')
    config, base, bundle = tmp_path / 'named.yaml', tmp_path / 'base.yaml', tmp_path / 'fit'
    bundle.mkdir()
    config.write_text('named: true\n')
    base.write_text('base: true\n')
    (bundle / 'fit.json').write_text(json.dumps({'status': 'accepted'}))
    (bundle / 'identity.json').write_text(json.dumps({'base_config': {'path': str(base), 'sha256': dual_sha256(base)}}))
    (bundle / 'pre-dev.json').write_text(json.dumps({'dev_scoring_batches': 0, 'dev_labels_opened': False,
                                                   'proposed_config': {'sha256': dual_sha256(config)}}))
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'pin')
    commit = git('rev-parse', 'HEAD')
    monkeypatch.setattr(dev, 'CODE', tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        dev.require_pushed(commit, config, bundle)
    git('update-ref', 'refs/remotes/origin/campaign930/i964-2-tracking', commit)
    dev.require_pushed(commit, config, bundle)
    config.write_text('named: changed\n')
    with pytest.raises(ValueError, match='Uncommitted'):
        dev.require_pushed(commit, config, bundle)
    config.write_text('named: true\n')
    (bundle / 'fit.json').write_text(json.dumps({'status': 'rejected'}))
    with pytest.raises(ValueError, match='Uncommitted'):
        dev.require_pushed(commit, config, bundle)


def test_projected_ids_keep_raw_axes_and_never_create_observations(dev: Any) -> None:
    raw = SimpleNamespace(observed=np.array([[True, False], [False, True]]), track_ids=np.array([11, 12]))
    mapping = np.array([[0, 1]], np.int64)
    np.testing.assert_array_equal(dev.project_ids(raw, mapping, np.array([[3, 3]])), [[3, -1], [-1, 3]])
    with pytest.raises(ValueError, match='synthetic'):
        dev.project_ids(raw, np.array([[1, 0]]), np.array([[3, 3]]))
    with pytest.raises(ValueError, match='Multiple'):
        dev.project_ids(raw, np.array([[0, 1], [0, 1]]), np.array([[3, 3], [4, 4]]))


def test_pool_retains_undecided_positive_pairs(dev: Any) -> None:
    def result(tp: int, fn: int, status: str) -> dict[str, Any]:
        return {'clip': status, 'status': status, 'reason': 'margin', 'metrics': {
            'pairs': {'tp': tp, 'fp': 0, 'fn': fn}, 'exclusion': {'tp': 0, 'fp': fn, 'fn': 0},
            'id_switch': {'true': 1, 'predicted': 0, 'matched': 0},
            'group_accuracy': {'frames_correct': tp, 'frames_scored': tp + fn},
            'coverage': {f'cam{i}': {'label_boxes': 10, 'matched_label_boxes': 8} for i in range(3)}}}
    pooled = dev.pooled([result(10, 0, 'ok'), result(0, 10, 'undecided')])
    assert pooled['decided_clips'] == 1 and pooled['total_clips'] == 2
    assert pooled['pairs'] == {'tp': 10, 'fp': 0, 'fn': 10, 'f1': 2 / 3}
    assert pooled['group_accuracy']['accuracy'] == .5
    assert pooled['coverage']['cam0']['label_boxes'] == 20


def test_existing_dev_receipt_stops_before_opening_labels(dev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, 'require_pushed', lambda *_: pytest.fail('must not enter another batch'))
    with pytest.raises(FileExistsError, match='one immutable'):
        dev.run(SimpleNamespace(report=tmp_path))
