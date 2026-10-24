"""CPU checks for resource boundaries and the six-condition controller."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def load(name: str) -> ModuleType:
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = load('session_guard')
pilot = load('pilot')
summary = load('summarize')
budget_projection = load('estimate_budget')


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(inputs=tmp_path / 'inputs', root=tmp_path / 'runs',
                              receipts=tmp_path / 'receipts', remote_root='gdrive:tennis_lab/unused',
                              created_at=datetime.now(UTC).isoformat())


def test_session_deadline_uses_allocation_not_job_start() -> None:
    created = datetime(2026, 10, 24, tzinfo=UTC)
    assert guard.elapsed(created.isoformat(), now=created + timedelta(hours=3, minutes=30)) == 12600
    with pytest.raises(ValueError, match='timezone'):
        guard.elapsed('2026-10-24T00:00:00')


def test_unmeasured_condition_never_gets_estimated_metrics() -> None:
    row = summary.measured_row(dict(condition='hybrid-T128', status='unmeasured', reason='reserve'), None)
    assert row['optimizer_updates'] is None
    assert row['warm_step_mean_seconds'] is None
    assert row['peak_allocated_bytes'] is None
    assert row['evaluation_seconds'] is None


def test_budget_refuses_to_fill_unmeasured_conditions(tmp_path: Path) -> None:
    records = [dict(condition=f'{a}-T{t}', status='unmeasured', reason='resource limit') for a, t in pilot.CONDITIONS]
    pilot.write(tmp_path / 'matrix.json', dict(conditions=records))
    with pytest.raises(ValueError, match='all six measured'):
        budget_projection.completed_measurements(tmp_path)


def test_completed_claim_requires_verified_checkpoint() -> None:
    result = dict(status='completed', completed_marker=True, latest=dict(update=10),
                  records=[dict(update=i) for i in range(1, 11)], publications=[])
    with pytest.raises(ValueError, match='verified checkpoint'):
        summary.measured_row(dict(condition='dpt-T32', status='completed', seconds=10), result)


def test_mislabeled_architecture_is_rejected() -> None:
    result = dict(status='completed', completed_marker=True, latest=dict(update=10),
                  records=[dict(update=i) for i in range(1, 11)], architecture='hybrid',
                  gpu=dict(name='NVIDIA L4'),
                  publications=[dict(stage='checkpoint', update=i, status='verified') for i in (5, 10)])
    with pytest.raises(ValueError, match='architecture'):
        summary.measured_row(dict(condition='dpt-T32', status='completed', seconds=10), result)


def test_no_condition_starts_in_stop_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, 'remaining', lambda _: 1199)
    monkeypatch.setattr(pilot.subprocess, 'Popen', lambda *_a, **_k: pytest.fail('must not start'))
    args = arguments(tmp_path)
    pilot.matrix(args)
    records = json.loads((args.receipts / 'matrix.json').read_text())['conditions']
    assert len(records) == 6
    assert all(r['status'] == 'unmeasured' for r in records)


@pytest.mark.parametrize('failed_code', [2, 3, 124])  # type: ignore[untyped-decorator]
def test_oom_timeout_and_storage_failure_have_distinct_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_code: int,
) -> None:
    args = arguments(tmp_path)
    started = []
    killed = []

    class Process:
        pid = 12345

        def __init__(self, argv: list[str], **_: Any) -> None:
            self.name = Path(argv[argv.index('--root') + 1]).name
            started.append(self.name)
            self.receipts = Path(argv[argv.index('--receipts') + 1])
            self.first = True

        def wait(self, *, timeout: float) -> int:
            assert timeout <= 1180
            if self.name == 'dpt-T32' and failed_code == 124 and self.first:
                self.first = False
                raise subprocess.TimeoutExpired('condition', timeout)
            if self.name == 'dpt-T32':
                code, status = failed_code, {2: 'oom', 3: 'storage_failed', 124: 'timeout'}[failed_code]
            else:
                code, status = 0, 'completed'
                pilot.write(args.root / self.name / 'COMPLETED.json', {})
            pilot.write(self.receipts / 'condition.json', dict(status=status))
            return code

    monkeypatch.setattr(pilot.subprocess, 'Popen', Process)
    monkeypatch.setattr(pilot.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    if failed_code == 3:
        with pytest.raises(RuntimeError, match='storage_failed'):
            pilot.matrix(args)
        assert started == ['dpt-T32']
    else:
        pilot.matrix(args)
        assert started == ['dpt-T32', 'hybrid-T32', 'hybrid-T64', 'hybrid-T128']
        records = json.loads((args.receipts / 'matrix.json').read_text())['conditions']
        assert [r['status'] for r in records if r['condition'] in ('dpt-T64', 'dpt-T128')] == ['unmeasured'] * 2
    assert bool(killed) == (failed_code == 124)
